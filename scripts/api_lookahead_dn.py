#!/usr/bin/env python
"""Does Phi-lookahead keep paying past two plies? Depth ladder on truss.

Measured (scripts/api_lookahead_d2.py, n=199): d2 0.693 vs budget-matched d1_wide
0.583, +11.1pp, p=0.0021. The control carried it -- d1_wide sees the same 8+3*8
candidates at ONE ply and scored WORSE than plain d1 (0.633), so the margin is the
ply, not the candidate count.

One depth comparison is a point. A dose-response over d1 < d2 < d3 is a curve, and it
is what would make "search depth over Phi pays" a claim rather than an observation.

Generalised to arbitrary depth. At each turn:
  ply 1      : ask the LLM for k candidates, simulate all
  ply 2..D   : take the top m surviving branches by Phi, ask the LLM for k2 follow-ups
               at each, simulate those
  score      : a ply-1 action is worth the best Phi reachable anywhere beneath it; its
               own Phi is a floor, and a feasible descendant makes the branch infinite

LLM calls per turn are 1 + m + m^2 + ... for depth D, so m is kept small (2) to make
depth 3 affordable. The budget control `d1_wide` asks for all of them in one call.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, threading, time
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import urllib.request, urllib.error
import ssl, socket, http.client, encodings.idna, hashlib, base64, email.utils  # noqa: F401
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals
from llm_finetune.data.processors import designbench_prompt as DP
from llm_finetune.training.rl.posterior.potential import (
    compute_potential_v2, program_from_truss_spec,
)
from api_grammar_2x2 import member_table, COMPOUND_RULE, parse_candidates, apply_candidate, call
from api_lookahead_d2 import propose, expand

_lock = threading.Lock()
_trace_fh = None


def branch_value(spec, initial, hist, truss, state, phi, goals, program,
                 depth, m, k2, model, region, token, alpha, tau, counters):
    """Best Phi reachable from this state within `depth` further plies."""
    if depth <= 0:
        return phi
    _t, cands = propose(spec, initial, hist, truss, k2, model, region, token)
    counters["llm"] += 1
    sc = expand(truss, goals, program, cands, alpha, tau)
    counters["sim"] += len(sc)
    if not sc:
        return phi
    if any(s.get("is_feasible") and program.is_valid(s) for _, _, s, _ in sc):
        return float("inf")
    best = phi
    for (c, t1, s1, p1) in sorted(sc, key=lambda x: x[3], reverse=True)[:m]:
        h2 = hist + [{"thinking": _t.strip(),
                      "obs": DP.format_eval_result(s1) + member_table(t1)}]
        v = branch_value(spec, initial, h2, t1, s1, p1, goals, program,
                         depth - 1, m, k2, model, region, token, alpha, tau, counters)
        if v > best:
            best = v
    return best


def run_episode(spec, arm, model, region, token, k=8, m=2, k2=8, max_steps=20,
                alpha=5.0, tau=0.05, trace=None):
    depth = {"d1": 0, "d2": 1, "d3": 2}.get(arm, 0)
    wide = k + m * k2 + m * m * k2
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    program = program_from_truss_spec(spec, initial_mass=state.get("mass"))
    history, n_turns = [], 0
    counters = {"llm": 0, "sim": 0}

    for _ in range(max_steps):
        kk = wide if arm == "d1_wide" else k
        text, cands = propose(spec, initial, history, truss, kk, model, region, token)
        counters["llm"] += 1
        if not cands:
            history.append({"thinking": f"No <action> parsed. Emit exactly {kk} "
                                        "<action>...</action> lines.",
                            "obs": DP.format_eval_result(state) + member_table(truss)})
            continue
        n_turns += 1
        scored = expand(truss, goals, program, cands, alpha, tau)
        counters["sim"] += len(scored)
        if not scored:
            history.append({"thinking": text.strip(),
                            "obs": DP.format_eval_result(state) + member_table(truss)})
            continue

        if trace is not None:
            trace.append({
                "problem_id": spec.get("problem_id"), "arm": arm, "turn": len(history),
                "raw": text,
                "state_before": {"mass": state.get("mass"),
                                 "fos_b": state.get("fos_buckling"),
                                 "fos_y": state.get("fos_yielding")},
                "candidates": [{"action": c, "phi": float(ph),
                                "mass": st.get("mass"),
                                "fos_b": st.get("fos_buckling"),
                                "fos_y": st.get("fos_yielding"),
                                "feasible": bool(st.get("is_feasible") and program.is_valid(st))}
                               for (c, _t, st, ph) in scored],
            })
        feas = [x for x in scored if x[2].get("is_feasible") and program.is_valid(x[2])]
        if feas:
            _c, truss, state, _p = max(feas, key=lambda x: x[3])
            history.append({"thinking": text.strip(),
                            "obs": DP.format_eval_result(state) + member_table(truss)})
            break

        if depth > 0:
            top = sorted(scored, key=lambda x: x[3], reverse=True)[:m]
            best, best_val = None, -float("inf")
            for (c1, t1, s1, p1) in top:
                h2 = history + [{"thinking": text.strip(),
                                 "obs": DP.format_eval_result(s1) + member_table(t1)}]
                v = branch_value(spec, initial, h2, t1, s1, p1, goals, program,
                                 depth, m, k2, model, region, token, alpha, tau, counters)
                if v > best_val:
                    best, best_val = (c1, t1, s1, p1), v
            chosen = best
        else:
            chosen = max(scored, key=lambda x: x[3])

        _c, truss, state, _p = chosen
        history.append({"thinking": text.strip(),
                        "obs": DP.format_eval_result(state) + member_table(truss)})
        if state.get("is_feasible") and program.is_valid(state):
            break

    return {
        "problem_id": spec.get("problem_id"), "arm": arm,
        "feasible": bool(state.get("is_feasible")) and bool(program.is_valid(state)),
        "turns": n_turns, "n_llm_calls": counters["llm"], "n_simulator_calls": counters["sim"],
        "mass_ratio": (state.get("mass") / program.objective_ref) if program.objective_ref else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-dir", default=str(DESIGNBENCH / "data/problems_hard"))
    ap.add_argument("--split-file", default=str(PROJECT / "data/splits/truss_hard_v1.json"))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n", type=int, default=202)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--k2", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="d1,d2,d3,d1_wide")
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/lookahead_dn.jsonl"))
    ap.add_argument("--trace-out", default="", help="jsonl of per-turn raw model text + all candidates")
    a = ap.parse_args()

    region = os.environ.get("AWS_REGION", "us-west-2")
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not token:
        raise SystemExit("AWS_BEARER_TOKEN_BEDROCK not set")

    wanted = set(json.load(open(a.split_file))[a.split]) if a.split_file else None
    specs = []
    for f in sorted(Path(a.problems_dir).glob("*.json")):
        try: d = json.load(open(f))
        except Exception: continue
        if not isinstance(d, dict) or "topology" not in d: continue
        d.setdefault("problem_id", f.stem)
        if wanted is None or d["problem_id"] in wanted:
            specs.append(d)
    random.Random(a.seed).shuffle(specs)
    specs = specs[: a.n]
    ARMS = [x for x in a.arms.split(",") if x]
    print("%d problems x %s  k=%d m=%d k2=%d  (d1_wide K=%d)"
          % (len(specs), ARMS, a.k, a.m, a.k2, a.k + a.m*a.k2 + a.m*a.m*a.k2), flush=True)

    t0 = time.time()
    _t, _g = _load_truss_and_goals(specs[0]); _st = _analyze_truss(_t, _g)
    compute_potential_v2(_st, program_from_truss_spec(specs[0], initial_mass=_st.get("mass")))
    member_table(_t); apply_candidate(_t, "SCALE_PARAM(0, r, 1.05)")
    _p = call([{"role": "user", "content": [{"text": "Reply with OK."}]}],
              None, a.model, region, token, max_tokens=8)
    print("warmup %.1fs api=%s" % (time.time()-t0, "ok" if _p.strip() else "FAILED"), flush=True)
    if not _p.strip():
        raise SystemExit("Bedrock probe empty")

    jobs = [(s, arm) for s in specs for arm in ARMS]
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    fh = open(outp, "w")
    global _trace_fh
    _trace_fh = open(a.trace_out, "w") if a.trace_out else None
    from concurrent.futures import ThreadPoolExecutor
    def work(j):
        s_, arm_ = j
        t = time.time()
        tr = [] if a.trace_out else None
        try:
            r = run_episode(s_, arm_, a.model, region, token, k=a.k, m=a.m, k2=a.k2,
                            max_steps=a.max_steps, trace=tr)
        except Exception as e:
            r = {"problem_id": s_.get("problem_id"), "arm": arm_, "error": str(e)[:200]}
        with _lock:
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if tr and _trace_fh is not None:
                for rec in tr:
                    _trace_fh.write(json.dumps(rec) + "\n")
                _trace_fh.flush()
            print("    [%6.1fs] %-20s %-8s feasible=%s llm=%s sims=%s %s"
                  % (time.time()-t, s_.get("problem_id"), arm_, r.get("feasible"),
                     r.get("n_llm_calls"), r.get("n_simulator_calls"), r.get("error","")),
                  flush=True)
        return r
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, _ in enumerate(ex.map(work, jobs), 1):
            if i % 50 == 0: print("  %d/%d" % (i, len(jobs)), flush=True)
    fh.close()
    print("wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
