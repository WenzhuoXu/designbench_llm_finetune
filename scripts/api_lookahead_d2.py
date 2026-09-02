#!/usr/bin/env python
"""Does a SECOND ply of Phi-lookahead help an LLM design agent?

Everything run with the LLM so far is depth-1: propose K actions, simulate all K,
take argmax Phi. Procedural search gains +7.0pp from depth 2 over depth 1
(grammar-legal) and +6.8pp (macro-armed) on `problems_hard` -- consistent across
both action spaces, so it is a real depth effect. The LLM proposer has never been
given more than one ply.

This is pure inference: no gradient steps, no training data. It exploits the two
measured properties of Phi -- it ranks siblings near-perfectly (319/320 conversion),
and it is cheap because the simulator has already run.

Three arms, same 202 held-out problems, all using the WINNING interface from the
2x2 (per-member observation + compound actions, +18.3pp, p=4.8e-05):

  d1        K=8 proposals, argmax Phi.                      [the 0.629 baseline]
  d2        K=8 at ply 1; the top m by Phi are each expanded by asking the LLM for
            K2 follow-ups; a ply-1 action is scored by the best Phi reachable under
            it. Second ply is LLM-expanded, so the whole search stays ON-POLICY.
  d1_wide   K = 8 + m*K2 proposals in ONE shot, argmax Phi.

`d1_wide` is the control that matters. d2 spends 1 + m LLM calls and sees 8 + m*K2
candidates per turn; d1_wide sees exactly the same number of candidates in one call.
If d2 > d1_wide the gain is from DEPTH. If d2 ~ d1_wide it was only ever more
proposals, and the honest conclusion is that breadth, not lookahead, is what pays.
"""
from __future__ import annotations
import argparse, copy, json, math, os, random, re, sys, threading, time
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

import urllib.request, urllib.error
import ssl, socket, http.client, encodings.idna, hashlib, base64, email.utils  # noqa: F401
from llm_finetune.envs.truss_env import (
    _analyze_truss, _load_truss_and_goals, normalize_action, parse_grammar_action,
)
from llm_finetune.data.processors import designbench_prompt as DP
from llm_finetune.training.rl.posterior.potential import (
    compute_potential_v2, program_from_truss_spec,
)
# Reuse the exact observation and action machinery the 2x2 established, so the only
# thing that differs between that experiment and this one is search depth.
from api_grammar_2x2 import (
    member_table, COMPOUND_RULE, parse_candidates, apply_candidate, call, MACRO_SEP,
)

_lock = threading.Lock()


def propose(spec, initial, history, truss, k, model, region, token):
    """One LLM call: k candidate compound actions at the current state."""
    system = DP.TRUSS_SYSTEM_PROMPT + (
        f"\n\nEach turn, propose {k} DIFFERENT candidate moves rather than one. Think "
        f"briefly, then emit exactly {k} lines of the form <action>...</action>. They "
        f"must be genuinely different candidates -- only one will be executed."
    ) + COMPOUND_RULE
    convo = [{"role": "user", "content": [{"text":
              DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
              + DP.format_eval_result(initial) + member_table(truss)}]}]
    for h in history:
        convo.append({"role": "assistant", "content": [{"text": h["thinking"]}]})
        convo.append({"role": "user", "content": [{"text":
                      "[Simulation Result]\nSTRUCTURAL ANALYSIS RESULT:\n" + h["obs"]}]})
    text = call(convo, system, model, region, token)
    return text, parse_candidates(text, k, True)


def expand(truss, goals, program, cands, alpha, tau):
    """Simulate every candidate once; return [(action, truss', state', phi')]."""
    out = []
    for c in cands:
        nt = apply_candidate(truss, c)
        if nt is None:
            continue
        st = _analyze_truss(nt, goals)
        out.append((c, nt, st, compute_potential_v2(st, program, alpha=alpha, tau=tau)))
    return out


def run_episode(spec, arm, model, region, token, k=8, m=3, k2=8, max_steps=20,
                alpha=5.0, tau=0.05):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    program = program_from_truss_spec(spec, initial_mass=state.get("mass"))
    history, n_sim, n_llm, n_turns = [], 0, 0, 0
    wide = k + m * k2

    for _ in range(max_steps):
        obs = DP.format_eval_result(state) + member_table(truss)
        kk = wide if arm == "d1_wide" else k
        text, cands = propose(spec, initial, history, truss, kk, model, region, token)
        n_llm += 1
        if not cands:
            history.append({"thinking": f"No <action> parsed. Emit exactly {kk} "
                                        "<action>...</action> lines.", "obs": obs})
            continue
        n_turns += 1
        scored = expand(truss, goals, program, cands, alpha, tau)
        n_sim += len(scored)
        if not scored:
            history.append({"thinking": text.strip(), "obs": obs})
            continue

        feas = [x for x in scored if x[2].get("is_feasible") and program.is_valid(x[2])]
        if feas:
            _c, truss, state, _p = max(feas, key=lambda x: x[3])
            history.append({"thinking": text.strip(),
                            "obs": DP.format_eval_result(state) + member_table(truss)})
            break

        if arm == "d2":
            # Score each surviving ply-1 action by the best Phi reachable one more ply
            # down. A candidate is never punished for bad children: its own Phi is a
            # floor, since the agent is free not to follow a child.
            top = sorted(scored, key=lambda x: x[3], reverse=True)[:m]
            best, best_val = None, -float("inf")
            for (c1, t1, s1, p1) in top:
                hist2 = history + [{"thinking": text.strip(),
                                    "obs": DP.format_eval_result(s1) + member_table(t1)}]
                _t2, c2s = propose(spec, initial, hist2, t1, k2, model, region, token)
                n_llm += 1
                sc2 = expand(t1, goals, program, c2s, alpha, tau)
                n_sim += len(sc2)
                val = max([p1] + [x[3] for x in sc2])
                if any(x[2].get("is_feasible") and program.is_valid(x[2]) for x in sc2):
                    val = float("inf")     # a child reaches feasibility: take this branch
                if val > best_val:
                    best, best_val = (c1, t1, s1, p1), val
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
        "turns": n_turns, "n_simulator_calls": n_sim, "n_llm_calls": n_llm,
        "final_mass": state.get("mass"),
        "mass_ratio": (state.get("mass") / program.objective_ref) if program.objective_ref else None,
        "final_fos_b": state.get("fos_buckling"), "final_fos_y": state.get("fos_yielding"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-dir", default=str(DESIGNBENCH / "data/problems_hard"))
    ap.add_argument("--split-file", default=str(PROJECT / "data/splits/truss_hard_v1.json"))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n", type=int, default=202)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--k2", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="d1,d2,d1_wide")
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/lookahead_d2.jsonl"))
    a = ap.parse_args()

    region = os.environ.get("AWS_REGION", "us-west-2")
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not token:
        raise SystemExit("AWS_BEARER_TOKEN_BEDROCK not set in environment")

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
    print("%d problems x %s, k=%d m=%d k2=%d (d1_wide K=%d)"
          % (len(specs), ARMS, a.k, a.m, a.k2, a.k + a.m * a.k2), flush=True)

    t0 = time.time()
    _t, _g = _load_truss_and_goals(specs[0])
    _st = _analyze_truss(_t, _g)
    compute_potential_v2(_st, program_from_truss_spec(specs[0], initial_mass=_st.get("mass")))
    member_table(_t)
    apply_candidate(_t, "SCALE_PARAM(0, r, 1.05)")
    _p = call([{"role": "user", "content": [{"text": "Reply with OK."}]}],
              None, a.model, region, token, max_tokens=8)
    print("warmup %.1fs, api=%s" % (time.time() - t0, "ok" if _p.strip() else "FAILED"), flush=True)
    if not _p.strip():
        raise SystemExit("Bedrock probe empty -- aborting")

    jobs = [(s, arm) for s in specs for arm in ARMS]
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    fh = open(outp, "w")
    from concurrent.futures import ThreadPoolExecutor
    def work(j):
        s_, arm_ = j
        t = time.time()
        try:
            r = run_episode(s_, arm_, a.model, region, token, k=a.k, m=a.m, k2=a.k2,
                            max_steps=a.max_steps)
        except Exception as e:
            r = {"problem_id": s_.get("problem_id"), "arm": arm_, "error": str(e)[:200]}
        with _lock:
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print("    [%5.1fs] %-20s %-8s feasible=%s turns=%s llm=%s sims=%s %s"
                  % (time.time() - t, s_.get("problem_id"), arm_, r.get("feasible"),
                     r.get("turns"), r.get("n_llm_calls"), r.get("n_simulator_calls"),
                     r.get("error", "")), flush=True)
        return r
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, _ in enumerate(ex.map(work, jobs), 1):
            if i % 50 == 0: print("  %d/%d" % (i, len(jobs)), flush=True)
    fh.close()
    print("wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
