#!/usr/bin/env python
"""Cutoff-and-restart, measured rather than projected.

Trace and run evidence:
  * every FAILED episode runs the full 20 turns; solved episodes have median 4
    (p75 10, p90 15). The turn budget is spent almost entirely on episodes already
    lost.
  * 8 independent runs over the same 195 problems: mean single run 0.649, UNION
    0.938, only 12/195 never solved by any run. 60.5% of problems are solved by
    some runs and not others -- failure is stochastic, not structural.
  * every attempt to make a single episode smarter (K=4..32, depth 2/3, incumbent
    floor, regression feedback) was null or negative against the baseline.

So: cap an attempt at C turns and start over instead of grinding to 20. The
projection from run statistics is 0.855 at C=10 x 3 attempts (21 turns) vs a 0.613
baseline at 11.5 turns, and 0.680 at C=8 x 2 attempts (11.8 turns) -- better at the
SAME cost. Those assume independent attempts, which they are not: the empirical
ceiling is the 0.938 union above. This measures it directly.

Arms are (cutoff, attempts). Each attempt is a fresh episode from the initial design
with an independent sample; the run stops at the first feasible one. Reported cost is
total turns and total simulator calls across attempts, so arms are comparable on
budget rather than on episode count.
"""
from __future__ import annotations
import argparse, json, os, random, sys, threading, time
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
from api_grammar_2x2 import member_table, apply_candidate, call
from api_lookahead_d2 import propose, expand

_lock = threading.Lock()


def one_attempt(spec, model, region, token, k, max_turns, alpha, tau):
    """A single episode, capped at max_turns. Returns (feasible, turns, sims, best_phi)."""
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    program = program_from_truss_spec(spec, initial_mass=state.get("mass"))
    history, n_turns, n_sim = [], 0, 0
    best_phi = compute_potential_v2(state, program, alpha=alpha, tau=tau)

    for _ in range(max_turns):
        text, cands = propose(spec, initial, history, truss, k, model, region, token)
        if not cands:
            history.append({"thinking": f"No <action> parsed. Emit exactly {k} "
                                        "<action>...</action> lines.",
                            "obs": DP.format_eval_result(state) + member_table(truss)})
            continue
        n_turns += 1
        scored = expand(truss, goals, program, cands, alpha, tau)
        n_sim += len(scored)
        if not scored:
            history.append({"thinking": text.strip(),
                            "obs": DP.format_eval_result(state) + member_table(truss)})
            continue
        feas = [x for x in scored if x[2].get("is_feasible") and program.is_valid(x[2])]
        pick = max(feas, key=lambda x: x[3]) if feas else max(scored, key=lambda x: x[3])
        _c, truss, state, phi = pick
        best_phi = max(best_phi, phi)
        history.append({"thinking": text.strip(),
                        "obs": DP.format_eval_result(state) + member_table(truss)})
        if state.get("is_feasible") and program.is_valid(state):
            return True, n_turns, n_sim, float(best_phi)
    return False, n_turns, n_sim, float(best_phi)


def run_problem(spec, arm, model, region, token, k=8, alpha=5.0, tau=0.05):
    cutoff, attempts = arm
    tot_turns = tot_sims = 0
    best_phi = -float("inf")
    for i in range(attempts):
        ok, t, s, bp = one_attempt(spec, model, region, token, k, cutoff, alpha, tau)
        tot_turns += t; tot_sims += s; best_phi = max(best_phi, bp)
        if ok:
            return {"problem_id": spec.get("problem_id"), "arm": "c%d_a%d" % arm,
                    "feasible": True, "attempts_used": i + 1,
                    "turns": tot_turns, "n_simulator_calls": tot_sims,
                    "best_phi": best_phi}
    return {"problem_id": spec.get("problem_id"), "arm": "c%d_a%d" % arm,
            "feasible": False, "attempts_used": attempts,
            "turns": tot_turns, "n_simulator_calls": tot_sims, "best_phi": best_phi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-dir", default=str(DESIGNBENCH / "data/problems_hard"))
    ap.add_argument("--split-file", default=str(PROJECT / "data/splits/truss_hard_v1.json"))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n", type=int, default=202)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    # baseline 20x1 is the current default; 8x2 matches its turn budget; 10x3 buys more
    ap.add_argument("--arms", default="20:1,8:2,10:3")
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/restart.jsonl"))
    ap.add_argument("--resume-from", default="", help="jsonl of finished (problem,arm) rows to skip")
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
    ARMS = [tuple(int(x) for x in t.split(":")) for t in a.arms.split(",") if t]
    print("%d problems x %s  k=%d" % (len(specs), ARMS, a.k), flush=True)

    t0 = time.time()
    _t, _g = _load_truss_and_goals(specs[0]); _st = _analyze_truss(_t, _g)
    compute_potential_v2(_st, program_from_truss_spec(specs[0], initial_mass=_st.get("mass")))
    member_table(_t); apply_candidate(_t, "SCALE_PARAM(0, r, 1.05)")
    _p = call([{"role": "user", "content": [{"text": "Reply with OK."}]}],
              None, a.model, region, token, max_tokens=8)
    print("warmup %.1fs api=%s" % (time.time()-t0, "ok" if _p.strip() else "FAILED"), flush=True)
    if not _p.strip():
        raise SystemExit("Bedrock probe empty")

    done, carried = set(), []
    if a.resume_from and Path(a.resume_from).exists():
        for line in open(a.resume_from):
            try: r = json.loads(line)
            except Exception: continue
            if "feasible" in r:
                done.add((r["problem_id"], r["arm"])); carried.append(r)
        print("resuming: %d rows carried" % len(carried), flush=True)
    jobs = [(s, arm) for s in specs for arm in ARMS
            if (s.get("problem_id"), "c%d_a%d" % arm) not in done]
    print("%d jobs to run" % len(jobs), flush=True)
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    fh = open(outp, "w")
    for r in carried:
        fh.write(json.dumps(r) + "\n")
    fh.flush()
    from concurrent.futures import ThreadPoolExecutor
    def work(j):
        s_, arm_ = j
        t = time.time()
        try:
            r = run_problem(s_, arm_, a.model, region, token, k=a.k)
        except Exception as e:
            r = {"problem_id": s_.get("problem_id"), "arm": "c%d_a%d" % arm_,
                 "error": str(e)[:200]}
        with _lock:
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print("    [%5.1fs] %-20s %-8s feasible=%s attempts=%s turns=%s %s"
                  % (time.time()-t, s_.get("problem_id"), r.get("arm"), r.get("feasible"),
                     r.get("attempts_used"), r.get("turns"), r.get("error","")), flush=True)
        return r
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, _ in enumerate(ex.map(work, jobs), 1):
            if i % 50 == 0: print("  %d/%d" % (i, len(jobs)), flush=True)
    fh.close()
    print("wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
