#!/usr/bin/env python
"""Does depth-2 Phi-lookahead transfer to batteries?

On truss (n=199): d2 0.693 vs budget-matched d1_wide 0.583, +11.1pp, p=0.0021.
The control mattered -- d1_wide sees the same 8+3*8 candidates at ONE ply and scores
worse than plain d1, so the margin is the second ply, not the extra candidates.

Same three arms here, on PyBaMM. Battery has no closed-form macro anywhere in the
domain and a completely different simulator and objective, so a positive result makes
"depth over Phi pays" a claim about design search rather than about trusses.

Arms are identical in structure to scripts/api_lookahead_d2.py:
  d1        K candidates, simulate all, argmax Phi
  d2        K at ply 1; top m by Phi each expanded by a fresh LLM call for k2
            follow-ups; a ply-1 action scored by the best Phi reachable beneath it
  d1_wide   K + m*k2 candidates in ONE call, argmax Phi   [budget control]

Battery episodes use the settings that measured best on this domain.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import urllib.request, urllib.error
import ssl, socket, http.client, encodings.idna, hashlib, base64, email.utils  # noqa: F401

from api_battery_2x2 import (
    sim, call, SYSTEM, obs_text, parse_candidates, apply_candidate, in_bounds,
)

_SEP = " ; "
import threading
_lock = threading.Lock()


def propose(pid, program, obs0, history, k, model, region, token):
    system = SYSTEM
    convo = [{"role": "user", "content": [{"text":
              "PROBLEM: %s\nMinimise %s subject to the constraints below.\n\n"
              "INITIAL STATE:\n%s" % (pid, program.objective_key, obs0)}]}]
    for h in history:
        convo.append({"role": "assistant", "content": [{"text": h["thinking"]}]})
        convo.append({"role": "user", "content": [{"text": "[Simulation Result]\n" + h["obs"]}]})
    convo[-1]["content"][0]["text"] += (
        "\n\nPropose %d DIFFERENT candidate moves, each as <action>...</action>." % k)
    text = call(convo, system, model, region, token)
    return text, parse_candidates(text, k, False)


def expand(s, program, params, bounds, cands, alpha):
    from llm_finetune.training.rl.posterior.potential import compute_potential_v2
    out = []
    for c in cands:
        np_ = apply_candidate(params, c)
        if np_ is None or not in_bounds(np_, bounds):
            continue
        st = s.evaluate(np_)
        out.append((c, np_, st, compute_potential_v2(st, program, alpha=alpha)))
    return out


def run_episode(job):
    (path, arm, model, region, token, k, m, k2, max_steps, alpha) = job
    from battery_ladder import load_problem, param_bounds, initial_params, make_program
    t0 = time.time()
    spec = load_problem(Path(path))
    pid = spec.get("problem_id", Path(path).stem)
    try:
        s = sim()
        bounds = param_bounds(spec)
        params = initial_params(spec, s)
        program = make_program(spec, use_scales=True)
        state = s.evaluate(params)
        obs0 = obs_text(program, state, params, bounds, "full")
        history, n_sim, n_llm, n_turns = [], 0, 0, 0
        wide = k + m * k2

        for _ in range(max_steps):
            kk = wide if arm == "d1_wide" else k
            text, cands = propose(pid, program, obs0, history, kk, model, region, token)
            n_llm += 1
            if not cands:
                history.append({"thinking": "No <action> parsed.",
                                "obs": obs_text(program, state, params, bounds, "full")})
                continue
            n_turns += 1
            scored = expand(s, program, params, bounds, cands, alpha)
            n_sim += len(scored)
            if not scored:
                history.append({"thinking": text.strip(),
                                "obs": obs_text(program, state, params, bounds, "full")})
                continue

            feas = [x for x in scored if program.is_feasible(x[2])]
            if feas:
                _c, params, state, _p = max(feas, key=lambda x: x[3])
                break

            if arm == "d2":
                top = sorted(scored, key=lambda x: x[3], reverse=True)[:m]
                best, best_val = None, -float("inf")
                for (c1, p1_, s1, phi1) in top:
                    h2 = history + [{"thinking": text.strip(),
                                     "obs": obs_text(program, s1, p1_, bounds, "full")}]
                    _t2, c2s = propose(pid, program, obs0, h2, k2, model, region, token)
                    n_llm += 1
                    sc2 = expand(s, program, p1_, bounds, c2s, alpha)
                    n_sim += len(sc2)
                    val = max([phi1] + [x[3] for x in sc2])
                    if any(program.is_feasible(x[2]) for x in sc2):
                        val = float("inf")
                    if val > best_val:
                        best, best_val = (c1, p1_, s1, phi1), val
                chosen = best
            else:
                chosen = max(scored, key=lambda x: x[3])

            _c, params, state, _p = chosen
            history.append({"thinking": text.strip(),
                            "obs": obs_text(program, state, params, bounds, "full")})
            if program.is_feasible(state):
                break

        return {"problem_id": pid, "arm": arm,
                "feasible": bool(program.is_feasible(state)),
                "turns": n_turns, "n_llm_calls": n_llm, "n_simulator_calls": n_sim,
                "objective": float(state.get(program.objective_key, float("nan"))),
                "wall_s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"problem_id": pid, "arm": arm,
                "error": "%s: %s" % (type(e).__name__, str(e)[:180])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-dir", default="data/battery_grid_q015")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--k2", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--arms", default="d1,d2,d1_wide")
    ap.add_argument("--out", default="results/api_guidance/battery_d2.jsonl")
    a = ap.parse_args()

    region = os.environ.get("AWS_REGION", "us-west-2")
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not token:
        raise SystemExit("AWS_BEARER_TOKEN_BEDROCK not set")

    files = sorted(Path(a.problems_dir).glob("*.json"))[: a.n]
    ARMS = [x for x in a.arms.split(",") if x]
    print("%d problems x %s  k=%d m=%d k2=%d (d1_wide K=%d)"
          % (len(files), ARMS, a.k, a.m, a.k2, a.k + a.m * a.k2), flush=True)

    jobs = [(str(f), arm, a.model, region, token, a.k, a.m, a.k2, a.max_steps, a.alpha)
            for f in files for arm in ARMS]
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    fh = open(outp, "w")
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_episode, jobs), 1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print("  [%3d/%d %5.0fs] %-20s %-8s feasible=%s llm=%s sims=%s %s"
                  % (i, len(jobs), time.time() - t0, r.get("problem_id"), r.get("arm"),
                     r.get("feasible"), r.get("n_llm_calls"), r.get("n_simulator_calls"),
                     r.get("error", "")), flush=True)
    fh.close()
    print("wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
