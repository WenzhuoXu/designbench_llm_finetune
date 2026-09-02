#!/usr/bin/env python
"""How good must a policy's ranking be before it designs well?

The gap this measures. The search ranks candidates PERFECTLY (it simulates each
one) and reaches 73.8% feasible at 0.644 mass. A policy must PREDICT that ranking
instead, and errors compound over ~14 turns. Preference training reached 0.744
pairwise accuracy on held-out problems -- but the warmstart reaches 0.606 pairwise
and is 0% feasible, while the champion is 0.548 pairwise and 61.8% feasible. So
pairwise accuracy plainly does not map monotonically onto task success across
those policies, and nobody has measured what the map actually is.

This measures it directly, holding everything else fixed. Take the procedural
search and corrupt ONLY its ranking: score candidates by Phi + Normal(0, sigma *
spread) instead of Phi. Sweeping sigma traces a family of choosers from perfect
(sigma = 0) to random (sigma large). For each one, report BOTH

  * the induced pairwise accuracy on same-move comparisons -- the identical
    quantity scripts/eval_preference_margin.py measures on a policy; and
  * the task outcome (feasibility, mass) that chooser achieves.

The result is the link function: what pairwise ranking accuracy buys what design
quality. It says whether a policy at 0.744 could ever reach the search's outcome,
and therefore whether more preference data is worth collecting or whether the
ceiling is somewhere else entirely.

CPU only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm_finetune.training.rl.posterior.potential import (  # noqa: E402
    compute_potential_v2, program_from_truss_spec,
)
from scripts.search_ladder import (  # noqa: E402
    candidate_actions, fea, make_truss, param_bounds, step,
)


def _move_sig(action: str):
    """(type, parameter, factor) -- everything except which element is touched."""
    import re
    kind = action.split("(", 1)[0].strip()
    m = re.search(r"[:,]\s*([0-9.]+)\s*\]?\)\s*$", action.strip())
    p = re.search(r"[,\[]\s*([rt])\s*[,:]", action)
    return (kind, p.group(1) if p else "?", round(float(m.group(1)), 4) if m else None)


def run_problem(args_tuple):
    path, sigmas, max_steps, alpha, tau, seed, min_gap = args_tuple
    spec = json.load(open(path))
    if not (isinstance(spec, dict) and "topology" in spec):
        return None
    spec.setdefault("problem_id", Path(path).stem)
    goals = spec.get("goals", {}) or {}
    bounds = param_bounds(spec)
    base = make_truss(spec)
    s0 = fea(base, goals)
    program = program_from_truss_spec(spec, initial_mass=s0.get("mass"))
    out = []

    for sigma in sigmas:
        rng = random.Random(hash((spec["problem_id"], sigma, seed)) & 0xFFFFFFFF)
        truss = copy.deepcopy(base)
        state = dict(s0)
        best = (state["mass"], state) if program.is_feasible(state) else None
        agree_num = agree_den = 0

        for _ in range(max_steps):
            acts = candidate_actions(truss, bounds, macros=False,
                                     target_fos_b=program.limit_for("fos_buckling") or 1.5,
                                     target_fos_y=program.limit_for("fos_yielding") or 1.5)
            scored = []
            for a in acts:
                res = step(truss, goals, a, program)
                if res is None:
                    continue
                nt, ns = res
                scored.append((compute_potential_v2(ns, program, alpha=alpha, tau=tau), a, nt, ns))
            if len(scored) < 2:
                break
            phis = [s[0] for s in scored]
            spread = (max(phis) - min(phis)) or 1.0
            noisy = [(p + rng.gauss(0.0, sigma * spread), p, a, nt, ns)
                     for (p, a, nt, ns) in scored]

            # Pairwise accuracy on SAME-MOVE comparisons, the identical quantity
            # eval_preference_margin.py measures on a policy.
            by_move = {}
            for np_, tp, a, _, _ in noisy:
                by_move.setdefault(_move_sig(a), []).append((tp, np_))
            for group in by_move.values():
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        (t1, n1), (t2, n2) = group[i], group[j]
                        # Only comparisons the POLICY was scored on are comparable.
                        # scripts/distill_preference_pairs.py keeps a pair only when
                        # the potential gap exceeds min_gap, so near-ties -- which no
                        # ranker can be expected to resolve and which would drag this
                        # accuracy down -- are excluded there and must be here too.
                        if abs(t1 - t2) <= min_gap:
                            continue
                        agree_den += 1
                        if (t1 > t2) == (n1 > n2):
                            agree_num += 1

            noisy.sort(key=lambda x: x[0], reverse=True)
            _, _, _, truss, state = noisy[0]
            if program.is_feasible(state) and (best is None or state["mass"] < best[0]):
                best = (state["mass"], state)

        final = best[1] if best else state
        out.append({
            "problem_id": spec["problem_id"], "sigma": sigma,
            "feasible": best is not None,
            "mass_ratio": (final.get("mass") / program.objective_ref)
                          if best and program.objective_ref else None,
            "pairwise_accuracy": (agree_num / agree_den) if agree_den else None,
            "n_comparisons": agree_den,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(DESIGNBENCH / "data/problems"))
    ap.add_argument("--split-file", default=str(PROJECT / "data/splits/truss_v1_auto.json"))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--sigmas", default="0,0.02,0.05,0.1,0.2,0.35,0.6,1.0,2.0,5.0")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--min-gap", type=float, default=0.01,
                    help="Match distill_preference_pairs.py so the accuracy scale is comparable.")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(PROJECT / "results/search_ladder/ranking_link.jsonl"))
    args = ap.parse_args()

    sigmas = [float(x) for x in args.sigmas.split(",")]
    wanted = set(json.load(open(args.split_file))[args.split]) if args.split_file else None
    files, seen = [], set()
    for f in sorted(Path(args.problems).glob("*.json")):
        try:
            spec = json.load(open(f))
        except Exception:
            continue
        if not (isinstance(spec, dict) and "topology" in spec):
            continue
        pid = spec.get("problem_id", f.stem)
        if pid in seen or (wanted is not None and pid not in wanted):
            continue
        seen.add(pid)
        files.append(str(f))
    print(f"{len(files)} problems x {len(sigmas)} noise levels")

    tasks = [(f, sigmas, args.max_steps, args.alpha, args.tau, 0, args.min_gap) for f in files]
    t0 = time.time()
    rows = []
    import multiprocessing as mp
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, got in enumerate(pool.imap_unordered(run_problem, tasks), 1):
            if got:
                rows.extend(got)
            if i % 10 == 0:
                print(f"  {i}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    import statistics as st
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[r["sigma"]].append(r)
    print(f"\nLINK FUNCTION: ranking accuracy -> design outcome  ({time.time()-t0:.0f}s)")
    print(f"  {'sigma':>7}{'pairwise acc':>14}{'feasible':>10}{'median mass/ref':>17}")
    for s in sorted(by):
        v = by[s]
        accs = [r["pairwise_accuracy"] for r in v if r["pairwise_accuracy"] is not None]
        feas = [r for r in v if r["feasible"]]
        mr = [r["mass_ratio"] for r in feas if r["mass_ratio"]]
        print(f"  {s:>7.2f}{(st.mean(accs) if accs else float('nan')):>14.3f}"
              f"{len(feas)/len(v):>10.3f}{(st.median(mr) if mr else float('nan')):>17.3f}")
    print("\n  Measured policy pairwise accuracies on the SAME comparisons:")
    print("    DPO on search preferences 0.744 | warmstart 0.606 | T6 SFT 0.598")
    print("    champion 0.548 | best GRPO arm 0.562 | chance 0.500")


if __name__ == "__main__":
    main()
