#!/usr/bin/env python
"""How much can the potential possibly change GRPO's gradient?

TRL runs GRPO with scale_rewards='group': advantages are mean-centred and
std-normalised inside each group of K rollouts on the same problem. So the only
thing a reward function contributes is the ORDER it induces over those K
trajectories. Two potentials that order every group identically produce
identical training, whatever their formulas look like.

This measures that directly. For each problem it samples K trajectories the way
a stochastic policy would -- pick among the top-m candidate actions at each step
rather than always the argmax -- scores each finished trajectory under Phi_v1 and
Phi_v2, and compares the two orderings.

The result BOUNDS the achievable effect size of the T5a-vs-T5b and T5ao-vs-T5bo
arms. If the orderings agree almost always, a null result in training is
explained rather than mysterious, and the place to intervene is the reward's
resolution, not its formula.

CPU only.

    python scripts/group_ordering_divergence.py --group-size 8 --workers 16
"""

from __future__ import annotations

import argparse
import json
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
    compute_potential, compute_potential_v2, program_from_truss_spec,
)
from scripts.search_ladder import candidate_actions, fea, make_truss, param_bounds, step  # noqa: E402


def sample_trajectory(spec, program, bounds, goals, *, max_steps, top_m, rng, alpha, tau):
    """One stochastic rollout: choose among the top-m candidates by Phi_v2 each step."""
    truss = make_truss(spec)
    state = fea(truss, goals)
    initial = dict(state)
    for _ in range(max_steps):
        if program.is_feasible(state):
            break
        acts = candidate_actions(truss, bounds, macros=False,
                                 target_fos_b=program.limit_for("fos_buckling") or 1.5,
                                 target_fos_y=program.limit_for("fos_yielding") or 1.5)
        if not acts:
            break
        # A policy does not enumerate; sample a handful and take a good one.
        subset = rng.sample(acts, min(len(acts), 12))
        scored = []
        for a in subset:
            res = step(truss, goals, a, program)
            if res is None:
                continue
            nt, ns = res
            scored.append((compute_potential_v2(ns, program, alpha=alpha, tau=tau), nt, ns))
        if not scored:
            break
        scored.sort(key=lambda x: x[0], reverse=True)
        _, truss, state = scored[rng.randrange(min(top_m, len(scored)))]
    return initial, state


def _kendall_tau(a: list[float], b: list[float]) -> float:
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total else 1.0


def run_problem(args_tuple):
    path, group_size, max_steps, alpha, tau, top_m, seed = args_tuple
    spec = json.load(open(path))
    if not (isinstance(spec, dict) and "topology" in spec):
        return None
    spec.setdefault("problem_id", Path(path).stem)
    goals = spec.get("goals", {}) or {}
    bounds = param_bounds(spec)
    rng = random.Random(seed)
    base = make_truss(spec)
    s0 = fea(base, goals)
    program = program_from_truss_spec(spec, initial_mass=s0.get("mass"))
    m0 = s0.get("mass", 1.0)

    finals = []
    for _ in range(group_size):
        try:
            _, final = sample_trajectory(spec, program, bounds, goals, max_steps=max_steps,
                                         top_m=top_m, rng=rng, alpha=alpha, tau=tau)
        except Exception:
            continue
        finals.append(final)
    if len(finals) < 3:
        return None

    v1 = [compute_potential(f, initial_mass=m0, alpha=alpha) for f in finals]
    v2 = [compute_potential_v2(f, program, alpha=alpha, tau=tau) for f in finals]
    feas = [bool(program.is_feasible(f)) for f in finals]
    return {
        "problem_id": spec["problem_id"],
        "k": len(finals),
        "argmax_agrees": int(max(range(len(v1)), key=lambda i: v1[i])
                             == max(range(len(v2)), key=lambda i: v2[i])),
        "kendall_tau": _kendall_tau(v1, v2),
        "n_feasible": sum(feas),
        # Does the potential's top pick actually reach feasibility?
        "v1_best_feasible": int(feas[max(range(len(v1)), key=lambda i: v1[i])]),
        "v2_best_feasible": int(feas[max(range(len(v2)), key=lambda i: v2[i])]),
        "spread_v1": max(v1) - min(v1),
        "spread_v2": max(v2) - min(v2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(DESIGNBENCH / "data/problems"))
    ap.add_argument("--split-file", default=str(PROJECT / "data/splits/truss_v1_auto.json"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=5, help="matches rl.max_turns")
    ap.add_argument("--top-m", type=int, default=4, help="policy stochasticity")
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--repeats", type=int, default=4, help="independent groups per problem")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(PROJECT / "results/search_ladder/group_ordering.jsonl"))
    args = ap.parse_args()

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

    tasks = [(f, args.group_size, args.max_steps, args.alpha, args.tau, args.top_m, 1000 * r + i)
             for r in range(args.repeats) for i, f in enumerate(files)]
    print(f"{len(files)} problems x {args.repeats} groups of {args.group_size}")
    t0 = time.time()
    rows = []
    import multiprocessing as mp
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_problem, tasks), 1):
            if r:
                rows.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    import statistics as st
    n = len(rows)
    agree = sum(r["argmax_agrees"] for r in rows) / n
    taus = [r["kendall_tau"] for r in rows]
    informative = [r for r in rows if 0 < r["n_feasible"] < r["k"]]
    print(f"\n{n} groups scored -> {args.out}")
    print(f"  Phi_v1 and Phi_v2 pick the SAME best rollout : {agree:.1%}")
    print(f"  Kendall tau between the two orderings        : median {st.median(taus):.3f}, "
          f"mean {st.mean(taus):.3f}")
    print(f"  groups where tau < 0.5 (orderings differ a lot): "
          f"{sum(1 for t in taus if t < 0.5)/n:.1%}")
    if informative:
        a1 = sum(r["v1_best_feasible"] for r in informative) / len(informative)
        a2 = sum(r["v2_best_feasible"] for r in informative) / len(informative)
        print(f"\n  On the {len(informative)} groups with a MIX of feasible and infeasible rollouts")
        print(f"  (the only groups where the ordering can matter for feasibility):")
        print(f"    Phi_v1's top-ranked rollout is feasible : {a1:.1%}")
        print(f"    Phi_v2's top-ranked rollout is feasible : {a2:.1%}")
    print(f"\n  This bounds how different the T5a/T5b arms can be: identical orderings")
    print(f"  produce identical gradients under scale_rewards='group'.")


if __name__ == "__main__":
    main()
