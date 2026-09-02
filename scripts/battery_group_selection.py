#!/usr/bin/env python
"""Does the potential pick the good design out of a group -- on a NON-truss domain?

On truss, Phi_v2 identifies the feasible member of a group of K rollouts 98-100%
of the time across three independent trajectory samplers
(scripts/composite_ordering_divergence.py). If that is a property of the
potential rather than of trusses, it must reproduce on a domain whose simulator,
constraints and action grammar share nothing with truss FEA.

Phi_v1 is NOT the control here. It reads mass / fos_buckling / fos_yielding,
none of which a battery simulation produces, so it scores every battery state
identically -- comparing against it would be rigged. The honest control is
RANDOM selection from the same group, which is exactly what a policy without a
ranker does.

Reported: random-pick, Phi_v2-pick and oracle(any-feasible) feasibility over the
same K trajectories, paired per problem.

CPU only.
"""
from __future__ import annotations
import argparse, json, math, random, statistics as st, sys
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.battery_ladder import (  # noqa: E402
    candidate_actions, initial_params, load_problem, make_program, make_sim,
    param_bounds, phi_v2, step,
)


def sample_trajectory(sim, spec, program, params, bounds, *, max_steps, rng):
    """Neutral rollout: uniformly random legal action each step.

    Deliberately NOT potential-guided -- the state distribution must not be
    chosen by the thing under test.
    """
    state = sim.evaluate(params)
    obj0 = state.get(program.objective_key)
    for _ in range(max_steps):
        if program.is_feasible(state):
            break
        acts = candidate_actions(params, bounds)
        if not acts:
            break
        rng.shuffle(acts)
        moved = False
        for a in acts[:12]:
            res = step(sim, params, a)
            if res is not None:
                params, state = res
                moved = True
                break
        if not moved:
            break
    return params, state, obj0


def run_problem(t):
    path, group_size, max_steps, alpha, fidelity, use_scales, seed = t
    spec = load_problem(Path(path))
    if spec is None:
        return None
    pid = spec.get("problem_id", Path(path).stem)
    program = make_program(spec, use_scales)
    sim = make_sim(spec, fidelity)
    p0 = initial_params(spec, sim)
    bounds = param_bounds(spec)
    rng = random.Random(seed)

    finals = []
    for _ in range(group_size):
        try:
            _, state, obj0 = sample_trajectory(sim, spec, program, dict(p0), bounds,
                                               max_steps=max_steps, rng=rng)
        except Exception:
            continue
        finals.append((state, obj0))
    if len(finals) < 3:
        return None

    feas = [bool(program.is_feasible(s)) for s, _ in finals]
    phis = [phi_v2(s, program, o if o is not None else 1.0, alpha) for s, o in finals]
    pick_phi = max(range(len(phis)), key=lambda i: phis[i])
    pick_rand = rng.randrange(len(finals))
    return {
        "problem_id": pid, "k": len(finals), "n_feasible": sum(feas),
        "phi_pick_feasible": int(feas[pick_phi]),
        "random_pick_feasible": int(feas[pick_rand]),
        "oracle_feasible": int(any(feas)),
        "phi_spread": (max(phis) - min(phis)) if phis else 0.0,
        "family": spec.get("family", "battery"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(PROJECT / "data/battery_problems"))
    ap.add_argument("--out", default=str(PROJECT / "results/battery_ladder/group_selection.jsonl"))
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--fidelity", default="spme_lean")
    ap.add_argument("--scales", default="problem")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    paths = sorted(str(p) for p in Path(a.problems).glob("*.json"))
    use_scales = a.scales == "problem"
    tasks = [(p, a.group_size, a.max_steps, a.alpha, a.fidelity, use_scales, 1000 * s + i)
             for s in range(a.seeds) for i, p in enumerate(paths)]
    print(f"{len(paths)} battery problems x {a.seeds} seeds = {len(tasks)} groups", flush=True)

    from multiprocessing import Pool
    out = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_problem, tasks, chunksize=1):
            if r:
                out.append(r)
                if len(out) % 10 == 0:
                    print(f"  {len(out)}/{len(tasks)}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    dec = [r for r in out if 0 < r["n_feasible"] < r["k"]]
    print(f"\n=== {len(out)} groups, {len(dec)} decisive (feasibility varies) ===")
    if dec:
        pr = sum(r["random_pick_feasible"] for r in dec)
        pp = sum(r["phi_pick_feasible"] for r in dec)
        n01 = sum(1 for r in dec if not r["random_pick_feasible"] and r["phi_pick_feasible"])
        n10 = sum(1 for r in dec if r["random_pick_feasible"] and not r["phi_pick_feasible"])
        N = n01 + n10
        p = (sum(math.comb(N, i) for i in range(min(n01, n10) + 1)) / 2 ** N * 2) if N else 1.0
        print(f"  random pick feasible : {pr}/{len(dec)} = {pr/len(dec):.3f}")
        print(f"  Phi_v2 pick feasible : {pp}/{len(dec)} = {pp/len(dec):.3f}")
        print(f"  delta                : {(pp-pr)/len(dec):+.3f}")
        print(f"  discordant phi-only={n01} random-only={n10}  McNemar p={min(p,1.0):.4f}")
    print(f"  oracle (any feasible): {sum(r['oracle_feasible'] for r in out)}/{len(out)}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
