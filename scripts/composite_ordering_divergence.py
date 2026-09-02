#!/usr/bin/env python
"""How much can the potential move the gradient AS THE T5 ARMS WERE CONFIGURED?

group_ordering_divergence.py compares Phi_v1 vs Phi_v2 orderings in isolation and
finds them genuinely different (argmax agrees 47% among groups with spread). But
that is not what trained. Two facts change the question:

  1. Shaping telescopes. sum_t gamma^t (gamma*Phi(s_{t+1}) - Phi(s_t))
     = gamma^T Phi(s_T) - Phi(s_0), and Phi(s_0) is a group constant that
     mean-centring deletes. See tests/test_shaping_invisibility.py. So the
     shaping term contributes ONLY terminal potential -- no per-step credit.

  2. The arms did not train on Phi alone. grpo_mt_t5b_phi2.yaml carries
     feasibility 1.0 + fos_improvement 0.5 + grammar 0.1 + step_efficiency 0.1
     ALONGSIDE the potential at 1.0, and those terms are functions of the same
     terminal quantities Phi is built from.

So the effect size of the T5a-vs-T5b A/B is bounded by how often the COMPOSITE
ordering differs, not the Phi ordering. This measures that, plus the more basic
question of whether adding the potential perturbs the composite ordering at all
relative to the base reward alone.

CPU only.
"""
from __future__ import annotations
import argparse, json, random, sys, statistics as st
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
from scripts.group_ordering_divergence import _kendall_tau  # noqa: E402

TARGET_FOS = 1.5


def _min_fos(s):
    vals = [s.get("fos_buckling"), s.get("fos_yielding")]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return min(vals) if vals else 0.0


def base_reward(initial, final, n_steps, max_steps, feasible):
    """The four non-potential terms, at the weights grpo_mt_t5b_phi2.yaml sets.

    grammar_compliance is omitted: these are scripted rollouts, so it is 1.0 for
    every member of the group and a group constant cannot change an ordering.
    """
    r = 1.0 * (1.0 if feasible else 0.0)
    delta = (_min_fos(final) - _min_fos(initial)) / TARGET_FOS
    r += 0.5 * max(-1.0, min(1.0, delta))
    frac = n_steps / max(max_steps, 1)
    r += 0.1 * ((1.0 - frac) if feasible else -0.1 * frac)
    return r


_M0_CACHE = {}


def _m0_of(spec, goals):
    pid = spec.get("problem_id")
    if pid not in _M0_CACHE:
        _M0_CACHE[pid] = fea(make_truss(spec), goals).get("mass", 1.0)
    return _M0_CACHE[pid]


def sample_trajectory(spec, program, bounds, goals, *, max_steps, top_m, rng, alpha, tau,
                      sampler="v2"):
    """sampler: which rule proposes the trajectory.

    'v2' biases the visited states toward the potential under test, which
    inflates its apparent ranking advantage. 'random' and 'v1' are the controls:
    the state distribution is then chosen by something other than the winner.
    """
    truss = make_truss(spec)
    state = fea(truss, goals)
    initial = dict(state)
    steps = 0
    for _ in range(max_steps):
        if program.is_feasible(state):
            break
        acts = candidate_actions(truss, bounds, macros=False,
                                 target_fos_b=program.limit_for("fos_buckling") or 1.5,
                                 target_fos_y=program.limit_for("fos_yielding") or 1.5)
        if not acts:
            break
        subset = rng.sample(acts, min(len(acts), 12))
        scored = []
        for a in subset:
            res = step(truss, goals, a, program)
            if res is None:
                continue
            nt, ns = res
            if sampler == "v1":
                sc = compute_potential(ns, initial_mass=_m0_of(spec, goals), alpha=alpha)
            elif sampler == "random":
                sc = rng.random()
            else:
                sc = compute_potential_v2(ns, program, alpha=alpha, tau=tau)
            scored.append((sc, nt, ns))
        if not scored:
            break
        scored.sort(key=lambda x: x[0], reverse=True)
        _, truss, state = scored[rng.randrange(min(top_m, len(scored)))]
        steps += 1
    return initial, state, steps


def run_problem(t):
    path, group_size, max_steps, alpha, tau, top_m, seed, gamma, sampler = t
    spec = json.load(open(path))
    if not (isinstance(spec, dict) and "topology" in spec):
        return None
    spec.setdefault("problem_id", Path(path).stem)
    goals = spec.get("goals", {}) or {}
    bounds = param_bounds(spec)
    rng = random.Random(seed)
    s0 = fea(make_truss(spec), goals)
    program = program_from_truss_spec(spec, initial_mass=s0.get("mass"))
    m0 = s0.get("mass", 1.0)

    rows = []
    for _ in range(group_size):
        try:
            init, final, nsteps = sample_trajectory(
                spec, program, bounds, goals, max_steps=max_steps,
                top_m=top_m, rng=rng, alpha=alpha, tau=tau, sampler=sampler)
        except Exception:
            continue
        rows.append((init, final, nsteps))
    if len(rows) < 3:
        return None

    feas = [bool(program.is_feasible(f)) for _, f, _ in rows]
    base = [base_reward(i, f, n, max_steps, ft) for (i, f, n), ft in zip(rows, feas)]
    # gamma^T * Phi(s_T); Phi(s_0) is a group constant and is dropped.
    v1 = [gamma**n * compute_potential(f, initial_mass=m0, alpha=alpha) for _, f, n in rows]
    v2 = [gamma**n * compute_potential_v2(f, program, alpha=alpha, tau=tau) for _, f, n in rows]
    comp1 = [b + p for b, p in zip(base, v1)]
    comp2 = [b + p for b, p in zip(base, v2)]
    am = lambda xs: max(range(len(xs)), key=lambda i: xs[i])
    return {
        "problem_id": spec["problem_id"], "k": len(rows), "n_feasible": sum(feas),
        # the actual A/B that ran: composite+Phi_v1 vs composite+Phi_v2
        "composite_argmax_agrees": int(am(comp1) == am(comp2)),
        "composite_tau": _kendall_tau(comp1, comp2),
        # does the potential perturb the ordering at ALL vs the base reward alone?
        "v2_vs_base_argmax_agrees": int(am(comp2) == am(base)),
        "v2_vs_base_tau": _kendall_tau(comp2, base),
        # Phi in isolation, for comparison with group_ordering_divergence.py
        "phi_only_argmax_agrees": int(am(v1) == am(v2)),
        "phi_only_tau": _kendall_tau(v1, v2),
        "v1_best_feasible": int(feas[am(v1)]),
        "v2_best_feasible": int(feas[am(v2)]),
        "sampler": sampler,
        "base_spread": max(base) - min(base),
        "phi2_spread": max(v2) - min(v2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(DESIGNBENCH / "data/problems"))
    ap.add_argument("--split-file", default=str(PROJECT / "data/splits/truss_v1_auto.json"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--top-m", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sampler", default="v2", choices=["v2", "v1", "random"],
                    help="which rule proposes trajectories; v1/random are the controls")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(PROJECT / "results/search_ladder/composite_ordering.jsonl"))
    a = ap.parse_args()

    split = json.load(open(a.split_file))
    ids = split[a.split] if isinstance(split.get(a.split), list) else []
    paths = [str(Path(a.problems) / f"{i}.json") for i in ids]
    paths = [p for p in paths if Path(p).exists()]
    tasks = [(p, a.group_size, a.max_steps, a.alpha, a.tau, a.top_m, 1000 * s + i,
              a.gamma, a.sampler)
             for s in range(a.seeds) for i, p in enumerate(paths)]
    print(f"{len(paths)} problems x {a.seeds} seeds = {len(tasks)} groups", flush=True)

    from multiprocessing import Pool
    out = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_problem, tasks, chunksize=1):
            if r:
                out.append(r)
                if len(out) % 25 == 0:
                    print(f"  {len(out)}/{len(tasks)}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    def summarise(tag, ak, tk):
        ag = [r[ak] for r in out]; ta = [r[tk] for r in out]
        print(f"  {tag:34s} argmax_agrees={sum(ag):4d}/{len(ag)} ({100*sum(ag)/max(len(ag),1):5.1f}%)  "
              f"mean_tau={st.mean(ta):.4f}")

    print(f"\n=== {len(out)} groups ===")
    summarise("composite v1 vs v2 (the A/B)", "composite_argmax_agrees", "composite_tau")
    summarise("composite+Phi2 vs base alone", "v2_vs_base_argmax_agrees", "v2_vs_base_tau")
    summarise("Phi alone v1 vs v2", "phi_only_argmax_agrees", "phi_only_tau")
    dec = [r for r in out if 0 < r["n_feasible"] < r["k"]]
    if dec:
        b1 = sum(r["v1_best_feasible"] for r in dec); b2 = sum(r["v2_best_feasible"] for r in dec)
        n01 = sum(1 for r in dec if not r["v1_best_feasible"] and r["v2_best_feasible"])
        n10 = sum(1 for r in dec if r["v1_best_feasible"] and not r["v2_best_feasible"])
        import math
        N = n01 + n10
        pval = (sum(math.comb(N, i) for i in range(min(n01, n10) + 1)) / 2 ** N * 2) if N else 1.0
        print(f"\n  DECISIVE groups (feasibility varies): {len(dec)}/{len(out)}")
        print(f"    Phi_v1 top pick feasible: {b1}/{len(dec)} = {b1/len(dec):.3f}")
        print(f"    Phi_v2 top pick feasible: {b2}/{len(dec)} = {b2/len(dec):.3f}")
        print(f"    discordant v2-only={n01} v1-only={n10}  McNemar p={min(pval,1.0):.4f}")
    print(f"\n  mean base spread = {st.mean([r['base_spread'] for r in out]):.4f}")
    print(f"  mean Phi2 spread = {st.mean([r['phi2_spread'] for r in out]):.4f}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
