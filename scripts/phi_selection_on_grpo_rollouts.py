#!/usr/bin/env python
"""Does Phi_v2 pick the feasible rollout out of REAL GRPO groups?

The eval harness answered this on 34 held-out problems with the champion policy
(29/29 oracle recovery). This is the same question against every GRPO group this
project ever logged: 43 runs x 10 problems x K=8, with ground-truth feasibility
from the simulator, at zero GPU cost.

Terminal state is reconstructed as initial + delta (the tables log deltas), and
`feasible` is the simulator's own verdict, not a recomputation.

Controls: random pick from the same group, and first-row pick (what a policy
without a ranker commits to).
"""
from __future__ import annotations
import json, glob, math, random, sys, statistics as st
from pathlib import Path
from collections import defaultdict

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals  # noqa: E402
from llm_finetune.training.rl.posterior.potential import (  # noqa: E402
    compute_potential_v2, program_from_truss_spec,
)

specs, init_cache = {}, {}
for pf in glob.glob(str(DESIGNBENCH / "data/problems/auto_problem_*.json")):
    try:
        d = json.load(open(pf))
        specs[d["problem_id"]] = d
    except Exception:
        pass


DEGEN_CACHE = {}


def is_degenerate(pid):
    """Reject problems whose FEA is physically meaningless.

    28 of the 100 originals are mechanisms: raw deflection is either inf (11) or
    ~1e13-1e14 m (17). The split file marks only 11 of them. They report huge FOS
    at low mass, so they score as spurious successes and any selection rule looks
    good on them. The clean problems top out at 0.106 m deflection, so the
    separation is unambiguous.
    """
    if pid not in DEGEN_CACHE:
        sp = specs.get(pid)
        if sp is None:
            DEGEN_CACHE[pid] = True
        else:
            try:
                import math
                from validation.truss_executor import analyze_truss, load_truss_from_problem
                _, goals = _load_truss_and_goals(sp)
                raw = analyze_truss(load_truss_from_problem(sp), goals)
                d = raw.get("deflection")
                fb, fy = raw.get("fos_buckling"), raw.get("fos_yielding")
                DEGEN_CACHE[pid] = (d is None or not math.isfinite(d) or abs(d) > 1.0
                                    or fb is None or fy is None
                                    or not math.isfinite(fb) or not math.isfinite(fy))
            except Exception:
                DEGEN_CACHE[pid] = True
    return DEGEN_CACHE[pid]


def initial_of(pid):
    if pid not in init_cache:
        sp = specs.get(pid)
        if sp is None:
            init_cache[pid] = None
        else:
            truss, goals = _load_truss_and_goals(sp)
            s0 = _analyze_truss(truss, goals)
            init_cache[pid] = (s0, program_from_truss_spec(sp, initial_mass=s0.get("mass")))
    return init_cache[pid]


def main():
    groups = defaultdict(list)
    seen = set()
    runs = set()
    for f in glob.glob(str(PROJECT / "wandb/**/media/table/rollouts/*.table.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        cols = d.get("columns") or []
        rows = d.get("data") or []
        need = ("problem_id", "feasible", "delta_FOS_buckling", "delta_FOS_yielding", "delta_mass")
        if not rows or any(n not in cols for n in need):
            continue
        idx = {n: cols.index(n) for n in need}
        run = f.split("/wandb/")[1].split("/")[0]
        runs.add(run)
        for k, r in enumerate(rows):
            pid = r[idx["problem_id"]]
            sig = (run, pid, round(float(r[idx["delta_mass"]] or 0), 6),
                   round(float(r[idx["delta_FOS_buckling"]] or 0), 6))
            if sig in seen:
                continue
            seen.add(sig)
            groups[(run, Path(f).name, pid)].append((k, r, idx))

    rng = random.Random(0)
    n_dec = 0
    n_all=[0]; u_phi=[0]; u_first=[0]; u_orac=[0]; n_skipped=[0]
    phi_hit = rand_hit = first_hit = oracle_hit = 0
    n01 = n10 = 0
    for (run, tbl, pid), rows in groups.items():
        if len(rows) < 3:
            continue
        if is_degenerate(pid):
            n_skipped[0] += 1
            continue
        got = initial_of(pid)
        if got is None:
            continue
        s0, program = got
        finals, feas = [], []
        for _, r, idx in rows:
            try:
                state = {
                    "mass": (s0.get("mass") or 0) + float(r[idx["delta_mass"]] or 0),
                    "fos_buckling": (s0.get("fos_buckling") or 0) + float(r[idx["delta_FOS_buckling"]] or 0),
                    "fos_yielding": (s0.get("fos_yielding") or 0) + float(r[idx["delta_FOS_yielding"]] or 0),
                    "deflection": s0.get("deflection", 0.0),
                }
            except Exception:
                continue
            finals.append(state)
            feas.append(bool(r[idx["feasible"]]))
        if len(finals) < 3:
            continue
        # unconditional tallies: every group, including those with no feasible
        # rollout at all. The decisive-only rate conditions on a feasible member
        # existing and so flatters every rule equally.
        n_all[0] += 1
        decisive = 0 < sum(feas) < len(feas)
        if decisive:
            n_dec += 1
        phis = []
        for s in finals:
            try:
                phis.append(float(compute_potential_v2(s, program, alpha=5.0, tau=0.05)))
            except Exception:
                phis.append(float("-inf"))
        p_i = max(range(len(phis)), key=lambda i: phis[i])
        r_i = rng.randrange(len(finals))
        u_phi[0] += feas[p_i]; u_first[0] += feas[0]; u_orac[0] += any(feas)
        if not decisive:
            continue
        phi_hit += feas[p_i]; rand_hit += feas[r_i]; first_hit += feas[0]; oracle_hit += 1
        if not feas[r_i] and feas[p_i]: n01 += 1
        if feas[r_i] and not feas[p_i]: n10 += 1

    N = n01 + n10
    p = (sum(math.comb(N, i) for i in range(min(n01, n10) + 1)) / 2 ** N * 2) if N else 1.0
    print(f"runs scanned          : {len(runs)}")
    print(f"groups skipped (degenerate problem): {n_skipped[0]}")
    print(f"decisive K=8 groups   : {n_dec}")
    print(f"  first-rollout pick  : {first_hit}/{n_dec} = {first_hit/max(n_dec,1):.3f}")
    print(f"  random pick         : {rand_hit}/{n_dec} = {rand_hit/max(n_dec,1):.3f}")
    print(f"  Phi_v2 pick         : {phi_hit}/{n_dec} = {phi_hit/max(n_dec,1):.3f}")
    print(f"  oracle (by defn)    : {oracle_hit}/{n_dec} = 1.000")
    print(f"  Phi vs random: discordant phi-only={n01} random-only={n10}  McNemar p={min(p,1.0):.6f}")
    print(f"  oracle recovery     : {phi_hit/max(oracle_hit,1):.3f}")
    na=max(n_all[0],1)
    print(f"\nUNCONDITIONAL over ALL {n_all[0]} groups (incl. groups with no feasible rollout):")
    print(f"  policy first rollout: {u_first[0]}/{n_all[0]} = {u_first[0]/na:.3f}")
    print(f"  Phi_v2 pick         : {u_phi[0]}/{n_all[0]} = {u_phi[0]/na:.3f}")
    print(f"  oracle ceiling      : {u_orac[0]}/{n_all[0]} = {u_orac[0]/na:.3f}")
    print(f"  Phi gain over policy: {(u_phi[0]-u_first[0])/na:+.3f}")


if __name__ == "__main__":
    main()
