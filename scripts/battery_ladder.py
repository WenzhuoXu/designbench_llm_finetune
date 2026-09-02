#!/usr/bin/env python
"""Search ladder for the BATTERY domain — the transfer test for the potential.

A line-for-line mirror of ``scripts/search_ladder.py``: the same four policies
(random / greedy-on-the-binding-constraint / 1-step lookahead under the
potential / depth-2 beam), the same runs.jsonl schema, the same summary table.
Only the simulator and the action grammar change.

The point of the mirror is that NOTHING in the scoring path is battery-specific.
The potential is imported from the shared module —

    program_from_battery_goals(goals) -> DesignProgram
    compute_potential_v2(state, program, alpha=...)

— and that adapter is nine lines that name an objective; the constraints are
recovered from the problem's own goals dict by the same naming convention the
truss problems use (``minimum_x``/``min_x`` => x >= limit, ``maximum_x``/``max_x``
=> x <= limit).  ``lookahead_v1_*`` runs the SHIPPED truss potential on battery
states to show what a domain-specific potential does off its domain.

Must be run in the isolated battery venv (pybamm is deliberately absent from
my_env):
    /ocean/projects/mch250030p/wxu7/envs/battery/bin/python scripts/battery_ladder.py

CPU only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── the SHARED, domain-general potential (not reimplemented here) ────────────
from llm_finetune.training.rl.posterior.potential import (  # noqa: E402
    DesignProgram,
    compute_potential,          # v1: the shipped truss-specific potential
    compute_potential_v2,       # v2: negated Lagrangian of the design program
    program_from_battery_goals,
    program_from_goals,
    _BATTERY_STATE_KEYS,
)

from battery_env import (  # noqa: E402
    ALIASES,
    DESIGN_PARAMS,
    BatterySimulator,
    execute_action,
    resolve,
    short,
)

# Battery-scale sizing moves. The truss ladder uses (0.85,1.1,1.25,1.5,2.0) on
# member radii; electrode dimensions tolerate a similar multiplicative range.
SCALE_FACTORS = (0.75, 0.9, 1.1, 1.25)
# The battery analogue of the truss "all members" global action: symmetric moves
# on the anode/cathode pair.
PARAM_GROUPS = (
    ("neg_thickness", "pos_thickness"),
    ("neg_porosity", "pos_porosity"),
    ("neg_radius", "pos_radius"),
    ("neg_am_fraction", "pos_am_fraction"),
)


# ── problem I/O ──────────────────────────────────────────────────────────────

def load_problem(path: Path) -> Optional[dict]:
    spec = json.load(open(path))
    return spec if isinstance(spec, dict) and "goals" in spec and "optimization" in spec else None


def param_bounds(spec: dict) -> Dict[str, Tuple[float, float]]:
    """Per-parameter [min,max] box, exactly like the truss shape_params block."""
    opt = spec.get("optimization") or {}
    dp = opt.get("design_params") or {}
    bounds: Dict[str, Tuple[float, float]] = {}
    for name, rng in dp.items():
        if isinstance(rng, dict) and "min" in rng and "max" in rng:
            bounds[resolve(name)] = (float(rng["min"]), float(rng["max"]))
    return bounds


def initial_params(spec: dict, sim: BatterySimulator) -> Dict[str, float]:
    base = sim.baseline_params()
    opt = spec.get("optimization") or {}
    for name, rng in (opt.get("design_params") or {}).items():
        if isinstance(rng, dict) and "initial" in rng:
            base[resolve(name)] = float(rng["initial"])
    return base


def make_program(spec: dict, use_scales: bool) -> DesignProgram:
    """goals dict -> DesignProgram.  No battery code beyond naming an objective."""
    goals = spec.get("goals") or {}
    obj = spec.get("objective") or {}
    key = obj.get("key", "charge_time")
    ref = obj.get("ref")
    sense = obj.get("sense", "min")
    if not use_scales:
        # the shipped adapter, verbatim
        return program_from_battery_goals(
            goals, objective_key=key, objective_ref=ref, objective_sense=sense)
    # identical program, but with the per-constraint normalisers the problem
    # statement supplies.  program_from_battery_goals has no `scales`
    # passthrough, so the generic constructor is called directly.
    return program_from_goals(
        goals,
        state_keys=_BATTERY_STATE_KEYS,
        objective_key=key,
        objective_ref=ref if ref else float(goals.get("max_charge_time", 900.0)),
        objective_sense=sense,
        scales=spec.get("scales") or {},
        domain="battery",
    )


def make_sim(spec: dict, fidelity: str) -> BatterySimulator:
    op = spec.get("operating_conditions") or {}
    return BatterySimulator(
        fidelity=fidelity,
        current_a=op.get("charge_current_a"),
        c_rate=op.get("c_rate", 1.5),
        ambient_k=op.get("ambient_temperature_k", 298.15),
        htc=op.get("heat_transfer_coefficient", 10.0),
    )


# ── candidate generation (mirrors search_ladder.candidate_actions) ───────────

def candidate_actions(params: Mapping[str, float],
                      bounds: Mapping[str, Tuple[float, float]],
                      include_global: bool = True) -> List[str]:
    """Sizing actions that stay inside the problem's own parameter bounds."""
    actions: List[str] = []
    for full in DESIGN_PARAMS:
        cur = params.get(full)
        if cur is None:
            continue
        lo, hi = bounds.get(full, (-math.inf, math.inf))
        for f in SCALE_FACTORS:
            new = cur * f
            if new < lo or new > hi:
                continue
            actions.append(f"SCALE_PARAM({short(full)}, {f})")
    if include_global:
        for group in PARAM_GROUPS:
            fulls = [resolve(g) for g in group]
            if any(params.get(x) is None for x in fulls):
                continue
            for f in SCALE_FACTORS:
                ok = True
                for x in fulls:
                    lo, hi = bounds.get(x, (-math.inf, math.inf))
                    if params[x] * f < lo or params[x] * f > hi:
                        ok = False
                        break
                if ok:
                    actions.append(f"SCALE_MULTI_PARAM([{','.join(group)}], {f})")
    return actions


def step(sim: BatterySimulator, params: Mapping[str, float], action: str):
    """Apply one action; return (new_params, new_state) or None if illegal.

    Mirrors search_ladder.step: an action the executor refuses yields None and is
    dropped from the candidate set. Here "refuses" also covers a parameter vector
    that is not a buildable cell (porosity + active material > 1), which the box
    bounds cannot express.
    """
    nxt = execute_action(dict(params), action)
    if nxt is None or not sim.admissible(nxt):
        return None
    return nxt, sim.evaluate(nxt)


# ── potentials ───────────────────────────────────────────────────────────────

def phi_v1(state: Mapping, program: DesignProgram, initial_obj: float, alpha: float) -> float:
    """The SHIPPED truss potential, applied unchanged to a battery state.

    It reads mass / fos_buckling / fos_yielding / deflection, none of which a
    battery simulation produces, so every battery state scores identically.
    """
    return compute_potential(dict(state), initial_mass=initial_obj, alpha=alpha)


# Set from --feasibility-offset / --tau. A module global so the PHI table keeps
# its signature; the ladder is single-config per process.
_PHI_OPTS = {"feasibility_offset": 0.0, "tau": 0.05}


def phi_v2(state: Mapping, program: DesignProgram, initial_obj: float, alpha: float) -> float:
    return compute_potential_v2(state, program, alpha=alpha, **_PHI_OPTS)


PHI = {"v1": phi_v1, "v2": phi_v2}


# ── result record (same shape as search_ladder.RunResult) ────────────────────

@dataclass
class RunResult:
    problem_id: str
    policy: str
    feasible: bool
    steps: int
    final_mass: float        # generic slot: the problem's objective at the end
    initial_mass: float      # ... at the start
    mass_ref: float          # ... its normaliser (truss: LP optimum)
    final_fos_b: float       # generic slot: worst relative constraint slack
    final_fos_y: float       # generic slot: total relative violation
    fea_calls: int           # simulator calls
    family: str
    domain: str = "battery"
    objective_key: str = "charge_time"
    final_metrics: Dict[str, float] = field(default_factory=dict)
    violations: Dict[str, float] = field(default_factory=dict)
    sim_seconds: float = 0.0


def _record(pid, policy, program, state, steps, obj0, calls, family, sim_seconds) -> dict:
    viol = program.violations(state, tau=0.0)
    obj = state.get(program.objective_key, float("nan"))
    slack = min(
        ((c.limit - float(state.get(c.key, 0.0))) / max(abs(c.limit), 1e-12)
         if c.sense == "upper" else
         (float(state.get(c.key, 0.0)) - c.limit) / max(abs(c.limit), 1e-12))
        for c in program.constraints
    ) if program.constraints else 0.0
    r = RunResult(
        problem_id=pid, policy=policy, feasible=bool(program.is_feasible(state)),
        steps=steps,
        final_mass=float(obj) if obj is not None else float("nan"),
        initial_mass=float(obj0), mass_ref=float(program.objective_ref),
        final_fos_b=float(slack), final_fos_y=float(sum(viol.values())),
        fea_calls=int(calls), family=family,
        objective_key=program.objective_key,
        final_metrics={k: float(v) for k, v in state.items() if isinstance(v, (int, float))},
        violations={k: float(v) for k, v in viol.items()},
        sim_seconds=float(sim_seconds),
    )
    return asdict(r)


# ── policies (mirroring search_ladder) ───────────────────────────────────────

def _binding(program: DesignProgram, state: Mapping) -> Optional[str]:
    """Constraint with the largest RELATIVE violation — the battery analogue of
    the FEA-reported critical member."""
    worst, worst_key = 0.0, None
    for c in program.constraints:
        v = c.violation(state, tau=0.0)
        if v > worst:
            worst, worst_key = v, c.key
    return worst_key


# Domain heuristic: which knob relieves which binding constraint. This is the
# battery counterpart of "scale the buckling-critical member's radius", i.e.
# the expert baseline the search has to beat -- NOT part of the potential.
_RELIEF: Dict[str, Tuple[str, ...]] = {
    "charge_time": ("SCALE_PARAM(neg_thickness, 0.9)",
                    "SCALE_PARAM(neg_porosity, 1.1)",
                    "SCALE_PARAM(neg_radius, 0.9)",
                    "SCALE_MULTI_PARAM([neg_thickness,pos_thickness], 0.9)"),
    "max_plating": ("SCALE_PARAM(neg_thickness, 0.9)",
                    "SCALE_PARAM(neg_porosity, 1.1)",
                    "SCALE_PARAM(neg_radius, 0.9)",
                    "SCALE_PARAM(neg_am_fraction, 0.9)"),
    "min_neg_potential": ("SCALE_PARAM(neg_thickness, 0.9)",
                          "SCALE_PARAM(neg_porosity, 1.1)",
                          "SCALE_PARAM(neg_radius, 0.9)"),
    "max_temperature": ("SCALE_MULTI_PARAM([neg_porosity,pos_porosity], 1.1)",
                        "SCALE_MULTI_PARAM([neg_thickness,pos_thickness], 0.9)",
                        "SCALE_PARAM(sep_thickness, 0.9)"),
    "temperature_rise": ("SCALE_MULTI_PARAM([neg_porosity,pos_porosity], 1.1)",
                         "SCALE_MULTI_PARAM([neg_thickness,pos_thickness], 0.9)",
                         "SCALE_PARAM(sep_thickness, 0.9)"),
    "energy_density": ("SCALE_MULTI_PARAM([neg_am_fraction,pos_am_fraction], 1.1)",
                       "SCALE_MULTI_PARAM([neg_porosity,pos_porosity], 0.9)",
                       "SCALE_PARAM(sep_thickness, 0.9)"),
    "true_capacity": ("SCALE_MULTI_PARAM([neg_thickness,pos_thickness], 1.1)",
                      "SCALE_MULTI_PARAM([neg_am_fraction,pos_am_fraction], 1.1)"),
}
_DEFAULT_RELIEF = ("SCALE_PARAM(neg_thickness, 0.9)",
                   "SCALE_PARAM(neg_porosity, 1.1)")


def _in_bounds(params: Mapping[str, float], bounds) -> bool:
    for full, (lo, hi) in bounds.items():
        v = params.get(full)
        if v is not None and (v < lo or v > hi):
            return False
    return True


def run_greedy_binding(sim, spec, program, params, max_steps, bounds):
    """Expert heuristic: relieve the most-violated constraint, one sim per step."""
    state = sim.evaluate(params)
    obj0 = state.get(program.objective_key, float("nan"))
    calls, taken = 1, 0
    hit_count: Dict[str, int] = defaultdict(int)
    best = (params, state) if program.is_feasible(state) else None
    for _ in range(max_steps):
        if program.is_feasible(state):
            break
        key = _binding(program, state)
        menu = _RELIEF.get(key, _DEFAULT_RELIEF)
        moved = False
        for _try in range(len(menu)):
            action = menu[hit_count[key] % len(menu)]
            hit_count[key] += 1
            nxt = execute_action(dict(params), action)
            if nxt is None or not _in_bounds(nxt, bounds) or not sim.admissible(nxt):
                continue
            ns = sim.evaluate(nxt)
            calls += 1
            if ns.get("success", 0.0) == 0.0:
                continue   # the simulator rejected the design; try the next knob
            params, state = nxt, ns
            moved = True
            break
        if not moved:
            break
        taken += 1
        if program.is_feasible(state):
            best = (params, state)
            break
    if best is not None:
        return best[0], best[1], calls, obj0, taken
    return params, state, calls, obj0, taken


def run_random(sim, spec, program, params, max_steps, bounds, seed=0):
    import random
    rng = random.Random(seed)
    state = sim.evaluate(params)
    obj0 = state.get(program.objective_key, float("nan"))
    calls, taken = 1, 0
    for _ in range(max_steps):
        if program.is_feasible(state):
            break
        acts = candidate_actions(params, bounds)
        if not acts:
            break
        out = step(sim, params, rng.choice(acts))
        calls += 1
        if out is None:
            continue
        params, state = out
        taken += 1
    return params, state, calls, obj0, taken


def run_lookahead(sim, spec, program, params, max_steps, phi_name, alpha, bounds,
                  depth=1, beam=4, stop_on_feasible=True):
    """Greedy (or depth-2 beam) search maximising the chosen potential."""
    phi = PHI[phi_name]
    state = sim.evaluate(params)
    obj0 = state.get(program.objective_key, float("nan"))
    calls, taken = 1, 0
    best = None
    if program.is_feasible(state):
        best = (program.objective(state), params, state)
    for _ in range(max_steps):
        if program.is_feasible(state) and stop_on_feasible:
            break
        acts = candidate_actions(params, bounds)
        if not acts:
            break
        children = []
        for a in acts:
            out = step(sim, params, a)
            calls += 1
            if out is None:
                continue
            np_, ns = out
            children.append((phi(ns, program, obj0, alpha), a, np_, ns))
        if not children:
            break
        feas = [c for c in children if program.is_feasible(c[3])]
        if feas:
            feas.sort(key=lambda c: program.objective(c[3]))
            cand = (program.objective(feas[0][3]), feas[0][2], feas[0][3])
            if best is None or cand[0] < best[0]:
                best = cand
            if stop_on_feasible:
                _, _, params, state = feas[0]
                taken += 1
                break
        children.sort(key=lambda c: c[0], reverse=True)
        if depth >= 2:
            rescored = []
            for val, a, np_, ns in children[:beam]:
                sub = candidate_actions(np_, bounds)
                best_child, child_feasible = -math.inf, False
                for a2 in sub:
                    out2 = step(sim, np_, a2)
                    calls += 1
                    if out2 is None:
                        continue
                    _, ns2 = out2
                    if program.is_feasible(ns2):
                        child_feasible = True
                    best_child = max(best_child, phi(ns2, program, obj0, alpha))
                if best_child == -math.inf:
                    best_child = val
                rescored.append((val + 0.99 * best_child + (5.0 if child_feasible else 0.0),
                                 a, np_, ns))
            rescored.sort(key=lambda c: c[0], reverse=True)
            children = rescored
        _, _, params, state = children[0]
        taken += 1
        if program.is_feasible(state):
            cand = (program.objective(state), params, state)
            if best is None or cand[0] < best[0]:
                best = cand
    if best is not None:
        return best[1], best[2], calls, obj0, taken
    return params, state, calls, obj0, taken


# ── driver ───────────────────────────────────────────────────────────────────

def run_problem(args_tuple):
    path, max_steps, alpha, policies, fidelity, use_scales, beam = args_tuple
    spec = load_problem(Path(path))
    if spec is None:
        return []
    pid = spec.get("problem_id", Path(path).stem)
    family = spec.get("family", "battery")
    program = make_program(spec, use_scales)
    sim = make_sim(spec, fidelity)
    p0 = initial_params(spec, sim)
    bounds = param_bounds(spec)

    results = []
    for policy in policies:
        t_wall = time.time()
        n0 = sim.n_sims
        try:
            if policy == "greedy_critical":
                pf, sf, calls, obj0, taken = run_greedy_binding(
                    sim, spec, program, dict(p0), max_steps, bounds)
            elif policy == "random":
                pf, sf, calls, obj0, taken = run_random(
                    sim, spec, program, dict(p0), max_steps, bounds,
                    seed=abs(hash(pid)) % 10000)
            elif policy.startswith("lookahead"):
                parts = policy.split("_")
                phi_name, dstr = parts[1], parts[2]
                minobj = len(parts) > 3 and parts[3] == "minobj"
                pf, sf, calls, obj0, taken = run_lookahead(
                    sim, spec, program, dict(p0), max_steps, phi_name, alpha, bounds,
                    depth=int(dstr[1:]), beam=beam, stop_on_feasible=not minobj)
            else:
                continue
        except Exception as exc:
            results.append({"problem_id": pid, "policy": policy, "feasible": False,
                            "steps": -1, "final_mass": float("nan"),
                            "initial_mass": float("nan"), "mass_ref": program.objective_ref,
                            "final_fos_b": float("nan"), "final_fos_y": float("nan"),
                            "fea_calls": 0, "family": family, "domain": "battery",
                            "error": f"{type(exc).__name__}: {exc}"})
            continue
        rec = _record(pid, policy, program, sf, taken, obj0, calls, family,
                      time.time() - t_wall)
        rec["new_sims"] = sim.n_sims - n0
        rec["design"] = {short(k): v for k, v in pf.items()}
        results.append(rec)
        print(f"  [{pid}] {policy:<18} feasible={rec['feasible']}  "
              f"{program.objective_key}={rec['final_mass']:.3f}  "
              f"calls={calls} sims={rec['new_sims']} {rec['sim_seconds']:.0f}s", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(PROJECT / "data/battery_problems"))
    ap.add_argument("--out", default=str(PROJECT / "results/battery_ladder"))
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--tau", type=float, default=0.05,
                    help="Hinge softness in RELATIVE units. Battery violations are ~0.02, so the "
                         "truss default of 0.05 never saturates and satisfied constraints keep paying.")
    ap.add_argument("--feasibility-offset", type=float, default=0.0,
                    help="Rank any violating design below any satisfying one. Needed where the "
                         "objective can outbid a small violation: on thermal_hard a cell 2%% over "
                         "its temperature limit but charging twice as fast scores ABOVE a feasible "
                         "one, because alpha*V = 0.10 against an objective gap of 0.33.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--fidelity", default="spme_lean",
                    choices=("spme_lean", "spme_full", "dfn_full"))
    ap.add_argument("--scales", choices=("none", "problem"), default="none",
                    help="'none' = program_from_battery_goals verbatim (limit-relative "
                         "normalisers); 'problem' = the scales block in the problem JSON")
    ap.add_argument("--policies",
                    default="random,greedy_critical,lookahead_v1_d1,lookahead_v2_d1,lookahead_v2_d2")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    _PHI_OPTS["feasibility_offset"] = args.feasibility_offset
    _PHI_OPTS["tau"] = args.tau

    files = sorted(Path(args.problems).glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    policies = [p for p in args.policies.split(",") if p]
    tasks = [(str(f), args.max_steps, args.alpha, policies, args.fidelity,
              args.scales == "problem", args.beam) for f in files]

    out_dir = Path(args.out) / (args.tag or f"{args.fidelity}_scales-{args.scales}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_results = []
    if args.workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(args.workers, len(tasks))) as pool:
            for i, res in enumerate(pool.imap_unordered(run_problem, tasks), 1):
                all_results.extend(res)
                print(f"  {i}/{len(tasks)} problems  ({time.time()-t0:.0f}s)", flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            all_results.extend(run_problem(task))
            print(f"  {i}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)

    with open(out_dir / "runs.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nDone in {time.time()-t0:.0f}s -> {out_dir}")
    summarize(all_results)


def summarize(results):
    by = defaultdict(list)
    for r in results:
        by[(r["policy"], r.get("family", "battery"))].append(r)
        by[(r["policy"], "ALL")].append(r)
    print(f"\n{'policy':<20} {'family':<11} {'n':>4} {'feas':>7} "
          f"{'obj/ref(feas)':>15} {'sims':>7}")
    print("-" * 72)
    for (policy, family), rows in sorted(by.items()):
        n = len(rows)
        feas = [r for r in rows if r["feasible"]]
        mr = [r["final_mass"] / r["mass_ref"] for r in feas
              if r["mass_ref"] > 0 and math.isfinite(r["final_mass"])]
        calls = st.mean([r["fea_calls"] for r in rows]) if rows else 0
        print(f"{policy:<20} {family:<11} {n:>4} {len(feas)/max(n,1):>6.1%} "
              f"{(st.mean(mr) if mr else float('nan')):>15.3f} {calls:>7.0f}")


if __name__ == "__main__":
    main()
