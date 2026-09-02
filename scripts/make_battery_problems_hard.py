#!/usr/bin/env python
"""Tier-2 (multi-step) battery problems.

The tier-1 set is solvable by a single grammar action on most problems, which
makes every lookahead policy identical: the episode ends the moment ANY child is
feasible, so the potential is never consulted. The potential only decides
anything when no 1-step child is feasible and the search must choose which
INFEASIBLE state to move to. Tier 2 tightens the binding threshold of each
tier-1 problem until that is true:

    0 of the 52 one-step neighbours is feasible, but at least one of the scanned
    two-step compositions is.

Both facts are checked against recorded simulator output, not assumed.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts")
import make_battery_problems as T1
from battery_env import HARD_BOUNDS, ALIASES

OUT = T1.OUT
BINDING = {  # the metric each family drives to its limit
    "rate": "max_charge_time", "thermal": "max_temperature",
    "energy": "minimum_energy_density", "coupled": None,   # inferred
}


def tighten(limit, sense, f):
    return limit * (1.0 - f) if sense == "max" else limit * (1.0 + f)


def main():
    made, skipped = [], []
    for pid, family, variant, current, goals in T1.SPECS:
        rows = T1.load_scan(variant, current)
        base = next(r for r in rows if r["tag"] == "BASE")
        d1 = [r for r in rows if r["depth"] == 1]
        d2 = [r for r in rows if r["depth"] == 2]
        key = BINDING.get(family)
        if key is None:  # coupled: whichever goal the start violates
            v = T1.violated(base, goals)
            key = next((k for k in goals
                        if {"max_charge_time": "charge_time",
                            "minimum_energy_density": "energy_density"}.get(k, k)
                        in (x.split("=")[0] for x in v)), None)
        if key is None:
            skipped.append((pid, "no binding key")); continue
        sense = "max" if key.startswith("max") else "min"

        chosen = None
        for f in [i / 200 for i in range(1, 161)]:
            g = dict(goals)
            g[key] = tighten(goals[key], sense, f)
            n1 = sum(1 for r in d1 if T1.satisfies(r, g))
            n2 = sum(1 for r in d2 if T1.satisfies(r, g))
            if T1.satisfies(base, g):
                continue
            if n1 == 0 and n2 >= 2:
                chosen = (g, n1, n2, f)
                break
        if chosen is None:
            skipped.append((pid, "no threshold with 0 one-step and >=1 two-step"))
            continue
        g, n1, n2, f = chosen
        hid = pid.replace("bat_", "bat_h_")
        init = dict(T1.CHEN, **T1.VARIANTS[variant])
        design_params = {a: {"min": HARD_BOUNDS[full][0], "max": HARD_BOUNDS[full][1],
                             "initial": float(init[a])}
                         for a, full in ALIASES.items()}
        f2 = [r for r in d2 if T1.satisfies(r, g)]
        spec = {
            "problem_id": hid, "domain": "battery", "family": family + "_hard",
            "description": (f"Li-ion cell design ({variant} starting point) charged at "
                            f"{current:g} A. Tier-2: no single grammar action reaches "
                            f"feasibility; at least two are required."),
            "operating_conditions": {
                "charge_current_a": current, "ambient_temperature_k": 298.15,
                "heat_transfer_coefficient": 10.0,
                "protocol": ["Rest for 2 min", f"Charge at {current:g} A until 4.2V",
                             "Hold at 4.2V until C/50", "Rest for 30 min"],
                "simulator": "pybamm DFN + lumped thermal + partially-reversible plating + solvent-diffusion SEI",
                "parameter_set": "Chen2020"},
            "goals": {k: float(v) for k, v in g.items()},
            "objective": {"key": "charge_time", "sense": "min",
                          "ref": float(g["max_charge_time"])},
            "optimization": {"design_params": design_params},
            "scales": {"charge_time": float(g["max_charge_time"]),
                       "max_temperature": 30.0,
                       "energy_density": float(g["minimum_energy_density"]),
                       "max_plating": 1.0},
            "_metadata": {
                "tier": 2, "derived_from": pid, "tightened_goal": key,
                "tightening_fraction": f,
                "initial_metrics": {k: base[k] for k in
                                    ("charge_time", "energy_density", "max_temperature",
                                     "max_plating", "true_capacity")},
                "initial_feasible": False,
                "initial_violations": T1.violated(base, g),
                "n_1step_neighbours_scanned": len(d1), "n_1step_feasible": 0,
                "n_2step_probes_scanned": len(d2), "n_2step_feasible": n2,
                "best_known_charge_time": min(r["charge_time"] for r in f2),
                "scan_source": f"results/battery_ladder/scan_{variant}_{int(current)}A.jsonl",
                "design_variant": variant},
        }
        json.dump(spec, open(OUT / f"{hid}.json", "w"), indent=2)
        made.append((hid, family, key, goals[key], g[key], n2,
                     spec["_metadata"]["best_known_charge_time"]))

    print(f"{'problem':<19}{'family':<9}{'tightened':<24}{'was':>10}{'now':>10}"
          f"{'d2feas':>7}{'best_t':>9}")
    for m in made:
        print(f"{m[0]:<19}{m[1]:<9}{m[2]:<24}{m[3]:>10.4g}{m[4]:>10.4g}{m[5]:>7}{m[6]:>9.2f}")
    for s in skipped:
        print(f"  SKIPPED {s[0]}: {s[1]}")
    print(f"\n{len(made)} tier-2 problems -> {OUT}")


if __name__ == "__main__":
    main()
