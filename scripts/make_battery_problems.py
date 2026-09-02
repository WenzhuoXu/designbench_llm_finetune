#!/usr/bin/env python
"""Build the battery problem set from the measured design-space scans.

Every problem is calibrated against real simulator output (results/battery_ladder/
scan_{variant}_{I}A.jsonl, which holds the BASE design plus the COMPLETE set of
1-step grammar neighbours), so two properties are checked, not assumed:

  * the initial design is genuinely INFEASIBLE (>=1 goal violated), and
  * feasibility is genuinely REACHABLE (>=1 neighbour in the scan satisfies
    every goal simultaneously — the joint check, not a per-metric one).

Written in the same JSON spirit as DesignBench/data/problems/*.json:
    problem_id / description / goals / optimization{design_params{min,max,initial}}
    / _metadata, plus operating_conditions (the load case) and objective.
"""
from __future__ import annotations
import json, sys, itertools
from pathlib import Path

sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts")
from battery_env import HARD_BOUNDS, ALIASES, short

RES = Path("/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder")
OUT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune/data/battery_problems")

VARIANTS = {
    "std":    {"neg_porosity": 0.28, "neg_am_fraction": 0.68,
               "pos_porosity": 0.335, "pos_am_fraction": 0.62},
    "dense":  {"neg_thickness": 1.30e-4, "neg_porosity": 0.22, "neg_am_fraction": 0.76,
               "pos_thickness": 1.05e-4, "pos_porosity": 0.30, "pos_am_fraction": 0.66},
    "thin":   {"neg_thickness": 5.5e-5, "neg_porosity": 0.32, "neg_am_fraction": 0.62,
               "pos_thickness": 4.9e-5, "pos_porosity": 0.36, "pos_am_fraction": 0.58},
    "coarse": {"neg_radius": 1.20e-5, "pos_radius": 1.05e-5,
               "neg_porosity": 0.28, "neg_am_fraction": 0.68,
               "pos_porosity": 0.335, "pos_am_fraction": 0.62},
}
CHEN = {  # PyBaMM Chen2020 defaults, for parameters a variant does not override
    "neg_thickness": 8.52e-05, "pos_thickness": 7.56e-05, "sep_thickness": 1.2e-05,
    "neg_porosity": 0.25, "pos_porosity": 0.335,
    "neg_radius": 5.86e-06, "pos_radius": 5.22e-06,
    "neg_am_fraction": 0.75, "pos_am_fraction": 0.665,
}

# (id, family, variant, current, goals)  — thresholds fixed below after calibration
SPECS = [
    ("bat_rate_00",    "rate",    "coarse",  8.0, dict(max_charge_time=50.0,  max_temperature=340.0, max_plating=1e-5, minimum_energy_density=400.0)),
    ("bat_rate_01",    "rate",    "coarse", 12.0, dict(max_charge_time=45.0,  max_temperature=340.0, max_plating=1e-5, minimum_energy_density=400.0)),
    ("bat_rate_02",    "rate",    "coarse", 16.0, dict(max_charge_time=35.0,  max_temperature=340.0, max_plating=1e-5, minimum_energy_density=400.0)),
    ("bat_thermal_00", "thermal", "std",    12.0, dict(max_temperature=328.0, max_charge_time=32.0,  max_plating=1e-5, minimum_energy_density=500.0)),
    ("bat_thermal_01", "thermal", "std",    16.0, dict(max_temperature=329.0, max_charge_time=32.0,  max_plating=1e-5, minimum_energy_density=505.0)),
    ("bat_thermal_02", "thermal", "dense",   8.0, dict(max_temperature=332.0, max_charge_time=50.0,  max_plating=1e-5, minimum_energy_density=850.0)),
    ("bat_energy_00",  "energy",  "thin",    8.0, dict(minimum_energy_density=460.0, max_temperature=325.0, max_charge_time=30.0, max_plating=1e-5)),
    ("bat_energy_01",  "energy",  "thin",   12.0, dict(minimum_energy_density=470.0, max_temperature=330.0, max_charge_time=28.0, max_plating=1e-5)),
    ("bat_energy_02",  "energy",  "std",     8.0, dict(minimum_energy_density=780.0, max_temperature=330.0, max_charge_time=40.0, max_plating=1e-5)),
    ("bat_coupled_00", "coupled", "std",    12.0, dict(minimum_energy_density=700.0, max_temperature=338.0, max_charge_time=28.0, max_plating=1e-5)),
    ("bat_coupled_01", "coupled", "dense",  12.0, dict(minimum_energy_density=1030.0, max_temperature=355.0, max_charge_time=52.0, max_plating=1e-5)),
    ("bat_coupled_02", "coupled", "coarse", 12.0, dict(minimum_energy_density=640.0, max_charge_time=45.0, max_temperature=330.0, max_plating=1e-5)),
]


def load_scan(variant, current):
    """Scan rows for a design variant + load case.

    The 1-step layer is exhaustive (every legal grammar action from BASE). The
    2-step layer is whatever has been certified: the cheap hand-picked probes
    from battery_scan.py, plus -- when it exists -- the exhaustive depth-2
    expansion of the 12 most promising 1-step states from battery_oracle_d2.py.
    """
    rows = [json.loads(l) for l in open(RES / f"scan_{variant}_{int(current)}A.jsonl")]
    orc = RES / f"oracle2_{variant}_{int(current)}A.jsonl"
    if orc.exists():
        seen = {(r["tag"], r["depth"]) for r in rows}
        for line in open(orc):
            r = json.loads(line)
            if (r["tag"], r["depth"]) not in seen:
                rows.append(r)
                seen.add((r["tag"], r["depth"]))
    return rows


def satisfies(row, goals):
    if row.get("success", 0.0) != 1.0:
        return False
    for k, lim in goals.items():
        metric = {"max_charge_time": "charge_time",
                  "minimum_energy_density": "energy_density"}.get(k, k)
        v = row.get(metric)
        if v is None:
            return False
        if k.startswith("min") and v < lim:
            return False
        if k.startswith("max") and v > lim:
            return False
    return True


def violated(row, goals):
    out = []
    for k, lim in goals.items():
        metric = {"max_charge_time": "charge_time",
                  "minimum_energy_density": "energy_density"}.get(k, k)
        v = row.get(metric)
        if v is None:
            continue
        if (k.startswith("min") and v < lim) or (k.startswith("max") and v > lim):
            out.append(f"{metric}={v:.4g}{'<' if k.startswith('min') else '>'}{lim:g}")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for pid, family, variant, current, goals in SPECS:
        rows = load_scan(variant, current)
        base = next(r for r in rows if r["tag"] == "BASE")
        d1 = [r for r in rows if r["depth"] == 1]
        d2 = [r for r in rows if r["depth"] == 2]
        base_ok = satisfies(base, goals)
        f1 = [r for r in d1 if satisfies(r, goals)]
        f2 = [r for r in d2 if satisfies(r, goals)]

        init = dict(CHEN, **VARIANTS[variant])
        design_params = {}
        for alias, full in ALIASES.items():
            lo, hi = HARD_BOUNDS[full]
            design_params[alias] = {"min": lo, "max": hi, "initial": float(init[alias])}

        spec = {
            "problem_id": pid,
            "domain": "battery",
            "family": family,
            "description": (
                f"Li-ion cell design ({variant} starting point) charged at "
                f"{current:g} A from 0% SOC to 4.2 V with a CV hold. Resize the "
                f"electrode geometry so the cell meets every goal."),
            "operating_conditions": {
                "charge_current_a": current,
                "ambient_temperature_k": 298.15,
                "heat_transfer_coefficient": 10.0,
                "protocol": ["Rest for 2 min", f"Charge at {current:g} A until 4.2V",
                             "Hold at 4.2V until C/50", "Rest for 30 min"],
                "simulator": "pybamm DFN + lumped thermal + partially-reversible plating + solvent-diffusion SEI",
                "parameter_set": "Chen2020",
            },
            "goals": goals,
            "objective": {"key": "charge_time", "sense": "min",
                          "ref": float(goals["max_charge_time"])},
            "optimization": {"design_params": design_params},
            "scales": {  # per-constraint normalisers (see report: NOT used by the
                         # default adapter; only by --scales problem)
                "charge_time": float(goals["max_charge_time"]),
                "max_temperature": 30.0,          # K of overshoot, not absolute K
                "energy_density": float(goals["minimum_energy_density"]),
                "max_plating": 1.0,               # mol/m3 of plated lithium
            },
            "_metadata": {
                "initial_metrics": {k: base[k] for k in
                                    ("charge_time", "energy_density", "max_temperature",
                                     "max_plating", "true_capacity")},
                "initial_feasible": base_ok,
                "initial_violations": violated(base, goals),
                "n_1step_neighbours_scanned": len(d1),
                "n_1step_feasible": len(f1),
                "n_2step_probes_scanned": len(d2),
                "n_2step_feasible": len(f2),
                "best_1step_charge_time": (min(r["charge_time"] for r in f1) if f1 else None),
                "best_known_charge_time": (min([r["charge_time"] for r in (f1 + f2)])
                                           if (f1 or f2) else None),
                "scan_source": f"results/battery_ladder/scan_{variant}_{int(current)}A.jsonl",
                "design_variant": variant,
            },
        }
        json.dump(spec, open(OUT / f"{pid}.json", "w"), indent=2)
        report.append((pid, family, variant, current, base_ok, len(f1), len(f2),
                       spec["_metadata"]["initial_violations"],
                       spec["_metadata"]["best_known_charge_time"]))

    print(f"{'problem':<17}{'family':<9}{'var':<8}{'I(A)':>5} {'init_feas':>10} "
          f"{'d1feas':>7} {'d2feas':>7}  best_t  violations_at_start")
    bad = 0
    for r in report:
        flag = ""
        if r[4]:
            flag = "  <<< START ALREADY FEASIBLE"; bad += 1
        if r[5] == 0 and r[6] == 0:
            flag += "  <<< NO KNOWN FEASIBLE NEIGHBOUR"; bad += 1
        bt = f"{r[8]:.2f}" if r[8] is not None else "  -  "
        print(f"{r[0]:<17}{r[1]:<9}{r[2]:<8}{r[3]:>5.0f} {str(r[4]):>10} "
              f"{r[5]:>7} {r[6]:>7}  {bt:>6}  {','.join(r[7])}{flag}")
    print(f"\n{len(report)} problems -> {OUT}   ({bad} calibration warnings)")


if __name__ == "__main__":
    main()
