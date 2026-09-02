#!/usr/bin/env python
"""Enumerate the 1-step reachable set from a candidate base design, so the
battery problem goals can be calibrated: initial genuinely infeasible, and
feasibility genuinely reachable through the SCALE_PARAM grammar."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts")
from battery_env import BatterySimulator, DESIGN_PARAMS, ALIASES, short, execute_action

FID = "dfn_full"
FACTORS = (0.75, 0.9, 1.1, 1.25)
GROUPS = (("neg_thickness", "pos_thickness"), ("neg_porosity", "pos_porosity"),
          ("neg_radius", "pos_radius"), ("neg_am_fraction", "pos_am_fraction"))

VARIANTS = {
    # roomy starting point: porosity+AM leave headroom so the porosity knob works
    "std": {"Negative electrode porosity": 0.28,
            "Negative electrode active material volume fraction": 0.68,
            "Positive electrode porosity": 0.335,
            "Positive electrode active material volume fraction": 0.62},
    # thick, dense anode: the plating-prone regime
    "dense": {"Negative electrode thickness [m]": 1.30e-4,
              "Negative electrode porosity": 0.22,
              "Negative electrode active material volume fraction": 0.76,
              "Positive electrode thickness [m]": 1.05e-4,
              "Positive electrode porosity": 0.30,
              "Positive electrode active material volume fraction": 0.66},
    # thin, under-built cell: low energy density, must gain material
    "thin": {"Negative electrode thickness [m]": 5.5e-5,
             "Negative electrode porosity": 0.32,
             "Negative electrode active material volume fraction": 0.62,
             "Positive electrode thickness [m]": 4.9e-5,
             "Positive electrode porosity": 0.36,
             "Positive electrode active material volume fraction": 0.58},
    # coarse particles: kinetically slow, must shrink radii
    "coarse": {"Negative particle radius [m]": 1.20e-5,
               "Positive particle radius [m]": 1.05e-5,
               "Negative electrode porosity": 0.28,
               "Negative electrode active material volume fraction": 0.68,
               "Positive electrode porosity": 0.335,
               "Positive electrode active material volume fraction": 0.62},
}


def main(current: float, variant: str):
    sim = BatterySimulator(fidelity=FID, current_a=current)
    base = dict(sim.baseline_params(), **VARIANTS[variant])
    rows = []

    def rec(tag, params, depth):
        m = sim.evaluate(params)
        rows.append({"current_a": current, "variant": variant, "tag": tag, "depth": depth,
                     **{k: round(float(v), 6) for k, v in m.items()}})
        print(json.dumps(rows[-1]), flush=True)
        return m

    rec("BASE", base, 0)
    acts = [f"SCALE_PARAM({short(p)}, {f})" for p in DESIGN_PARAMS for f in FACTORS]
    acts += [f"SCALE_MULTI_PARAM([{','.join(g)}], {f})" for g in GROUPS for f in FACTORS]
    for a in acts:
        nxt = execute_action(dict(base), a)
        if nxt is None:
            continue
        rec(a, nxt, 1)

    # a few depth-2 compositions along the physically obvious directions
    probes = ("SCALE_MULTI_PARAM([neg_thickness,pos_thickness], 0.75)",
              "SCALE_MULTI_PARAM([neg_radius,pos_radius], 0.75)",
              "SCALE_MULTI_PARAM([neg_porosity,pos_porosity], 1.1)",
              "SCALE_MULTI_PARAM([neg_thickness,pos_thickness], 1.25)")
    for a1 in probes:
        for a2 in probes + ("SCALE_MULTI_PARAM([neg_am_fraction,pos_am_fraction], 1.1)",):
            p1 = execute_action(dict(base), a1)
            if p1 is None:
                continue
            p2 = execute_action(p1, a2)
            if p2 is None:
                continue
            rec(f"{a1} ; {a2}", p2, 2)

    out = Path("/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder"
               f"/scan_{variant}_{int(current)}A.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", out, "n=", len(rows), "sims=", sim.n_sims, "sim_s=", round(sim.sim_seconds, 1))


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "std")
