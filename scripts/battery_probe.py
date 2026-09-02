#!/usr/bin/env python
"""Probe the battery design space: pick charge currents and design moves that
make a problem genuinely infeasible at the start yet reachable by the grammar."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts")
from battery_env import BatterySimulator, DESIGN_PARAMS, ALIASES

FID = sys.argv[1] if len(sys.argv) > 1 else "spme_lean"
rows = []

def probe(tag, sim, params):
    t0 = time.time()
    m = sim.evaluate(params)
    rows.append({"tag": tag, "fidelity": sim.fidelity, "current_a": sim.current_a,
                 "wall_s": round(time.time()-t0, 2),
                 **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}})
    print(json.dumps(rows[-1]), flush=True)

for I in (5.0, 10.0, 15.0, 20.0, 25.0):
    sim = BatterySimulator(fidelity=FID, current_a=I)
    base = sim.baseline_params()
    probe(f"baseline@{I}A", sim, base)

# design sweeps at the two most stressful currents
for I in (15.0, 20.0):
    sim = BatterySimulator(fidelity=FID, current_a=I)
    base = sim.baseline_params()
    moves = {
        "neg_thin_0.6":  {"Negative electrode thickness [m]": base["Negative electrode thickness [m]"]*0.6},
        "neg_thin_0.4":  {"Negative electrode thickness [m]": base["Negative electrode thickness [m]"]*0.4},
        "both_thin_0.6": {"Negative electrode thickness [m]": base["Negative electrode thickness [m]"]*0.6,
                          "Positive electrode thickness [m]": base["Positive electrode thickness [m]"]*0.6},
        "neg_por_1.6":   {"Negative electrode porosity": base["Negative electrode porosity"]*1.6},
        "both_por_1.5":  {"Negative electrode porosity": base["Negative electrode porosity"]*1.5,
                          "Positive electrode porosity": base["Positive electrode porosity"]*1.5},
        "neg_rad_0.5":   {"Negative particle radius [m]": base["Negative particle radius [m]"]*0.5},
        "both_rad_0.5":  {"Negative particle radius [m]": base["Negative particle radius [m]"]*0.5,
                          "Positive particle radius [m]": base["Positive particle radius [m]"]*0.5},
        "neg_thick_1.5": {"Negative electrode thickness [m]": base["Negative electrode thickness [m]"]*1.5},
        "both_thick_1.4":{"Negative electrode thickness [m]": base["Negative electrode thickness [m]"]*1.4,
                          "Positive electrode thickness [m]": base["Positive electrode thickness [m]"]*1.4},
        "am_up_1.15":    {"Negative electrode active material volume fraction": base["Negative electrode active material volume fraction"]*1.15,
                          "Positive electrode active material volume fraction": base["Positive electrode active material volume fraction"]*1.15},
        "combo_fast":    {"Negative electrode thickness [m]": base["Negative electrode thickness [m]"]*0.6,
                          "Negative electrode porosity": base["Negative electrode porosity"]*1.4,
                          "Negative particle radius [m]": base["Negative particle radius [m]"]*0.6},
    }
    for tag, upd in moves.items():
        probe(f"{tag}@{I}A", sim, dict(base, **upd))

out = Path(f"/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/probe_{FID}.jsonl")
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", out)
