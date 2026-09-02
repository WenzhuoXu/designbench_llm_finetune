#!/usr/bin/env python
"""Depth-2 oracle: expand the K most promising 1-step neighbours exhaustively.

Gives (a) a certified 2-step reachable set for problem calibration and (b) the
best charge time known to be attainable within two grammar actions -- the
battery analogue of the truss problems' recorded LP optimum.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts")
from battery_env import BatterySimulator, DESIGN_PARAMS, short, execute_action
import battery_scan as SC   # reuse the variant table

CURRENT = float(sys.argv[1]); VARIANT = sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 12
FACTORS = (0.75, 0.9, 1.1, 1.25)
GROUPS = (("neg_thickness","pos_thickness"), ("neg_porosity","pos_porosity"),
          ("neg_radius","pos_radius"), ("neg_am_fraction","pos_am_fraction"))
ACTS = [f"SCALE_PARAM({short(p)}, {f})" for p in DESIGN_PARAMS for f in FACTORS] + \
       [f"SCALE_MULTI_PARAM([{','.join(g)}], {f})" for g in GROUPS for f in FACTORS]

sim = BatterySimulator(fidelity="dfn_full", current_a=CURRENT)
base = dict(sim.baseline_params(), **SC.VARIANTS[VARIANT])
rows = []

def rec(tag, params, depth):
    if not sim.admissible(params):
        return None
    m = sim.evaluate(params)
    rows.append({"current_a": CURRENT, "variant": VARIANT, "tag": tag, "depth": depth,
                 **{k: round(float(v), 6) for k, v in m.items()}})
    return m

rec("BASE", base, 0)
d1 = []
for a in ACTS:
    p = execute_action(dict(base), a)
    if p is None:
        continue
    m = rec(a, p, 1)
    if m and m.get("success") == 1.0:
        d1.append((a, p, m))

# expand the K most diverse/promising: coolest, fastest, densest, plus the rest
picks = []
for key, rev in (("max_temperature", False), ("charge_time", False),
                 ("energy_density", True), ("max_plating", False)):
    for a, p, m in sorted(d1, key=lambda x: x[2][key], reverse=rev)[:K // 3 + 1]:
        if a not in [q[0] for q in picks]:
            picks.append((a, p, m))
picks = picks[:K]
print(f"expanding {len(picks)} of {len(d1)} one-step states", flush=True)
for a1, p1, _ in picks:
    for a2 in ACTS:
        p2 = execute_action(dict(p1), a2)
        if p2 is None:
            continue
        rec(f"{a1} ; {a2}", p2, 2)
    print(f"  done {a1}  (sims={sim.n_sims}, {sim.sim_seconds:.0f}s)", flush=True)

out = Path(f"/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/oracle2_{VARIANT}_{int(CURRENT)}A.jsonl")
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", out, "n=", len(rows), "sims=", sim.n_sims, "sim_s=", round(sim.sim_seconds, 1))
