#!/usr/bin/env python
"""One battery problem per (variant, current) scan, with a difficulty knob.

Why not `make_battery_problems.py`: it hardcodes 12 SPECS drawn from 10 scans, and
its `_hard` tier doubles that to 24 problems over the SAME 10 scans. The honest
clustering unit is the scan -- a problem and its tightened twin share a starting cell
and a load case -- so 24 problems buy ~10-24 effective units and an MDE of 20.2pp,
larger than the 18.3pp interface effect they are meant to test (Turn 19/21).

Each (variant, current) pair IS a genuinely different design task: a different
starting cell under a different charge current. So: scan the grid, emit ONE problem
per scan, and cluster by construction.

Difficulty knob `q`. For the binding metric the scan gives a BASE value and a best
achievable value over the certified 1- and 2-step neighbours. Setting the goal at

    limit = best + q * (base - best)          (for a "lower is better" metric)

makes BASE infeasible by construction, keeps feasibility reachable by construction,
and slides difficulty smoothly: q -> 0 admits only the very best neighbours, q -> 1
admits almost anything. Both facts are then re-checked against the recorded rows
rather than assumed. Calibrate q the way the truss hard set and the t030 battery set
were calibrated -- run the status-quo cell and pick the q landing near 0.45.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

RES = Path("/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder")

# metric -> (goal key, "lower better"?)
METRICS = {
    "charge_time": ("max_charge_time", True),
    "max_temperature": ("max_temperature", True),
    "max_plating": ("max_plating", True),
    "energy_density": ("minimum_energy_density", False),
}
# a metric is only usable as the binding one if the scan actually moves it
BINDING_CANDIDATES = ("charge_time", "energy_density", "max_temperature")


def satisfies(row, goals):
    for metric, (gk, lower_better) in METRICS.items():
        if gk not in goals or metric not in row:
            continue
        v, lim = float(row[metric]), float(goals[gk])
        if lower_better and v > lim:
            return False
        if not lower_better and v < lim:
            return False
    return True


def build(variant, current, q, slack=0.15):
    f = RES / f"scan_{variant}_{int(current)}A.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r.get("success", 1.0)]
    base = next((r for r in rows if r.get("tag") == "BASE"), None)
    nb = [r for r in rows if r.get("depth", 0) >= 1]
    if base is None or not nb:
        return None

    # pick the binding metric: the one the neighbourhood improves most, relatively
    best_metric, best_gain, best_val = None, 0.0, None
    for m in BINDING_CANDIDATES:
        if m not in base:
            continue
        lower = METRICS[m][1]
        vals = [float(r[m]) for r in nb if m in r]
        if not vals:
            continue
        bv = min(vals) if lower else max(vals)
        b0 = float(base[m])
        gain = (b0 - bv) / max(abs(b0), 1e-12) if lower else (bv - b0) / max(abs(b0), 1e-12)
        if gain > best_gain:
            best_metric, best_gain, best_val = m, gain, bv
    if best_metric is None or best_gain <= 0.02:
        return None                      # nothing to optimise here

    lower = METRICS[best_metric][1]
    b0 = float(base[best_metric])
    limit = best_val + q * (b0 - best_val)          # works for both senses

    # Non-binding goals cannot be derived from BASE alone. The neighbour that reaches
    # a high energy density also charges slowly, so a charge-time goal set at BASE+slack
    # excludes it and NOTHING is reachable -- which is why the first version emitted 0
    # problems from all 29 scans. Derive them from a TARGET neighbour that already
    # satisfies the binding constraint, loose enough that both BASE and the target pass.
    passing = []
    for r in nb:
        if best_metric not in r:
            continue
        v = float(r[best_metric])
        if (lower and v <= limit) or ((not lower) and v >= limit):
            passing.append(r)
    if not passing:
        return None
    obj_lower = METRICS["charge_time"][1]
    target = min(passing, key=lambda r: float(r.get("charge_time", 1e18))) if obj_lower \
        else passing[0]

    goals = {}
    for m, (gk, lb) in METRICS.items():
        if m == best_metric:
            goals[gk] = float(limit)
        elif m in base:
            vb, vt = float(base[m]), float(target.get(m, base[m]))
            goals[gk] = float(max(vb, vt) * (1 + slack) if lb
                              else min(vb, vt) * (1 - slack))
    if "max_plating" in goals:
        goals["max_plating"] = max(goals["max_plating"], 1e-5)

    if satisfies(base, goals):
        return None                       # must start INFEASIBLE
    reach = [r for r in nb if satisfies(r, goals)]
    if not reach:
        return None                       # must be REACHABLE

    from battery_env import ALIASES, HARD_BOUNDS
    import battery_scan as BS
    init = dict(BS.CHEN, **BS.VARIANTS[variant]) if hasattr(BS, "CHEN") else None
    design_params = {}
    for alias, full in ALIASES.items():
        lo, hi = HARD_BOUNDS[full]
        d = {"min": lo, "max": hi}
        if init is not None and alias in init:
            d["initial"] = float(init[alias])
        design_params[alias] = d

    pid = f"batg_{variant}_{int(current)}A"
    return {
        "problem_id": pid, "domain": "battery", "family": f"grid_{best_metric}",
        "description": (f"Li-ion cell design ({variant} starting point) charged at "
                        f"{current:g} A. Resize the electrode geometry to meet every goal."),
        "operating_conditions": {"charge_current_a": float(current),
                                 "ambient_temperature_k": 298.15,
                                 "heat_transfer_coefficient": 10.0,
                                 "parameter_set": "Chen2020"},
        "goals": goals,
        "objective": {"key": "charge_time", "sense": "min",
                      "ref": float(goals.get("max_charge_time", base["charge_time"]))},
        "optimization": {"design_params": design_params},
        "scales": {"charge_time": float(goals.get("max_charge_time", 1.0)),
                   "max_temperature": 30.0,
                   "energy_density": float(goals.get("minimum_energy_density", 1.0)),
                   "max_plating": 1.0},
        "_metadata": {
            "generated_by": "make_battery_problems_grid.py",
            "binding_metric": best_metric, "q": q,
            "base_value": b0, "best_reachable": best_val, "limit": float(limit),
            "n_neighbours_scanned": len(nb), "n_neighbours_feasible": len(reach),
            "initial_metrics": {k: float(base[k]) for k in
                                ("charge_time", "energy_density", "max_temperature",
                                 "max_plating") if k in base},
            "initial_feasible": False,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True)
    ap.add_argument("--q", type=float, default=0.25)
    a = ap.parse_args()
    import sys
    sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts")

    dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    made, skipped = 0, 0
    for f in sorted(RES.glob("scan_*.jsonl")):
        stem = f.stem[len("scan_"):]
        if "_" not in stem:
            skipped += 1; continue        # legacy scan_8A.jsonl with no variant
        variant, cur = stem.rsplit("_", 1)
        try:
            current = float(cur.rstrip("A"))
        except ValueError:
            skipped += 1; continue
        spec = build(variant, current, a.q)
        if spec is None:
            skipped += 1; continue
        json.dump(spec, open(dst / (spec["problem_id"] + ".json"), "w"), indent=2)
        made += 1
    print("q=%.2f -> %d problems in %s  (skipped %d scans)" % (a.q, made, dst, skipped))
    for p in sorted(dst.glob("*.json"))[:5]:
        m = json.load(open(p))["_metadata"]
        print("  %-22s binding=%-16s base=%.4g best=%.4g limit=%.4g  reachable %d/%d"
              % (p.stem, m["binding_metric"], m["base_value"], m["best_reachable"],
                 m["limit"], m["n_neighbours_feasible"], m["n_neighbours_scanned"]))


if __name__ == "__main__":
    main()
