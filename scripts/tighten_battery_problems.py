#!/usr/bin/env python
"""Re-tighten battery problems so they are hard for a STRONG LLM AGENT.

Why this is needed. `make_battery_problems_hard.py` tightens each family's binding
constraint until "0 of the 52 one-step neighbours is feasible, but a two-step
composition is". That criterion is calibrated to PROCEDURAL SEARCH DEPTH. An LLM
allowed a heterogeneous compound move covers a two-step composition in a single
turn, so the tier-2 set is not hard for it at all: measured status-quo feasibility
on the existing 24 problems is 0.875 overall and 0.917 on the `_hard` families
(Turn 19), leaving 8-17pp of headroom -- far too little to detect a ~17pp effect.

That made the battery generality test a NON-MEASUREMENT rather than a refutation.

This script scales each problem's binding constraint by an additional factor so the
difficulty can be swept, exactly the calibration loop that produced the hard truss
set (status quo 0.380, oracle 0.663 -- a 28pp gap comfortably containing the effect).
Pick the factor whose measured status-quo rate lands near 0.35, then run the 2x2.

Only the goals are changed; topology, initial parameters, bounds and objective are
untouched, so a tightened problem is the SAME design task with a stricter target.
"""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path

# Which goal each family drives to its limit, and which direction is "harder".
# `minimum_*` are lower bounds (raise to tighten); `max_*` are upper bounds (lower).
LOWER_IS_TIGHTER = ("max_temperature", "max_charge_time", "max_plating")
HIGHER_IS_TIGHTER = ("minimum_energy_density",)

BINDING = {"rate": "max_charge_time", "thermal": "max_temperature",
           "energy": "minimum_energy_density"}


def binding_key(spec):
    fam = (spec.get("family") or "").replace("_hard", "")
    k = BINDING.get(fam)
    if k and k in (spec.get("goals") or {}):
        return k
    # `coupled` has no single declared binding metric. `_metadata.initial_violations`
    # is a LIST OF STRINGS like "energy_density=675.6<700" / "max_temperature=359.5>355",
    # so parse it and take the constraint violated by the largest RELATIVE amount.
    goals = spec.get("goals") or {}
    viol = (spec.get("_metadata") or {}).get("initial_violations") or []
    best, best_rel = None, -1.0
    for entry in viol:
        m = re.match(r"\s*([A-Za-z_]+)\s*=\s*([-\d.eE]+)\s*([<>])\s*([-\d.eE]+)", str(entry))
        if not m:
            continue
        metric, val, _op, lim = m.group(1), float(m.group(2)), m.group(3), float(m.group(4))
        cand = [g for g in goals if metric in g] or [g for g in goals if g in metric]
        if not cand:
            continue
        rel = abs(val - lim) / max(abs(lim), 1e-12)
        if rel > best_rel:
            best, best_rel = cand[0], rel
    if best:
        return best
    obj = (spec.get("objective") or {}).get("key")
    for g in goals:
        if obj and obj in g:
            return g
    return None


def tighten_spec(spec, f):
    goals = dict(spec.get("goals") or {})
    key = binding_key(spec)
    if key is None or key not in goals:
        return None
    old = float(goals[key])
    if key in LOWER_IS_TIGHTER:
        new = old * (1.0 - f)
    elif key in HIGHER_IS_TIGHTER:
        new = old * (1.0 + f)
    else:
        return None
    goals[key] = new
    out = json.loads(json.dumps(spec))
    out["goals"] = goals
    out.setdefault("_metadata", {})["retightened"] = {
        "binding_key": key, "factor": f, "old_limit": old, "new_limit": new,
        "note": "tightened for LLM-agent calibration; see tighten_battery_problems.py",
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/battery_problems")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--factor", type=float, required=True,
                    help="relative tightening of the binding constraint, e.g. 0.10")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = 0
    for f in sorted(src.glob("*.json")):
        spec = json.load(open(f))
        out = tighten_spec(spec, a.factor)
        if out is None:
            n_skip += 1
            continue
        json.dump(out, open(dst / f.name, "w"), indent=2)
        n_ok += 1
    print("tightened %d problems by %.3f -> %s   (skipped %d)" % (n_ok, a.factor, dst, n_skip))
    for f in sorted(dst.glob("*.json"))[:4]:
        m = json.load(open(f))["_metadata"]["retightened"]
        print("  %-22s %-26s %.4g -> %.4g" % (f.stem, m["binding_key"],
                                              m["old_limit"], m["new_limit"]))


if __name__ == "__main__":
    main()
