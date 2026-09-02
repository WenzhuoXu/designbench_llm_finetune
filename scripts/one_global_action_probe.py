#!/usr/bin/env python
"""How many problems fall to a SINGLE global rescaling of every member?

The argmax diagnostic in search_ladder answers "does the potential prefer the
global action", which is a property of the potential as much as of the problem.
This answers the blunt question directly: is there ANY one-shot uniform rescale

    SCALE_MULTI_PARAM([all], [r:a])   /   [t:b]   /   both

that takes the posed design straight to feasibility?  A problem with such an
action contains no allocation decision -- the whole design task is one scalar.
The grid is far denser than the search's own five factors (0.85..2.0), so this
is an upper bound on what one global action can do, not a policy result.

CPU only.  Usage:
    python scripts/one_global_action_probe.py --dirs <d1> <d2> ... [--workers 32]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm_finetune.training.rl.posterior.potential import program_from_truss_spec  # noqa: E402

R_MIN, R_MAX, T_MIN, T_MAX = 0.005, 0.15, 0.001, 0.02
# 21 factors per parameter over [0.5, 4]; both-parameter sweep is the outer product.
FACTORS = tuple(0.5 * (8.0 ** (i / 20.0)) for i in range(21))


def _state(spec, topology, goals):
    from validation.truss_executor import analyze_truss, load_truss_from_problem
    s = dict(spec)
    s["topology"] = topology
    truss = load_truss_from_problem(s)
    return analyze_truss(truss, goals)


# "declared" = the action must stay inside the problem's own shape_params bounds,
# which is exactly the filter search_ladder.candidate_actions applies, so this is
# the one-action reachability of the POLICIES' action space.  "physical" drops
# that filter (the executor itself does not enforce shape_params) and asks only
# for a pipe the simulator accepts -- an upper bound on what any agent emitting
# raw grammar could do in one move.
_BOUND_MODE = {"mode": "declared"}


def probe(path: str) -> dict | None:
    try:
        spec = json.load(open(path))
    except Exception:
        return None
    if not isinstance(spec, dict) or "topology" not in spec:
        return None
    goals = spec.get("goals") or {}
    try:
        s0 = _state(spec, spec["topology"], goals)
        m0 = float(s0.get("mass", 0.0))
    except Exception:
        return None
    prog = program_from_truss_spec(spec, initial_mass=m0)
    pid = spec.get("problem_id", Path(path).stem)
    base = spec["topology"]
    out = {"problem_id": pid, "one_global_r": False, "one_global_t": False,
           "one_global_rt": False}
    if prog.is_feasible(s0):
        return out | {"already_feasible": True}

    def try_scale(fr: float, ft: float) -> bool:
        top = copy.deepcopy(base)
        for m in top["members"]:
            sh = m["shape"]
            r = sh["r"] * fr
            t = sh["t"] * ft
            if t >= r or r <= 0 or t <= 0:
                return False
            if _BOUND_MODE["mode"] == "declared" and not (
                    R_MIN <= r <= R_MAX and T_MIN <= t <= T_MAX):
                return False
            sh["r"], sh["t"] = r, t
        try:
            st = _state(spec, top, goals)
        except Exception:
            return False
        if not prog.is_valid(st):
            return False
        return bool(prog.is_feasible(st))

    for f in FACTORS:
        if not out["one_global_r"] and try_scale(f, 1.0):
            out["one_global_r"] = True
        if not out["one_global_t"] and try_scale(1.0, f):
            out["one_global_t"] = True
    for fr in FACTORS:
        if out["one_global_rt"]:
            break
        for ft in FACTORS:
            if try_scale(fr, ft):
                out["one_global_rt"] = True
                break
    out["any_one_global"] = out["one_global_r"] or out["one_global_t"] or out["one_global_rt"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bounds", choices=("declared", "physical"), default="declared")
    a = ap.parse_args()
    _BOUND_MODE["mode"] = a.bounds
    print(f"bounds mode: {a.bounds}", flush=True)
    from multiprocessing import Pool
    rows_all = {}
    for d in a.dirs:
        files = sorted(str(p) for p in Path(d).glob("*.json"))
        with Pool(a.workers) as pool:
            rows = [r for r in pool.map(probe, files, chunksize=2) if r]
        n = len(rows)
        solvable = sum(1 for r in rows if r.get("any_one_global"))
        r_only = sum(1 for r in rows if r.get("one_global_r"))
        t_only = sum(1 for r in rows if r.get("one_global_t"))
        rt = sum(1 for r in rows if r.get("one_global_rt"))
        print(f"{Path(d).name:<24} n={n:>4}  solved by ONE global action: "
              f"{solvable}/{n} = {solvable/max(n,1):.3f}   "
              f"(r-only {r_only/max(n,1):.3f}, t-only {t_only/max(n,1):.3f}, "
              f"r&t {rt/max(n,1):.3f})", flush=True)
        rows_all[d] = rows
    if a.out:
        with open(a.out, "w") as fh:
            for d, rows in rows_all.items():
                for r in rows:
                    fh.write(json.dumps({"dir": d} | r) + "\n")


if __name__ == "__main__":
    main()
