#!/usr/bin/env python
"""Smoke + timing test for the battery domain.

Runs (1) the DesignBench battery_app model as shipped (DFN + thermal + plating +
SEI) and (2) progressively cheaper surrogates, and reports metric keys and
wall-clock per simulation.  Must be run in the isolated battery venv:
    /ocean/projects/mch250030p/wxu7/envs/battery/bin/python
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pybamm

print("pybamm", pybamm.__version__, flush=True)

report = {}


def timeit(label, fn):
    t0 = time.time()
    try:
        out = fn()
        dt = time.time() - t0
        keys = sorted(k for k in out) if isinstance(out, dict) else None
        report[label] = {"ok": True, "seconds": dt, "keys": keys,
                         "metrics": {k: v for k, v in (out or {}).items()
                                     if isinstance(v, (int, float))}}
        print(f"[{label}] OK in {dt:.2f}s  keys={keys}", flush=True)
    except Exception as exc:
        dt = time.time() - t0
        report[label] = {"ok": False, "seconds": dt,
                         "error": f"{type(exc).__name__}: {exc}",
                         "tb": traceback.format_exc()[-2000:]}
        print(f"[{label}] FAIL in {dt:.2f}s: {type(exc).__name__}: {exc}", flush=True)


# ── 1. DesignBench model as shipped ─────────────────────────────────────────
def shipped():
    from battery_app.battery_model import BatteryModel
    m = BatteryModel()
    out = m.run_simulation({})
    return {k: v for k, v in out.items() if not k.startswith("_")}


timeit("designbench_dfn_shipped", shipped)


# ── 2/3/4. handmade variants ────────────────────────────────────────────────
PATCH = {
    "Dead lithium decay rate [s-1]": 1e-06,
    "Typical plated lithium concentration [mol.m-3]": 1000,
    "Initial plated lithium concentration [mol.m-3]": 0,
    "Lithium metal partial molar volume [m3.mol-1]": 1.3e-5,
    "Exchange-current density for plating [A.m-2]": 0.001,
    "Exchange-current density for stripping [A.m-2]": 0.001,
    "Lithium plating transfer coefficient": 0.5,
    "Lithium stripping transfer coefficient": 0.5,
    "Ambient temperature [K]": 298.15,
    "Total heat transfer coefficient [W.m-2.K-1]": 10,
}


def build(model_cls, options, var_pts, solver_name="idaklu"):
    def _run():
        model = model_cls(options)
        pv = pybamm.ParameterValues("Chen2020")
        pv.update(PATCH, check_already_exists=False)
        pv.set_initial_stoichiometries(0)
        exp = pybamm.Experiment([
            "Rest for 2 min",
            "Charge at 1.5C until 4.2V",
            "Hold at 4.2V until C/50",
            "Rest for 30 min",
        ])
        sim = pybamm.Simulation(model, parameter_values=pv, experiment=exp, var_pts=var_pts)
        solver = pybamm.IDAKLUSolver() if solver_name == "idaklu" else pybamm.CasadiSolver(mode="safe", dt_max=120)
        sol = sim.solve(solver=solver)
        out = {}
        cur = sol["Current [A]"].entries
        th = sol["Time [s]"].entries / 3600.0
        import scipy.integrate
        ah = scipy.integrate.cumulative_trapezoid(-cur, th, initial=0)
        cap = float(np.max(ah))
        out["true_capacity"] = cap
        soc = ah / cap if cap > 0 else np.zeros_like(ah)
        if soc[-1] < 0.8:
            out["charge_time"] = 999.0
        else:
            i10 = int(np.argmax(soc >= 0.1)); i80 = int(np.argmax(soc >= 0.8))
            out["charge_time"] = float((sol["Time [s]"].entries[i80] - sol["Time [s]"].entries[i10]) / 60.0)
        v = sol["Terminal voltage [V]"].entries
        out["energy_density"] = float(np.trapezoid(v * abs(cur), th) / (pv["Cell volume [m3]"] * 1000) * 0.86)
        try:
            out["max_plating"] = float(np.max(sol["Negative lithium plating concentration [mol.m-3]"].entries))
        except Exception:
            out["max_plating"] = 0.0
        try:
            out["max_temperature"] = float(np.max(sol["X-averaged cell temperature [K]"].entries))
        except Exception:
            out["max_temperature"] = 298.15
        out["success"] = 1.0
        return out
    return _run


FULL_OPTS = {"thermal": "lumped", "lithium plating": "partially reversible",
             "SEI": "solvent-diffusion limited"}
LEAN_OPTS = {"thermal": "lumped", "lithium plating": "irreversible"}

timeit("dfn_full_20pts", build(pybamm.lithium_ion.DFN, FULL_OPTS,
                               {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 20, "r_p": 20}))
timeit("spme_full_20pts", build(pybamm.lithium_ion.SPMe, FULL_OPTS,
                                {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 20, "r_p": 20}))
timeit("spme_lean_10pts", build(pybamm.lithium_ion.SPMe, LEAN_OPTS,
                                {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 10, "r_p": 10}))
timeit("spm_lean_10pts", build(pybamm.lithium_ion.SPM, LEAN_OPTS,
                               {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 10, "r_p": 10}))

out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/smoke.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(report, open(out_path, "w"), indent=2)
print("\nwrote", out_path)
