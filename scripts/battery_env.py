#!/usr/bin/env python
"""Battery design environment — the second DesignBench domain.

Mirrors the role that ``DesignBench/validation/truss_executor.py`` plays for the
truss domain: a deterministic simulator that maps a design-parameter vector to a
flat dict of metrics, plus the SCALE_PARAM/MODIFY_PARAM grammar executor.

Physics is PyBaMM.  This module MUST be run in the isolated battery venv
(/ocean/projects/mch250030p/wxu7/envs/battery/bin/python) — pybamm is
deliberately absent from ``my_env`` so it cannot perturb the torch/TRL stack.

The metric dict is intentionally flat and JSON-serialisable so that the SAME
domain-general potential (llm_finetune.training.rl.posterior.potential) can be
applied to it with no battery-specific code.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

# ── design parameter vocabulary (Chen2020 names, PyBaMM >= 25) ───────────────
# NOTE: DesignBench/battery_app/actions.py lists "Negative electrode particle
# radius [m]" / "Positive electrode particle radius [m]"; neither exists in the
# Chen2020 set of current PyBaMM (the names are "Negative particle radius [m]"
# / "Positive particle radius [m]"), so every particle-size action in the
# shipped grammar silently no-ops.  We use the names PyBaMM actually exposes.
DESIGN_PARAMS: Tuple[str, ...] = (
    "Negative electrode thickness [m]",
    "Positive electrode thickness [m]",
    "Separator thickness [m]",
    "Negative electrode porosity",
    "Positive electrode porosity",
    "Negative particle radius [m]",
    "Positive particle radius [m]",
    "Negative electrode active material volume fraction",
    "Positive electrode active material volume fraction",
)

# short aliases so an LLM (and a JSON file) never has to type a bracketed unit
ALIASES: Dict[str, str] = {
    "neg_thickness": "Negative electrode thickness [m]",
    "pos_thickness": "Positive electrode thickness [m]",
    "sep_thickness": "Separator thickness [m]",
    "neg_porosity": "Negative electrode porosity",
    "pos_porosity": "Positive electrode porosity",
    "neg_radius": "Negative particle radius [m]",
    "pos_radius": "Positive particle radius [m]",
    "neg_am_fraction": "Negative electrode active material volume fraction",
    "pos_am_fraction": "Positive electrode active material volume fraction",
}
REVERSE_ALIASES = {v: k for k, v in ALIASES.items()}

# physical admissibility (porosity + active fraction + binder/CBD must be < 1)
HARD_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Negative electrode thickness [m]": (2.0e-5, 3.0e-4),
    "Positive electrode thickness [m]": (2.0e-5, 3.0e-4),
    "Separator thickness [m]": (5.0e-6, 4.0e-5),
    "Negative electrode porosity": (0.15, 0.55),
    "Positive electrode porosity": (0.15, 0.55),
    "Negative particle radius [m]": (1.0e-6, 2.0e-5),
    "Positive particle radius [m]": (1.0e-6, 2.0e-5),
    "Negative electrode active material volume fraction": (0.35, 0.80),
    "Positive electrode active material volume fraction": (0.35, 0.80),
}

_PATCH = {
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

FAILURE_METRICS: Dict[str, float] = {
    "charge_time": 999.0,
    "max_temperature": 1000.0,
    "max_plating": 1000.0,
    "energy_density": 0.0,
    "true_capacity": 0.0,
    "min_neg_potential": -1.0,
    "temperature_rise": 700.0,
    "success": 0.0,
}


def resolve(name: str) -> str:
    """Alias or full PyBaMM name -> full PyBaMM name."""
    return ALIASES.get(name, name)


def short(name: str) -> str:
    return REVERSE_ALIASES.get(name, name)


# ── simulator ────────────────────────────────────────────────────────────────

class BatterySimulator:
    """PyBaMM design evaluator with an in-process cache.

    ``fidelity`` selects the physics/cost trade-off:
      "dfn_full"  DFN + lumped thermal + partially-reversible plating + SEI
                  (the model DesignBench/battery_app/battery_model.py ships)
      "spme_full" SPMe with the same physics options
      "spme_lean" SPMe, lumped thermal + irreversible plating, coarse mesh
    """

    def __init__(self, fidelity: str = "spme_lean", c_rate: float = 1.5,
                 current_a: Optional[float] = None,
                 var_pts: Optional[dict] = None, ambient_k: float = 298.15,
                 htc: float = 10.0):
        import pybamm

        self.fidelity = fidelity
        self.c_rate = c_rate
        # A fixed charge CURRENT (A) is the honest XFC framing: the charger does
        # not know the cell's capacity, and PyBaMM's C-rate is derived from the
        # static "Nominal cell capacity [A.h]", which does not track the design.
        self.current_a = current_a
        self.ambient_k = ambient_k
        self.htc = htc
        opts_full = {"thermal": "lumped",
                     "lithium plating": "partially reversible",
                     "SEI": "solvent-diffusion limited"}
        opts_lean = {"thermal": "lumped", "lithium plating": "irreversible"}
        table = {
            "dfn_full": (pybamm.lithium_ion.DFN, opts_full,
                         {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 20, "r_p": 20}),
            "spme_full": (pybamm.lithium_ion.SPMe, opts_full,
                          {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 20, "r_p": 20}),
            "spme_lean": (pybamm.lithium_ion.SPMe, opts_lean,
                          {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 10, "r_p": 10}),
        }
        if fidelity not in table:
            raise ValueError(f"unknown fidelity {fidelity!r}; have {sorted(table)}")
        cls, opts, default_pts = table[fidelity]
        self.model = cls(opts)
        self.var_pts = var_pts or default_pts
        self._base = pybamm.ParameterValues("Chen2020")
        self._base.update(dict(_PATCH, **{"Ambient temperature [K]": ambient_k,
                                          "Total heat transfer coefficient [W.m-2.K-1]": htc}),
                          check_already_exists=False)
        self._cache: Dict[str, Dict[str, float]] = {}
        self.n_sims = 0
        self.sim_seconds = 0.0

    # -- defaults -------------------------------------------------------------
    def baseline_params(self) -> Dict[str, float]:
        return {p: float(self._base[p]) for p in DESIGN_PARAMS}

    # -- caching --------------------------------------------------------------
    @staticmethod
    def key(params: Mapping[str, float]) -> str:
        items = sorted((resolve(k), float(v)) for k, v in params.items())
        blob = ";".join(f"{k}={v:.10e}" for k, v in items)
        return hashlib.sha1(blob.encode()).hexdigest()

    # -- evaluation -----------------------------------------------------------
    def evaluate(self, params: Mapping[str, float]) -> Dict[str, float]:
        k = self.key(params)
        hit = self._cache.get(k)
        if hit is not None:
            return dict(hit)
        out = self._simulate(params)
        self._cache[k] = dict(out)
        return dict(out)

    def admissible(self, params: Mapping[str, float]) -> bool:
        """Is this parameter vector a physically buildable cell?

        The counterpart of the truss executor refusing an illegal grammar action:
        porosity + active-material fraction must leave room for binder/CBD, and
        every dimension must stay in its manufacturable range. An action that
        lands outside is ILLEGAL, not merely bad, and is dropped from the
        candidate set rather than scored.
        """
        return self._admissible(params)

    def _admissible(self, params: Mapping[str, float]) -> bool:
        for name, value in params.items():
            full = resolve(name)
            lo, hi = HARD_BOUNDS.get(full, (-math.inf, math.inf))
            if not (lo - 1e-15 <= float(value) <= hi + 1e-15):
                return False
        # solid + pore volume must leave room for binder/conductive additive
        for side in ("Negative", "Positive"):
            eps = float(params.get(f"{side} electrode porosity",
                                   self._base[f"{side} electrode porosity"]))
            am = float(params.get(f"{side} electrode active material volume fraction",
                                  self._base[f"{side} electrode active material volume fraction"]))
            if eps + am > 1.0 + 1e-9:
                return False
        return True

    def _simulate(self, params: Mapping[str, float]) -> Dict[str, float]:
        import pybamm
        import scipy.integrate

        if not self._admissible(params):
            return dict(FAILURE_METRICS)

        pv = self._base.copy()
        for name, value in params.items():
            full = resolve(name)
            if full in pv:
                pv[full] = float(value)
        try:
            pv.set_initial_stoichiometries(0)
        except Exception:
            return dict(FAILURE_METRICS)

        drive = (f"Charge at {self.current_a} A until 4.2V" if self.current_a
                 else f"Charge at {self.c_rate}C until 4.2V")
        exp = pybamm.Experiment([
            "Rest for 2 min",
            drive,
            "Hold at 4.2V until C/50",
            "Rest for 30 min",
        ])
        t0 = time.time()
        try:
            sim = pybamm.Simulation(self.model, parameter_values=pv,
                                    experiment=exp, var_pts=self.var_pts)
            try:
                solver = pybamm.IDAKLUSolver()
            except Exception:
                solver = pybamm.CasadiSolver(mode="safe", dt_max=120)
            sol = sim.solve(solver=solver)
            # A PyBaMM experiment that aborts mid-protocol still returns a
            # (truncated) Solution; the callback only logs. Treat a run that did
            # not complete all four steps as a failed evaluation, otherwise a
            # design that kills the solver looks like a 0.1 Ah cell that charges
            # in 20 seconds -- the single most dangerous artefact for a search
            # that maximises "charges fast".
            n_done = len(getattr(sol, "cycles", []) or [])
            if n_done < 4 or getattr(sol, "termination", "") != "final time":
                out = dict(FAILURE_METRICS)
            else:
                out = self._metrics(sol, pv)
        except Exception:
            out = dict(FAILURE_METRICS)
        finally:
            dt = time.time() - t0
            self.n_sims += 1
            self.sim_seconds += dt
        out["sim_seconds"] = float(dt)
        return out

    def _metrics(self, sol, pv) -> Dict[str, float]:
        import scipy.integrate

        out: Dict[str, float] = {}
        current = sol["Current [A]"].entries
        t_s = sol["Time [s]"].entries
        t_h = t_s / 3600.0
        ah = scipy.integrate.cumulative_trapezoid(-current, t_h, initial=0.0)
        capacity = float(np.max(ah))
        out["true_capacity"] = capacity
        soc = ah / capacity if capacity > 0 else np.zeros_like(ah)

        # charge time, 10% -> 80% SOC, in MINUTES
        if soc[-1] < 0.8 or capacity <= 0:
            out["charge_time"] = 999.0
        else:
            i10 = int(np.argmax(soc >= 0.1))
            i80 = int(np.argmax(soc >= 0.8))
            out["charge_time"] = float((t_s[i80] - t_s[i10]) / 60.0)

        voltage = sol["Terminal voltage [V]"].entries
        cell_vol = float(pv["Cell volume [m3]"])
        energy_wh = float(np.trapezoid(voltage * np.abs(current), t_h))
        out["energy_wh"] = energy_wh
        out["energy_density"] = float(energy_wh / (cell_vol * 1000.0) * 0.86)  # Wh/L

        try:
            out["max_plating"] = float(np.max(
                sol["Negative lithium plating concentration [mol.m-3]"].entries))
        except Exception:
            out["max_plating"] = 0.0
        try:
            tk = float(np.max(sol["X-averaged cell temperature [K]"].entries))
        except Exception:
            tk = self.ambient_k
        out["max_temperature"] = tk
        out["temperature_rise"] = tk - self.ambient_k
        try:
            out["min_neg_potential"] = float(
                np.min(sol["Negative electrode potential [V]"].entries))
        except Exception:
            out["min_neg_potential"] = 0.0
        out["success"] = 1.0
        for k, v in list(out.items()):
            if not math.isfinite(v):
                out[k] = FAILURE_METRICS.get(k, 0.0)
                out["success"] = 0.0
        return out


# ── grammar ──────────────────────────────────────────────────────────────────
# Same surface syntax as DesignBench/battery_app/actions.py, extended with the
# multi-parameter form the truss grammar has (SCALE_MULTI_PARAM).

_SCALE = re.compile(r"^\s*SCALE_PARAM\s*\(\s*([^,]+?)\s*,\s*([-\d.eE+]+)\s*\)\s*$")
_MODIFY = re.compile(r"^\s*MODIFY_PARAM\s*\(\s*([^,]+?)\s*,\s*([-\d.eE+]+)\s*\)\s*$")
_MULTI = re.compile(r"^\s*SCALE_MULTI_PARAM\s*\(\s*\[(.+?)\]\s*,\s*([-\d.eE+]+)\s*\)\s*$")


def execute_action(params: Dict[str, float], action: str) -> Optional[Dict[str, float]]:
    """Apply one grammar action; return a NEW param dict, or None if invalid."""
    m = _SCALE.match(action)
    if m:
        name, factor = resolve(m.group(1).strip()), float(m.group(2))
        if name not in params:
            return None
        nxt = dict(params)
        nxt[name] = params[name] * factor
        return nxt
    m = _MODIFY.match(action)
    if m:
        name, value = resolve(m.group(1).strip()), float(m.group(2))
        if name not in params:
            return None
        nxt = dict(params)
        nxt[name] = value
        return nxt
    m = _MULTI.match(action)
    if m:
        names = [resolve(x.strip()) for x in m.group(1).split(",")]
        factor = float(m.group(2))
        if any(n not in params for n in names):
            return None
        nxt = dict(params)
        for n in names:
            nxt[n] = params[n] * factor
        return nxt
    return None
