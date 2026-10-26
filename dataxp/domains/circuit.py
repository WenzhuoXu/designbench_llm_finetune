"""Analog two-stage Miller op-amp sizing via square-law hand equations.

The textbook analytic model (no SPICE): a 5-element sizing problem over the
classic two-stage Miller-compensated CMOS op-amp,

    element 0  M1   input diff-pair width          W1  = 2.0  * x  [um]
    element 1  M2   second-stage gm device width   W2  = 4.0  * x  [um]
    element 2  MB   tail->stage-2 mirror ratio     B   = 0.5  * x  [-]
    element 3  CC   Miller compensation cap        Cc  = 0.2  * x  [pF]
    element 4  IB   tail/bias current              Ib  = 2.0  * x  [uA]

with ONE uniform dimensionless knob ``x`` in [1, 100] per element (so the
generic scale-factor move set and the shared ``bounds()`` apply to all five).

Square-law model (units: uA, uS, MOhm, pF, MHz, V/us, mW, um^2):
    Id1 = Ib/2 ;  I2 = B*Ib
    gm  = sqrt(2 k (W/L) Id) ;  ro-pair R = 1/(2 lambda Id)
    Av  = gm1 R1 * gm2 R2                       -> gain_db
    GBW = gm1/(2 pi Cc) ;  p2 = gm2/(2 pi CL) ;  RHP zero z = gm2/(2 pi Cc)
    PM  = 90 - atan(GBW/p2) - atan(GBW/z)       [deg]
    SR  = min(Ib/Cc, I2/CL)                     [V/us]
    P   = (2 Ib + I2) Vdd                       [mW]
    area= 2 W1 L + W2 L + (1+B) Wmb L + 200*Cc  [um^2]  (objective, minimise)

Validity (the classic analytic-model exploit): the square law only holds in
strong inversion + saturation. ``min_vov`` is the minimum overdrive across
devices; ``vov_defect`` = hinge(0.03 - min_vov) + hinge(max_vov - 0.35 Vdd)
must be exactly 0 inside the envelope. A search that pumps W at fixed Id gets
sqrt(W) gm forever from the formula while the real device sits in weak
inversion - the envelope kills those states.

Stochastic mode = process mismatch Monte-Carlo: per-seed multiplicative draws
on k per device (sigma 3-8 %) and a mirror current error from Vth mismatch
(dI/I = 2 dVth / Vov_mirror), plus a bias-reference draw. Chance constraints
over seeds = yield.

Problems are solvable by construction: goals are set as margins off a random
valid WITNESS design (rejection-sampled with the problem's rng). ~78 % of
problems start at minimum sizes, which fail the speed/PM specs (constraint
restoration data); the rest start AT the witness with relaxed goals (pure
area minimisation from a feasible start), which also gives the initial
violation distribution its required spread.

Compatibility: python >= 3.10, numpy only (also runs on 3.12 / numpy 1.26).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from ..domain import Domain, stable_seed
from ..potential import DesignProgram, program_from_goals

_ELEMENTS = ("M1", "M2", "MB", "CC", "IB")
# physical value at x = 1 :   W1 um  W2 um  ratio  Cc pF  Ib uA
_BASE = (2.0, 4.0, 0.5, 0.2, 2.0)
_X_LO, _X_HI = 1.0, 100.0

_L_UM = 1.0                 # channel length, all devices
_W_MB_UM = 5.0              # mirror unit-device width
_CAP_UM2_PER_PF = 200.0     # cap area density
_VOV_MIN = 0.03             # V; below this the square law is fiction
_TWO_PI = 2.0 * math.pi

_FAMILIES = ("low_power", "high_speed", "general")

_WORST = {
    "gain_db": float("-inf"),
    "gbw_mhz": float("-inf"),
    "phase_margin_deg": float("-inf"),
    "slew_v_per_us": float("-inf"),
    "power_mw": float("inf"),
    "area": float("inf"),
    "min_vov": float("-inf"),
    "max_vov": float("inf"),
    "vov_defect": float("inf"),
}


def _square_law(tech: dict, W1, W2, B, Cc, Ib,
                kf1=1.0, kf2=1.0, kfb=1.0, mirror_err=0.0) -> dict:
    """Vectorised hand analysis. All args scalar or broadcastable arrays."""
    kn = tech["kn_ua_v2"]
    lam = tech["lambda_per_v"]
    vdd = tech["vdd_v"]
    cl = tech["cl_pf"]
    L = tech["l_um"]

    Id1 = 0.5 * Ib                                   # uA per input device
    I2 = np.maximum(B * Ib * (1.0 + mirror_err), 1e-3)
    k1 = kn * kf1
    k2 = kn * kf2
    kb = kn * kfb

    gm1 = np.sqrt(2.0 * k1 * (W1 / L) * Id1)         # uS
    gm2 = np.sqrt(2.0 * k2 * (W2 / L) * I2)
    r1 = 1.0 / (2.0 * lam * Id1)                     # MOhm ~ ro2 || ro4
    r2 = 1.0 / (2.0 * lam * I2)                      # MOhm ~ ro6 || ro7
    av = gm1 * r1 * gm2 * r2
    gain_db = 20.0 * np.log10(np.maximum(av, 1e-9))

    gbw = gm1 / (_TWO_PI * Cc)                       # MHz
    p2 = gm2 / (_TWO_PI * cl)                        # MHz
    zrhp = gm2 / (_TWO_PI * Cc)                      # MHz (RHP zero)
    pm = (90.0 - np.degrees(np.arctan(gbw / p2))
          - np.degrees(np.arctan(gbw / zrhp)))

    slew = np.minimum(Ib / Cc, I2 / cl)              # V/us
    power = (2.0 * Ib + I2) * vdd * 1e-3             # mW

    vov1 = np.sqrt(2.0 * Id1 / (k1 * (W1 / L)))      # V
    vov2 = np.sqrt(2.0 * I2 / (k2 * (W2 / L)))
    vovb = np.sqrt(2.0 * Ib / (kb * (_W_MB_UM / L)))
    min_vov = np.minimum(np.minimum(vov1, vov2), vovb)
    max_vov = np.maximum(np.maximum(vov1, vov2), vovb)
    vov_hi = max(0.4, 0.35 * vdd)
    vov_defect = (np.maximum(0.0, _VOV_MIN - min_vov)
                  + np.maximum(0.0, max_vov - vov_hi))

    area = (2.0 * W1 * L + W2 * L + (1.0 + B) * _W_MB_UM * L
            + _CAP_UM2_PER_PF * Cc)

    return {
        "gain_db": gain_db,
        "gbw_mhz": gbw,
        "phase_margin_deg": pm,
        "slew_v_per_us": slew,
        "power_mw": power,
        "area": area,
        "min_vov": min_vov,
        "max_vov": max_vov,
        "vov_defect": vov_defect,
    }


def _phys(x):
    """x (..., 5) in knob units -> physical (W1, W2, B, Cc, Ib)."""
    x = np.asarray(x, dtype=float)
    return (_BASE[0] * x[..., 0], _BASE[1] * x[..., 1], _BASE[2] * x[..., 2],
            _BASE[3] * x[..., 3], _BASE[4] * x[..., 4])


_GOAL_KEYS = ("minimum_gain_db", "minimum_gbw_mhz", "minimum_phase_margin_deg",
              "minimum_slew_v_per_us", "maximum_power_mw")
_GOAL_METRIC = {
    "minimum_gain_db": "gain_db",
    "minimum_gbw_mhz": "gbw_mhz",
    "minimum_phase_margin_deg": "phase_margin_deg",
    "minimum_slew_v_per_us": "slew_v_per_us",
    "maximum_power_mw": "power_mw",
}


def _goal_ok(goals: dict, metrics: dict) -> bool:
    for gk, mk in _GOAL_METRIC.items():
        if gk not in goals:
            continue
        v = float(metrics[mk])
        if not math.isfinite(v):
            return False
        if gk.startswith("minimum_"):
            if v < goals[gk]:
                return False
        elif v > goals[gk]:
            return False
    return True


class CircuitDomain(Domain):
    """Two-stage Miller op-amp sizing (square-law hand equations)."""

    name = "circuit"
    params = ("x",)
    # Coarser than the default ladder: sizing spans two decades, and under
    # mismatch noise the top-2 Phi margin must clear the replicate band.
    factors = (0.7, 0.85, 1.2, 1.5)
    reference_class = "heuristic"   # best feasible of a seeded coarse search

    def __init__(self, stochastic: bool = False):
        super().__init__(stochastic)

    # -- problems ------------------------------------------------------------
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[index % len(_FAMILIES)]

        # technology corner: fixed per family with mild per-problem draws
        kn = float(rng.uniform(160.0, 240.0))            # uA/V^2
        lam = float(rng.uniform(0.12, 0.22))             # 1/V
        vdd_base = {"low_power": 1.8, "high_speed": 3.3, "general": 2.5}[family]
        vdd = float(vdd_base * rng.uniform(0.95, 1.05))
        cl_lo, cl_hi = {"low_power": (2.0, 6.0), "high_speed": (1.0, 3.0),
                        "general": (3.0, 8.0)}[family]
        cl = float(rng.uniform(cl_lo, cl_hi))            # pF
        tech = {"kn_ua_v2": kn, "lambda_per_v": lam, "vdd_v": vdd,
                "cl_pf": cl, "l_um": _L_UM}

        mc = {"sigma_k": float(rng.uniform(0.03, 0.06)),
              "sigma_vth_v": float(rng.uniform(0.002, 0.004))}

        pm_goal = float(rng.uniform(45.0, 60.0))

        # witness: a random VALID design the goals will be anchored to,
        # so a feasible design exists by construction.
        ib_lo, ib_hi = {"low_power": (1.0, 12.0), "high_speed": (8.0, 100.0),
                        "general": (2.0, 40.0)}[family]
        n_w = 300
        Xw = np.exp(rng.uniform(
            np.log([2.0, 2.0, 1.0, 3.0, ib_lo]),
            np.log([100.0, 100.0, 30.0, 90.0, ib_hi]),
            size=(n_w, 5)))
        mw = _square_law(tech, *_phys(Xw))
        vov_hi = max(0.4, 0.35 * vdd)
        vov_ok = (mw["min_vov"] >= 0.04) & (mw["max_vov"] <= vov_hi - 0.02)
        sane = (vov_ok & (mw["gain_db"] >= 52.0) & (mw["gbw_mhz"] >= 1.0)
                & (mw["slew_v_per_us"] >= 0.4) & (mw["power_mw"] <= 40.0))

        # ~22 % of problems start AT the witness with relaxed goals: pure
        # area minimisation from a feasible start. This is real training
        # signal AND gives the initial-violation distribution its spread.
        feasible_start = bool(rng.random() < 0.18)
        x_w = None
        if feasible_start:
            pm_goal = min(pm_goal, 50.0)   # keep a strong witness findable
            strong = (sane & (mw["phase_margin_deg"] >= pm_goal + 6.0)
                      & (mw["max_vov"] <= 0.9 * vov_hi))
            if strong.any():
                x_w = Xw[int(np.argmax(strong))]
            else:
                feasible_start = False
        if x_w is None:
            accept = sane & (mw["phase_margin_deg"] >= pm_goal + 2.0)
            if not accept.any():             # relax: any sane witness, PM adapts
                accept = sane
            if accept.any():
                x_w = Xw[int(np.argmax(accept))]     # first accepted draw
            else:                            # essentially unreachable fallback
                x_w = np.array([10.0, 20.0, 8.0, 25.0, 10.0])
        w = {k: float(np.asarray(v).reshape(-1)[0]) for k, v in
             _square_law(tech, *_phys(x_w.reshape(1, 5))).items()}
        pm_margin = 6.0 if feasible_start else 2.0
        pm_goal = float(min(pm_goal, w["phase_margin_deg"] - pm_margin))
        pm_goal = float(max(pm_goal, 30.0))

        # spec corners: which constraints are tight is the family identity
        if feasible_start:
            gn_lo, gn_hi = 0.82, 0.93
            sp_lo, sp_hi = 0.45, 0.75
            pw_lo, pw_hi = 1.25, 1.7
        else:
            gn_lo, gn_hi = 0.85, 0.97
            sp_lo, sp_hi = {"low_power": (0.35, 0.65), "high_speed": (0.72, 0.95),
                            "general": (0.50, 0.85)}[family]
            pw_lo, pw_hi = {"low_power": (1.05, 1.35), "high_speed": (1.6, 3.0),
                            "general": (1.25, 2.0)}[family]
        goals = {
            "minimum_gain_db": round(w["gain_db"] * float(rng.uniform(gn_lo, gn_hi)), 4),
            "minimum_gbw_mhz": round(w["gbw_mhz"] * float(rng.uniform(sp_lo, sp_hi)), 5),
            "minimum_phase_margin_deg": round(pm_goal, 3),
            "minimum_slew_v_per_us": round(w["slew_v_per_us"] * float(rng.uniform(sp_lo, sp_hi)), 5),
            "maximum_power_mw": round(w["power_mw"] * float(rng.uniform(pw_lo, pw_hi)), 5),
        }

        if feasible_start:
            x0 = np.array(x_w, dtype=float)
        else:
            # initial design: minimum sizes (valid, but misses the speed specs)
            x0 = None
            for _ in range(30):
                cand = np.array([rng.uniform(1.0, 2.2), rng.uniform(1.0, 1.8),
                                 rng.uniform(1.0, 2.0), rng.uniform(2.5, 12.5),
                                 rng.uniform(1.0, 2.5)])
                m0 = _square_law(tech, *_phys(cand))
                if (float(m0["min_vov"]) >= 0.036
                        and float(m0["max_vov"]) <= vov_hi - 0.05):
                    x0 = cand
                    break
            if x0 is None:
                x0 = np.array([1.0, 1.0, 1.0, 5.0, 2.0])
            m0 = {k: float(v) for k, v in _square_law(tech, *_phys(x0)).items()}

            # if the draw happens to start feasible, pull the speed goals up
            # toward the witness (difficulty stays graded; witness stays feasible)
            if _goal_ok(goals, m0):
                for gk, wv, iv in (("minimum_gbw_mhz", w["gbw_mhz"], m0["gbw_mhz"]),
                                   ("minimum_slew_v_per_us", w["slew_v_per_us"],
                                    m0["slew_v_per_us"])):
                    uplift = float(rng.uniform(1.4, 2.4))
                    goals[gk] = round(max(goals[gk],
                                          min(0.95 * wv, iv * uplift)), 5)

        return {
            "problem_id": "circuit_%04d" % index,
            "family": family,
            "goals": goals,
            "tech": tech,
            "mc": mc,
            "x0": [float(v) for v in x0],
            "x_witness": [float(v) for v in x_w],
            "starts_feasible_by_design": bool(feasible_start),
        }

    def load(self, spec: dict):
        x = np.clip(np.asarray(spec["x0"], dtype=float), _X_LO, _X_HI)
        return {"spec": spec, "x": x}

    def clone(self, st):
        # spec is never mutated by set(); the knob vector is.
        return {"spec": st["spec"], "x": np.array(st["x"], dtype=float)}

    # -- simulation ----------------------------------------------------------
    def evaluate(self, st, seed: int = 0) -> dict:
        self.calls += 1
        try:
            spec = st["spec"]
            tech = spec["tech"]
            x = np.clip(np.asarray(st["x"], dtype=float), _X_LO, _X_HI)
            W1, W2, B, Cc, Ib = _phys(x)

            kf1 = kf2 = kfb = 1.0
            mirror_err = 0.0
            if self.stochastic:
                r = np.random.default_rng(int(seed) & 0xFFFFFFFF)
                sk = float(spec.get("mc", {}).get("sigma_k", 0.05))
                svth = float(spec.get("mc", {}).get("sigma_vth_v", 0.006))
                zk = r.standard_normal(3)          # k mismatch M1, M2, MB
                zi = r.standard_normal()           # bias reference error
                zv = r.standard_normal()           # mirror Vth mismatch
                kf1 = float(np.clip(1.0 + sk * zk[0], 0.7, 1.3))
                kf2 = float(np.clip(1.0 + sk * zk[1], 0.7, 1.3))
                kfb = float(np.clip(1.0 + sk * zk[2], 0.7, 1.3))
                Ib = Ib * float(np.clip(1.0 + 0.5 * sk * zi, 0.75, 1.25))
                vovb = math.sqrt(2.0 * Ib / (tech["kn_ua_v2"] * kfb
                                             * (_W_MB_UM / tech["l_um"])))
                # Pelgrom: Vth mismatch shrinks with mirror device area
                area_mb = (1.0 + float(B)) * _W_MB_UM * tech["l_um"]
                svth_eff = svth / math.sqrt(max(area_mb / 10.0, 0.25))
                mirror_err = float(np.clip(
                    2.0 * svth_eff / max(vovb, 0.08) * zv, -0.6, 0.6))

            m = _square_law(tech, W1, W2, B, Cc, Ib,
                            kf1=kf1, kf2=kf2, kfb=kfb, mirror_err=mirror_err)
            out = {k: float(v) for k, v in m.items()}
            for v in out.values():
                if math.isnan(v):
                    return dict(_WORST)
            return out
        except Exception:
            return dict(_WORST)

    # -- design variables ------------------------------------------------------
    def n_elements(self, st) -> int:
        return len(_ELEMENTS)

    def get(self, st, i: int, p: str) -> Optional[float]:
        if p != "x" or not (0 <= i < len(_ELEMENTS)):
            return None
        return float(st["x"][i])

    def set(self, st, i: int, p: str, v: float) -> None:
        if p != "x" or not (0 <= i < len(_ELEMENTS)):
            return
        st["x"][i] = float(min(_X_HI, max(_X_LO, float(v))))

    def bounds(self, st, p: str) -> Tuple[float, float]:
        return (_X_LO, _X_HI)

    # -- program / reference ---------------------------------------------------
    def program(self, spec: dict) -> DesignProgram:
        prog = program_from_goals(
            spec["goals"],
            objective_key="area",
            objective_ref=self.reference(spec),
            objective_sense="min",
            domain=self.name,
        )
        # Envelope: square-law validity (saturation / strong inversion), a
        # sane phase margin magnitude, and a bounded layout area.
        return replace(prog, validity_bounds=(
            ("vov_defect", 1e-9),
            ("phase_margin_deg", 120.0),
            ("area", 1.0e6),
        ))

    def reference(self, spec: dict) -> Optional[float]:
        cached = spec.get("_ref_area")
        if cached is not None:
            return float(cached)
        rng = np.random.default_rng(
            stable_seed(spec.get("problem_id", ""), spec.get("family", ""),
                        "circuit_ref"))
        n = 2000
        X = np.exp(rng.uniform(math.log(_X_LO), math.log(_X_HI), size=(n, 5)))
        if spec.get("x_witness"):
            X[0] = np.clip(np.asarray(spec["x_witness"], dtype=float),
                           _X_LO, _X_HI)
        m = _square_law(spec["tech"], *_phys(X))
        goals = spec["goals"]
        feas = np.isfinite(m["area"]) & (m["vov_defect"] <= 0.0)
        for gk, mk in _GOAL_METRIC.items():
            if gk not in goals:
                continue
            if gk.startswith("minimum_"):
                feas &= m[mk] >= goals[gk]
            else:
                feas &= m[mk] <= goals[gk]
        if feas.any():
            best = float(np.min(np.asarray(m["area"])[feas]))
        else:  # unreachable when x_witness is present; still return a scale
            best = float(np.median(np.asarray(m["area"])))
        best = max(best, 1e-6)
        spec["_ref_area"] = best
        return best

    # -- depth-0 feedback hint ---------------------------------------------------
    def hint(self, st) -> Optional[Tuple[int, str, str]]:
        try:
            spec = st["spec"]
            x = np.clip(np.asarray(st["x"], dtype=float), _X_LO, _X_HI)
            m = {k: float(v) for k, v in
                 _square_law(spec["tech"], *_phys(x)).items()}
        except Exception:
            return None
        if not all(math.isfinite(v) for v in m.values()):
            return None
        goals = spec["goals"]

        rel = {}
        for gk, mk in _GOAL_METRIC.items():
            if gk not in goals:
                continue
            limit = float(goals[gk])
            scale = max(abs(limit), 1e-9)
            if gk.startswith("minimum_"):
                rel[gk] = (limit - m[mk]) / scale
            else:
                rel[gk] = (m[mk] - limit) / scale
        if not rel:
            return None
        worst_key = max(rel, key=lambda k: rel[k])
        if rel[worst_key] <= 0.0:
            # feasible: chase the objective - the cap dominates the area
            if x[3] > _X_LO + 1e-6:
                return (3, "x", "down")
            return None

        at_hi = lambda i: x[i] >= _X_HI - 1e-6
        at_lo = lambda i: x[i] <= _X_LO + 1e-6
        if worst_key == "minimum_gain_db":                 # gain -> W1 up
            return (0, "x", "up") if not at_hi(0) else (1, "x", "up")
        if worst_key == "minimum_gbw_mhz":                 # gbw -> W1 up | Cc dn
            return (0, "x", "up") if not at_hi(0) else (3, "x", "down")
        if worst_key == "minimum_phase_margin_deg":        # PM -> Cc up
            return (3, "x", "up") if not at_hi(3) else (1, "x", "up")
        if worst_key == "minimum_slew_v_per_us":           # slew -> Ibias up
            return (4, "x", "up") if not at_hi(4) else (3, "x", "down")
        if worst_key == "maximum_power_mw":                # power -> Ibias down
            return (4, "x", "down") if not at_lo(4) else (2, "x", "down")
        return None

    # -- observation (policy-facing text) ---------------------------------------
    def describe(self, st, spec) -> str:
        """Self-contained plain-text observation of the current design:
        PROBLEM (topology, technology constants, knob meaning), DESIGN (one line
        per element) and RESULTS (per-device operating point and the
        performance quantity each element drives). Per-element results are the
        nominal (mismatch-free) square-law evaluation; aggregate metrics and
        their limits are rendered elsewhere. Pure: st is not mutated."""
        spec = spec if spec is not None else st["spec"]
        x = np.clip(np.asarray(st["x"], dtype=float).reshape(-1), _X_LO, _X_HI)
        op = _obs_operating_point(spec["tech"], x)
        return _obs_text(spec, x, op, self.stochastic)

    def element_label(self, st, i) -> str:
        """Short role label for element i, used inline in the action line."""
        i = int(i)
        if 0 <= i < len(_OBS_LABELS):
            return _OBS_LABELS[i]
        return "element %d" % i


# ---------------------------------------------------------------------------
# describe() helpers (observation only; nothing above depends on them)
# ---------------------------------------------------------------------------
_OBS_LABELS = (
    "M1 (element 0, input differential pair)",
    "M2 (element 1, second-stage gain device)",
    "MB (element 2, tail-to-stage-2 current-mirror ratio)",
    "CC (element 3, Miller compensation capacitor)",
    "IB (element 4, tail/bias current)",
)


def _obs_num(v, sig: int = 4) -> str:
    """Significant-figure formatting without exponents (a nonzero 0.00013
    prints as 0.00013, never 0) and without thousands separators."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return str(f)
    if f == 0.0:
        return "0"
    mag = int(math.floor(math.log10(abs(f))))
    dec = sig - 1 - mag
    if dec <= 0:
        return "%d" % int(round(f))
    s = "%.*f" % (dec, f)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _obs_operating_point(tech: dict, x) -> dict:
    """Nominal per-device hand analysis (same square-law equations as
    _square_law with no mismatch), itemised per element."""
    kn = float(tech["kn_ua_v2"])
    lam = float(tech["lambda_per_v"])
    vdd = float(tech["vdd_v"])
    cl = float(tech["cl_pf"])
    L = float(tech["l_um"])
    W1, W2, B, Cc, Ib = (float(v) for v in _phys(x))

    id1 = 0.5 * Ib
    i2 = max(B * Ib, 1e-3)
    gm1 = math.sqrt(2.0 * kn * (W1 / L) * id1)
    gm2 = math.sqrt(2.0 * kn * (W2 / L) * i2)
    r1 = 1.0 / (2.0 * lam * id1)
    r2 = 1.0 / (2.0 * lam * i2)
    a1 = gm1 * r1
    a2 = gm2 * r2
    gbw = gm1 / (_TWO_PI * Cc)
    p2 = gm2 / (_TWO_PI * cl)
    z = gm2 / (_TWO_PI * Cc)
    vov1 = math.sqrt(2.0 * id1 / (kn * (W1 / L)))
    vov2 = math.sqrt(2.0 * i2 / (kn * (W2 / L)))
    vovb = math.sqrt(2.0 * Ib / (kn * (_W_MB_UM / L)))
    area = {
        0: 2.0 * W1 * L,
        1: W2 * L,
        2: (1.0 + B) * _W_MB_UM * L,
        3: _CAP_UM2_PER_PF * Cc,
    }
    return {
        "W1": W1, "W2": W2, "B": B, "Cc": Cc, "Ib": Ib,
        "kn": kn, "lam": lam, "vdd": vdd, "cl": cl, "L": L,
        "id1": id1, "i2": i2, "gm1": gm1, "gm2": gm2, "r1": r1, "r2": r2,
        "a1": a1, "a2": a2,
        "a1_db": 20.0 * math.log10(max(a1, 1e-9)),
        "a2_db": 20.0 * math.log10(max(a2, 1e-9)),
        "p2": p2, "z": z,
        "p2_over_gbw": p2 / gbw, "z_over_gbw": z / gbw,
        "lag_p2": math.degrees(math.atan(gbw / p2)),
        "lag_z": math.degrees(math.atan(gbw / z)),
        "sr_int": Ib / Cc, "sr_out": i2 / cl,
        "p_bias": 2.0 * Ib * vdd * 1e-3, "p_stage2": i2 * vdd * 1e-3,
        "vov1": vov1, "vov2": vov2, "vovb": vovb,
        "vov_lo": _VOV_MIN, "vov_hi": max(0.4, 0.35 * vdd),
        "area": area,
    }


def _obs_vov_note(v: float, lo: float, hi: float) -> str:
    if v < lo:
        return "BELOW the %s V floor: weak inversion, invalid" % _obs_num(lo)
    if v > hi:
        return "ABOVE the %s V ceiling: out of saturation, invalid" % _obs_num(hi)
    return "%s V above floor, %s V below ceiling" % (_obs_num(v - lo, 3),
                                                     _obs_num(hi - v, 3))


def _obs_text(spec: dict, x, op: dict, stochastic: bool) -> str:
    n = _obs_num
    area_tot = sum(op["area"].values())
    pw_tot = op["p_bias"] + op["p_stage2"]

    def a_pct(i):
        return n(100.0 * op["area"][i] / area_tot, 3) if area_tot > 0 else "0"

    def p_pct(v):
        return n(100.0 * v / pw_tot, 3) if pw_tot > 0 else "0"

    fam = str(spec.get("family", "general"))
    L = n(op["L"])
    lines = [
        "PROBLEM",
        "Two-stage Miller-compensated CMOS op-amp (%s family), sized with "
        "square-law hand equations (no SPICE). Signal path: differential input "
        "pair M1 biased by the tail current Ib -> high-impedance stage-1 output "
        "node -> common-source stage-2 device M2, whose current I2 = B*Ib is "
        "mirrored from the tail -> output node driving the load capacitor CL. "
        "The Miller capacitor Cc connects the stage-2 input and output "
        "(pole splitting)." % fam.replace("_", " "),
        "Technology and load (fixed): process transconductance k = %s uA/V^2, "
        "channel-length modulation lambda = %s 1/V, Vdd = %s V, channel length "
        "L = %s um (all devices), load CL = %s pF, mirror unit-device width "
        "%s um, capacitor area density %s um^2/pF. The MB mirror is a %s um "
        "input device carrying Ib plus a B-times-wider output device carrying "
        "I2, both at the same Vov, hence its (1+B)*%s um width."
        % (n(op["kn"]), n(op["lam"]), n(op["vdd"]), L, n(op["cl"]),
           n(_W_MB_UM), n(_CAP_UM2_PER_PF), n(_W_MB_UM), n(_W_MB_UM)),
        "Tunable: each element has one dimensionless knob x, bounds [%s, %s]; "
        "physical value = unit * x: M1 width W1 = %s um * x; M2 width W2 = %s um "
        "* x; MB mirror ratio B = %s * x; CC Cc = %s pF * x; IB Ib = %s uA * x."
        % (n(_X_LO), n(_X_HI), n(_BASE[0]), n(_BASE[1]), n(_BASE[2]),
           n(_BASE[3]), n(_BASE[4])),
        "Equations (uA, uS, MOhm, pF, MHz, V/us, mW, um^2): Id1 = Ib/2 per input "
        "device; I2 = B*Ib; gm = sqrt(2 k (W/L) Id); Vov = sqrt(2 Id / (k W/L)); "
        "stage output resistance R = 1/(2 lambda Id) (two ro in parallel); "
        "gain = (gm1 R1)(gm2 R2); GBW = gm1/(2 pi Cc); non-dominant pole "
        "p2 = gm2/(2 pi CL); RHP zero z = gm2/(2 pi Cc); PM = 90 - atan(GBW/p2) "
        "- atan(GBW/z) deg; slew = min(Ib/Cc, I2/CL); power = (2 Ib + I2) Vdd; "
        "area = 2 W1 L + W2 L + (1+B)(%s um)L + (%s um^2/pF) Cc."
        % (n(_W_MB_UM), n(_CAP_UM2_PER_PF)),
        "Validity: the square law holds only in strong inversion and "
        "saturation, so every device (M1, M2 and the MB mirror input device, "
        "which carries Ib) must keep %s V <= Vov <= %s V; a design outside this "
        "window is invalid." % (n(op["vov_lo"]), n(op["vov_hi"])),
    ]
    if stochastic:
        mc = spec.get("mc", {}) or {}
        sk = float(mc.get("sigma_k", 0.05))
        svth = float(mc.get("sigma_vth_v", 0.006))
        lines.append(
            "Process mismatch (this variant): every evaluation draws independent "
            "k errors on M1, M2 and MB (1-sigma %s%% each), an Ib bias-source "
            "error (1-sigma %s%%) and a mirror Vth mismatch (1-sigma %s mV at "
            "10 um^2 mirror area, shrinking as 1/sqrt(area)) that shifts I2 by "
            "about 2*dVth/Vov(MB). Each limit is judged on an adverse quantile "
            "across draws (yield), so constraint values can be worse than the "
            "nominal figures. RESULTS below are the nominal (mismatch-free) "
            "evaluation."
            % (n(100.0 * sk, 3), n(50.0 * sk, 3), n(1000.0 * svth, 3)))

    def at_bound(i):
        if x[i] <= _X_LO + 1e-6:
            return "  (at lower bound, cannot decrease)"
        if x[i] >= _X_HI - 1e-6:
            return "  (at upper bound, cannot increase)"
        return ""

    lines += [
        "",
        "DESIGN (element: role, knob x, physical value)",
        "  0 M1  input differential pair (2 matched devices)  x=%s  W1=%s um%s"
        % (n(x[0]), n(op["W1"]), at_bound(0)),
        "  1 M2  stage-2 common-source gain device  x=%s  W2=%s um%s"
        % (n(x[1]), n(op["W2"]), at_bound(1)),
        "  2 MB  tail-to-stage-2 current-mirror ratio  x=%s  B=%s (I2 = B*Ib)%s"
        % (n(x[2]), n(op["B"]), at_bound(2)),
        "  3 CC  Miller compensation capacitor  x=%s  Cc=%s pF%s"
        % (n(x[3]), n(op["Cc"]), at_bound(3)),
        "  4 IB  tail/bias current  x=%s  Ib=%s uA%s"
        % (n(x[4]), n(op["Ib"]), at_bound(4)),
    ]

    # mirror mismatch (stochastic only): nominal 1-sigma of the I2 error
    mb_mis = ""
    mb_up = ""
    ib_up = ""
    if stochastic:
        mc = spec.get("mc", {}) or {}
        svth = float(mc.get("sigma_vth_v", 0.006))
        area_mb = op["area"][2]
        svth_eff = svth / math.sqrt(max(area_mb / 10.0, 0.25))
        sig_i2 = 2.0 * svth_eff / max(op["vovb"], 0.08)
        mb_mis = ", mirror I2 mismatch 1-sigma about %s%%" % n(100.0 * sig_i2, 3)
        mb_up = "; the larger mirror area lowers the I2 mismatch"
        if op["vovb"] >= 0.08:
            ib_up = "; the higher mirror Vov lowers the I2 mismatch"
        else:
            ib_up = ("; the I2 mismatch only starts to fall once the mirror Vov "
                     "exceeds 0.08 V")

    lim = "internal (Ib/Cc)" if op["sr_int"] <= op["sr_out"] else "output (I2/CL)"
    lo, hi = op["vov_lo"], op["vov_hi"]
    vovs = (("M1", op["vov1"]), ("M2", op["vov2"]), ("MB", op["vovb"]))
    tight = min(vovs, key=lambda kv: min(kv[1] - lo, hi - kv[1]))
    lines += [
        "",
        "RESULTS (per element, nominal square-law evaluation)",
        "  0 M1: Id1=%s uA per device, gm1=%s uS, Vov=%s V (%s), R1=%s MOhm, "
        "stage-1 gain gm1*R1=%s V/V (%s dB), area 2*W1*L=%s um^2 (%s%% of area). "
        "Drives GBW = gm1/(2 pi Cc) and the stage-1 gain. Raising x (W1 at fixed "
        "Id1): gm1, GBW and stage-1 gain rise as sqrt(W1), both phase lags grow "
        "(PM falls), Vov falls as 1/sqrt(W1), area rises; slew and power "
        "unchanged."
        % (n(op["id1"]), n(op["gm1"]), n(op["vov1"]),
           _obs_vov_note(op["vov1"], lo, hi), n(op["r1"]), n(op["a1"]),
           n(op["a1_db"]), n(op["area"][0]), a_pct(0)),
        "  1 M2: Id=I2=%s uA, gm2=%s uS, Vov=%s V (%s), R2=%s MOhm, stage-2 gain "
        "gm2*R2=%s V/V (%s dB), area W2*L=%s um^2 (%s%% of area). Drives the "
        "non-dominant pole p2=%s MHz (p2/GBW=%s, phase lag atan(GBW/p2)=%s deg) "
        "and the RHP zero z=%s MHz (z/GBW=gm2/gm1=%s, phase lag atan(GBW/z)=%s "
        "deg). Raising x (W2 at fixed I2): gm2, p2, z and stage-2 gain rise as "
        "sqrt(W2), both lags shrink (PM rises), Vov falls, area rises; GBW, "
        "slew and power unchanged."
        % (n(op["i2"]), n(op["gm2"]), n(op["vov2"]),
           _obs_vov_note(op["vov2"], lo, hi), n(op["r2"]), n(op["a2"]),
           n(op["a2_db"]), n(op["area"][1]), a_pct(1), n(op["p2"]),
           n(op["p2_over_gbw"]), n(op["lag_p2"]), n(op["z"]),
           n(op["z_over_gbw"]), n(op["lag_z"])),
        "  2 MB: I2=B*Ib=%s uA, output slew limb I2/CL=%s V/us, stage-2 power "
        "I2*Vdd=%s mW (%s%% of power), mirror input device (W=%s um, carries "
        "Ib) Vov=%s V (%s)%s, area (1+B)*%s um*L=%s um^2 (%s%% of area). "
        "Raising x: I2 rises in proportion, so gm2, p2 and z rise as sqrt(I2) "
        "(PM rises), the output slew limb and stage-2 power rise linearly, "
        "stage-2 gain falls as 1/sqrt(I2), M2 Vov rises as sqrt(I2), mirror "
        "area rises; GBW and the mirror Vov unchanged%s."
        % (n(op["i2"]), n(op["sr_out"]), n(op["p_stage2"]), p_pct(op["p_stage2"]),
           n(_W_MB_UM), n(op["vovb"]), _obs_vov_note(op["vovb"], lo, hi),
           mb_mis, n(_W_MB_UM), n(op["area"][2]), a_pct(2), mb_up),
        "  3 CC: Cc=%s pF, internal slew limb Ib/Cc=%s V/us, area %s*Cc=%s um^2 "
        "(%s%% of area). GBW and z scale as 1/Cc but p2 does not, so p2/GBW = "
        "(gm2/gm1)(Cc/CL) = %s; the zero lag atan(gm1/gm2) does not depend on "
        "Cc. Raising x: GBW and z fall, the p2 lag shrinks (PM rises), the "
        "internal slew limb falls, area rises; gain, power and Vov unchanged."
        % (n(op["Cc"]), n(op["sr_int"]), n(_CAP_UM2_PER_PF), n(op["area"][3]),
           a_pct(3), n(op["p2_over_gbw"])),
        "  4 IB: Ib=%s uA, bias+tail power 2*Ib*Vdd=%s mW (%s%% of power), no "
        "area. Raising x scales Id1 and I2 together: gm1, gm2, GBW, p2 and z "
        "rise as sqrt(Ib) with their ratios fixed (PM unchanged), both slew "
        "limbs and total power rise linearly, total gain falls as 1/Ib, every "
        "Vov (M1, M2, MB mirror input) rises as sqrt(Ib)%s."
        % (n(op["Ib"]), n(op["p_bias"]), p_pct(op["p_bias"]), ib_up),
        "  Slew is set by the %s limb. Device closest to (or outside) the Vov "
        "window: %s at %s V." % (lim, tight[0], n(tight[1])),
    ]
    return "\n".join(lines)
