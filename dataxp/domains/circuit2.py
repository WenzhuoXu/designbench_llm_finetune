"""Two-stage Miller op-amp, v2: full action set (device shape, compensation
scheme, nulling resistor, stage-1 cascode) over square-law hand analysis.

v1 (dataxp/domains/circuit.py, untouched) let the agent scale ONE knob per
element at a fixed channel length with plain Miller compensation. v2 keeps
v1's square-law operating point and adds the moves a designer actually has.

ELEMENTS (fixed list; indices never change)

    0 M1   input differential pair (2 matched NMOS)   W [um], L [um]
    1 M2   stage-2 common-source gain device          W [um], L [um]
    2 MB   bias mirror (tail -> stage-2 ratio)        B [-]
    3 CC   compensation network                       Cc [pF], comp {0,1,2}
    4 IB   tail / bias current                        Ib [uA]
    5 RZ   nulling resistor in series with Cc         Rz [kOhm], rz_on {0,1}
    6 CAS  stage-1 cascode (input side + load side)   cas_on {0,1}

PARAMS (get() returns None where a param does not apply to an element)

    W       continuous, scaled   [0.5, 500] um            M1, M2
    L       continuous, scaled   [Lmin, 12 Lmin] um       M1, M2
    B       continuous, scaled   [0.25, 30]               MB
    Cc      continuous, scaled   [0.05, 20] pF            CC
    comp    categorical          0 direct Miller, 1 voltage-buffered Miller
                                 (source follower), 2 current-buffered
                                 (common-gate, "Ahuja") Miller          CC
    Ib      continuous, scaled   [0.5, 300] uA            IB
    Rz      continuous, scaled   [0.1, 200] kOhm          RZ
    rz_on   toggle               0/1                      RZ
    cas_on  toggle               0/1                      CAS

Values are legalised by set(): clamped to bounds, continuous values rounded
to 4 significant digits and at most 3 decimal places (so every legal value
prints exactly with 3 decimals), toggles to {0,1}, comp to the nearest code.

PHYSICS (units: uA, uS, MOhm, pF, rad/us, MHz, V/us, mW, um^2)

    lambda(L) = lambda0 * Lmin / L          (channel-length modulation)
    Id1 = Ib/2 ;  I2 = B * Ib
    gm  = sqrt(2 k (W/L) Id) ;  Vov = 2 Id / gm ;  ro = 1/(lambda(L) Id)
    R1  = ro(L1)/2  [x A_cas = 2/(0.1 V * lambda(L1)) with the cascode]
    R2  = ro(L2)/2 ;  DC gain A0 = gm1 R1 gm2 R2
    C1  = (2/3) Cox W2 L2 + Cj (drain widths at node 1) + Cwire
    CLe = CL + Cj (W2 + B*W_MB)
  Exact small-signal transfer function of the two-node hand model with the
  selected compensation branch (derived by nodal analysis, not dominant-pole
  approximations):
    A(s) = A0 (1 + n1 s) / (1 + (d1 s + d2 s^2 + d3 s^3)/d0)
           x mirror pole-zero doublet (pole gm3/Cm, zero 2 gm3/Cm)
           x cascode-node pole gm_cas/Cx            (cascode on only)
    comp 0 (direct Miller, optional series Rz): RHP zero 1/(Cc(1/gm2 - Rz)),
           -> LHP once Rz > 1/gm2; extra pole ~ 1/(Rz C1).
    comp 1 (voltage buffer, follower gm_b = 2*(Ib/2)/0.1 V): no feed-forward
           zero; LHP zero gm_b/Cc; p2 ~ gm2 Cc/((C1+Cc) CLe).
    comp 2 (current buffer gm_c): no feed-forward zero; LHP zero gm_c/Cc;
           non-dominant poles from C1 CLe Cc/gm_c s^2 + C1 (CLe+Cc) s + gm2 Cc
           (usually a complex pair, natural frequency ~ sqrt(gm2 gm_c/(C1
           CLe)) independent of Cc, Q ~ Cc/(CLe+Cc) sqrt(gm2 CLe/(gm_c C1))
           rising with Cc). gm_c = 2 Ib/0.1 V
           from a dedicated common-gate device biased at Ib (costs Ib of
           supply current and area). Above gm_c/Cc the inner loop holds the
           gain at gm1/gm_c, so the scheme needs gm_c > gm1.
    Poles: batched eigenvalues of the companion matrix (exact roots).
    gbw_mhz           = FIRST frequency where |A(j w)| falls to 1, / 2 pi
    phase_margin_deg  = 180 + phase(A), the minimum over EVERY unity crossing
                        (a resonant peak that re-crosses 0 dB is charged at its
                        own crossing); per-factor angles, continuous by
                        construction, no unwrapping
    slew              = min(Ib/Cc, I2/CL)
    power             = (2 Ib + I2 + I_buffer + I_commongate) Vdd
    icmr_v            = Vdd - 0.1 - Vth - Vov1 - Vov_MB - cas_on*(0.1 + 0.1)
    out_swing_v       = Vdd - Vov2 - (max(Vov_MB, Vth + 0.2) if comp == 1
                                      else Vov_MB)
    area              = 2 W1 L1 + W2 L2 + (2+B) W_MB L_MB + mirror loads
                        + 200 Cc + rz_on (4 Rz + 10) + cascodes (+25)
                        + buffer (+15) / common-gate device (+15)
  Objective: area (minimise). Constraints: gain, GBW, phase margin, slew,
  power, input common-mode range, output swing.
  Validity envelope: every signal device (M1, M2, MB) in
  0.075 V <= Vov <= max(0.4, 0.35 Vdd) (square law: below ~0.075 V the
  formula's gm/Id = 2/Vov exceeds the weak-inversion ceiling 1/(n UT));
  crossover <= 50 GHz (beyond any device fT of the node); area <= 1e6 um^2.
  Invalid or numerically failed designs return finite worst-case metrics.

STOCHASTIC (stochastic=True): per-seed process mismatch Monte-Carlo, as v1:
k errors on M1, M2, MB (sigma 3-6 %), bias-source error, mirror Vth mismatch
(Pelgrom, shifts I2 by 2 dVth/Vov_MB), plus poly-resistor sheet variation on
Rz (sigma 5-10 %) and capacitor variation on Cc (sigma 2-4 %). Area is
computed from the nominal sizes and does not vary with the seed.

SIMPLIFICATIONS (stated explicitly)
  * Square law everywhere; no velocity saturation, body effect, or
    subthreshold conduction (the Vov envelope rejects designs that need them).
  * One k for every signal device (as v1); PMOS auxiliary devices use
    k/2.5. |Vthp| = Vthn = Vth.
  * Auxiliary devices (mirror loads, cascodes, buffer, common-gate device,
    their current sources) are auto-sized to Vov = 0.1 V; mirror loads and
    cascodes have length L1, buffer/common-gate devices length 2 Lmin. MB is
    a unit device of length 1 um whose width is fixed PER PROBLEM (sized for
    Vov 0.2 V at the geometric mean of the family's current range; tail and
    stage-2 sink are copies), so only a ~10-30x band of Ib keeps it valid.
  * Cascode: telescopic on both stage-1 branches; its headroom cost is one
    auxiliary Vov plus a fixed 0.1 V margin, taken from the input
    common-mode range; it does not change the output swing.
  * Two-node model plus the compensation branch; Cgd/Miller of the devices,
    the stage-2 load's own pole, slew of the buffer devices, noise, offset
    and CMRR/PSRR are not modelled. The mirror pole and cascode-node pole are
    applied as separate first-order factors.
  * comp 1/2: the buffer's own output parasitic is ignored (its 1/gm sits in
    series with Cc, which is modelled). comp 2 always uses its own
    common-gate device (the stage-1 cascode, at Id1 and Vov 0.1 V, would give
    gm_c <= gm1 and is not used as the buffer). Rz is in the signal path
    only for comp 0; with comp 1/2 a resistor left on still costs area.
  * Slew is min(Ib/Cc, I2/CL) for every scheme.
  * GBW is the first unity crossing; PM is the worst over all crossings (no
    full Nyquist/gain-margin analysis).

PROBLEMS (make_problem): 5 families (low_power, high_speed, high_gain,
wide_swing, general; index % 5) x 3 tiers (easy, medium, hard; (index // 5)
% 3) x 3 process nodes (n180/n250/n350, drawn) with per-problem k, lambda0,
Vth, Vdd and CL draws. Goals are margins off a random sane WITNESS design
(sampled over every param incl. scheme/cascode/resistor, picked per family:
top quartile of gain / GBW / low power / headroom, or random for general),
so a feasible design exists by construction; tier sets the margins. The start
is a small valid plain-Miller design (no resistor, no cascode) moved part of
the way toward the witness in log space (tier sets how far), so it is
infeasible; ~12 % of problems instead start AT the witness with relaxed
goals (pure area minimisation).
REFERENCE: best feasible area of 3000 seeded random designs over the full
action space (plus the witness), each of the 3 best polished by coordinate
descent over every param incl. toggles and scheme (reference_class
"heuristic").

Compatibility: Python >= 3.8, numpy only.

Verb surface (docs/dataxp_verb_surface.md; owner's decision of 2026-10-23)
------------
The move set (candidate_values over the element/param table above), set(),
Phi and the stored {i, p, v, v_old} dicts are the v2 ones and are untouched;
only the spelling in prompts, action lines and reply parsing changes (class
attributes verb_surface and verb_table; format_move / parse_move / grammar are
the table-driven defaults of dataxp.domain). The bijection, per parameter
(the first argument is always the SET_PARAM element number):
  W v   on 0 / 1        <-> SET_WIDTH(device, v)                     [um]
  L v   on 0 / 1        <-> SET_LENGTH(device, v)                    [um]
  B v   on 2            <-> SET_MIRROR_RATIO(2, v)                   [-]
  Cc v  on 3            <-> SET_COMP_CAP(3, v)                       [pF]
  comp k on 3           <-> SET_COMPENSATION(3, name), name =
                            DirectMiller / VoltageBufferedMiller /
                            CurrentBufferedMiller (codes 0 / 1 / 2)
  Ib v  on 4            <-> SET_BIAS_CURRENT(4, v)                   [uA]
  Rz v  on 5            <-> SET_NULLING_RESISTOR(5, v)               [kOhm]
  rz_on 1 / 0 on 5      <-> ENABLE_NULLING(5) / DISABLE_NULLING(5)
  cas_on 1 / 0 on 6     <-> ENABLE_CASCODE(6) / DISABLE_CASCODE(6)
Judgment calls, recorded here as the surface rules ask:
  * The element argument is kept on every verb, including the five that only
    one element admits (SET_MIRROR_RATIO always 2, SET_COMP_CAP and
    SET_COMPENSATION always 3, SET_BIAS_CURRENT 4, SET_NULLING_RESISTOR /
    ENABLE_NULLING / DISABLE_NULLING 5, ENABLE_CASCODE / DISABLE_CASCODE 6):
    the table-driven surface spells SET_PARAM's element index as the first
    argument of every verb, and dropping it would need format_move /
    parse_move / grammar overrides for no change in the move set. The
    grammar states the fixed number of each; the DESIGN rows number the
    elements the same way.
  * W and L are two verbs (SET_WIDTH / SET_LENGTH on device 0 = M1 or
    1 = M2) rather than one SIZE_DEVICE(device, W|L, v): each has its own
    range (W 0.5..500 um; L Lmin..12 Lmin um with Lmin per process), and the
    grammar can state both only as separate lines.
  * Compensation schemes are spelled as identifiers (DirectMiller,
    VoltageBufferedMiller, CurrentBufferedMiller); parse_move also accepts
    the observation's names (direct Miller, voltage-buffered Miller,
    current-buffered Miller), the field's short names (Miller, SourceFollower
    / VoltageBuffer, CommonGate / CurrentBuffer / Ahuja) and any letter case;
    format_move writes the canonical identifier only, so the surface is
    one-to-one on the legal moves. A bare code number (SET_COMPENSATION(3, 2))
    is not accepted. The DESIGN and RESULTS rows keep "comp=k (name)"; the
    Tunables paragraph ties each identifier to its code.
  * A continuous value is the new absolute setting as a plain number, printed
    to 6 significant digits (fmt_arg, as SET_PARAM printed it); every legal
    value of this domain has 4 significant digits and at most 3 decimals, so
    the text replays bit for bit. set() still clamps and rounds, so an
    out-of-range or off-grid number parses and lands on the legal grid, as it
    did under SET_PARAM.
  * SET_PARAM(...) text is no longer parsed on this domain: the grammar the
    policy is served lists the verbs only.
  * The toggle verbs on an element already in that state (ENABLE_NULLING with
    rz_on = 1, DISABLE_CASCODE with cas_on = 0) parse and apply() leaves the
    state unchanged, exactly as the SET_PARAM did; legal_moves never offers
    them. SET_NULLING_RESISTOR while the resistor is disabled or bypassed
    (comp != 0) parses and stores the value, as before; legal_moves does not
    offer it then. Nothing was added to or removed from the move set.
  * describe()'s Tunables paragraph (the "how to write a change" text) is
    written in the verbs (observation text only; no generated file contains
    it): the header names the ten verb spellings and each element line names
    its verbs; the physics text of every line is unchanged. The what-if row
    "Other schemes at these sizes" adds the scheme's identifier after its
    code. Every other observation line is as before.
  * History and "change you are about to make" lines print the replaced value
    ("was 12.5", "scheme DirectMiller -> CurrentBufferedMiller") for the
    continuous and categorical verbs; the enable / disable pairs need none.

Set-parameter spelling (docs/dataxp_spellings.md; DATAXP_SPELLING=set_parameter)
-----------------------
The same moves through the one wrapper tool, generated from the verb table
by dataxp.domain (no override here): SET_WIDTH(0, 12.5) <-> SET_PARAM(0, W,
12.5), SET_COMPENSATION(3, CurrentBufferedMiller) <-> SET_PARAM(3, comp, 2)
(the stored code; the grammar names the codes), ENABLE_NULLING(5) /
DISABLE_NULLING(5) <-> SET_PARAM(5, rz_on, 1) / SET_PARAM(5, rz_on, 0), and
so on for every row. Judgment calls: the phrases are "Set a device's channel
width / length", "Set the mirror ratio", "Set the compensation capacitor",
"Select the compensation scheme", "Set the bias current", "Set the nulling
resistor's value", "Enable / Disable the nulling resistor", "Enable / Disable
the cascode"; the Tunables paragraph of the observation reverts to its
SET_PARAM spelling (verbs=False, the run2 text) under this spelling
(spelled_in_verbs); the history "was ..." prints the stored comp code.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from ..domain import (
    ArgSpec,
    Candidate,
    Categorical,
    Continuous,
    Domain,
    ParamVerb,
    Toggle,
    spelled_in_verbs,
    stable_seed,
)
from ..potential import DesignProgram, program_from_goals

# ---------------------------------------------------------------------------
# layout of the design vector
# ---------------------------------------------------------------------------
_ELEMENTS = ("M1", "M2", "MB", "CC", "IB", "RZ", "CAS")
_PARAMS = ("W", "L", "B", "Cc", "comp", "Ib", "Rz", "rz_on", "cas_on")
# (element, param) -> slot in the design vector
_SLOT = {
    (0, "W"): 0, (0, "L"): 1,
    (1, "W"): 2, (1, "L"): 3,
    (2, "B"): 4,
    (3, "Cc"): 5, (3, "comp"): 6,
    (4, "Ib"): 7,
    (5, "Rz"): 8, (5, "rz_on"): 9,
    (6, "cas_on"): 10,
}
_NSLOT = 11
S_W1, S_L1, S_W2, S_L2, S_B, S_CC, S_COMP, S_IB, S_RZ, S_RZON, S_CAS = range(_NSLOT)
_KIND = {"W": "cont", "L": "cont", "B": "cont", "Cc": "cont", "Ib": "cont",
         "Rz": "cont", "comp": "cat", "rz_on": "toggle", "cas_on": "toggle"}
_SLOT_KIND = ["cont", "cont", "cont", "cont", "cont", "cont", "cat", "cont",
              "cont", "toggle", "toggle"]
_N_COMP = 3
_COMP_NAMES = ("direct Miller", "voltage-buffered Miller",
               "current-buffered Miller")
# verb-surface identifiers of the comp codes (docstring "Verb surface")
COMP_NAMES = ("DirectMiller", "VoltageBufferedMiller", "CurrentBufferedMiller")
_COMP_ALIASES = (
    ("direct Miller", 0), ("Miller", 0), ("direct", 0),
    ("voltage-buffered Miller", 1), ("VoltageBuffer", 1), ("SourceFollower", 1),
    ("current-buffered Miller", 2), ("CurrentBuffer", 2), ("CommonGate", 2), ("Ahuja", 2),
)
_COMP_DOC = ("DirectMiller (comp 0) = direct Miller: Cc straight across stage 2, a "
             "right-half-plane zero at gm2/Cc unless the nulling resistor is enabled; "
             "VoltageBufferedMiller (comp 1) = source-follower buffered Miller: no "
             "feed-forward zero, costs Ib/2 and lifts the lowest output voltage; "
             "CurrentBufferedMiller (comp 2) = common-gate (Ahuja) buffered Miller: no "
             "feed-forward zero, usually a complex non-dominant pair, costs Ib.")
_DEVICE_ARG = ArgSpec("device", "int", "0 = M1 (input pair) or 1 = M2 (stage-2 device), the "
                                       "element number")
_MIRROR_ARG = ArgSpec("mirror", "int", "always 2 (MB, the tail-to-stage-2 current mirror)")
_NETWORK_ARG = ArgSpec("network", "int", "always 3 (CC, the compensation network)")
_SOURCE_ARG = ArgSpec("source", "int", "always 4 (IB, the tail / bias current source)")
_RESISTOR_ARG = ArgSpec("resistor", "int", "always 5 (RZ, the nulling resistor)")
_CASCODE_ARG = ArgSpec("cascode", "int", "always 6 (CAS, the stage-1 cascode)")
_LEGAL_DOC = " (clamped to the bounds and rounded to 4 significant digits, at most 3 decimals)"

# ---------------------------------------------------------------------------
# technology constants
# ---------------------------------------------------------------------------
_NODES = {
    "n180": {"lmin_um": 0.18, "vdd_v": 1.8, "vth_v": 0.45, "cox": 8.5e-3,
             "kn": (250.0, 320.0), "lam0": (0.30, 0.45)},
    "n250": {"lmin_um": 0.25, "vdd_v": 2.5, "vth_v": 0.50, "cox": 6.0e-3,
             "kn": (190.0, 250.0), "lam0": (0.24, 0.36)},
    "n350": {"lmin_um": 0.35, "vdd_v": 3.3, "vth_v": 0.55, "cox": 4.6e-3,
             "kn": (150.0, 200.0), "lam0": (0.18, 0.28)},
}
_NODE_NAMES = ("n180", "n250", "n350")

_W_MB_UM = 5.0              # MB unit device width
_L_MB_UM = 1.0              # MB unit device length
_CAP_UM2_PER_PF = 200.0     # compensation-capacitor density (as v1)
_RES_UM2_PER_KOHM = 4.0     # poly resistor density
_RES_OVH_UM2 = 10.0         # resistor contacts/spacing
_CAS_OVH_UM2 = 25.0         # cascode bias generator
_BUF_OVH_UM2 = 15.0         # follower bias
_CG_OVH_UM2 = 15.0          # common-gate device bias
_VAUX = 0.10                # V, Vov of auto-sized auxiliary devices
_CAS_MARGIN_V = 0.10        # V, cascode headroom margin beyond its Vov
_CAS_HEADROOM_V = _VAUX + _CAS_MARGIN_V
_VOV_MIN = 0.075            # V, square-law floor (weak-inversion ceiling)
_CJ_PF_PER_UM = 1.0e-3      # junction capacitance per um of drain width
_C_WIRE_PF = 0.02           # wiring parasitic per internal node
_KP_RATIO = 2.5             # kn / kp
_GBW_VALID_MHZ = 5.0e4      # crossover beyond 50 GHz: outside the model
_TWO_PI = 2.0 * math.pi

# frequency grid for the crossover bracket (rad/us), 20 points per decade
_W_GRID = np.logspace(-6.0, 6.0, 241)
_LOG_W_GRID = np.log(_W_GRID)
_SECANT_ITERS = 12

_FAMILIES = ("low_power", "high_speed", "high_gain", "wide_swing", "general")
_TIERS = ("easy", "medium", "hard")
_CL_RANGE = {"low_power": (2.0, 6.0), "high_speed": (1.0, 3.0),
             "high_gain": (2.0, 6.0), "wide_swing": (2.0, 8.0),
             "general": (3.0, 8.0)}
_IB_RANGE = {"low_power": (1.0, 12.0), "high_speed": (15.0, 150.0),
             "high_gain": (2.0, 40.0), "wide_swing": (2.0, 40.0),
             "general": (2.0, 60.0)}

_GOAL_METRIC = {
    "minimum_gain_db": "gain_db",
    "minimum_gbw_mhz": "gbw_mhz",
    "minimum_phase_margin_deg": "phase_margin_deg",
    "minimum_slew_v_per_us": "slew_v_per_us",
    "maximum_power_mw": "power_mw",
    "minimum_icmr_v": "icmr_v",
    "minimum_out_swing_v": "out_swing_v",
}

# finite worst case: maximally violating, outside the validity envelope
_WORST = {
    "gain_db": -100.0,
    "gbw_mhz": 0.0,
    "phase_margin_deg": -180.0,
    "slew_v_per_us": 0.0,
    "power_mw": 1.0e4,
    "icmr_v": -10.0,
    "out_swing_v": -10.0,
    "area": 1.0e7,
    "min_vov": 0.0,
    "max_vov": 10.0,
    "vov_defect": 10.0,
}


def _round_sig(v: float, sig: int = 4) -> float:
    return float("%.*g" % (sig, float(v)))


def _round_val(v: float) -> float:
    """Legal grid of a continuous design value: 4 significant digits and at
    most 3 decimal places. Every value on this grid prints exactly under a
    3-decimal formatter, so an action rendered as text replays bit for bit
    (4 significant digits alone would print 0.7903 as 0.79)."""
    v = _round_sig(v)
    if abs(v) < 1.0:
        v = round(v, 3)
    return float(v)


def _slot_bounds(tech: dict) -> Tuple[np.ndarray, np.ndarray]:
    lmin = float(tech["lmin_um"])
    lmax = round(12.0 * lmin, 6)       # 12*0.35 is 4.199999999999999 in floats
    lo = np.array([0.5, lmin, 0.5, lmin, 0.25, 0.05, 0.0, 0.5, 0.1, 0.0, 0.0])
    hi = np.array([500.0, lmax, 500.0, lmax, 30.0, 20.0,
                   float(_N_COMP - 1), 300.0, 200.0, 1.0, 1.0])
    return lo, hi


def _legalize(tech: dict, slot: int, v: float) -> float:
    lo, hi = _slot_bounds(tech)
    kind = _SLOT_KIND[slot]
    if kind == "toggle":
        return 1.0 if v >= 0.5 else 0.0
    if kind == "cat":
        return float(min(hi[slot], max(lo[slot], math.floor(v + 0.5))))
    v = min(hi[slot], max(lo[slot], float(v)))
    v = _round_val(v)
    return float(min(hi[slot], max(lo[slot], v)))


def _vov_hi(vdd: float) -> float:
    return max(0.4, 0.35 * vdd)


# ---------------------------------------------------------------------------
# core analysis (vectorised over designs)
# ---------------------------------------------------------------------------
def _factor_logmag(a, b, w):
    """log|1 - j w u| for inverse roots u = a + j b (broadcast)."""
    return 0.5 * np.log((1.0 + w * b) ** 2 + (w * a) ** 2)


def _factor_angle(a, b, w):
    """angle(1 - j w u): continuous in w for every u with Re(u) != 0."""
    return np.arctan2(-w * a, 1.0 + w * b)


def _logmag(logA0, pa, pb, za, zb, w):
    """log|A(j w)|; pa/pb (n, P), za/zb (n, Z), w (n, k) -> (n, k)."""
    wz = w[:, None, :]
    num = _factor_logmag(za[:, :, None], zb[:, :, None], wz).sum(axis=1)
    den = _factor_logmag(pa[:, :, None], pb[:, :, None], wz).sum(axis=1)
    return logA0[:, None] + num - den


def _phase(pa, pb, za, zb, w):
    """phase of A(j w) in rad; w (n,) -> (n,)."""
    wz = w[:, None]
    return (_factor_angle(za, zb, wz).sum(axis=1)
            - _factor_angle(pa, pb, wz).sum(axis=1))


def _refine_scalar(lm, xa, fa, xb, fb):
    """Illinois regula falsi for a sign change of lm(exp(x)) on [xa, xb]."""
    for _ in range(_SECANT_ITERS):
        den = fb - fa
        xm = xb - fb * (xb - xa) / den if abs(den) > 1e-300 else 0.5 * (xa + xb)
        xm = min(max(xm, min(xa, xb)), max(xa, xb))
        fm = lm(math.exp(xm))
        if (fm >= 0.0) == (fb >= 0.0):
            fa *= 0.5
        else:
            xa, fa = xb, fb
        xb, fb = xm, fm
    return xb if abs(fb) <= abs(fa) else xa


_GOLD = 0.5 * (math.sqrt(5.0) - 1.0)
_EXTREMUM_BAND = 0.5      # |log|A|| below which a grid extremum is re-examined


def _golden_extremum(f, xa, xb, sign):
    """Golden-section search for max (sign=+1) or min (sign=-1) of f."""
    c = xb - _GOLD * (xb - xa)
    d = xa + _GOLD * (xb - xa)
    fc, fd = sign * f(c), sign * f(d)
    for _ in range(40):
        if fc >= fd:
            xb, d, fd = d, c, fc
            c = xb - _GOLD * (xb - xa)
            fc = sign * f(c)
        else:
            xa, c, fc = c, d, fd
            d = xa + _GOLD * (xb - xa)
            fd = sign * f(d)
    return (c, sign * fc) if fc >= fd else (d, sign * fd)


def _near_zero_extrema(lm_row) -> np.ndarray:
    """Grid indices of local extrema close to 0 dB that do not themselves
    change sign: a narrow resonant hump (or dip) can cross 0 dB twice
    between two grid points."""
    mid = lm_row[1:-1]
    left, right = lm_row[:-2], lm_row[2:]
    hump = (mid >= left) & (mid >= right) & (mid < 0.0) & (mid > -_EXTREMUM_BAND)
    dip = (mid <= left) & (mid <= right) & (mid >= 0.0) & (mid < _EXTREMUM_BAND)
    return np.flatnonzero(hump | dip) + 1


def _crossover_one(logA0, pa, pb, za, zb, lm_row):
    """All unity crossings of one design (Python floats)."""
    P = list(zip(pa.tolist(), pb.tolist()))
    Z = list(zip(za.tolist(), zb.tolist()))
    log, atan2, exp = math.log, math.atan2, math.exp

    def lm(w):
        s = logA0
        for a, b in Z:
            s += 0.5 * log((1.0 + w * b) ** 2 + (w * a) ** 2)
        for a, b in P:
            s -= 0.5 * log((1.0 + w * b) ** 2 + (w * a) ** 2)
        return s

    def lmx(x):
        return lm(exp(x))

    def phase(w):
        ph = 0.0
        for a, b in Z:
            ph += atan2(-w * a, 1.0 + w * b)
        for a, b in P:
            ph -= atan2(-w * a, 1.0 + w * b)
        return ph

    pos = lm_row >= 0.0
    if not pos[0]:                       # DC gain below 1: no bandwidth
        wu = float(_W_GRID[0])
        return wu, phase(wu)
    brackets = []                        # (xa, fa, xb, fb) with a sign change
    for k in np.flatnonzero(pos[:-1] != pos[1:]).tolist():
        brackets.append((float(_LOG_W_GRID[k]), float(lm_row[k]),
                         float(_LOG_W_GRID[k + 1]), float(lm_row[k + 1])))
    for j in _near_zero_extrema(lm_row).tolist():
        xl, xr = float(_LOG_W_GRID[j - 1]), float(_LOG_W_GRID[j + 1])
        sign = 1.0 if lm_row[j] < 0.0 else -1.0
        xe, fe = _golden_extremum(lmx, xl, xr, sign)
        if (fe >= 0.0) != (lm_row[j] >= 0.0):
            fl, fr = float(lm_row[j - 1]), float(lm_row[j + 1])
            brackets.append((xl, fl, xe, fe))
            brackets.append((xe, fe, xr, fr))
    if not brackets:                     # never falls below 1 on the grid
        wu = float(_W_GRID[-1])
        return wu, phase(wu)
    brackets.sort(key=lambda t: t[0])
    wu = None
    ph_min = None
    for xa, fa, xb, fb in brackets:
        wc = exp(_refine_scalar(lm, xa, fa, xb, fb))
        if wu is None:
            wu = wc                      # first crossing = GBW
        ph = phase(wc)
        ph_min = ph if ph_min is None else min(ph_min, ph)
    return wu, ph_min


def _crossover(logA0, pa, pb, za, zb):
    """GBW = FIRST unity crossing; phase taken as the worst (minimum) over
    every unity crossing, so a resonant peak that re-crosses 0 dB is charged
    at its own crossing instead of being credited as bandwidth."""
    n = logA0.shape[0]
    grid = np.broadcast_to(_W_GRID, (n, _W_GRID.size))
    lm = _logmag(logA0, pa, pb, za, zb, grid)
    if n == 1:
        wu, ph = _crossover_one(float(logA0[0]), pa[0], pb[0], za[0], zb[0], lm[0])
        return np.array([wu]), np.array([ph])
    pos = lm >= 0.0
    chg = pos[:, :-1] != pos[:, 1:]
    n_cross = chg.sum(axis=1)
    k1 = np.argmax(chg, axis=1)
    rows = np.arange(n)
    xa, xb = _LOG_W_GRID[k1].copy(), _LOG_W_GRID[k1 + 1].copy()
    fa, fb = lm[rows, k1].copy(), lm[rows, k1 + 1].copy()
    for _ in range(_SECANT_ITERS):
        den = fb - fa
        safe = np.abs(den) > 1e-300
        xm = np.where(safe, xb - fb * (xb - xa) / np.where(safe, den, 1.0),
                      0.5 * (xa + xb))
        xm = np.clip(xm, np.minimum(xa, xb), np.maximum(xa, xb))
        fm = _logmag(logA0, pa, pb, za, zb, np.exp(xm)[:, None])[:, 0]
        same = (fm >= 0.0) == (fb >= 0.0)
        fa = np.where(same, 0.5 * fa, fb)
        xa = np.where(same, xa, xb)
        xb, fb = xm, fm
    wu = np.exp(np.where(np.abs(fb) <= np.abs(fa), xb, xa))
    wu = np.where(n_cross == 0, np.where(pos[:, 0], _W_GRID[-1], _W_GRID[0]), wu)
    wu = np.where(~pos[:, 0], _W_GRID[0], wu)
    ph = _phase(pa, pb, za, zb, wu)
    mid, left, right = lm[:, 1:-1], lm[:, :-2], lm[:, 2:]
    odd = (((mid >= left) & (mid >= right) & (mid < 0.0) & (mid > -_EXTREMUM_BAND))
           | ((mid <= left) & (mid <= right) & (mid >= 0.0)
              & (mid < _EXTREMUM_BAND))).any(axis=1)
    for r in np.flatnonzero(((n_cross > 1) | odd) & pos[:, 0]).tolist():
        w1, p1 = _crossover_one(float(logA0[r]), pa[r], pb[r], za[r], zb[r], lm[r])
        wu[r], ph[r] = w1, p1
    return wu, ph


def _analyze(tech: dict, X, *, kf=(1.0, 1.0, 1.0), ib_fac=1.0,
             mirror_err=0.0, rz_fac=1.0, cc_fac=1.0,
             detail: bool = False) -> dict:
    """Hand analysis of N designs X (N, 11). Returns arrays (N,)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    W1, L1, W2, L2, B, Cc, comp, Ib0, Rz, rz_on, cas = (X[:, k] for k in range(_NSLOT))
    comp = np.clip(np.floor(comp + 0.5), 0, _N_COMP - 1)
    rzon = rz_on >= 0.5
    cason = cas >= 0.5
    is0, is1, is2 = comp == 0, comp == 1, comp == 2

    kn = float(tech["kn_ua_v2"])
    lam0 = float(tech["lambda0_per_v"])
    lmin = float(tech["lmin_um"])
    vdd = float(tech["vdd_v"])
    vth = float(tech["vth_v"])
    cox = float(tech["cox_pf_per_um2"])
    cl = float(tech["cl_pf"])
    kp = kn / _KP_RATIO
    cg = (2.0 / 3.0) * cox
    l_aux = 2.0 * lmin

    # -- auxiliary sizing from NOMINAL currents (layout does not vary) -------
    Id1n = 0.5 * Ib0
    W3 = 2.0 * Id1n * L1 / (kp * _VAUX ** 2)       # PMOS mirror load (at L1)
    Wcn = 2.0 * Id1n * L1 / (kn * _VAUX ** 2)      # NMOS cascode (at L1)
    Wcp = W3                                       # PMOS cascode (at L1)

    # -- operating point -----------------------------------------------------
    Ib = Ib0 * ib_fac
    lam1 = lam0 * lmin / L1
    lam2 = lam0 * lmin / L2
    Id1 = 0.5 * Ib
    I2 = np.maximum(B * Ib * (1.0 + mirror_err), 1e-3)
    k1, k2, kb = kn * kf[0], kn * kf[1], kn * kf[2]
    gm1 = np.sqrt(2.0 * k1 * (W1 / L1) * Id1)
    gm2 = np.sqrt(2.0 * k2 * (W2 / L2) * I2)
    vov1 = 2.0 * Id1 / gm1
    vov2 = 2.0 * I2 / gm2
    w_mb = float(tech.get("mb_w_um", _W_MB_UM))
    l_mb = float(tech.get("mb_l_um", _L_MB_UM))
    vovb = np.sqrt(2.0 * Ib / (kb * (w_mb / l_mb)))
    a_cas = 2.0 / (_VAUX * lam1)
    R1 = np.where(cason, a_cas, 1.0) / (2.0 * lam1 * Id1)
    R2 = 1.0 / (2.0 * lam2 * I2)
    G1 = 1.0 / R1
    G2 = 1.0 / R2

    # -- node capacitances ---------------------------------------------------
    C1 = (cg * W2 * L2 + _CJ_PF_PER_UM * np.where(cason, Wcn + Wcp, W1 + W3)
          + _C_WIRE_PF)
    CLe = cl + _CJ_PF_PER_UM * (W2 + B * w_mb)
    gm3 = 2.0 * Id1 / _VAUX
    Cm = 2.0 * cg * W3 * L1 + _CJ_PF_PER_UM * (W1 + W3) + _C_WIRE_PF
    wm = gm3 / Cm                                   # mirror pole (rad/us)
    gmx = 2.0 * Id1 / _VAUX                         # stage-1 cascode gm
    Cx = _CJ_PF_PER_UM * W1 + cg * Wcn * L1 + _C_WIRE_PF
    wx = gmx / Cx                                   # cascode-node pole

    # -- compensation branch -------------------------------------------------
    Ccx = Cc * cc_fac
    Rzm = np.where(is0 & rzon, Rz * 1e-3 * rz_fac, 0.0)     # MOhm
    I_buf = np.where(is1, 0.5 * Ib, 0.0)
    I_cg = np.where(is2, Ib, 0.0)
    gmb = 2.0 * (0.5 * Ib) / _VAUX
    gmc = 2.0 * Ib / _VAUX
    tau = np.where(is0, Rzm * Ccx, np.where(is1, Ccx / gmb, Ccx / gmc))

    P0 = G1 * G2
    P1 = G1 * CLe + G2 * C1
    P2 = C1 * CLe
    d1 = P1 + tau * P0 + Ccx * np.where(
        is0, G1 + G2 + gm2, np.where(is1, G2 + gm2, G1 + gm2))
    d2 = P2 + tau * P1 + Ccx * np.where(is0, C1 + CLe, np.where(is1, CLe, C1))
    d3 = tau * P2
    n1 = np.where(is0, tau - Ccx / gm2, tau)

    # inverse loop poles u = 1/p: roots of u^3 + (d1/d0) u^2 + (d2/d0) u + d3/d0
    M = np.zeros((n, 3, 3))
    M[:, 0, 0] = -d1 / P0
    M[:, 0, 1] = -d2 / P0
    M[:, 0, 2] = -d3 / P0
    M[:, 1, 0] = 1.0
    M[:, 2, 1] = 1.0
    u_loop = np.linalg.eigvals(M)                    # (n, 3) complex

    pa = np.concatenate([u_loop.real, (-1.0 / wm)[:, None],
                         np.where(cason, -1.0 / wx, 0.0)[:, None]], axis=1)
    pb = np.concatenate([u_loop.imag, np.zeros((n, 2))], axis=1)
    za = np.stack([-n1, -0.5 / wm], axis=1)
    zb = np.zeros_like(za)

    A0 = gm1 * R1 * gm2 * R2
    logA0 = np.log(np.maximum(A0, 1e-300))
    wu, ph = _crossover(logA0, pa, pb, za, zb)

    gain_db = 20.0 * np.log10(np.maximum(A0, 1e-12))
    gbw = wu / _TWO_PI
    pm = 180.0 + np.degrees(ph)
    slew = np.minimum(Ib / Ccx, I2 / cl)
    power = (2.0 * Ib + I2 + I_buf + I_cg) * vdd * 1e-3
    icmr = (vdd - _VAUX - vth - vov1 - vovb
            - np.where(cason, _CAS_HEADROOM_V, 0.0))
    swing = vdd - vov2 - np.where(is1, np.maximum(vovb, vth + 2.0 * _VAUX), vovb)

    # -- area (nominal sizes) --------------------------------------------------
    a_m1 = 2.0 * W1 * L1
    a_m2 = W2 * L2
    a_mb = (2.0 + B) * w_mb * l_mb
    a_load = 2.0 * W3 * L1
    a_cc = _CAP_UM2_PER_PF * Cc
    a_rz = np.where(rzon, _RES_UM2_PER_KOHM * Rz + _RES_OVH_UM2, 0.0)
    # (named area_cas: a_cas above is the cascode GAIN multiplier, which
    # describe() reports; reusing the name printed the area as the gain)
    area_cas = np.where(cason, 2.0 * (Wcn + Wcp) * L1 + _CAS_OVH_UM2, 0.0)
    wl_buf = 2.0 * (0.5 * Ib0) / (kn * _VAUX ** 2)
    a_buf = np.where(is1, 2.0 * wl_buf * l_aux ** 2 + _BUF_OVH_UM2, 0.0)
    wl_cg = 2.0 * Ib0 / (kn * _VAUX ** 2)
    a_cg = np.where(is2, (1.0 + _KP_RATIO) * wl_cg * l_aux ** 2 + _CG_OVH_UM2, 0.0)
    area = a_m1 + a_m2 + a_mb + a_load + a_cc + a_rz + area_cas + a_buf + a_cg

    vovs = np.stack([vov1, vov2, vovb], axis=1)
    min_vov = vovs.min(axis=1)
    max_vov = vovs.max(axis=1)
    vov_defect = (np.maximum(0.0, _VOV_MIN - min_vov)
                  + np.maximum(0.0, max_vov - _vov_hi(vdd)))

    out = {
        "gain_db": gain_db,
        "gbw_mhz": gbw,
        "phase_margin_deg": pm,
        "slew_v_per_us": slew,
        "power_mw": power,
        "icmr_v": icmr,
        "out_swing_v": swing,
        "area": area,
        "min_vov": min_vov,
        "max_vov": max_vov,
        "vov_defect": vov_defect,
    }
    if detail:
        out.update({
            "Id1": Id1, "I2": I2, "gm1": gm1, "gm2": gm2, "vov1": vov1,
            "vov2": vov2, "vovb": vovb, "lam1": lam1, "lam2": lam2,
            "R1": R1, "R2": R2, "a_cas_gain": a_cas, "C1": C1, "CLe": CLe,
            "Cgs2": cg * W2 * L2, "wm": wm, "wx": wx, "gmb": gmb, "gmc": gmc,
            "I_buf": I_buf, "I_cg": I_cg, "n1": n1, "u_loop": u_loop,
            "pa": pa, "pb": pb, "za": za, "zb": zb, "wu": wu, "A0": A0,
            "Rzm": Rzm, "W3": W3, "Wcn": Wcn, "Wcp": Wcp,
            "a_m1": a_m1, "a_m2": a_m2, "a_mb": a_mb, "a_load": a_load,
            "a_cc": a_cc, "a_rz": a_rz, "a_cas": area_cas, "a_buf": a_buf,
            "a_cg": a_cg, "comp": comp, "rzon": rzon, "cason": cason,
        })
    return out


def _valid_mask(m: dict) -> np.ndarray:
    ok = np.ones_like(m["area"], dtype=bool)
    for k in m:
        if k in _WORST:
            ok &= np.isfinite(m[k])
    ok &= (m["vov_defect"] <= 1e-9) & (m["gbw_mhz"] <= _GBW_VALID_MHZ)
    ok &= (m["area"] > 0.0) & (m["area"] <= 1.0e6)
    return ok


def _feasible_mask(goals: dict, m: dict) -> np.ndarray:
    ok = _valid_mask(m)
    for gk, mk in _GOAL_METRIC.items():
        if gk not in goals:
            continue
        if gk.startswith("minimum_"):
            ok &= m[mk] >= float(goals[gk])
        else:
            ok &= m[mk] <= float(goals[gk])
    return ok


def _scalar(m: dict) -> dict:
    return {k: float(np.asarray(m[k]).reshape(-1)[0]) for k in _WORST}


def _mismatch_draws(spec: dict, x: np.ndarray, seed: int) -> dict:
    """Seeded process-mismatch draw (cheap: one 7-normal vector)."""
    tech = spec["tech"]
    mc = spec.get("mc", {}) or {}
    sk = float(mc.get("sigma_k", 0.05))
    svth = float(mc.get("sigma_vth_v", 0.003))
    sr = float(mc.get("sigma_r", 0.07))
    sc = float(mc.get("sigma_c", 0.03))
    r = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    z = r.standard_normal(7)
    kf = tuple(float(np.clip(1.0 + sk * z[j], 0.7, 1.3)) for j in range(3))
    ib_fac = float(np.clip(1.0 + 0.5 * sk * z[3], 0.75, 1.25))
    Ib = float(x[S_IB]) * ib_fac
    B = float(x[S_B])
    w_mb = float(tech.get("mb_w_um", _W_MB_UM))
    l_mb = float(tech.get("mb_l_um", _L_MB_UM))
    vovb = math.sqrt(2.0 * Ib / (float(tech["kn_ua_v2"]) * kf[2] * (w_mb / l_mb)))
    area_mb = (2.0 + B) * w_mb * l_mb
    svth_eff = svth / math.sqrt(max(area_mb / 10.0, 0.25))
    mirror_err = float(np.clip(2.0 * svth_eff / max(vovb, 0.08) * z[4], -0.6, 0.6))
    rz_fac = float(np.clip(1.0 + sr * z[5], 0.6, 1.4))
    cc_fac = float(np.clip(1.0 + sc * z[6], 0.85, 1.15))
    return {"kf": kf, "ib_fac": ib_fac, "mirror_err": mirror_err,
            "rz_fac": rz_fac, "cc_fac": cc_fac}


# ---------------------------------------------------------------------------
# problem generation helpers
# ---------------------------------------------------------------------------
def _sample_designs(rng: np.random.Generator, tech: dict, n: int,
                    family: str) -> np.ndarray:
    lmin = float(tech["lmin_um"])
    ib_lo, ib_hi = _IB_RANGE[family]
    lo = np.array([2.0, lmin, 4.0, lmin, 0.5, 0.1, ib_lo, 0.5])
    hi = np.array([200.0, 8.0 * lmin, 400.0, 6.0 * lmin, 15.0, 8.0, ib_hi, 60.0])
    cont = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(n, 8)))
    X = np.zeros((n, _NSLOT))
    X[:, S_W1], X[:, S_L1], X[:, S_W2], X[:, S_L2] = cont[:, 0], cont[:, 1], cont[:, 2], cont[:, 3]
    X[:, S_B], X[:, S_CC], X[:, S_IB], X[:, S_RZ] = cont[:, 4], cont[:, 5], cont[:, 6], cont[:, 7]
    if family == "wide_swing":
        X[:, S_COMP] = np.where(rng.random(n) < 0.5, 0.0, 2.0)
        X[:, S_CAS] = (rng.random(n) < 0.15).astype(float)
    elif family == "high_gain":
        X[:, S_COMP] = rng.integers(0, _N_COMP, size=n).astype(float)
        X[:, S_CAS] = (rng.random(n) < 0.7).astype(float)
    else:
        X[:, S_COMP] = rng.integers(0, _N_COMP, size=n).astype(float)
        X[:, S_CAS] = (rng.random(n) < 0.5).astype(float)
    X[:, S_RZON] = (rng.random(n) < 0.6).astype(float)
    return X


def _legalize_vec(tech: dict, x) -> np.ndarray:
    return np.array([_legalize(tech, s, float(x[s])) for s in range(_NSLOT)])


def _goal_ok(goals: dict, m: dict) -> bool:
    return bool(_feasible_mask(goals, {k: np.atleast_1d(v) for k, v in m.items()})[0])


class Circuit2Domain(Domain):
    """Two-stage op-amp with shape, topology and compensation-scheme moves."""
    # stochastic variants: phi_select compares candidates with a fresh
    # incumbent estimate (dataxp/ladder.py); no effect when deterministic
    rescore_incumbent = True

    name = "circuit2"
    params = _PARAMS
    factors = (0.7, 0.85, 1.2, 1.5)        # greedy fallback only
    reference_class = "heuristic"           # best of seeded sampling + polish
    max_elements = 24
    max_candidates = 160
    # per-kind move values
    _CONT_FACTORS = (0.5, 0.7, 0.85, 0.93, 1.07, 1.2, 1.4, 2.0)
    _RZ_FACTORS = (0.5, 0.7, 1.4, 2.0)

    # ── verb surface (docstring "Verb surface"; docs/dataxp_verb_surface.md) ──
    # Spelling only: every SET_PARAM move candidate_values enumerates has one
    # verb text and parses back to the same move. format_move / parse_move /
    # grammar are the table-driven defaults of dataxp.domain.
    verb_surface = True
    verb_table = (
        ParamVerb("W", "SET_WIDTH", _DEVICE_ARG,
                  Continuous("width", "um", "0.5..500, the new channel width of that device"
                                            + _LEGAL_DOC),
                  doc="resize a device's channel width: gm and Vov follow W/L, the pair area "
                      "2*W*L (M1) or W*L (M2), and the gate / junction capacitances.",
                  phrase="Set a device's channel width"),
        ParamVerb("L", "SET_LENGTH", _DEVICE_ARG,
                  Continuous("length", "um", "Lmin..12*Lmin (Lmin of the process, stated in "
                                             "the problem), the new channel length of that "
                                             "device" + _LEGAL_DOC),
                  doc="resize a device's channel length: lambda = lambda0*Lmin/L, so ro and "
                      "gain rise with L while W/L, area and gate capacitance change too; L1 "
                      "also sizes the mirror loads and cascodes.",
                  phrase="Set a device's channel length"),
        ParamVerb("B", "SET_MIRROR_RATIO", _MIRROR_ARG,
                  Continuous("ratio", "", "0.25..30, the new stage-2 to tail current ratio "
                                          "(I2 = ratio*Ib)" + _LEGAL_DOC),
                  doc="set the MB mirror ratio: stage-2 current, its power, slew limb I2/CL "
                      "and the sink's area and junction capacitance follow it.",
                  phrase="Set the mirror ratio"),
        ParamVerb("Cc", "SET_COMP_CAP", _NETWORK_ARG,
                  Continuous("cc", "pF", "0.05..20, the new compensation capacitance"
                                         + _LEGAL_DOC),
                  doc="set the compensation capacitor: dominant pole and crossover ~ "
                      "gm1/(2 pi Cc), internal slew limb Ib/Cc, area 200 um^2/pF.",
                  phrase="Set the compensation capacitor"),
        ParamVerb("comp", "SET_COMPENSATION", _NETWORK_ARG,
                  Categorical("scheme", COMP_NAMES, aliases=_COMP_ALIASES, doc=_COMP_DOC),
                  doc="select the compensation scheme around Cc; Cc and the resistor setting "
                      "are kept.",
                  phrase="Select the compensation scheme"),
        ParamVerb("Ib", "SET_BIAS_CURRENT", _SOURCE_ARG,
                  Continuous("ib", "uA", "0.5..300, the new tail current (Id1 = ib/2; the "
                                         "MB mirror copies it)" + _LEGAL_DOC),
                  doc="set the tail / bias current: every branch current, the power, the "
                      "helper-device areas and every Vov (~ sqrt(Ib)) scale with it.",
                  phrase="Set the bias current"),
        ParamVerb("Rz", "SET_NULLING_RESISTOR", _RESISTOR_ARG,
                  Continuous("rz", "kOhm", "0.1..200, the new resistance; in the signal path "
                                           "only while enabled and the scheme is DirectMiller"
                                           + _LEGAL_DOC),
                  doc="set the nulling resistor's value: the compensation zero moves to "
                      "1/(2 pi Cc (1/gm2 - Rz)) (left half plane once Rz > 1/gm2) and a pole "
                      "appears near 1/(2 pi Rz C1).",
                  phrase="Set the nulling resistor's value"),
        ParamVerb("rz_on", "ENABLE_NULLING", _RESISTOR_ARG,
                  Toggle("DISABLE_NULLING",
                         off_doc="take the nulling resistor out of the Cc branch (its stored "
                                 "value is kept); it then costs no area.",
                         off_phrase="Disable the nulling resistor"),
                  doc="put the nulling resistor (at its stored value) in series with Cc; it "
                      "acts only with the DirectMiller scheme but costs its area under every "
                      "scheme.",
                  phrase="Enable the nulling resistor"),
        ParamVerb("cas_on", "ENABLE_CASCODE", _CASCODE_ARG,
                  Toggle("DISABLE_CASCODE",
                         off_doc="remove the stage-1 cascode: node-1 resistance back to "
                                 "ro/2, the input range regained, its area and pole gone.",
                         off_phrase="Disable the cascode"),
                  doc="add a telescopic cascode to both stage-1 branches: node-1 resistance "
                      "x gm_cas*ro_cas (tens to hundreds), costs input common-mode range, "
                      "cascode area and a cascode-node pole.",
                  phrase="Enable the cascode"),
    )

    def __init__(self, stochastic: bool = False):
        super().__init__(stochastic)

    # -- problems ------------------------------------------------------------
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[index % len(_FAMILIES)]
        tier = _TIERS[(index // len(_FAMILIES)) % len(_TIERS)]
        node = _NODE_NAMES[int(rng.integers(len(_NODE_NAMES)))]
        nd = _NODES[node]
        tech = {
            "node": node,
            "lmin_um": nd["lmin_um"],
            "vdd_v": round(nd["vdd_v"] * float(rng.uniform(0.97, 1.03)), 4),
            "vth_v": round(nd["vth_v"] * float(rng.uniform(0.95, 1.05)), 4),
            "cox_pf_per_um2": nd["cox"],
            "kn_ua_v2": round(float(rng.uniform(*nd["kn"])), 3),
            "lambda0_per_v": round(float(rng.uniform(*nd["lam0"])), 4),
            "cl_pf": round(float(rng.uniform(*_CL_RANGE[family])), 4),
        }
        # MB unit device: sized by the designer for this family's current
        # range (Vov 0.2 V at the range's geometric mean), then FIXED.
        ib_mid = math.sqrt(_IB_RANGE[family][0] * _IB_RANGE[family][1])
        tech["mb_l_um"] = _L_MB_UM
        tech["mb_w_um"] = round(float(np.clip(
            2.0 * ib_mid * _L_MB_UM / (tech["kn_ua_v2"] * 0.2 ** 2), 0.5, 60.0)), 3)
        mc = {"sigma_k": round(float(rng.uniform(0.03, 0.06)), 5),
              "sigma_vth_v": round(float(rng.uniform(0.002, 0.004)), 6),
              "sigma_r": round(float(rng.uniform(0.05, 0.10)), 5),
              "sigma_c": round(float(rng.uniform(0.02, 0.04)), 5)}
        vdd = tech["vdd_v"]
        vov_hi = _vov_hi(vdd)

        # ---- witness: a random sane design the goals are anchored to -------
        x_w = None
        for attempt in range(8):
            Xw = _sample_designs(rng, tech, 900, family)
            mw = _analyze(tech, Xw)
            sane = (_valid_mask(mw)
                    & (mw["min_vov"] >= _VOV_MIN + 0.01)
                    & (mw["max_vov"] <= vov_hi - 0.03)
                    & (mw["gain_db"] >= 55.0) & (mw["gain_db"] <= 130.0)
                    & (mw["phase_margin_deg"] >= 52.0)
                    & (mw["phase_margin_deg"] <= 95.0)
                    & (mw["gbw_mhz"] >= 0.5)
                    & (mw["slew_v_per_us"] >= 0.3)
                    & (mw["power_mw"] <= 25.0)
                    & (mw["icmr_v"] >= 0.15)
                    & (mw["out_swing_v"] >= 0.35)
                    & (mw["area"] <= 2.0e4))
            if family == "wide_swing":
                pref = sane & (Xw[:, S_CAS] < 0.5) & (Xw[:, S_COMP] != 1)
                if pref.any():
                    sane = pref
            idx = np.flatnonzero(sane)
            if idx.size == 0:
                continue
            if family == "general" or idx.size < 4:
                pick = int(idx[int(rng.integers(idx.size))])
            else:
                key = {"high_gain": mw["gain_db"],
                       "high_speed": mw["gbw_mhz"],
                       "low_power": -mw["power_mw"],
                       "wide_swing": mw["icmr_v"] + mw["out_swing_v"]}[family][idx]
                order = idx[np.argsort(-key)]
                top = order[:max(2, order.size // 4)]
                pick = int(top[int(rng.integers(top.size))])
            x_w = _legalize_vec(tech, Xw[pick])
            break
        if x_w is None:  # essentially unreachable fallback: a plain design
            lmin = tech["lmin_um"]
            x_w = _legalize_vec(tech, [40.0, 2 * lmin, 80.0, 2 * lmin, 2.0, 2.0,
                                       0.0, 10.0, 5.0, 1.0, 0.0])
        w = _scalar(_analyze(tech, x_w[None, :]))

        # ---- goals ---------------------------------------------------------
        feasible_start = bool(rng.random() < 0.12)
        t = "easy" if feasible_start else tier
        f_gain = {"easy": (0.82, 0.90), "medium": (0.88, 0.95), "hard": (0.93, 0.985)}[t]
        f_speed = {"easy": (0.40, 0.65), "medium": (0.55, 0.80), "hard": (0.72, 0.93)}[t]
        f_pow = {"easy": (1.4, 2.0), "medium": (1.15, 1.5), "hard": (1.03, 1.25)}[t]
        pm_margin = {"easy": 8.0, "medium": 4.0, "hard": 1.5}[t]
        hr_margin = {"easy": (0.12, 0.30), "medium": (0.06, 0.18), "hard": (0.02, 0.09)}[t]
        if family == "wide_swing" and not feasible_start:
            hr_margin = (0.01, 0.06)
        if family == "high_speed":
            f_speed = (min(f_speed[0] + 0.1, 0.9), min(f_speed[1] + 0.05, 0.97))
        if family == "low_power" and not feasible_start:
            f_pow = (1.02, min(f_pow[1], 1.2))
        pm_goal = float(min(rng.uniform(50.0, 65.0), w["phase_margin_deg"] - pm_margin))
        pm_goal = float(max(pm_goal, 40.0))
        goals = {
            "minimum_gain_db": round(w["gain_db"] * float(rng.uniform(*f_gain)), 3),
            "minimum_gbw_mhz": _round_sig(w["gbw_mhz"] * float(rng.uniform(*f_speed)), 4),
            "minimum_phase_margin_deg": round(pm_goal, 2),
            "minimum_slew_v_per_us": _round_sig(w["slew_v_per_us"] * float(rng.uniform(*f_speed)), 4),
            "maximum_power_mw": _round_sig(w["power_mw"] * float(rng.uniform(*f_pow)), 4),
            "minimum_icmr_v": round(max(0.05, w["icmr_v"] - float(rng.uniform(*hr_margin))), 4),
            "minimum_out_swing_v": round(max(0.1, w["out_swing_v"] - float(rng.uniform(*hr_margin))), 4),
        }

        # ---- initial design -------------------------------------------------
        lmin = tech["lmin_um"]
        if feasible_start:
            x0 = np.array(x_w, dtype=float)
        else:
            # small currents, plain Miller, no resistor, no cascode; widths
            # follow from a target overdrive so the start is a VALID design
            x0 = None
            kn = float(tech["kn_ua_v2"])
            for _ in range(60):
                ib = float(ib_mid * rng.uniform(0.2, 0.45))
                bb = float(rng.uniform(0.6, 2.0))
                dev = []
                for cur in (0.5 * ib, bb * ib):
                    ll = lmin * float(rng.uniform(1.0, 2.5))
                    vt = float(rng.uniform(0.10, 0.22))
                    wl = 2.0 * cur / (kn * vt ** 2)
                    ww = wl * ll
                    if ww < 0.5:
                        ll = min(12.0 * lmin, 0.5 / wl)
                        ww = 0.5
                    dev += [ww, ll]
                cand = np.array([dev[0], dev[1], dev[2], dev[3], bb,
                                 rng.uniform(0.4, 2.0), 0.0, ib,
                                 rng.uniform(2.0, 20.0), 0.0, 0.0])
                cand = _legalize_vec(tech, cand)
                m0 = _scalar(_analyze(tech, cand[None, :]))
                if (m0["min_vov"] >= _VOV_MIN + 0.008
                        and m0["max_vov"] <= vov_hi - 0.05
                        and m0["vov_defect"] <= 0.0):
                    x0 = cand
                    break
            if x0 is None:
                x0 = _legalize_vec(tech, [4.0, lmin, 8.0, lmin, 1.0, 1.0, 0.0,
                                          2.0, 5.0, 0.0, 0.0])
            # Move the small design part of the way toward the witness in log
            # space (continuous sizes only; topology stays plain Miller, no
            # resistor, no cascode). Every Vov is log-linear in (W, L, I), so
            # the blend of two valid designs is valid. The remaining distance
            # is the tier's difficulty.
            t_lo, t_hi = {"easy": (0.45, 0.65), "medium": (0.30, 0.50),
                          "hard": (0.15, 0.35)}[tier]
            tb = float(rng.uniform(t_lo, t_hi))
            xb = np.array(x0, dtype=float)
            for sl in (S_W1, S_L1, S_W2, S_L2, S_B, S_CC, S_IB):
                xb[sl] = math.exp((1.0 - tb) * math.log(x0[sl])
                                  + tb * math.log(x_w[sl]))
            xb = _legalize_vec(tech, xb)
            mb = _scalar(_analyze(tech, xb[None, :]))
            if (mb["vov_defect"] <= 0.0 and mb["min_vov"] >= _VOV_MIN + 0.005
                    and not _goal_ok(goals, mb)):
                x0 = xb
            # stored resistor value: what a designer would pencil in, around
            # 1/gm2 of the start (it only acts once rz_on is set)
            d0 = _analyze(tech, x0[None, :], detail=True)
            x0[S_RZ] = _legalize(tech, S_RZ, 1e3 / float(d0["gm2"][0])
                                 * float(rng.uniform(0.6, 2.5)))
            m0 = _scalar(_analyze(tech, x0[None, :]))
            # a start that happens to meet everything: pull the speed goals up
            # toward the witness (graded difficulty; witness stays feasible)
            if _goal_ok(goals, m0):
                for gk, wv, iv in (("minimum_gbw_mhz", w["gbw_mhz"], m0["gbw_mhz"]),
                                   ("minimum_slew_v_per_us", w["slew_v_per_us"],
                                    m0["slew_v_per_us"])):
                    uplift = float(rng.uniform(1.4, 2.4))
                    goals[gk] = _round_sig(max(goals[gk], min(0.95 * wv, iv * uplift)), 4)

        return {
            "problem_id": "circuit2_%04d" % index,
            "family": family,
            "tier": tier,
            "goals": goals,
            "tech": tech,
            "mc": mc,
            "x0": [float(v) for v in x0],
            "x_witness": [float(v) for v in x_w],
            "starts_feasible_by_design": bool(feasible_start),
        }

    def load(self, spec: dict):
        tech = spec["tech"]
        x = np.array([_legalize(tech, s, float(v)) for s, v in enumerate(spec["x0"])])
        return {"spec": spec, "v": x}

    def clone(self, st):
        return {"spec": st["spec"], "v": np.array(st["v"], dtype=float)}

    # -- simulation ----------------------------------------------------------
    def evaluate(self, st, seed: int = 0) -> dict:
        self.calls += 1
        try:
            spec = st["spec"]
            x = np.asarray(st["v"], dtype=float)
            kw = {}
            if self.stochastic:
                kw = _mismatch_draws(spec, x, seed)
            out = _scalar(_analyze(spec["tech"], x[None, :], **kw))
            for v in out.values():
                if not math.isfinite(v):
                    return dict(_WORST)
            return out
        except Exception:
            return dict(_WORST)

    # -- design variables ------------------------------------------------------
    def n_elements(self, st) -> int:
        return len(_ELEMENTS)

    def get(self, st, i: int, p: str) -> Optional[float]:
        slot = _SLOT.get((int(i), p))
        if slot is None:
            return None
        return float(st["v"][slot])

    def set(self, st, i: int, p: str, v: float) -> None:
        slot = _SLOT.get((int(i), p))
        if slot is None:
            return
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v):
            return
        st["v"][slot] = _legalize(st["spec"]["tech"], slot, v)

    def bounds(self, st, p: str) -> Tuple[float, float]:
        lo, hi = _slot_bounds(st["spec"]["tech"])
        for (i, q), s in _SLOT.items():
            if q == p:
                return float(lo[s]), float(hi[s])
        return (0.0, 0.0)

    # -- typed move set (v2 hook) ----------------------------------------------
    def candidate_values(self, st, i: int, p: str, cur: float) -> List[float]:
        slot = _SLOT.get((int(i), p))
        if slot is None:
            return []
        tech = st["spec"]["tech"]
        x = st["v"]
        kind = _KIND[p]
        if kind == "toggle":
            return [1.0 - float(round(cur))]
        if kind == "cat":
            c = int(round(cur))
            return [float(k) for k in range(_N_COMP) if k != c]
        if p == "Rz":
            # the resistor value only acts when it is in the signal path
            if not (x[S_RZON] >= 0.5 and int(round(x[S_COMP])) == 0):
                return []
            vals = [cur * f for f in self._RZ_FACTORS]
            try:
                d = _analyze(tech, x[None, :], detail=True)
                inv_gm2 = 1e3 / float(d["gm2"][0])                 # kOhm
                ratio = float(d["CLe"][0]) / max(float(x[S_CC]), 1e-9)
                vals += [inv_gm2, inv_gm2 * (1.0 + ratio)]
            except Exception:
                pass
        else:
            vals = [cur * f for f in self._CONT_FACTORS]
        out = []
        for v in vals:
            lv = _legalize(tech, slot, v)
            if abs(lv - cur) > 1e-12 * max(1.0, abs(cur)) and lv not in out:
                out.append(lv)
        return out

    # -- program / reference ---------------------------------------------------
    def program(self, spec: dict) -> DesignProgram:
        prog = program_from_goals(
            spec["goals"],
            objective_key="area",
            objective_ref=self.reference(spec),
            objective_sense="min",
            domain=self.name,
        )
        return replace(prog, validity_bounds=(
            ("vov_defect", 1e-9),
            ("gbw_mhz", _GBW_VALID_MHZ),
            ("area", 1.0e6),
        ))

    def reference(self, spec: dict) -> Optional[float]:
        cached = spec.get("_ref_area")
        if cached is not None:
            return float(cached)
        tech = spec["tech"]
        goals = spec["goals"]
        rng = np.random.default_rng(
            stable_seed(spec.get("problem_id", ""), spec.get("family", ""),
                        "circuit2_ref"))
        lo, hi = _slot_bounds(tech)
        n = 3000
        X = np.zeros((n, _NSLOT))
        cont = [S_W1, S_L1, S_W2, S_L2, S_B, S_CC, S_IB, S_RZ]
        for s in cont:
            X[:, s] = np.exp(rng.uniform(math.log(lo[s]), math.log(hi[s]), size=n))
        X[:, S_COMP] = rng.integers(0, _N_COMP, size=n)
        X[:, S_RZON] = rng.integers(0, 2, size=n)
        X[:, S_CAS] = rng.integers(0, 2, size=n)
        extra = [np.asarray(spec["x_witness"], dtype=float)] if spec.get("x_witness") else []
        if extra:
            X = np.vstack([np.array(extra), X])
        m = _analyze(tech, X)
        feas = _feasible_mask(goals, m)
        if not feas.any():
            best = float(np.median(m["area"]))
            spec["_ref_area"] = max(best, 1e-6)
            return spec["_ref_area"]
        order = np.flatnonzero(feas)[np.argsort(m["area"][feas])]
        starts = [X[k] for k in order[:3]]
        best = float(m["area"][order[0]])
        for x in starts:
            a = self._polish(tech, goals, np.array(x), lo, hi)
            best = min(best, a)
        best = max(best, 1e-6)
        spec["_ref_area"] = best
        return best

    @staticmethod
    def _polish(tech, goals, x, lo, hi, rounds: int = 40) -> float:
        """Coordinate descent over every param (feasible, lower area)."""
        cur_area = float(_analyze(tech, x[None, :])["area"][0])
        facs = (0.6, 0.8, 0.9, 1.1, 1.25)
        for _ in range(rounds):
            nb = []
            for s in (S_W1, S_L1, S_W2, S_L2, S_B, S_CC, S_IB, S_RZ):
                for f in facs:
                    y = x.copy()
                    y[s] = min(hi[s], max(lo[s], x[s] * f))
                    nb.append(y)
            for s in (S_RZON, S_CAS):
                y = x.copy()
                y[s] = 1.0 - x[s]
                nb.append(y)
            for c in range(_N_COMP):
                if c != int(round(x[S_COMP])):
                    y = x.copy()
                    y[S_COMP] = float(c)
                    nb.append(y)
            Y = np.array(nb)
            m = _analyze(tech, Y)
            ok = _feasible_mask(goals, m) & (m["area"] < cur_area * (1 - 1e-6))
            if not ok.any():
                break
            k = int(np.flatnonzero(ok)[np.argmin(m["area"][ok])])
            x = Y[k]
            cur_area = float(m["area"][k])
        return cur_area

    # -- depth-0 feedback: blamed element and the greedy move -------------------
    def _advice(self, st) -> Optional[Tuple[int, str, float, float]]:
        spec = st["spec"]
        tech = spec["tech"]
        goals = spec["goals"]
        x = np.asarray(st["v"], dtype=float)
        try:
            d = _analyze(tech, x[None, :], detail=True)
        except Exception:
            return None
        m = {k: float(np.asarray(v).reshape(-1)[0]) for k, v in d.items()
             if k in _WORST or k in ("vov1", "vov2", "vovb", "gm2", "CLe",
                                     "I2", "I_buf", "I_cg")}
        if not all(math.isfinite(v) for v in m.values()):
            return None
        lo, hi = _slot_bounds(tech)

        def move(slot, value):
            nv = _legalize(tech, slot, value)
            if abs(nv - x[slot]) <= 1e-12 * max(1.0, abs(x[slot])):
                return None
            for (i, p), s in _SLOT.items():
                if s == slot:
                    return (i, p, nv, float(x[slot]))
            return None

        def first(*opts):
            for o in opts:
                if o is not None:
                    return o
            return None

        rel = {}
        for gk, mk in _GOAL_METRIC.items():
            if gk not in goals:
                continue
            lim = float(goals[gk])
            sc = max(abs(lim), 1e-9)
            rel[gk] = ((lim - m[mk]) / sc if gk.startswith("minimum_")
                       else (m[mk] - lim) / sc)
        comp = int(round(x[S_COMP]))
        rzon = x[S_RZON] >= 0.5
        cas = x[S_CAS] >= 0.5
        if m["vov_defect"] > 0.0:
            # outside the square-law window: undo the offending device
            if m["vov1"] < _VOV_MIN:
                return first(move(S_W1, x[S_W1] * 0.67), move(S_IB, x[S_IB] * 1.5))
            if m["vov2"] < _VOV_MIN:
                return first(move(S_W2, x[S_W2] * 0.67), move(S_B, x[S_B] * 1.5))
            if m["vovb"] < _VOV_MIN:
                return move(S_IB, x[S_IB] * 1.5)
            vhi = _vov_hi(tech["vdd_v"])
            if m["vovb"] > vhi:
                return move(S_IB, x[S_IB] * 0.67)
            if m["vov1"] > vhi:
                return first(move(S_W1, x[S_W1] * 1.5), move(S_L1, x[S_L1] * 0.67),
                             move(S_IB, x[S_IB] * 0.67))
            return first(move(S_W2, x[S_W2] * 1.5), move(S_L2, x[S_L2] * 0.67),
                         move(S_B, x[S_B] * 0.67))
        if not rel:
            return None
        worst = max(rel, key=lambda k: rel[k])
        if rel[worst] <= 0.0:
            # feasible: drop a bypassed resistor (area only), else shrink the
            # largest area contributor
            if rzon and comp != 0:
                return move(S_RZON, 0.0)
            parts = {"cc": float(d["a_cc"][0]), "m1": float(d["a_m1"][0]),
                     "m2": float(d["a_m2"][0])}
            big = max(parts, key=lambda k: parts[k])
            if big == "cc":
                return move(S_CC, x[S_CC] * 0.85)
            if big == "m1":
                return first(move(S_L1, x[S_L1] * 0.85), move(S_W1, x[S_W1] * 0.85))
            return first(move(S_L2, x[S_L2] * 0.85), move(S_W2, x[S_W2] * 0.85))
        if worst == "minimum_gain_db":
            if (not cas and "minimum_icmr_v" in goals
                    and m["icmr_v"] - float(goals["minimum_icmr_v"]) > _CAS_HEADROOM_V + 0.02):
                return move(S_CAS, 1.0)
            rel_l1 = x[S_L1] / lo[S_L1]
            rel_l2 = x[S_L2] / lo[S_L2]
            order = (S_L1, S_L2) if rel_l1 <= rel_l2 else (S_L2, S_L1)
            return first(move(order[0], x[order[0]] * 1.5),
                         move(order[1], x[order[1]] * 1.5),
                         move(S_W1, x[S_W1] * 1.5))
        if worst == "minimum_gbw_mhz":
            if m["vov1"] > 1.6 * _VOV_MIN:
                return first(move(S_W1, x[S_W1] * 1.5), move(S_IB, x[S_IB] * 1.5))
            return first(move(S_IB, x[S_IB] * 1.5), move(S_CC, x[S_CC] * 0.67))
        if worst == "minimum_phase_margin_deg":
            if comp == 0 and not rzon:
                return move(S_RZON, 1.0)
            if comp == 0:
                target = 1e3 / m["gm2"] * (1.0 + m["CLe"] / max(x[S_CC], 1e-9))
                if abs(x[S_RZ] - target) > 0.15 * target:
                    o = move(S_RZ, target)
                    if o is not None:
                        return o
            return first(move(S_CC, x[S_CC] * 1.5), move(S_B, x[S_B] * 1.5))
        if worst == "minimum_slew_v_per_us":
            if x[S_IB] / max(x[S_CC], 1e-9) <= m["I2"] / float(tech["cl_pf"]):
                return first(move(S_IB, x[S_IB] * 1.5), move(S_CC, x[S_CC] * 0.67))
            return first(move(S_B, x[S_B] * 1.5), move(S_IB, x[S_IB] * 1.5))
        if worst == "maximum_power_mw":
            if m["I_buf"] > 0.0 or m["I_cg"] > 0.0:
                return move(S_COMP, 0.0)
            return first(move(S_IB, x[S_IB] * 0.67), move(S_B, x[S_B] * 0.67))
        if worst == "minimum_icmr_v":
            if cas:
                return move(S_CAS, 0.0)
            if m["vov1"] >= m["vovb"]:
                return first(move(S_W1, x[S_W1] * 1.5), move(S_IB, x[S_IB] * 0.67))
            return move(S_IB, x[S_IB] * 0.67)
        if worst == "minimum_out_swing_v":
            if comp == 1:
                return move(S_COMP, 0.0)
            if m["vov2"] >= m["vovb"]:
                return first(move(S_W2, x[S_W2] * 1.5), move(S_B, x[S_B] * 0.67))
            return move(S_IB, x[S_IB] * 0.67)
        return None

    def hint(self, st) -> Optional[Tuple[int, str, str]]:
        a = self._advice(st)
        if a is None:
            return None
        i, p, v, v_old = a
        return (i, p, "up" if v > v_old else "down")

    def hint_move(self, st) -> Optional[Candidate]:
        a = self._advice(st)
        if a is None:
            return None
        i, p, v, v_old = a
        return Candidate(i=i, p=p, v=float(v), v_old=float(v_old))

    # -- observation -------------------------------------------------------------
    def describe(self, st, spec=None) -> str:
        """Self-contained plain-text observation: PROBLEM (setup, every tunable
        with units, bounds, codes and the cost of each kind of change), DESIGN
        (every element and its current values) and RESULTS (nominal per-element
        hand-analysis outputs, frequency response, headroom). Pure."""
        spec = spec if spec is not None else st["spec"]
        x = np.asarray(st["v"], dtype=float).reshape(-1)
        d = _analyze(spec["tech"], x[None, :], detail=True)
        d = {k: (np.asarray(v)[0] if np.ndim(v) >= 1 else v) for k, v in d.items()}
        # verbs under multitool; the SET_PARAM paragraph under set_parameter
        # (docstring "Set-parameter spelling")
        return _obs_text(spec, x, d, self.stochastic, verbs=spelled_in_verbs(self))

    def element_label(self, st, i) -> str:
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
    "CC (element 3, compensation capacitor and scheme)",
    "IB (element 4, tail/bias current)",
    "RZ (element 5, nulling resistor)",
    "CAS (element 6, stage-1 cascode)",
)
_DB_PER_NEPER = 20.0 / math.log(10.0)
# one-tunable x1.1 steps reported as local sensitivities
_OBS_SENS = ((S_W1, "W1"), (S_L1, "L1"), (S_W2, "W2"), (S_L2, "L2"),
             (S_B, "B"), (S_CC, "Cc"), (S_IB, "Ib"))
_OBS_SPREAD_DRAWS = 16


def _n(v, sig: int = 4) -> str:
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


def _sg(v, sig: int = 2, dec: int = 2) -> str:
    """Signed short number with at most `dec` decimals: +0.41, -12, 0."""
    try:
        v = round(float(v), dec)
    except (TypeError, ValueError):
        return str(v)
    s = _n(v, sig)
    if s in ("0", "-0"):
        return "0"
    return s if s.startswith("-") or not math.isfinite(float(v)) else "+" + s


def _mhz(w_rad_per_us: float) -> str:
    return _n(abs(w_rad_per_us) / _TWO_PI)


def _vov_note(v: float, lo: float, hi: float) -> str:
    if v < lo:
        return "BELOW the %s V floor: invalid" % _n(lo)
    if v > hi:
        return "ABOVE the %s V ceiling: invalid" % _n(hi)
    return "floor +%s, ceiling -%s V" % (_n(v - lo, 3), _n(hi - v, 3))


def _db(v: float) -> str:
    return _n(20.0 * math.log10(max(float(v), 1e-12)), 4)


def _row(dd: dict, r: int) -> dict:
    return {k: (np.asarray(v)[r] if np.ndim(v) >= 1 else v) for k, v in dd.items()}


def _obs_factors(d: dict):
    return tuple(np.asarray(d[k], float).reshape(-1) for k in ("pa", "pb", "za", "zb"))


def _obs_log_a0(d: dict) -> float:
    return math.log(max(float(d["A0"]), 1e-300))


def _obs_lm(d: dict, w) -> np.ndarray:
    """log|A(j w)| (nepers) of one analysed design at frequencies w (rad/us)."""
    pa, pb, za, zb = _obs_factors(d)
    w = np.atleast_1d(np.asarray(w, float))
    return _logmag(np.array([_obs_log_a0(d)]), pa[None, :], pb[None, :],
                   za[None, :], zb[None, :], w[None, :])[0]


def _obs_crossings(d: dict) -> List[Tuple[float, float]]:
    """Every unity crossing [(w, phase rad)] of one analysed design, located
    exactly as _crossover_one locates them (that routine keeps only the first
    frequency and the worst phase)."""
    pa, pb, za, zb = _obs_factors(d)
    logA0 = _obs_log_a0(d)
    P = list(zip(pa.tolist(), pb.tolist()))
    Z = list(zip(za.tolist(), zb.tolist()))

    def lm(w):
        s = logA0
        for a, b in Z:
            s += 0.5 * math.log((1.0 + w * b) ** 2 + (w * a) ** 2)
        for a, b in P:
            s -= 0.5 * math.log((1.0 + w * b) ** 2 + (w * a) ** 2)
        return s

    def lmx(xv):
        return lm(math.exp(xv))

    def phase(w):
        ph = 0.0
        for a, b in Z:
            ph += math.atan2(-w * a, 1.0 + w * b)
        for a, b in P:
            ph -= math.atan2(-w * a, 1.0 + w * b)
        return ph

    lm_row = _obs_lm(d, _W_GRID)
    pos = lm_row >= 0.0
    if not pos[0]:
        w = float(_W_GRID[0])
        return [(w, phase(w))]
    brackets = []
    for k in np.flatnonzero(pos[:-1] != pos[1:]).tolist():
        brackets.append((float(_LOG_W_GRID[k]), float(lm_row[k]),
                         float(_LOG_W_GRID[k + 1]), float(lm_row[k + 1])))
    for j in _near_zero_extrema(lm_row).tolist():
        xl, xr = float(_LOG_W_GRID[j - 1]), float(_LOG_W_GRID[j + 1])
        sign = 1.0 if lm_row[j] < 0.0 else -1.0
        xe, fe = _golden_extremum(lmx, xl, xr, sign)
        if (fe >= 0.0) != (lm_row[j] >= 0.0):
            brackets.append((xl, float(lm_row[j - 1]), xe, fe))
            brackets.append((xe, fe, xr, float(lm_row[j + 1])))
    if not brackets:
        w = float(_W_GRID[-1])
        return [(w, phase(w))]
    brackets.sort(key=lambda t: t[0])
    out = []
    for xa, fa, xb, fb in brackets:
        wc = math.exp(_refine_scalar(lm, xa, fa, xb, fb))
        out.append((wc, phase(wc)))
    return out


def _obs_peak(d: dict, w_from: float) -> Optional[Tuple[float, float]]:
    """Highest local maximum of |A| above w_from (searched over 4 decades,
    and past 10x the complex pair), as (w rad/us, dB); None when the gain
    falls monotonically there."""
    w_from = max(float(w_from), 1e-12)
    pair = _obs_pair(d)
    span = max(4.0 * math.log(10.0),
               math.log(10.0 * pair[0] / w_from) if pair is not None else 0.0)
    npts = max(401, int(100.0 * span / math.log(10.0)) + 1)
    xs = math.log(w_from) + np.linspace(0.0, span, npts)
    lm = _obs_lm(d, np.exp(xs))
    mid = lm[1:-1]
    idx = np.flatnonzero((mid > lm[:-2]) & (mid >= lm[2:])) + 1
    if idx.size == 0:
        return None

    def f(xv):
        return float(_obs_lm(d, [math.exp(xv)])[0])

    best = None
    for j in idx.tolist():
        xe, fe = _golden_extremum(f, float(xs[j - 1]), float(xs[j + 1]), 1.0)
        if best is None or fe > best[1]:
            best = (math.exp(xe), fe)
    return best[0], best[1] * _DB_PER_NEPER


def _obs_pair(d: dict) -> Optional[Tuple[float, float]]:
    """(|p| rad/us, Q) of the loop's complex pole pair, if it has one."""
    for v in np.asarray(d["u_loop"]).reshape(-1).tolist():
        u = complex(v)
        if abs(u) > 1e-30 and abs(u.imag) > 1e-9 * abs(u):
            p = 1.0 / u
            q = abs(p) / (2.0 * abs(p.real)) if p.real != 0 else float("inf")
            return abs(p), q
    return None


def _root_entries(d: dict):
    """Poles and zeros as [(short name, text, [(a, b, sign)])]; sign -1
    for a pole, +1 for a zero."""
    pa, pb, za, _ = _obs_factors(d)
    poles, used, order = [], set(), 0
    loop = [(k, complex(pa[k], pb[k])) for k in range(3)
            if abs(complex(pa[k], pb[k])) > 1e-30]
    loop.sort(key=lambda kv: -abs(kv[1]))      # largest time constant first
    for k, u in loop:
        if k in used:
            continue
        p = 1.0 / u
        used.add(k)
        if abs(u.imag) > 1e-9 * abs(u):
            facs = [(pa[k], pb[k], -1.0)]
            mate = [j for j, v in loop if j not in used
                    and abs(v - u.conjugate()) <= 1e-6 * abs(u)]
            if mate:
                used.add(mate[0])
                facs.append((pa[mate[0]], pb[mate[0]], -1.0))
            q = abs(p) / (2.0 * abs(p.real)) if p.real != 0 else float("inf")
            poles.append(("complex pair", "complex pair |p|=%s MHz Q=%s"
                          % (_mhz(abs(p)), _n(q, 3)), facs))
        else:
            order += 1
            tag = "dominant" if order == 1 else "non-dominant"
            side = "" if p.real < 0 else " RIGHT-half-plane"
            poles.append((tag, "%s%s %s MHz" % (tag, side, _mhz(p.real)),
                          [(pa[k], pb[k], -1.0)]))
    for k, name in ((3, "current-mirror"), (4, "cascode-node")):
        if abs(pa[k]) > 1e-30:
            poles.append(("%s pole" % name, "%s %s MHz" % (name, _mhz(1.0 / pa[k])),
                          [(pa[k], 0.0, -1.0)]))
    zeros = []
    for k, name in enumerate(("compensation", "current-mirror")):
        if k >= za.size or abs(za[k]) <= 1e-30:
            continue
        z = 1.0 / za[k]           # factor (1 - s/z)
        side = "RIGHT-half-plane" if z > 0 else "left-half-plane"
        zeros.append(("%s zero" % name, "%s %s zero %s MHz" % (name, side, _mhz(z)),
                      [(za[k], 0.0, 1.0)]))
    return poles, zeros


def _contrib(facs, w: float) -> float:
    """Phase contribution (deg) of a pole/zero entry at w."""
    return sum(sg * math.degrees(float(_factor_angle(a, b, w))) for a, b, sg in facs)


def _lag_lead(c: float) -> str:
    return ("lead %s deg" % _n(c, 3)) if c > 0 else ("lag %s deg" % _n(-c, 3))


def _roots_text(d: dict, w: float) -> Tuple[str, str]:
    """Poles and zeros with their phase lag/lead at frequency w (rad/us)."""
    poles, zeros = _root_entries(d)
    pt = "; ".join("%s (%s)" % (t, _lag_lead(_contrib(f, w))) for _, t, f in poles)
    zt = "; ".join("%s (%s)" % (t, _lag_lead(_contrib(f, w))) for _, t, f in zeros)
    return pt, (zt or "none")


def _phase_short(d: dict, w: float) -> str:
    """Compact per-factor phase at w plus the total."""
    poles, zeros = _root_entries(d)
    parts, tot = [], 0.0
    for name, _, f in poles + zeros:
        c = _contrib(f, w)
        tot += c
        parts.append("%s %s" % (name, _sg(c, 3)))
    return "%s (total %s deg)" % (", ".join(parts), _n(tot, 4))


def _obs_variants(tech: dict, x, mods) -> List[dict]:
    """Analyse copies of x with {slot: value} overrides (what-if rows)."""
    if not mods:
        return []
    X = np.tile(np.asarray(x, float), (len(mods), 1))
    for r, mod in enumerate(mods):
        for s, v in mod.items():
            X[r, s] = v
    dd = _analyze(tech, X, detail=True)
    return [_row(dd, r) for r in range(len(mods))]


def _obs_valid(r: dict) -> bool:
    return bool(_valid_mask({k: np.atleast_1d(r[k]) for k in _WORST})[0])


def _obs_spread(spec: dict, x) -> dict:
    """1-sigma mismatch spread of each constrained metric over fixed draws."""
    tech = spec["tech"]
    keys = ("gain_db", "gbw_mhz", "phase_margin_deg", "slew_v_per_us",
            "power_mw", "icmr_v", "out_swing_v")
    vals = {k: [] for k in keys}
    xv = np.asarray(x, float)
    for j in range(_OBS_SPREAD_DRAWS):
        kw = _mismatch_draws(spec, xv, 104729 + 7919 * j)
        m = _scalar(_analyze(tech, xv[None, :], **kw))
        for k in keys:
            vals[k].append(m[k])
    return {k: float(np.std(v, ddof=1)) for k, v in vals.items()}


def _obs_tunables(lo, hi, w_mb: float, l_mb: float, verbs: bool) -> List[str]:
    """The 'how to write a change' paragraph of the observation: the SET_PARAM
    spelling, or (verbs=True, docstring "Verb surface") the same facts with
    each element line naming its verbs. The physics text of every line is
    shared between the two spellings."""
    n = _n
    m1_tail = (" Pair area 2*W*L. L1 is also the length of "
               "the mirror loads and cascodes, so their area grows as Ib*L1^2, the "
               "current-mirror pole falls roughly as 1/L1^2, and the cascode gain "
               "grows as L1. W1 adds Cj*W1 to node 1 (to the cascode node instead "
               "when CAS is on) and to the mirror node.")
    m2_tail = (" Area W*L; its gate (2/3)Cox*W*L loads node 1 (C1), its drain junction "
               "Cj*W the output (CL_out).")
    mb_tail = (" Area (2+B)*%s um^2; stage-2 power B*Ib*Vdd; the sink copy adds Cj*B*%s um "
               "to CL_out." % (n(w_mb * l_mb), n(w_mb)))
    comp0 = ("feed-forward through Cc gives a right-half-plane zero at gm2/(2 pi Cc) "
             "unless RZ is on; non-dominant pole ~ gm2*Cc/(2 pi (C1*CL_out + Cc*(C1 + "
             "CL_out))) ~ gm2/(2 pi CL_out) once Cc >> C1, so a larger Cc buys margin "
             "only by lowering the crossover (~ gm1/(2 pi Cc)); a larger gm2 raises the "
             "pole.")
    comp1 = ("(source follower at Ib/2, gm_f = Ib/%s V): left-half-plane zero at "
             "gm_f/(2 pi Cc), non-dominant pole ~ gm2*Cc/(2 pi (C1 + Cc)*CL_out), third "
             "pole near gm_f/(2 pi C1); costs Ib/2, the follower area, and lifts the "
             "lowest output voltage to Vth + %s V." % (n(_VAUX), n(2 * _VAUX)))
    comp2 = ("(Cc returns into node 1 through a common-gate device at Ib, gm_c = 2*Ib/%s "
             "V): left-half-plane zero at gm_c/(2 pi Cc); the non-dominant poles usually "
             "form a complex pair of natural frequency ~ sqrt(gm2*gm_c/(C1*CL_out))/(2 pi), "
             "independent of Cc, and Q ~ [Cc/(CL_out + Cc)]*sqrt(gm2*CL_out/(gm_c*C1)): a "
             "larger Cc or gm2, or a smaller C1 or gm_c, raises Q and the resonant peak, "
             "and once the peak passes 0 dB the phase margin collapses; above gm_c/(2 pi "
             "Cc) the gain flattens at gm1/gm_c, so it needs gm_c > gm1; costs Ib and the "
             "common-gate device area." % n(_VAUX))
    ib_tail = (" Every current scales with it: total power = (2 + B, +0.5 if comp = 1, +1 "
               "if comp = 2)*Ib*Vdd; helper-device area (less fixed overheads) is "
               "proportional to Ib; DC gain ~ 1/Ib; every Vov ~ sqrt(Ib), so MB (fixed "
               "size) is valid only inside an Ib window (RESULTS).")
    rz_tail = ("the zero moves to 1/(2 pi Cc (1/gm2 - Rz)) (left half plane once Rz > "
               "1/gm2; Rz = (1/gm2)(1 + CL_out/Cc) puts it on the non-dominant pole) and a "
               "pole appears near 1/(2 pi Rz C1). While on it costs area whatever the "
               "scheme.")
    cas_tail = ("On: node-1 resistance x gm_cas*ro_cas = 2/(%s V * lambda(L1)); costs %s V "
                "of input common-mode range, the cascode area, and a pole gm_cas/(2 pi Cx) "
                "at the cascode node (gm_cas = Ib/%s V, Cx = Cj*W1 + NMOS-cascode gate + "
                "wiring)." % (n(_VAUX), n(_CAS_HEADROOM_V), n(_VAUX)))
    w1, w2 = (n(lo[S_W1]), n(hi[S_W1])), (n(lo[S_W2]), n(hi[S_W2]))
    l1, l2 = (n(lo[S_L1]), n(hi[S_L1])), (n(lo[S_L2]), n(hi[S_L2]))
    if verbs:
        return [
            "Tunables (one change per turn, one of SET_WIDTH(device, um), SET_LENGTH(device, "
            "um), SET_MIRROR_RATIO(2, B), SET_COMP_CAP(3, pF), SET_COMPENSATION(3, scheme), "
            "SET_BIAS_CURRENT(4, uA), SET_NULLING_RESISTOR(5, kOhm), ENABLE_NULLING(5) / "
            "DISABLE_NULLING(5), ENABLE_CASCODE(6) / DISABLE_CASCODE(6); the first argument "
            "is the element number below; a value is the new absolute setting, not a "
            "change: clamped to bounds, continuous values rounded to 4 significant digits "
            "and at most 3 decimals; the scheme is written by name):",
            "  element 0 M1: SET_WIDTH(0, W), W in [%s, %s] um; SET_LENGTH(0, L), L in [%s, "
            "%s] um (width and length of each pair device)." % (w1 + l1) + m1_tail,
            "  element 1 M2: SET_WIDTH(1, W), W in [%s, %s] um; SET_LENGTH(1, L), L in [%s, "
            "%s] um." % (w2 + l2) + m2_tail,
            "  element 2 MB: SET_MIRROR_RATIO(2, B), B = stage-2 to tail current ratio in "
            "[%s, %s]." % (n(lo[S_B]), n(hi[S_B])) + mb_tail,
            "  element 3 CC: SET_COMP_CAP(3, Cc), Cc = compensation capacitance in [%s, %s] "
            "pF. SET_COMPENSATION(3, scheme), scheme = DirectMiller (shown as comp=0): %s "
            "VoltageBufferedMiller (comp=1) %s CurrentBufferedMiller (comp=2) %s"
            % (n(lo[S_CC]), n(hi[S_CC]), comp0, comp1, comp2),
            "  element 4 IB: SET_BIAS_CURRENT(4, Ib), Ib = tail current in [%s, %s] uA."
            % (n(lo[S_IB]), n(hi[S_IB])) + ib_tail,
            "  element 5 RZ: ENABLE_NULLING(5) / DISABLE_NULLING(5) put the nulling "
            "resistor in / out of series with Cc (shown as rz_on=1 / 0); "
            "SET_NULLING_RESISTOR(5, Rz), Rz = its value in [%s, %s] kOhm (enabling uses "
            "the stored Rz). Acts only with DirectMiller: %s"
            % (n(lo[S_RZ]), n(hi[S_RZ]), rz_tail),
            "  element 6 CAS: ENABLE_CASCODE(6) / DISABLE_CASCODE(6) add / remove the "
            "stage-1 cascode on both branches (shown as cas_on=1 / 0). " + cas_tail,
        ]
    return [
        "Tunables (SET_PARAM(element, parameter, value); clamped to bounds, "
        "continuous values rounded to 4 significant digits and at most 3 "
        "decimals, comp to the nearest code, toggles to 0/1):",
        "  element 0 M1: W [%s, %s] um; L [%s, %s] um (continuous; width and "
        "length of each pair device)." % (w1 + l1) + m1_tail,
        "  element 1 M2: W [%s, %s] um; L [%s, %s] um (continuous)." % (w2 + l2) + m2_tail,
        "  element 2 MB: B = stage-2 to tail current ratio [%s, %s] "
        "(continuous)." % (n(lo[S_B]), n(hi[S_B])) + mb_tail,
        "  element 3 CC: Cc = compensation capacitance [%s, %s] pF (continuous). "
        "comp = scheme (categorical): 0 = direct Miller: %s 1 = voltage-buffered Miller "
        "%s 2 = current-buffered Miller %s"
        % (n(lo[S_CC]), n(hi[S_CC]), comp0, comp1, comp2),
        "  element 4 IB: Ib = tail current [%s, %s] uA (continuous)."
        % (n(lo[S_IB]), n(hi[S_IB])) + ib_tail,
        "  element 5 RZ: rz_on = nulling resistor in series with Cc (toggle); "
        "Rz = its value [%s, %s] kOhm (continuous; switching rz_on on uses the "
        "stored Rz). Acts only with comp = 0: %s"
        % (n(lo[S_RZ]), n(hi[S_RZ]), rz_tail),
        "  element 6 CAS: cas_on = stage-1 cascode on both branches (toggle). " + cas_tail,
    ]


def _obs_text(spec: dict, x, d: dict, stochastic: bool, verbs: bool = False) -> str:
    n = _n
    tech = spec["tech"]
    fam = str(spec.get("family", "general")).replace("_", " ")
    tier = str(spec.get("tier", ""))
    lmin = float(tech["lmin_um"])
    vdd = float(tech["vdd_v"])
    vth = float(tech["vth_v"])
    kn = float(tech["kn_ua_v2"])
    lam0 = float(tech["lambda0_per_v"])
    cox = float(tech["cox_pf_per_um2"])
    cl = float(tech["cl_pf"])
    lo, hi = _slot_bounds(tech)
    vov_hi = _vov_hi(vdd)
    w_mb = float(tech.get("mb_w_um", _W_MB_UM))
    l_mb = float(tech.get("mb_l_um", _L_MB_UM))
    x = np.asarray(x, float).reshape(-1)
    W1, L1, W2, L2, B, Cc, comp, Ib, Rz, rzon, cas = (float(v) for v in x)
    comp = int(round(comp))
    rzon = rzon >= 0.5
    cas = cas >= 0.5
    c_load = 500.0 / kn                      # um^2 per (uA um^2)
    c_cas = 700.0 / kn
    c_buf = 800.0 * lmin ** 2 / kn           # um^2 per uA
    c_cg = 2800.0 * lmin ** 2 / kn

    lines = [
        "PROBLEM",
        "Two-stage CMOS operational amplifier (%s family%s), square-law hand "
        "analysis with the exact small-signal transfer function of the hand "
        "model (no SPICE). Signal path: NMOS input pair M1 (two devices, "
        "current-mirror load) -> node 1 -> common-source M2 -> output, driving "
        "CL. The tail current Ib (IB) is copied by the MB mirror; stage-2 "
        "current I2 = B*Ib. Cc links the output and node 1 through the scheme "
        "on CC (comp), optionally with the series nulling resistor RZ; optional "
        "stage-1 cascode (CAS)."
        % (fam, (", %s tier" % tier) if tier else ""),
        "Technology and load (fixed): %s process, Lmin = %s um, Vdd = %s V, "
        "Vth = %s V, k = %s uA/V^2 for M1, M2 and MB, channel-length "
        "modulation lambda(L) = %s 1/V * Lmin/L, gate capacitance Cox = %s "
        "fF/um^2, junction capacitance Cj = %s fF per um of drain width, %s pF "
        "wiring per internal node, load CL = %s pF. Area densities: capacitor "
        "%s um^2/pF, resistor %s um^2/kOhm plus %s um^2 per resistor."
        % (tech.get("node", ""), n(lmin), n(vdd), n(vth), n(kn), n(lam0),
           n(1000.0 * cox), n(1000.0 * _CJ_PF_PER_UM), n(_C_WIRE_PF), n(cl),
           n(_CAP_UM2_PER_PF), n(_RES_UM2_PER_KOHM), n(_RES_OVH_UM2)),
        "Helper devices (auto-sized, not tunable; their area follows Ib and "
        "L1): MB is a %s um x %s um device "
        "carrying Ib, with a 1:1 tail copy and a B-times-wider stage-2 sink "
        "copy (all at the MB Vov). Mirror loads and cascodes (length L1), "
        "follower and common-gate device (length 2*Lmin) are sized to Vov = %s "
        "V (PMOS at k/%s); areas with Ib in uA, L1 in um: mirror loads "
        "(500/k)*Ib*L1^2 = %s*Ib*L1^2, cascodes %s*Ib*L1^2 + %s, follower "
        "%s*Ib + %s, common-gate device %s*Ib + %s um^2."
        % (n(w_mb), n(l_mb), n(_VAUX), n(_KP_RATIO), n(c_load), n(c_cas),
           n(_CAS_OVH_UM2), n(c_buf), n(_BUF_OVH_UM2), n(c_cg), n(_CG_OVH_UM2)),
        *_obs_tunables(lo, hi, w_mb, l_mb, verbs),
        "Equations (uA, uS, MOhm, pF, MHz, V/us, mW, um^2): Id1 = Ib/2 per input "
        "device; gm = sqrt(2 k (W/L) Id); Vov = 2 Id/gm; ro = 1/(lambda(L) Id); "
        "R1 = ro(L1)/2 (x gm_cas*ro_cas with the cascode); R2 = ro(L2)/2; DC "
        "gain = gm1 R1 gm2 R2. C1 = M2 gate + drain junctions at node 1 + "
        "wiring; CL_out = CL + M2 and MB-sink drain junctions. Open loop = "
        "exact transfer function of this two-node model with the selected "
        "compensation branch (dominant pole ~ 1/(2 pi gm2 R1 R2 Cc)), times the "
        "current-mirror doublet (pole gm3/(2 pi Cm), zero at twice it; gm3 = "
        "Ib/%s V, Cm = mirror gates + junctions) and the cascode-node pole "
        "when on. gbw_mhz = first frequency where the open-loop gain falls to "
        "1; phase_margin_deg = 180 + phase at a unity crossing, the worst one "
        "if a resonant peak crosses 1 again. slew = min(Ib/Cc, I2/CL). icmr_v "
        "= Vdd - %s (mirror-load Vov) - Vth - Vov(M1) - Vov(MB) (- %s with the "
        "cascode). out_swing_v = Vdd - Vov(M2) - Vov(MB) (comp = 1: Vdd - "
        "Vov(M2) - max(Vov(MB), Vth + %s)). area = sum of all areas above "
        "(objective: minimise). Scaling: Vov ~ sqrt(Id*L/W), gm ~ "
        "sqrt(Id*W/L), R ~ L/Id, so DC gain ~ sqrt(W1*L1*W2*L2)/(Ib*sqrt(B)) "
        "(x L1 again with the cascode)."
        % (n(_VAUX), n(_VAUX), n(_CAS_HEADROOM_V), n(2 * _VAUX)),
        "Validity (square law): M1, M2 and MB must keep %s V <= Vov <= %s V, "
        "and the crossover must stay "
        "below %s MHz; outside either window the design is invalid. In the "
        "status table min_vov / max_vov are the smallest / largest of those "
        "three Vov, and vov_defect (must be 0) is the total distance (V) by "
        "which they fall outside the window."
        % (n(_VOV_MIN), n(vov_hi), n(_GBW_VALID_MHZ)),
        "How designs are compared: area (relative to the best area known for "
        "this problem) and limit shortfalls are traded in one sum. Each "
        "shortfall is a fraction of its own limit (gain in dB: 60 dB against "
        ">= 105 dB is 0.43) and the fractions are added, so a 0.1 cut in any "
        "one is worth the same; missing any limit also costs an extra penalty "
        "that fades only as the shortfalls approach zero. Large shortfalls "
        "therefore usually matter most, but when the only shortfalls are small "
        "a large area cut can be worth more than closing them. Sitting exactly "
        "on a limit is mildly penalised (keep a small margin); among designs "
        "meeting every limit, lower area wins; an invalid design is worst.",
    ]
    if stochastic:
        mc = spec.get("mc", {}) or {}
        lines.append(
            "Process mismatch (this variant): each evaluation draws 1-sigma "
            "errors of %s%% on k of M1, M2 and MB, %s%% on the bias source, %s "
            "mV mirror Vth at 10 um^2 of MB area (~1/sqrt(area); shifts I2 by "
            "~2*dVth/Vov(MB)), %s%% on Rz and %s%% on Cc. Designs are judged "
            "over several draws, each paying its own shortfalls; a limit counts "
            "as met only at the adverse 10%% quantile (90%% yield); area does "
            "not vary. RESULTS are nominal; the spread is at the end."
            % (n(100.0 * float(mc.get("sigma_k", 0.05)), 3),
               n(50.0 * float(mc.get("sigma_k", 0.05)), 3),
               n(1000.0 * float(mc.get("sigma_vth_v", 0.003)), 3),
               n(100.0 * float(mc.get("sigma_r", 0.07)), 3),
               n(100.0 * float(mc.get("sigma_c", 0.03)), 3)))

    def at_bound(slot):
        if _SLOT_KIND[slot] != "cont":
            return ""
        if x[slot] <= lo[slot] * (1 + 1e-9):
            return " (at lower bound)"
        if x[slot] >= hi[slot] * (1 - 1e-9):
            return " (at upper bound)"
        return ""

    rz_state = "on" if rzon else "off"
    if rzon and comp != 0:
        rz_state += ", bypassed because comp != 0 (area only)"
    elif not rzon:
        rz_state += ", Rz has no effect"
    lines += [
        "",
        "DESIGN (element: role, current values)",
        "  0 M1  input differential pair (2 devices)  W=%s um%s  L=%s um%s  (W/L=%s)"
        % (n(W1), at_bound(S_W1), n(L1), at_bound(S_L1), n(W1 / L1)),
        "  1 M2  stage-2 common-source gain device  W=%s um%s  L=%s um%s  (W/L=%s)"
        % (n(W2), at_bound(S_W2), n(L2), at_bound(S_L2), n(W2 / L2)),
        "  2 MB  tail-to-stage-2 current-mirror ratio  B=%s%s  (I2 = B*Ib)"
        % (n(B), at_bound(S_B)),
        "  3 CC  compensation  Cc=%s pF%s  comp=%d (%s)"
        % (n(Cc), at_bound(S_CC), comp, _COMP_NAMES[comp]),
        "  4 IB  tail/bias current  Ib=%s uA%s" % (n(Ib), at_bound(S_IB)),
        "  5 RZ  nulling resistor  rz_on=%d (%s)  Rz=%s kOhm%s"
        % (int(rzon), rz_state, n(Rz), at_bound(S_RZ)),
        "  6 CAS stage-1 cascode  cas_on=%d (%s)" % (int(cas), "on" if cas else "off"),
    ]

    area_tot = float(d["area"])

    def pct(a):
        return n(100.0 * float(a) / area_tot, 3) if area_tot > 0 else "0"

    p_tail = 2.0 * Ib * vdd * 1e-3
    p_s2 = float(d["I2"]) * vdd * 1e-3
    p_aux = (float(d["I_buf"]) + float(d["I_cg"])) * vdd * 1e-3
    p_tot = float(d["power_mw"])

    def ppct(v):
        return n(100.0 * v / p_tot, 3) if p_tot > 0 else "0"

    gm1, gm2 = float(d["gm1"]), float(d["gm2"])
    gmb, gmc = float(d["gmb"]), float(d["gmc"])
    C1, CLe = float(d["C1"]), float(d["CLe"])
    a1 = gm1 * float(d["R1"])
    a2 = gm2 * float(d["R2"])
    inv_gm2_k = 1e3 / gm2
    gbw_est = gm1 / (_TWO_PI * Cc)
    wu = float(d["wu"])
    poles_t, zeros_t = _roots_text(d, wu)
    p2_miller = gm2 * Cc / (C1 * CLe + Cc * (C1 + CLe))

    # -- what-if rows: other schemes, toggles, x1.1 sensitivities ---------------
    mods, tags = [], []
    for c in range(_N_COMP):
        if c != comp:
            mods.append({S_COMP: float(c)})
            tags.append(("comp", c))
    mods.append({S_CAS: 0.0 if cas else 1.0})
    tags.append(("cas", None))
    if comp == 0:
        mods.append({S_RZON: 0.0 if rzon else 1.0})
        tags.append(("rz", None))
    for s, lab in _OBS_SENS:
        mods.append({s: x[s] * 1.1})
        tags.append(("sens", (s, lab)))
    try:
        wrows = _obs_variants(tech, x, mods)
    except Exception:
        wrows = []
    wi = {}
    for t, r in zip(tags, wrows):
        wi.setdefault(t[0], []).append((t[1], r))

    def fr_text(r):
        """crossover / PM of a what-if row."""
        s = "crossover %s MHz, phase margin %s deg" % (
            n(float(r["gbw_mhz"])), n(float(r["phase_margin_deg"])))
        if not _obs_valid(r):
            s += " (INVALID design)"
        return s

    if comp == 0:
        if rzon:
            zdesc = ("with Rz = %s kOhm vs 1/gm2 = %s kOhm the compensation zero "
                     "is in the %s half plane; resistor pole near 1/(2 pi Rz C1) "
                     "= %s MHz; non-dominant pole estimate %s MHz"
                     % (n(Rz), n(inv_gm2_k),
                        "right" if Rz < inv_gm2_k else "left",
                        _mhz(1.0 / max(Rz * 1e-3 * C1, 1e-30)), _mhz(p2_miller)))
        else:
            zdesc = ("no nulling resistor: right-half-plane zero at gm2/(2 pi Cc) "
                     "= %s MHz; non-dominant pole estimate %s MHz"
                     % (_mhz(gm2 / Cc), _mhz(p2_miller)))
    elif comp == 1:
        zdesc = ("follower gm_f = %s uS at %s uA: left-half-plane zero at "
                 "gm_f/(2 pi Cc) = %s MHz; non-dominant pole estimate %s MHz, "
                 "third pole near gm_f/(2 pi C1) = %s MHz"
                 % (n(gmb), n(float(d["I_buf"])), _mhz(gmb / Cc),
                    _mhz(gm2 * Cc / ((C1 + Cc) * CLe)), _mhz(gmb / C1)))
    else:
        w0 = math.sqrt(gm2 * gmc / (C1 * CLe))
        q_est = Cc / (CLe + Cc) * math.sqrt(gm2 * CLe / (gmc * C1))
        zdesc = ("common-gate gm_c = %s uS at %s uA (gm1/gm_c = %s): "
                 "left-half-plane zero at gm_c/(2 pi Cc) = %s MHz; complex-pair "
                 "estimate: natural frequency %s MHz, Q %s"
                 % (n(gmc), n(float(d["I_cg"])), n(gm1 / gmc, 3),
                    _mhz(gmc / Cc), _mhz(w0), n(q_est, 3)))

    vov1, vov2, vovb = float(d["vov1"]), float(d["vov2"]), float(d["vovb"])
    icmr_parts = ("Vdd %s - %s - Vth %s - Vov(M1) %s - Vov(MB) %s%s"
                  % (n(vdd), n(_VAUX), n(vth), n(vov1), n(vovb),
                     (" - cascode %s" % n(_CAS_HEADROOM_V)) if cas else ""))
    if comp == 1:
        sw_parts = ("Vdd %s - Vov(M2) %s - max(Vov(MB) %s, Vth + %s = %s)"
                    % (n(vdd), n(vov2), n(vovb), n(2 * _VAUX), n(vth + 2 * _VAUX)))
    else:
        sw_parts = "Vdd %s - Vov(M2) %s - Vov(MB) %s" % (n(vdd), n(vov2), n(vovb))

    # node capacitances
    cj_n1 = _CJ_PF_PER_UM * ((float(d["Wcn"]) + float(d["Wcp"])) if cas
                             else (W1 + float(d["W3"])))
    if cas:
        cj_txt = "the two cascodes, width ~ Ib*L1"
    else:
        cj_txt = "M1 %s, mirror load %s with width ~ Ib*L1" % (
            n(_CJ_PF_PER_UM * W1), n(_CJ_PF_PER_UM * float(d["W3"])))
    cj_m2 = _CJ_PF_PER_UM * W2
    cj_mb = _CJ_PF_PER_UM * B * w_mb

    # Ib windows (Vov of MB, and of M1, M2, MB together)
    v2lo, v2hi = _VOV_MIN ** 2, vov_hi ** 2
    kb_wl = kn * (w_mb / l_mb)
    ib_mb = (0.5 * kb_wl * v2lo, 0.5 * kb_wl * v2hi)
    ib_m1 = (kn * (W1 / L1) * v2lo, kn * (W1 / L1) * v2hi)
    ib_m2 = (kn * (W2 / L2) * v2lo / (2.0 * B), kn * (W2 / L2) * v2hi / (2.0 * B))
    ib_all = (max(ib_mb[0], ib_m1[0], ib_m2[0]), min(ib_mb[1], ib_m1[1], ib_m2[1]))
    win_all = ("%s <= Ib <= %s uA" % (n(ib_all[0]), n(ib_all[1]))
               if ib_all[0] <= ib_all[1] else "no Ib")
    a_helper_ib = (float(d["a_load"])
                   + (float(d["a_cas"]) - _CAS_OVH_UM2 if cas else 0.0)
                   + (float(d["a_buf"]) - _BUF_OVH_UM2 if comp == 1 else 0.0)
                   + (float(d["a_cg"]) - _CG_OVH_UM2 if comp == 2 else 0.0))
    pfac = 2.0 + B + (0.5 if comp == 1 else (1.0 if comp == 2 else 0.0))

    # RZ line
    rz_area = _RES_UM2_PER_KOHM * Rz + _RES_OVH_UM2
    tgt = inv_gm2_k * (1.0 + CLe / Cc)
    if comp != 0:
        rz_line = ("rz_on=%d; Rz acts only with comp = 0; while comp = %d, "
                   "rz_on = 1 only costs %s um^2%s"
                   % (int(rzon), comp, n(rz_area),
                      (" (now %s%% of area)" % pct(d["a_rz"])) if rzon else ""))
    elif not rzon:
        r = wi.get("rz", [(None, None)])[0][1]
        if r is not None:
            z_new = 1.0 / float(np.asarray(r["za"]).reshape(-1)[0]) if abs(
                float(np.asarray(r["za"]).reshape(-1)[0])) > 1e-30 else float("inf")
            rz_line = ("off (Rz = %s kOhm stored); turning it on at this Rz "
                       "would move the compensation zero from %s MHz "
                       "right-half-plane to %s MHz %s-half-plane, add a pole "
                       "near 1/(2 pi Rz C1) = %s MHz, give %s, and cost %s um^2 "
                       "(%s%% of area); 1/gm2 = %s kOhm, (1/gm2)(1 + CL_out/Cc) "
                       "= %s kOhm"
                       % (n(Rz), _mhz(gm2 / Cc), _mhz(z_new),
                          "right" if z_new > 0 else "left",
                          _mhz(1.0 / max(Rz * 1e-3 * C1, 1e-30)), fr_text(r),
                          n(rz_area), n(100.0 * rz_area / area_tot, 3),
                          n(inv_gm2_k), n(tgt)))
        else:
            rz_line = ("off (Rz = %s kOhm stored); 1/gm2 = %s kOhm, (1/gm2)(1 + "
                       "CL_out/Cc) = %s kOhm" % (n(Rz), n(inv_gm2_k), n(tgt)))
    else:
        r = wi.get("rz", [(None, None)])[0][1]
        rz_line = ("on, Rz = %s kOhm, area %s um^2 (%s%%); 1/gm2 = %s kOhm, "
                   "(1/gm2)(1 + CL_out/Cc) = %s kOhm%s"
                   % (n(Rz), n(float(d["a_rz"])), pct(d["a_rz"]), n(inv_gm2_k),
                      n(tgt), ("; turning it off would give %s" % fr_text(r))
                      if r is not None else ""))

    # CAS line
    r = wi.get("cas", [(None, None)])[0][1]
    a_cas_gain = float(d["a_cas_gain"])
    if cas:
        cas_line = ("on: gm_cas*ro_cas = %s (+%s dB of DC gain), cascode-node "
                    "pole %s MHz, costs %s V of input range, area %s um^2 (%s%%)"
                    % (n(a_cas_gain), _db(a_cas_gain), _mhz(float(d["wx"])),
                       n(_CAS_HEADROOM_V), n(float(d["a_cas"])), pct(d["a_cas"])))
        if r is not None:
            cas_line += ("; turning it off: DC gain %s dB, input range %s V, C1 "
                         "%s pF, %s"
                         % (n(float(r["gain_db"])), n(float(r["icmr_v"])),
                            n(float(r["C1"])), fr_text(r)))
    else:
        cas_line = ("off; turning it on: R1 x gm_cas*ro_cas = %s (+%s dB DC "
                    "gain), cascode-node pole %s MHz, input range -%s V (to %s "
                    "V), area +%s um^2"
                    % (n(a_cas_gain), _db(a_cas_gain), _mhz(float(d["wx"])),
                       n(_CAS_HEADROOM_V), n(float(d["icmr_v"]) - _CAS_HEADROOM_V),
                       n(c_cas * Ib * L1 * L1 + _CAS_OVH_UM2)))
        if r is not None:
            cas_line += (", C1 %s -> %s pF, %s"
                         % (n(C1), n(float(r["C1"])), fr_text(r)))
            pr = _obs_pair(r) if comp == 2 else None
            if pr is not None:
                cas_line += ", pair Q %s" % n(pr[1], 3)

    # scheme alternatives
    alt = []
    for c, r in wi.get("comp", []):
        # verb surface: the code and the identifier SET_COMPENSATION takes
        s = "comp %d%s -> %s" % (c, (" " + COMP_NAMES[c]) if verbs else "", fr_text(r))
        if c == 2:
            pr = _obs_pair(r)
            if pr is not None:
                s += ", pair %s MHz Q %s" % (_mhz(pr[0]), n(pr[1], 3))
                try:
                    pk = _obs_peak(r, float(r["wu"]))
                except Exception:
                    pk = None
                s += (", resonant peak %s dB" % n(pk[1], 3)) if pk else ", no resonant peak"
        s += ", power %s mW, area %s um^2, output swing %s V" % (
            n(float(r["power_mw"])), n(float(r["area"])), n(float(r["out_swing_v"])))
        alt.append(s)

    # sensitivities
    sens = []
    g0, f0, pm0 = float(d["gain_db"]), float(d["gbw_mhz"]), float(d["phase_margin_deg"])
    for (s, lab), r in wi.get("sens", []):
        f1 = float(r["gbw_mhz"])
        txt = "%s %s/%s/%s" % (lab, _sg(float(r["gain_db"]) - g0),
                               _sg(100.0 * (f1 / f0 - 1.0), 2, 1) if f0 > 0 else "n/a",
                               _sg(float(r["phase_margin_deg"]) - pm0))
        if not _obs_valid(r):
            txt += " (invalid)"
        sens.append(txt)

    # frequency response: resonance and the PM-setting crossing
    fr_extra = ""
    try:
        crossings = _obs_crossings(d)
        pair = _obs_pair(d)
        peak = _obs_peak(d, wu)
        if pair is not None:
            fr_extra += (" Complex pair: open-loop gain at its natural frequency "
                         "%s MHz is %s dB" % (_mhz(pair[0]),
                                              n(float(_obs_lm(d, [pair[0]])[0]) * _DB_PER_NEPER, 3)))
            if peak is not None:
                fr_extra += ("; resonant peak above the crossover %s dB near %s "
                             "MHz (a peak above 0 dB re-crosses unity)."
                             % (n(peak[1], 3), _mhz(peak[0])))
            else:
                fr_extra += ("; no resonant peak (the gain falls monotonically "
                             "above the crossover).")
        elif peak is not None:
            fr_extra += (" Above the crossover the gain peaks at %s dB near %s MHz."
                         % (n(peak[1], 3), _mhz(peak[0])))
        if len(crossings) > 1:
            w_pm = min(crossings, key=lambda t: t[1])[0]
            if abs(w_pm - wu) > 1e-6 * wu:
                fr_extra += (" Phase margin is set at a LATER unity crossing, %s "
                             "MHz (resonant re-crossing; unity crossings at %s MHz), "
                             "not at the first crossover %s MHz; phase there: %s."
                             % (_mhz(w_pm), ", ".join(_mhz(w) for w, _ in crossings),
                                _mhz(wu), _phase_short(d, w_pm)))
    except Exception:
        pass

    lines += [
        "",
        "RESULTS (nominal hand analysis)",
        "  0 M1: Id1=%s uA per device, gm1=%s uS, Vov=%s V (%s), lambda(L1)=%s "
        "1/V, R1=%s MOhm%s, stage-1 gain gm1*R1=%s (%s dB). "
        "Current-mirror pole %s MHz. Area: pair %s um^2 (%s%%), mirror loads %s "
        "um^2 (%s%%)."
        % (n(float(d["Id1"])), n(gm1), n(vov1), _vov_note(vov1, _VOV_MIN, vov_hi),
           n(float(d["lam1"])), n(float(d["R1"])),
           " (cascoded)" if cas else "", n(a1), _db(a1), _mhz(float(d["wm"])),
           n(float(d["a_m1"])), pct(d["a_m1"]), n(float(d["a_load"])),
           pct(d["a_load"])),
        "  1 M2: I2=%s uA, gm2=%s uS, Vov=%s V (%s), "
        "lambda(L2)=%s 1/V, R2=%s MOhm, stage-2 gain gm2*R2=%s (%s dB). Node-1 "
        "capacitance C1 = %s pF = M2 gate %s + drain junctions %s (%s) + wiring "
        "%s; output capacitance CL_out = CL %s + junctions %s (M2 %s, MB sink "
        "%s) = %s pF. Area %s um^2 (%s%%)."
        % (n(float(d["I2"])), n(gm2), n(vov2),
           _vov_note(vov2, _VOV_MIN, vov_hi), n(float(d["lam2"])),
           n(float(d["R2"])), n(a2), _db(a2), n(C1), n(float(d["Cgs2"])),
           n(cj_n1), cj_txt,
           n(_C_WIRE_PF), n(cl), n(cj_m2 + cj_mb), n(cj_m2), n(cj_mb), n(CLe),
           n(float(d["a_m2"])), pct(d["a_m2"])),
        "  2 MB: I2=B*Ib=%s uA, MB Vov=%s V (%s), output slew limb I2/CL=%s "
        "V/us, stage-2 power %s mW (%s%% of power), area %s um^2 (%s%%)."
        % (n(float(d["I2"])), n(vovb), _vov_note(vovb, _VOV_MIN, vov_hi),
           n(float(d["I2"]) / cl), n(p_s2), ppct(p_s2), n(float(d["a_mb"])),
           pct(d["a_mb"])),
        "  3 CC: Cc=%s pF, %s; dominant-pole GBW estimate gm1/(2 pi Cc)=%s MHz; "
        "internal slew limb Ib/Cc=%s V/us; %s. Area: capacitor %s um^2 (%s%%)%s."
        % (n(Cc), _COMP_NAMES[comp], n(gbw_est), n(Ib / Cc), zdesc,
           n(float(d["a_cc"])), pct(d["a_cc"]),
           (", follower %s um^2 (%s%%)" % (n(float(d["a_buf"])), pct(d["a_buf"])))
           if comp == 1 else
           ((", common-gate device %s um^2 (%s%%)" % (n(float(d["a_cg"])), pct(d["a_cg"])))
            if comp == 2 else "")),
        "  4 IB: Ib=%s uA, bias+tail power 2*Ib*Vdd=%s mW (%s%%)%s; total power "
        "%s*Ib*Vdd = %s mW. Helper area proportional to Ib %s um^2 (%s%%). Vov "
        "window: MB valid for %s <= Ib <= %s uA; M1, M2 and MB together (at the "
        "current W, L, B) for %s."
        % (n(Ib), n(p_tail), ppct(p_tail),
           (", buffer bias %s mW (%s%%)" % (n(p_aux), ppct(p_aux))) if p_aux > 0 else "",
           n(pfac), n(p_tot), n(a_helper_ib), pct(a_helper_ib),
           n(ib_mb[0]), n(ib_mb[1]), win_all),
        "  5 RZ: %s." % rz_line,
        "  6 CAS: %s." % cas_line,
    ]
    if alt:
        lines.append("  Other schemes at these sizes: %s." % "; ".join(alt))
    lines.append(
        "  Frequency response: DC gain %s dB, crossover %s MHz, phase margin %s "
        "deg. Poles (phase at the crossover): %s. Zeros: %s.%s"
        % (n(float(d["gain_db"])), _mhz(wu), n(pm0), poles_t, zeros_t, fr_extra))
    if sens:
        lines.append("  Sensitivity, one tunable x1.1 (DC gain dB / crossover "
                     "%% / phase margin deg): %s."
                     % "; ".join(sens))
    lines.append(
        "  Headroom: input common-mode range = %s = %s V; output swing = %s = "
        "%s V. Slew %s V/us (%s limb). Total area %s um^2."
        % (icmr_parts, n(float(d["icmr_v"])), sw_parts, n(float(d["out_swing_v"])),
           n(float(d["slew_v_per_us"])),
           "internal Ib/Cc" if Ib / Cc <= float(d["I2"]) / cl else "output I2/CL",
           n(area_tot)))
    if stochastic:
        try:
            sp = _obs_spread(spec, x)

            def rel(k):
                v = abs(float(d[k]))
                return n(100.0 * sp[k] / v, 2) if v > 0 else "n/a"

            lines.append(
                "  Mismatch spread at this design (1-sigma over %d fixed draws): "
                "gain %s dB, GBW %s%%, phase margin %s deg, slew %s%%, power %s%%, "
                "input range %s mV, output swing %s mV."
                % (_OBS_SPREAD_DRAWS, n(sp["gain_db"], 2), rel("gbw_mhz"),
                   n(sp["phase_margin_deg"], 2), rel("slew_v_per_us"),
                   rel("power_mw"), n(1000.0 * sp["icmr_v"], 2),
                   n(1000.0 * sp["out_swing_v"], 2)))
        except Exception:
            pass
    return "\n".join(lines)
