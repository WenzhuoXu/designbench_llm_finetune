"""Unit commitment + DC power-flow dispatch with line switching and
phase-shifting transformers on MESHED transmission grids (dataxp v2 domain
"grid2"; the v1 module dataxp/domains/grid.py is untouched).

Physics
-------
Lossless DC power flow on a connected synthetic grid of 30-120 buses. Each
line l (from-bus i, to-bus j) has a susceptance b_l (MW/rad), a thermal limit
and a flow f_l = b_l * (theta_i - theta_j + psi_l), where psi_l is the phase
shift of a phase-shifting transformer (PST) on that line (0 on ordinary
lines). Bus angles solve B * theta = P - A^T (b * psi) on the energised buses
with theta_0 = 0; P is the net injection (generation + renewables - load).
Bus 0 is the slack bus: a tie to a neighbouring system that imports any
shortfall and exports any surplus. Islanding: a bus with no path of closed
lines to bus 0 is DE-ENERGISED - its load is shed (counted in load_shed_mw
and billed at VOLL), renewables there are lost, and generators there are
tripped (zero output, no reserve; a committed unit still pays no-load cost).
Islanding is therefore PENALISED LOAD SHED (a constraint with limit 0 MW plus
a VOLL cost), not an invalid state; it never makes the solve singular, since
only the energised component around bus 0 is solved.

Metrics: total_cost ($/h, objective, min) = energy (sum c_g p_g over live
units) + commitment (no-load of every committed unit + start-up of every
committed unit that was offline in the initial design) + PST operation
(pst_cost_per_deg * sum |shift|) + switching (switch_cost per switchable line
whose status differs from the initial design) + import_price * import +
export_price * export + VOLL * load shed. Constraints (goals): line_loading =
max over in-service lines of |f|/limit <= 1.0; overload_mw = sum over in-service lines
of max(0, |f| - limit) <= 0 (the same feasible set as line_loading, but it
credits relieving ANY overloaded line, not only the worst one - without it a
depth-1 policy stalls when two lines are overloaded by similar amounts);
slack_import_mw / slack_export_mw <= per-problem limits; spinning_reserve_mw =
sum over committed, energised units of (p_max - p_set) >= per-problem
requirement; load_shed_mw <= 0. Reported only: energy_cost, commitment_cost,
n_overloaded, n_committed. Constraint scales (program_from_goals scales=):
line_loading 0.1, overload_mw 0.1 x median line limit, slack 0.5% of net
demand, reserve 10% of the requirement, load shed 0.1% of net demand - tight,
so the smooth hinge does not pay for margin on satisfied constraints.
Validity envelope: line_loading <= 10, slack import/export <= total load,
load shed <= 50% of total load. Any numerical failure returns FINITE metrics
outside that envelope (line_loading 1e3, ...): never NaN/inf.

Element types and parameters (indices fixed for the whole trajectory)
---------------------------------------------------------------------
elements 0..G-1          generators (6-24)
    p_set   continuous (scaled moves), MW, p_min <= p_set <= p_max while
            committed; clamped. get() returns None while the unit is off and
            set() is then a no-op (an offline unit has no setpoint).
    on      toggle 0/1 (unit commitment). 0 -> 1 starts the unit at exactly
            p_min (and charges its start-up cost if it was off initially);
            1 -> 0 sets its output to 0. Values >= 0.5 mean 1.
elements G..G+S-1        switchable lines (fixed pool of 4-13: normally-closed
                         lines with breakers, sometimes the single feeder of a
                         radial spur bus, + 2-6 normally-open tie lines)
    closed  toggle 0/1 (1 = in service). Switching cost while the status
            differs from the initial design (a state function).
elements G+S..G+S+P-1    phase-shifting transformers (2-4 per problem)
    shift   additive, degrees, step 5, bounds [-30, +30]; set() rounds to the
            nearest multiple of 5 and clamps. Operating cost per degree.
get(st, i, p) returns None when p does not apply to element i; set() of an
inapplicable (known) parameter is a no-op; an unknown parameter name or an
out-of-range element index raises ValueError. There are no categorical
parameters in this domain.

Move set (candidate_moves, every turn): every commitment toggle, every switch
toggle and every PST shift (+-5, +-10, +-15 deg) is always offered; p_set values
(x0.6/0.8/0.9/0.97/1.03/1.1/1.25/1.5, p_min, p_max, "take up the current slack
mismatch", "bring the worst overloaded line to 98% of its limit via this
unit's shift factor") fill the rest of a 150-move budget round-robin over the
running units. 40-150 moves per turn.

Problems
--------
Four families x ten sizes (30..120 buses): "meshed" (one ring-plus-chords
area), "two_area" (exporting area with cheap units and renewables, importing
area with load and peakers, 2-3 tie lines), "load_pocket" (a main area and a
load pocket fed by 2-3 lines with expensive local units), "three_area" (three
areas in a ring). Areas of >= 8 buses get a radial spur bus with probability
0.6 (its feeder is switchable w.p. 0.7: opening it sheds that bus's load).
Susceptances are 4 x the per-bus load size x U(1.5, 6) MW/rad (ties
x U(0.6, 2.8) instead), so a 5 deg PST step moves a median 6% (p10 2%, p90
28%) of the PST line's own limit; the best single PST move cuts the worst
line's loading by a median 0.18 at the initial state (dispatch / commitment /
switching moves 0.5-0.6). At 1 x the per-bus load size PSTs moved ~2% per
5 deg and were decorative. Generators come in three classes (base / mid /
peaker) with different energy cost, p_min fraction, no-load and start-up
cost. A hidden FEASIBLE design (topology, PST shifts, commitment, dispatch)
is drawn first; line limits are calibrated to its flows with headroom
1.06-1.5 (x a per-line 1.0-1.25). The initial design then carries defects
relative to it - normally-open ties that should be closed, weak lines that
should be open, PSTs reset, a needed unit decommitted, an unneeded unit
committed, too few units for the reserve, dispatch scheduled for a 4-14%
lower demand, merit-order dispatch - blended so the worst line starts near
1.1-2.5x its limit (<= 2.6x, <= max(10, L/12) lines over). A per-problem
difficulty d ~ U(0,1) (stored as spec["difficulty"], never shown to the
agent) sets that target and scales every defect probability; size and d are
drawn once per problem (a draw rejected 60 times in a row - a few problems
per thousand - is retried at d/2, then at a new size, so one unlucky draw
does not kill a generation job). Usually infeasible, and solvable: the hidden
design is not stored in the spec, but its topology, PST shifts and
commitment are reachable exactly with the move set and its dispatch approximately.

reference(spec): heuristic best of a deterministic coarse configuration
search: DC-OPF LP (scipy linprog/HiGHS) economic dispatch under the hidden
design's configuration, the initial configuration and their crosses, then 3
rounds of best-improvement local moves (flip one switch, PST +-5 deg, toggle
one unit) each re-solved by LP. reference_class = "heuristic" (an upper
bound on the true mixed-integer optimum; a policy can beat it slightly).

Stochastic variant (grid2_stoch): every bus load is scaled by an independent
lognormal factor (mean 1, sigma 0.03-0.08 by family) and every renewable
injection by one with sigma 0.08-0.18, drawn from default_rng(seed) only; the
slack bus absorbs the net error. Both slack limits include 1.5 sigma of that
net-demand error (the same spec is used by both variants), so the 90% chance
constraints on import and export can hold together. The reference is the
nominal LP (heuristic).

Simplifications (stated explicitly)
-----------------------------------
* DC power flow: no losses, no reactive power, no voltage limits, flat
  voltage magnitudes; one snapshot (no ramping, minimum up/down times).
* Start-up cost is a state function: charged while a unit that was offline in
  the initial design is committed (shutting it down again refunds it).
* No N-1 security, no protection or stability limits on switching. PST
  operation costs a small linear amount per degree (~0.01% of system cost per
  degree; stands in for extra losses and tap wear) and every switchable line
  whose status differs from the initial design costs 2.5 degrees' worth of
  PST operation (~0.03% of system cost; switching operations, breaker wear).
  Both are tiny next to dispatch costs; they exist because free moves let
  the gold policy keep nudging PSTs / toggling lines after reaching
  feasibility for ~1e-5 of smooth-hinge margin (verified: with free switching
  9% of gold moves were zero-cost margin-only toggles).
* Islanded buses are simply de-energised (no island self-balancing).
* Thermal limits are synthetic (calibrated to the hidden design), not ratings.
* No categorical parameters (none is physically natural here); generator
  classes are fixed data, not tunables.
* Renewables are fixed, uncurtailable injections.

numpy + scipy + stdlib only; no I/O, no prints, no global randomness.

Verb surface (docs/dataxp_verb_surface.md; owner's decision of 2026-10-23)
------------
The move set (candidate_moves), set(), Phi and the stored {i, p, v, v_old}
dicts are the v2 ones above and are untouched; only the spelling in prompts,
action lines and reply parsing changes (class attributes verb_surface and
verb_table; format_move / parse_move / grammar come from the table in
dataxp.domain). The bijection, per parameter (# is the SET_PARAM element
number, the number in brackets in the observation):
  p_set v   on a generator      <-> DISPATCH(#, v)                      [MW]
  on 1 / 0  on a generator      <-> COMMIT_UNIT(#) / DECOMMIT_UNIT(#)
  closed 1 / 0 on a switch line <-> CLOSE_LINE(#) / OPEN_LINE(#)
  shift v   on a PST            <-> SET_PHASE_SHIFT(#, v)               [deg]
Judgment calls, recorded here as the surface rules ask:
  * The element argument is the element number SET_PARAM used (generators
    0..G-1, switchable lines G..G+S-1, phase shifters G+S..G+S+P-1: the
    number in brackets before G# / SW# / PST#), never the G, SW, L or PST
    number; the grammar says so for each verb.
  * DISPATCH's value is the new absolute setpoint in MW, not a change; it is
    printed to 6 significant digits (fmt_arg, as SET_PARAM printed it) while
    the stored dict keeps the full value, so the text replays to within 1e-6
    relative, exactly as before (set() clamps to [p_min, p_max]).
  * SET_PHASE_SHIFT's value is the new TOTAL shift in degrees, not a step;
    set() still rounds to the nearest multiple of 5 and clamps to +-30, so an
    off-grid number parses and lands on the grid, as under SET_PARAM.
  * SET_PARAM(...) text is no longer parsed on this domain: the grammar the
    policy is served lists the verbs only.
  * A verb the state does not admit parses and apply() behaves as set() did:
    COMMIT_UNIT on a running unit, DECOMMIT_UNIT on an offline one, CLOSE_LINE
    on a closed line and OPEN_LINE on an open one change nothing; DISPATCH on
    an offline unit is a no-op (an offline unit has no setpoint); an element
    number outside the table raises ValueError from set(), as before.
    legal_moves never offers any of these. Nothing was added to or removed
    from the move set.
  * describe()'s change paragraph (the SET_PARAM line and the three element
    bullets) and the slack-bus sentence that names the actions ("a p_set
    change; on=1 adds p_min; on=0 removes p_set") are written in the verbs
    (observation text only; no generated file contains it). The table headers
    and rows keep "p_set", "ON / OFF", "CLOSED / OPEN" and "shift" as the
    names of the quantities they print; the bullets tie each verb to them.
  * History and "change you are about to make" lines print the replaced value
    ("was 123.457", "degrees 5 -> 10") for DISPATCH and SET_PHASE_SHIFT; the
    commit / decommit and close / open pairs need none.

Set-parameter spelling (docs/dataxp_spellings.md; DATAXP_SPELLING=set_parameter)
-----------------------
The same moves through the one wrapper tool, generated from the verb table
by dataxp.domain (no override here): DISPATCH(2, 123.457) <-> SET_PARAM(2,
p_set, 123.457), COMMIT_UNIT(2) / DECOMMIT_UNIT(2) <-> SET_PARAM(2, on, 1) /
SET_PARAM(2, on, 0), CLOSE_LINE(7) / OPEN_LINE(7) <-> SET_PARAM(7, closed,
1 / 0), SET_PHASE_SHIFT(9, 10) <-> SET_PARAM(9, shift, 10). Judgment calls:
the phrases are "Dispatch a unit", "Commit a unit" / "Decommit a unit",
"Close a line" / "Open a line", "Set a phase shifter's angle"; the CHANGES
paragraph and the slack-bus sentence revert to their SET_PARAM wording (the
run2 text) under this spelling (spelled_in_verbs), so the two prompts differ
in action syntax only.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..domain import ArgSpec, Candidate, Continuous, Domain, ParamVerb, Toggle, spelled_in_verbs
from ..potential import DesignProgram, program_from_goals

_FAMILIES = ("meshed", "two_area", "load_pocket", "three_area")
_SIZE_LADDER = (30, 36, 44, 52, 60, 70, 80, 92, 106, 120)
_PARAMS = ("p_set", "on", "closed", "shift")
_SHIFT_STEP = 5.0
_SHIFT_MAX = 30.0
_LOADING_CAP = 10.0
_MAKE_ATTEMPTS = 60
# Line susceptance scale (b = _B_SCALE x per-bus load size x U(...), MW/rad).
# DC flows driven by injections do not depend on a uniform b scale; the PST
# effect does (b * shift). At 1.0 a 5 deg step moved a median 2% of the PST
# line's own limit and the best PST move cut the worst line's loading by a
# median 0.05 (vs 0.6-0.7 for a dispatch or switching move): the PSTs were
# decorative. At 4.0 a 5 deg step moves a median ~7% of the line limit.
_B_SCALE = 4.0
# switching cost per changed line, in degrees-of-PST-operation equivalents
_SWITCH_COST_DEG = 2.5
_METRIC_KEYS = ("total_cost", "energy_cost", "commitment_cost", "line_loading",
                "overload_mw", "n_overloaded", "slack_import_mw", "slack_export_mw",
                "spinning_reserve_mw", "load_shed_mw", "n_committed")
_DEG = math.pi / 180.0

# verb-surface argument specs (docstring "Verb surface"): the element number
# is the number in brackets, never the G / SW / L / PST number
_GEN_ARG = ArgSpec("generator", "int", "the generator's element number 0..G-1 (the number in "
                                       "brackets before G#)")
_LINE_ARG = ArgSpec("line", "int", "the switchable line's element number G..G+S-1 (the number "
                                   "in brackets before SW#, not its L number)")
_PST_ARG = ArgSpec("pst", "int", "the phase shifter's element number G+S..G+S+P-1 (the number "
                                 "in brackets before PST#)")

_LOAD_W = {"export": 0.7, "mixed": 1.0, "main": 1.0, "import": 1.6, "pocket": 1.9}
_GEN_W = {
    "base": {"export": 3.0, "main": 2.0, "mixed": 1.5, "import": 0.4, "pocket": 0.2},
    "mid": {"export": 1.0, "main": 1.0, "mixed": 1.0, "import": 1.0, "pocket": 0.6},
    "peak": {"export": 0.5, "main": 0.8, "mixed": 1.2, "import": 3.0, "pocket": 4.0},
}
_AREA_NAMES = ("A", "B", "C")


class _Reject(Exception):
    pass


# ── network numerics ────────────────────────────────────────────────────────

def _energized(n: int, fr: np.ndarray, to: np.ndarray) -> np.ndarray:
    """Buses connected to bus 0 through the given (closed) lines."""
    if fr.size == 0:
        e = np.zeros(n, dtype=bool)
        e[0] = True
        return e
    A = csr_matrix((np.ones(fr.size), (fr, to)), shape=(n, n))
    ncomp, labels = connected_components(A, directed=False)
    if ncomp == 1:
        return np.ones(n, dtype=bool)
    return labels == labels[0]


def _bmat(n: int, fa: np.ndarray, ta: np.ndarray, ba: np.ndarray) -> np.ndarray:
    """Susceptance Laplacian (no parallel duplicates by construction)."""
    B = np.zeros((n, n))
    deg = np.bincount(fa, ba, n) + np.bincount(ta, ba, n)
    B[np.arange(n), np.arange(n)] = deg
    B[fa, ta] = -ba
    B[ta, fa] = -ba
    return B


def _solve_flows(n, fr, to, b, act, energized, inj, psi):
    """DC flows on active lines; inj is the net injection per bus (dead buses
    must be 0); psi the PST angle per line (rad). Returns (flow, theta)."""
    fa, ta, ba = fr[act], to[act], b[act]
    pa = psi[act]
    rhs = np.array(inj, dtype=float)
    bp = ba * pa
    if np.any(bp != 0.0):
        rhs = rhs - np.bincount(fa, bp, n) + np.bincount(ta, bp, n)
    theta = np.zeros(n)
    idx = np.flatnonzero(energized)
    idx = idx[idx != 0]
    if idx.size:
        B = _bmat(n, fa, ta, ba)
        theta[idx] = np.linalg.solve(B[np.ix_(idx, idx)], rhs[idx])
    flow = np.zeros(fr.shape[0])
    flow[act] = ba * (theta[fa] - theta[ta] + pa)
    return flow, theta


class Grid2Domain(Domain):
    """Unit commitment + dispatch + line switching + PST shifts on a meshed
    DC grid: clear overloads, keep the slack tie, reserve and load-shed
    within limits, at minimum generation + commitment cost."""
    # stochastic variants: phi_select compares candidates with a fresh
    # incumbent estimate (dataxp/ladder.py); no effect when deterministic
    rescore_incumbent = True

    name = "grid2"
    params = _PARAMS
    factors = (0.8, 0.9, 0.97, 1.03, 1.1, 1.25)
    reference_class = "heuristic"
    max_elements = 64
    max_candidates = 160
    _cand_budget = 150

    # ── verb surface (docstring "Verb surface"; docs/dataxp_verb_surface.md) ──
    # Spelling only: every SET_PARAM move candidate_moves enumerates has one
    # verb text and parses back to the same move. format_move / parse_move /
    # grammar are the table-driven defaults of dataxp.domain.
    verb_surface = True
    verb_table = (
        ParamVerb("p_set", "DISPATCH", _GEN_ARG,
                  Continuous("mw", "MW", "the unit's new output setpoint, within its "
                                         "p_min..p_max (clamped); only a running unit takes "
                                         "it"),
                  doc="set a running unit's real-power output; the MW added or removed is "
                      "balanced at the slack bus 0 (import / export), so moving output "
                      "between two units takes two changes.",
                  phrase="Dispatch a unit"),
        ParamVerb("on", "COMMIT_UNIT", _GEN_ARG,
                  Toggle("DECOMMIT_UNIT",
                         off_doc="shut a running unit down: output 0, no no-load cost; a "
                                 "start-up cost charged for it is refunded.",
                         off_phrase="Decommit a unit"),
                  doc="start an offline unit at exactly its p_min; it then pays its no-load "
                      "$/h and, if it was off in the initial design, its start-up cost.",
                  phrase="Commit a unit"),
        ParamVerb("closed", "CLOSE_LINE", _LINE_ARG,
                  Toggle("OPEN_LINE",
                         off_doc="take a switchable line out of service; a bus left without "
                                 "a closed path to bus 0 is de-energised (its load shed at "
                                 "VOLL, its units tripped).",
                         off_phrase="Open a line"),
                  doc="put a switchable line in service; every switchable line whose status "
                      "differs from the initial design costs the switching charge.",
                  phrase="Close a line"),
        ParamVerb("shift", "SET_PHASE_SHIFT", _PST_ARG,
                  Continuous("degrees", "deg", "the new TOTAL shift, a multiple of 5 in "
                                               "-30..30 (rounded to the nearest 5 and "
                                               "clamped), not a step"),
                  doc="set a phase-shifting transformer's angle: + pushes more power "
                      "from->to through its own line and less through the parallel paths; "
                      "costs the per-degree charge on |shift|.",
                  phrase="Set a phase shifter's angle"),
    )

    def __init__(self, stochastic: bool = False):
        super().__init__(stochastic)

    # ══ problems ═════════════════════════════════════════════════════════════
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[index % len(_FAMILIES)]
        allow_easy = bool(rng.random() < 0.05)
        # the size is drawn ONCE per problem: redrawing it on every rejected
        # attempt skewed the accepted set toward small grids (large ones are
        # rejected more often)
        base = int(_SIZE_LADDER[int(rng.integers(len(_SIZE_LADDER)))])
        n_bus = int(np.clip(base + int(rng.integers(-2, 3)), 30, 120))
        # difficulty in [0, 1], also drawn once: sets the target initial
        # overload and scales every defect probability (G1's graded axis)
        diff = float(rng.random())
        fallback = None
        for attempt in range(3 * _MAKE_ATTEMPTS):
            # A rare (size, difficulty) draw is rejected on every attempt
            # (a few problems per thousand); relax deterministically
            # instead of failing the whole generation job: halve the
            # difficulty after _MAKE_ATTEMPTS tries, also redraw the size after
            # 2 * _MAKE_ATTEMPTS.
            if attempt == _MAKE_ATTEMPTS:
                diff = 0.5 * diff
            elif attempt == 2 * _MAKE_ATTEMPTS:
                base = int(_SIZE_LADDER[int(rng.integers(len(_SIZE_LADDER)))])
                n_bus = int(np.clip(base + int(rng.integers(-2, 3)), 30, 120))
            try:
                spec = self._draw_spec(index, family, rng, n_bus, diff)
            except Exception:          # _Reject or any numerical failure: redraw
                continue
            if spec["initially_infeasible_nominal"] or allow_easy:
                return spec
            if fallback is None:
                fallback = spec
        if fallback is not None:
            return fallback
        raise RuntimeError("grid2.make_problem: no valid problem in %d attempts "
                           "(index=%d)" % (3 * _MAKE_ATTEMPTS, index))

    def _draw_spec(self, index: int, family: str, rng: np.random.Generator,
                   n: int, diff: float = 0.5) -> dict:
        U = lambda a, c: float(rng.uniform(a, c))  # noqa: E731

        # ── areas ──
        if family == "meshed":
            sizes, roles = [n], ["mixed"]
            pairs = []
        elif family == "two_area":
            nA = int(round(n * U(0.42, 0.58)))
            sizes, roles = [nA, n - nA], ["export", "import"]
            pairs = [(0, 1)]
        elif family == "load_pocket":
            nP = int(round(n * U(0.15, 0.25)))
            sizes, roles = [n - nP, nP], ["main", "pocket"]
            pairs = [(0, 1)]
        else:
            a1 = int(round(n * U(0.28, 0.38)))
            a2 = int(round(n * U(0.28, 0.38)))
            sizes, roles = [a1, a2, n - a1 - a2], ["export", "mixed", "import"]
            pairs = [(0, 1), (1, 2), (2, 0)]
        starts = np.cumsum([0] + sizes[:-1]).tolist()
        area = np.repeat(np.arange(len(sizes)), sizes)
        bus_role = [roles[int(area[k])] for k in range(n)]

        # Susceptances scale with the bus load size so a 5 deg PST step moves
        # roughly 5-15% of a main-line flow (DC flows from injections do not
        # depend on a uniform b scale; only the PST effect does).
        load_scale = U(15.0, 40.0)
        bs = _B_SCALE * load_scale

        # ── base lines ──
        key_set: Dict[Tuple[int, int], int] = {}
        fr: List[int] = []
        to: List[int] = []
        bb: List[float] = []
        tie: List[bool] = []

        def add(a: int, c: int, bval: float, is_tie: bool) -> bool:
            if a == c:
                return False
            k = (min(a, c), max(a, c))
            if k in key_set:
                return False
            key_set[k] = len(fr)
            fr.append(k[0])
            to.append(k[1])
            bb.append(bval)
            tie.append(is_tie)
            return True

        spur_bus: List[int] = []
        spur_line: List[int] = []
        for s, m in zip(starts, sizes):
            buses = list(range(s, s + m))
            # optional radial spur: bus q hangs on its ring neighbour q-1 by a
            # single line (a bridge); the ring bypasses it via (q-1, q+1)
            q = None
            if m >= 8 and rng.random() < 0.6:
                q = int(rng.integers(2, m - 2))
            for k in range(m):
                if q is not None and k in (q - 1, q):
                    continue
                add(buses[k], buses[(k + 1) % m], bs * U(1.5, 6.0), False)
            if q is not None:
                add(buses[q - 1], buses[q + 1], bs * U(1.5, 6.0), False)
                add(buses[q - 1], buses[q], bs * U(1.5, 6.0), False)
                spur_bus.append(buses[q])
                spur_line.append(len(fr) - 1)
            n_ch = int(round(m * U(0.2, 0.4)))
            added = tries = 0
            while added < n_ch and tries < 20 * n_ch + 20:
                tries += 1
                k = int(rng.integers(m))
                d = int(rng.integers(2, max(3, m // 4) + 1))
                if q is not None and q in (k, (k + d) % m):
                    continue
                if add(buses[k], buses[(k + d) % m], bs * U(1.5, 6.0), False):
                    added += 1
        for (x, y) in pairs:
            nt = int(rng.integers(2, 4))
            added = tries = 0
            while added < nt and tries < 60:
                tries += 1
                a = starts[x] + int(rng.integers(sizes[x]))
                c = starts[y] + int(rng.integers(sizes[y]))
                if add(a, c, bs * U(0.6, 2.4), True):
                    added += 1
        n_base = len(fr)

        # ── loads / renewables ──
        loads = np.zeros(n)
        for k in range(1, n):
            if rng.random() < 0.75:
                loads[k] = load_scale * U(0.3, 1.0) * _LOAD_W[bus_role[k]]
        for k in spur_bus:
            if loads[k] == 0.0:
                loads[k] = load_scale * U(0.3, 1.0) * _LOAD_W[bus_role[k]]
        if np.count_nonzero(loads) < 5:
            raise _Reject()
        total_load = float(loads.sum())
        renew = np.zeros(n)
        n_ren = int(rng.integers(0, 4))
        if n_ren:
            pool = [k for k in range(1, n) if bus_role[k] in ("export", "main", "mixed")]
            if family == "meshed":
                pool = list(range(1, n))
            sites = rng.choice(np.asarray(pool), size=min(n_ren, len(pool)), replace=False)
            share = U(0.05, 0.25)
            w = rng.dirichlet(np.ones(len(sites)))
            renew[sites] = share * total_load * w
        net_load = float(total_load - renew.sum())

        # ── generators ──
        G = int(np.clip(int(round(n * U(0.13, 0.20))), 6, 24))
        n_b = max(2, int(round(0.35 * G)))
        n_p = max(2, int(round(0.30 * G)))
        n_m = G - n_b - n_p
        if n_m < 1:
            raise _Reject()
        cls = ["base"] * n_b + ["mid"] * n_m + ["peak"] * n_p
        used = set()
        gbus = []
        for c in cls:
            free = [k for k in range(1, n) if k not in used]
            w = np.asarray([_GEN_W[c][bus_role[k]] for k in free], dtype=float)
            k = int(rng.choice(np.asarray(free), p=w / w.sum()))
            used.add(k)
            gbus.append(k)
        cap_total = U(1.5, 1.9) * net_load
        share = {"base": 0.45, "mid": 0.35, "peak": 0.20}
        frac = rng.uniform(0.6, 1.4, G)
        pmax = np.zeros(G)
        for c in ("base", "mid", "peak"):
            idx = [g for g in range(G) if cls[g] == c]
            pmax[idx] = share[c] * cap_total * frac[idx] / frac[idx].sum()
        pmin_f = {"base": (0.40, 0.55), "mid": (0.30, 0.45), "peak": (0.15, 0.30)}
        cost_r = {"base": (12.0, 24.0), "mid": (28.0, 48.0), "peak": (55.0, 100.0)}
        nl_r = {"base": (2.0, 4.0), "mid": (1.0, 2.5), "peak": (0.4, 1.2)}
        su_r = {"base": (3.0, 6.0), "mid": (1.5, 3.0), "peak": (0.3, 1.0)}
        pmin = np.array([pmax[g] * U(*pmin_f[cls[g]]) for g in range(G)])
        cost = np.array([U(*cost_r[cls[g]]) for g in range(G)])
        noload = np.array([pmax[g] * U(*nl_r[cls[g]]) for g in range(G)])
        startup = np.array([pmax[g] * U(*su_r[cls[g]]) for g in range(G)])
        order = np.argsort(np.asarray(gbus), kind="stable")    # display by bus
        gbus = np.asarray(gbus, dtype=int)[order]
        cls = [cls[g] for g in order]
        pmax, pmin, cost = pmax[order], pmin[order], cost[order]
        noload, startup = noload[order], startup[order]

        # ── hidden feasible design: commitment + dispatch ──
        avg = cost + noload / pmax
        on_a = np.zeros(G, dtype=bool)
        target_cap = net_load * U(1.15, 1.30)
        for g in np.argsort(avg, kind="stable"):
            if float(pmax[on_a].sum()) >= target_cap:
                break
            if float(pmin[on_a].sum()) + pmin[g] > 0.8 * net_load:
                continue
            on_a[g] = True
        if family != "meshed":
            imp = [g for g in range(G) if bus_role[gbus[g]] in ("import", "pocket")]
            if imp and not on_a[imp].any():
                g = min(imp, key=lambda q: avg[q])
                if float(pmin[on_a].sum()) + pmin[g] <= 0.85 * net_load:
                    on_a[g] = True
        if float(pmax[on_a].sum()) < 1.08 * net_load:
            raise _Reject()
        p_merit = _merit(on_a, pmin, pmax, cost, net_load)
        p_spread = _spread(on_a, pmin, pmax, net_load)
        if p_merit is None or p_spread is None:
            raise _Reject()
        wb = U(0.3, 0.75)
        p_a = (1.0 - wb) * p_merit + wb * p_spread
        reserve_req = min(U(0.05, 0.10) * net_load,
                          0.85 * float((pmax - p_a)[on_a].sum()))

        # ── preliminary flows (base topology, no PST) to place PSTs/switches ──
        fr_b = np.asarray(fr, dtype=int)
        to_b = np.asarray(to, dtype=int)
        b_b = np.asarray(bb, dtype=float)
        inj = renew - loads
        np.add.at(inj, gbus, p_a)
        act_b = np.ones(n_base, dtype=bool)
        f_pre, _ = _solve_flows(n, fr_b, to_b, b_b, act_b, np.ones(n, dtype=bool),
                                inj, np.zeros(n_base))
        absf = np.abs(f_pre)
        rank = np.argsort(-absf, kind="stable")
        n_pst = int(rng.integers(2, 5))
        pst_lines: List[int] = []
        spur_set = set(spur_line)               # a PST on a bridge does nothing
        rank = np.asarray([l for l in rank if int(l) not in spur_set], dtype=int)
        tie_idx = [l for l in rank if tie[l]]
        if tie_idx:
            k_t = min(len(tie_idx), int(rng.integers(1, 3)))
            pst_lines += [int(x) for x in rng.choice(np.asarray(tie_idx[:max(3, k_t)]),
                                                     size=k_t, replace=False)]
        top = [int(l) for l in rank[:max(6, int(0.4 * n_base))]
               if int(l) not in pst_lines and not tie[l]]
        need = n_pst - len(pst_lines)
        if need > 0:
            if len(top) < need:
                raise _Reject()
            pst_lines += [int(x) for x in rng.choice(np.asarray(top), size=need,
                                                     replace=False)]
        pst_set = set(pst_lines)
        k_sw = int(np.clip(2 + n // 35 + int(rng.integers(0, 3)), 2, 7))
        pool = [int(l) for l in rank[:max(8, int(0.6 * n_base))] if int(l) not in pst_set]
        if len(pool) < k_sw:
            raise _Reject()
        sw_base = [int(x) for x in rng.choice(np.asarray(pool), size=k_sw, replace=False)]
        for l in spur_line:                    # a radial feeder with a breaker
            if l not in sw_base and l not in pst_set and rng.random() < 0.7:
                sw_base.append(int(l))
        sw_base = sorted(sw_base)
        # normally-open tie lines (new edges)
        k_open = int(np.clip(2 + n // 40 + int(rng.integers(0, 2)), 2, 6))
        sw_new: List[int] = []
        tries = 0
        while len(sw_new) < k_open and tries < 200:
            tries += 1
            if pairs and (len(sw_new) == 0 or rng.random() < 0.4):
                x, y = pairs[int(rng.integers(len(pairs)))]
                a = starts[x] + int(rng.integers(sizes[x]))
                c = starts[y] + int(rng.integers(sizes[y]))
                bval, is_t = bs * U(0.8, 2.8), True
            else:
                ar = int(rng.integers(len(sizes)))
                m = sizes[ar]
                k = int(rng.integers(m))
                d = int(rng.integers(max(2, m // 6), max(3, m // 2) + 1))
                a, c = starts[ar] + k, starts[ar] + (k + d) % m
                bval, is_t = bs * U(1.0, 5.0), False
            if add(a, c, bval, is_t):
                sw_new.append(len(fr) - 1)
        if len(sw_new) < 2:
            raise _Reject()
        L = len(fr)
        fr_a = np.asarray(fr, dtype=int)
        to_a = np.asarray(to, dtype=int)
        b_a = np.asarray(bb, dtype=float)
        sw_lines = sorted(sw_base + sw_new)
        S = len(sw_lines)
        sw_pos = {l: s for s, l in enumerate(sw_lines)}
        normal = np.array([0.0 if l in sw_new else 1.0 for l in sw_lines])
        P = len(pst_lines)
        pst_lines = sorted(pst_lines)

        # hidden topology / PST
        closed_a = normal.copy()
        new_pos = [sw_pos[l] for l in sw_new]
        base_pos = [sw_pos[l] for l in sw_base]
        n_close = int(rng.integers(1, len(new_pos) + 1)) if rng.random() < 0.9 else 0
        for s in rng.choice(np.asarray(new_pos), size=n_close, replace=False):
            closed_a[int(s)] = 1.0
        n_open = int(rng.choice([0, 1, 2], p=[0.3, 0.45, 0.25]))
        opened = []
        for s in rng.permutation(np.asarray(base_pos)):
            if len(opened) >= n_open:
                break
            trial = closed_a.copy()
            trial[int(s)] = 0.0
            cl = _closed_mask(L, sw_lines, trial)
            if _energized(n, fr_a[cl], to_a[cl]).all():
                closed_a = trial
                opened.append(int(s))
        shift_a = np.array([float(rng.choice([-25, -20, -15, -10, -5, 5, 10, 15, 20, 25]))
                            if rng.random() < 0.8 else 0.0 for _ in range(P)])

        # hidden-design flows -> thermal limits
        cl_a = _closed_mask(L, sw_lines, closed_a)
        psi_a = np.zeros(L)
        psi_a[pst_lines] = shift_a * _DEG
        inj_a = renew - loads
        np.add.at(inj_a, gbus, p_a)
        f_a, _ = _solve_flows(n, fr_a, to_a, b_a, cl_a, np.ones(n, dtype=bool),
                              inj_a, psi_a)
        af = np.abs(f_a)
        nz = af[cl_a & (af > 1e-6)]
        if nz.size < 5:
            raise _Reject()
        floor = U(0.7, 1.1) * float(np.median(nz))
        h = {"meshed": U(1.10, 1.45), "two_area": U(1.08, 1.40),
             "load_pocket": U(1.08, 1.40), "three_area": U(1.06, 1.35)}[family]
        limit = np.maximum(h * rng.uniform(1.0, 1.25, L) * af, floor)
        for s in opened:                       # weak lines best left open
            limit[sw_lines[s]] = floor * U(0.5, 0.9)
        for s in new_pos:
            if closed_a[s] < 0.5:
                limit[sw_lines[s]] = floor * U(0.8, 1.4)
        limit = np.maximum(limit, 0.5)

        # ── initial (defective) design ──
        closed0 = closed_a.copy()
        for s in new_pos:
            if closed_a[s] > 0.5 and rng.random() < 0.55 + 0.4 * diff:
                closed0[s] = 0.0
        for s in opened:
            if rng.random() < 0.55 + 0.4 * diff:
                closed0[s] = 1.0
        cl0 = _closed_mask(L, sw_lines, closed0)
        if not _energized(n, fr_a[cl0], to_a[cl0]).all():
            raise _Reject()
        shift0 = shift_a.copy()
        for k in range(P):
            r = rng.random()
            if shift_a[k] != 0.0 and r < 0.45 + 0.45 * diff:
                shift0[k] = 0.0
            elif r < 0.9:
                shift0[k] = float(np.clip(shift_a[k] + float(rng.choice([-15, -10, -5, 5, 10, 15])),
                                          -_SHIFT_MAX, _SHIFT_MAX))
        on0 = on_a.copy()
        if rng.random() < 0.25 + 0.55 * diff:
            w = np.array([(3.0 if bus_role[gbus[g]] in ("import", "pocket") else 1.0)
                          * (2.0 if cls[g] != "base" else 0.7) for g in range(G)])
            w = w * on_a
            if w.sum() > 0 and on_a.sum() > 2:
                g = int(rng.choice(G, p=w / w.sum()))
                on0[g] = False
        if rng.random() < 0.2 + 0.4 * diff:
            w = np.where(on_a, 0.0, 1.0 + (cost > np.median(cost)))
            if w.sum() > 0:
                g = int(rng.choice(G, p=w / w.sum()))
                on0[g] = True
        if rng.random() < 0.1 + 0.4 * diff:
            # short commitment: drop units (priciest first, never base) until
            # the running fleet cannot also carry the reserve requirement
            for g in np.argsort(-avg, kind="stable"):
                g = int(g)
                if float(pmax[on0].sum()) < net_load + reserve_req:
                    break
                if (on0[g] and cls[g] != "base"
                        and float(pmax[on0].sum()) - pmax[g] >= 0.97 * net_load):
                    on0[g] = False
        if float(pmin[on0].sum()) > net_load or on0.sum() < 2:
            raise _Reject()
        # dispatch scheduled for a lower demand forecast (import shortfall)
        sched = (net_load * (1.0 - U(0.04, 0.06 + 0.08 * diff))
                 if rng.random() < 0.15 + 0.4 * diff else net_load)
        sched = max(sched, float(pmin[on0].sum()))

        p_m0 = _merit_capped(on0, pmin, pmax, cost, sched)
        p_proj = np.where(on0, np.where(on_a, p_a, pmin), 0.0)
        p_proj = np.clip(p_proj, pmin, pmax) * on0
        p_proj = _balance(p_proj, on0, pmin, pmax, sched)

        psi0 = np.zeros(L)
        psi0[pst_lines] = shift0 * _DEG

        def _load_at(wt: float) -> float:
            pp = (1.0 - wt) * p_proj + wt * p_m0
            inj_t = renew - loads
            np.add.at(inj_t, gbus, pp)
            f_t, _ = _solve_flows(n, fr_a, to_a, b_a, cl0, np.ones(n, dtype=bool),
                                  inj_t, psi0)
            return float(np.max(np.abs(f_t)[cl0] / limit[cl0]))

        target = 1.1 + 1.4 * min(max(diff + U(-0.08, 0.08), 0.0), 1.0)
        l1, l0 = _load_at(1.0), _load_at(0.0)
        if l1 <= target:
            wt = 1.0
        elif l0 >= target:
            wt = 0.0
        else:
            lo_w, hi_w = 0.0, 1.0
            for _ in range(22):
                mid = 0.5 * (lo_w + hi_w)
                if _load_at(mid) > target:
                    hi_w = mid
                else:
                    lo_w = mid
            wt = hi_w
        p0 = (1.0 - wt) * p_proj + wt * p_m0
        if _load_at(wt) > 3.0:
            raise _Reject()

        sig_l = {"meshed": (0.03, 0.05), "two_area": (0.04, 0.06),
                 "load_pocket": (0.05, 0.08), "three_area": (0.04, 0.07)}[family]
        sigma_load = round(U(*sig_l), 4)
        sigma_renew = round(U(0.08, 0.18), 4)
        # 1-sigma error of net demand under the stochastic variant; both slack
        # limits carry 1.5 of it so the 90% chance constraint on import AND
        # export can hold at once (identical problems in both variants)
        sig_net = math.sqrt(float(np.sum((sigma_load * loads) ** 2)
                                  + np.sum((sigma_renew * renew) ** 2)))
        imp_lim = max(2.0, U(0.02, 0.05) * net_load + 1.5 * sig_net)
        exp_lim = max(2.0, U(0.01, 0.03) * net_load + 1.5 * sig_net)

        r5 = lambda x: [round(float(v), 5) for v in x]  # noqa: E731
        kind = ["fixed"] * L
        for l in sw_lines:
            kind[l] = "switch"
        for l in pst_lines:
            kind[l] = "pst"
        spec = {
            "problem_id": "grid2_%04d" % int(index),
            "family": family,
            "difficulty": round(diff, 4),
            "n_bus": n,
            "n_lines": L,
            "areas": [_AREA_NAMES[int(a)] for a in area],
            "area_roles": list(roles),
            "lines": [[int(fr_a[l]), int(to_a[l]), round(float(b_a[l]), 5),
                       round(float(limit[l]), 4), kind[l]] for l in range(L)],
            "switch_lines": [int(l) for l in sw_lines],
            "switch_normal": [int(x) for x in normal],
            "switch_init": [int(x) for x in closed0],
            "pst_lines": [int(l) for l in pst_lines],
            "pst_init": [float(x) for x in shift0],
            "loads": r5(loads),
            "renew": r5(renew),
            "gen_bus": [int(x) for x in gbus],
            "gen_class": list(cls),
            "gen_cost": r5(cost),
            "gen_pmin": r5(pmin),
            "gen_pmax": r5(pmax),
            "gen_noload": r5(noload),
            "gen_startup": r5(startup),
            "gen_on_init": [int(x) for x in on0],
            "p_init": r5(p0),
            "pst_cost_per_deg": round(U(0.8, 1.6) * 1e-4 * float(
                np.dot(cost, p_a) + noload[on_a].sum()), 4),
            "import_price": round(U(1.3, 2.0) * float(cost.max()), 4),
            "export_price": round(U(0.2, 0.5) * float(cost.mean()), 4),
            "voll": round(U(8.0, 15.0) * float(cost.max()), 4),
            "sigma_load": sigma_load,
            "sigma_renew": sigma_renew,
            "net_load_mw": round(net_load, 5),
            "goals": {
                "maximum_line_loading": 1.0,
                "maximum_slack_import_mw": round(imp_lim, 4),
                "maximum_slack_export_mw": round(exp_lim, 4),
                "minimum_spinning_reserve_mw": round(reserve_req, 4),
                "maximum_load_shed_mw": 0.0,
                "maximum_overload_mw": 0.0,
            },
        }
        # switching cost per line whose status differs from the start (no
        # extra draw, so the rest of the problem is unchanged by it)
        spec["switch_cost"] = round(_SWITCH_COST_DEG * spec["pst_cost_per_deg"], 4)
        # everything below runs on the ROUNDED, loaded problem
        st = self.load(spec)
        m0 = self._metrics(st, st["loads"], st["renew"])
        if m0["load_shed_mw"] > 0.0 or not all(math.isfinite(v) for v in m0.values()):
            raise _Reject()
        if m0["line_loading"] > 2.6 or m0["n_overloaded"] > max(10, L // 12):
            raise _Reject()
        hidden = (np.asarray(closed_a), np.asarray(shift_a), np.asarray(on_a))
        initial = (np.asarray(closed0, dtype=float), np.asarray(shift0), np.asarray(on0))
        ref = self._reference_search(st, [hidden, initial,
                                          (hidden[0], hidden[1], initial[2]),
                                          (initial[0], initial[1], hidden[2])])
        if ref is None:
            raise _Reject()
        spec["ref_cost"] = round(float(ref), 6)
        spec["initially_infeasible_nominal"] = bool(self._violated(m0, spec))
        return spec

    # ══ state ════════════════════════════════════════════════════════════════
    def load(self, spec: dict):
        lines = spec["lines"]
        L = len(lines)
        n = int(spec["n_bus"])
        fr = np.asarray([int(l[0]) for l in lines], dtype=int)
        to = np.asarray([int(l[1]) for l in lines], dtype=int)
        b = np.asarray([float(l[2]) for l in lines], dtype=float)
        lim = np.asarray([float(l[3]) for l in lines], dtype=float)
        sw = np.asarray(spec["switch_lines"], dtype=int)
        pst = np.asarray(spec["pst_lines"], dtype=int)
        pmin = np.asarray(spec["gen_pmin"], dtype=float)
        pmax = np.asarray(spec["gen_pmax"], dtype=float)
        on = np.asarray(spec["gen_on_init"], dtype=float)
        p = np.where(on > 0.5, np.clip(np.asarray(spec["p_init"], dtype=float), pmin, pmax), 0.0)
        goals = spec["goals"]
        return {
            "n_bus": n,
            "fr": fr, "to": to, "b": b, "lim": lim,
            "sw_line": sw, "pst_line": pst,
            "sw_normal": np.asarray(spec["switch_normal"], dtype=float),
            "sw_closed": np.asarray(spec["switch_init"], dtype=float).copy(),
            "sw_init": np.asarray(spec["switch_init"], dtype=float) > 0.5,
            "sw_cost": float(spec.get("switch_cost", 0.0)),
            "pst_shift": np.asarray(spec["pst_init"], dtype=float).copy(),
            "loads": np.asarray(spec["loads"], dtype=float),
            "renew": np.asarray(spec["renew"], dtype=float),
            "g_bus": np.asarray(spec["gen_bus"], dtype=int),
            "g_class": list(spec["gen_class"]),
            "g_cost": np.asarray(spec["gen_cost"], dtype=float),
            "g_pmin": pmin, "g_pmax": pmax,
            "g_noload": np.asarray(spec["gen_noload"], dtype=float),
            "g_startup": np.asarray(spec["gen_startup"], dtype=float),
            "g_on_init": on.copy() > 0.5,
            "g_on": on.copy(),
            "g_p": p,
            "pst_cost": float(spec.get("pst_cost_per_deg", 0.0)),
            "import_price": float(spec["import_price"]),
            "export_price": float(spec["export_price"]),
            "voll": float(spec["voll"]),
            "sigma_load": float(spec["sigma_load"]),
            "sigma_renew": float(spec["sigma_renew"]),
            "imp_lim": float(goals["maximum_slack_import_mw"]),
            "exp_lim": float(goals["maximum_slack_export_mw"]),
            "reserve_req": float(goals["minimum_spinning_reserve_mw"]),
            "areas": list(spec.get("areas", ["A"] * n)),
            "L": L,
        }

    def clone(self, st):
        new = dict(st)
        for k in ("sw_closed", "pst_shift", "g_on", "g_p"):
            new[k] = np.array(st[k], dtype=float)
        return new

    # ══ simulation ═══════════════════════════════════════════════════════════
    def evaluate(self, st, seed: int = 0) -> dict:
        try:
            loads, renew = st["loads"], st["renew"]
            if self.stochastic:
                rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
                sl, sr = float(st["sigma_load"]), float(st["sigma_renew"])
                zl = rng.standard_normal(loads.shape[0])
                zr = rng.standard_normal(renew.shape[0])
                loads = loads * np.exp(sl * zl - 0.5 * sl * sl)
                renew = renew * np.exp(sr * zr - 0.5 * sr * sr)
            return self._metrics(st, loads, renew)
        except Exception:
            return self._failure_metrics(st)

    def _network(self, st, loads=None, renew=None) -> dict:
        """Full nominal (or given-load) solve with internals."""
        if loads is None:
            loads = st["loads"]
        if renew is None:
            renew = st["renew"]
        n = int(st["n_bus"])
        fr, to, b = st["fr"], st["to"], st["b"]
        closed = _closed_mask(st["L"], st["sw_line"], st["sw_closed"])
        energized = _energized(n, fr[closed], to[closed])
        act = closed & energized[fr]
        live = (st["g_on"] > 0.5) & energized[st["g_bus"]]
        pg = np.where(live, st["g_p"], 0.0)
        inj = np.where(energized, renew - loads, 0.0)
        inj = inj + np.bincount(st["g_bus"], pg, n)
        slack = -float(inj.sum())
        psi = np.zeros(st["L"])
        psi[st["pst_line"]] = st["pst_shift"] * _DEG
        flow, theta = _solve_flows(n, fr, to, b, act, energized, inj, psi)
        return {"closed": closed, "energized": energized, "act": act, "live": live,
                "pg": pg, "slack": slack, "flow": flow, "theta": theta, "psi": psi,
                "loads": loads, "renew": renew}

    def _metrics(self, st, loads, renew) -> dict:
        try:
            if not (np.all(np.isfinite(st["g_p"])) and np.all(np.isfinite(loads))
                    and np.all(np.isfinite(renew)) and np.all(np.isfinite(st["pst_shift"]))):
                return self._failure_metrics(st)
            net = self._network(st, loads, renew)
            flow, act = net["flow"], net["act"]
            loading = np.where(act, np.abs(flow) / st["lim"], 0.0)
            max_loading = float(loading.max()) if loading.size else 0.0
            slack = net["slack"]
            imp = max(slack, 0.0)
            exp = max(-slack, 0.0)
            shed = float(loads[~net["energized"]].sum())
            live = net["live"]
            on = st["g_on"] > 0.5
            reserve = float((st["g_pmax"] - st["g_p"])[live].sum())
            energy = float(np.dot(st["g_cost"], net["pg"]))
            commit = float(st["g_noload"][on].sum()
                           + st["g_startup"][on & ~st["g_on_init"]].sum())
            pst_op = float(st["pst_cost"] * np.abs(st["pst_shift"]).sum())
            sw_op = self._switching_cost(st, st["sw_closed"])
            total = (energy + commit + pst_op + sw_op + st["import_price"] * imp
                     + st["export_price"] * exp + st["voll"] * shed)
            over_mw = float(np.sum(np.maximum(np.abs(flow) - st["lim"], 0.0)[act]))
            out = {
                "total_cost": total,
                "energy_cost": energy,
                "commitment_cost": commit,
                "line_loading": max_loading,
                "overload_mw": over_mw,
                "n_overloaded": float(np.count_nonzero(loading > 1.0)),
                "slack_import_mw": imp,
                "slack_export_mw": exp,
                "spinning_reserve_mw": reserve,
                "load_shed_mw": shed,
                "n_committed": float(np.count_nonzero(on)),
            }
            if not all(math.isfinite(v) for v in out.values()):
                return self._failure_metrics(st)
            return out
        except Exception:
            return self._failure_metrics(st)

    @staticmethod
    def _switching_cost(st, sw_closed) -> float:
        """sw_cost per switchable line whose status differs from the start."""
        changed = (np.asarray(sw_closed, dtype=float) > 0.5) != st["sw_init"]
        return float(st["sw_cost"] * np.count_nonzero(changed))

    def _failure_metrics(self, st) -> dict:
        """Finite, maximally bad, outside the validity envelope."""
        try:
            tl = float(np.sum(st["loads"]))
            nl = float(st["L"])
        except Exception:
            tl, nl = 1e4, 1.0
        big = 10.0 * max(tl, 1.0)
        return {"total_cost": 1e9, "energy_cost": 1e9, "commitment_cost": 0.0,
                "line_loading": 1e3, "overload_mw": big, "n_overloaded": nl,
                "slack_import_mw": big, "slack_export_mw": big,
                "spinning_reserve_mw": 0.0, "load_shed_mw": big,
                "n_committed": 0.0}

    @staticmethod
    def _violated(m: dict, spec: dict) -> bool:
        g = spec["goals"]
        return (m["line_loading"] > 1.0 + 1e-9
                or m["slack_import_mw"] > float(g["maximum_slack_import_mw"]) + 1e-9
                or m["slack_export_mw"] > float(g["maximum_slack_export_mw"]) + 1e-9
                or m["spinning_reserve_mw"] < float(g["minimum_spinning_reserve_mw"]) - 1e-9
                or m["load_shed_mw"] > 1e-9
                or m.get("overload_mw", 0.0) > 1e-9)

    # ══ design variables ═════════════════════════════════════════════════════
    def _n(self, st) -> Tuple[int, int, int]:
        return (int(st["g_cost"].shape[0]), int(st["sw_line"].shape[0]),
                int(st["pst_line"].shape[0]))

    def _kind(self, st, i: int) -> Tuple[str, int]:
        G, S, P = self._n(st)
        i = int(i)
        if 0 <= i < G:
            return "gen", i
        if G <= i < G + S:
            return "switch", i - G
        if G + S <= i < G + S + P:
            return "pst", i - G - S
        raise ValueError("grid2: element index %r out of range [0, %d)" % (i, G + S + P))

    def n_elements(self, st) -> int:
        G, S, P = self._n(st)
        return G + S + P

    def get(self, st, i: int, p: str) -> Optional[float]:
        try:
            kind, k = self._kind(st, i)
        except ValueError:
            return None
        if kind == "gen":
            if p == "on":
                return 1.0 if st["g_on"][k] > 0.5 else 0.0
            if p == "p_set":
                return float(st["g_p"][k]) if st["g_on"][k] > 0.5 else None
            return None
        if kind == "switch":
            return (1.0 if st["sw_closed"][k] > 0.5 else 0.0) if p == "closed" else None
        return float(st["pst_shift"][k]) if p == "shift" else None

    def set(self, st, i: int, p: str, v: float) -> None:
        if p not in _PARAMS:
            raise ValueError("grid2 domain has no parameter %r" % (p,))
        kind, k = self._kind(st, i)
        v = float(v)
        if not math.isfinite(v):
            return
        if kind == "gen":
            if p == "on":
                want = v >= 0.5
                is_on = st["g_on"][k] > 0.5
                if want and not is_on:
                    st["g_on"][k] = 1.0
                    st["g_p"][k] = float(st["g_pmin"][k])
                elif (not want) and is_on:
                    st["g_on"][k] = 0.0
                    st["g_p"][k] = 0.0
            elif p == "p_set":
                if st["g_on"][k] > 0.5:
                    st["g_p"][k] = min(max(v, float(st["g_pmin"][k])), float(st["g_pmax"][k]))
            return
        if kind == "switch":
            if p == "closed":
                st["sw_closed"][k] = 1.0 if v >= 0.5 else 0.0
            return
        if p == "shift":
            st["pst_shift"][k] = _legal_shift(v)

    def bounds(self, st, p: str) -> Tuple[float, float]:
        if p == "p_set":
            return (0.0, float(np.max(st["g_pmax"])))
        if p in ("on", "closed"):
            return (0.0, 1.0)
        if p == "shift":
            return (-_SHIFT_MAX, _SHIFT_MAX)
        raise ValueError("grid2 domain has no parameter %r" % (p,))

    # ── typed move set ───────────────────────────────────────────────────────
    def candidate_values(self, st, i: int, p: str, cur: float) -> List[float]:
        """Per-(element, param) values (used by tools that ask for them; the
        ladder uses candidate_moves, which adds stratified budgeting)."""
        kind, k = self._kind(st, i)
        if kind == "gen" and p == "p_set":
            return self._pset_values(st, k, self._slack_now(st))
        if p in ("on", "closed"):
            return [1.0 - (1.0 if cur > 0.5 else 0.0)]
        if kind == "pst" and p == "shift":
            return self._shift_values(cur)
        return []

    def _relief_steps(self, st) -> Dict[int, float]:
        """Per live unit: the p_set change (MW, balanced at slack bus 0) that
        brings the worst overloaded line to 98% of its limit, from
        the unit's shift factor on that line (|factor| >= 0.02 only)."""
        try:
            net = self._network(st)
            act = net["act"]
            loading = np.where(act, np.abs(net["flow"]) / st["lim"], 0.0)
            if loading.size == 0 or float(loading.max()) <= 1.0:
                return {}
            l = int(np.argmax(loading))
            excess = abs(float(net["flow"][l])) - 0.98 * float(st["lim"][l])
            sign = 1.0 if net["flow"][l] >= 0.0 else -1.0
            sens = self._sens_bus(st, net, l)[st["g_bus"]]
            out = {}
            for g in np.flatnonzero(net["live"]):
                sg = float(sens[g]) * sign
                if abs(sg) >= 0.02:
                    out[int(g)] = -excess / sg
            return out
        except Exception:
            return {}

    def _slack_now(self, st) -> float:
        n = int(st["n_bus"])
        closed = _closed_mask(st["L"], st["sw_line"], st["sw_closed"])
        en = _energized(n, st["fr"][closed], st["to"][closed])
        live = (st["g_on"] > 0.5) & en[st["g_bus"]]
        return float((st["loads"] - st["renew"])[en].sum() - st["g_p"][live].sum())

    def _pset_values(self, st, g: int, slack: float,
                     relief: Optional[float] = None) -> List[float]:
        p = float(st["g_p"][g])
        lo, hi = float(st["g_pmin"][g]), float(st["g_pmax"][g])
        vals = [p * f for f in self.factors] + [p * 0.6, p * 1.5, lo, hi]
        if abs(slack) > 1e-6:
            vals.append(p + slack)            # take up the whole current mismatch
        if relief is not None and math.isfinite(relief):
            vals.append(p + relief)           # just clears the worst overload
        out: List[float] = []
        for v in vals:
            v = min(max(float(v), lo), hi)
            if abs(v - p) <= 1e-9 * max(1.0, abs(p)):
                continue
            if any(abs(v - u) <= 1e-9 * max(1.0, abs(u)) for u in out):
                continue
            out.append(v)
        return out

    @staticmethod
    def _shift_values(cur: float) -> List[float]:
        out: List[float] = []
        for d in (-15.0, -10.0, -5.0, 5.0, 10.0, 15.0):
            v = _legal_shift(cur + d)
            if v != cur and v not in out:
                out.append(v)
        return out

    def candidate_moves(self, st, rng: np.random.Generator) -> List[Candidate]:
        """Every element type and move type every turn: all commitment
        toggles, all switch toggles, all PST shifts are always offered;
        p_set values fill the remaining budget round-robin over generators
        (each committed, energised unit keeps at least one value)."""
        G, S, P = self._n(st)
        n = int(st["n_bus"])
        closed = _closed_mask(st["L"], st["sw_line"], st["sw_closed"])
        en = _energized(n, st["fr"][closed], st["to"][closed])
        live = (st["g_on"] > 0.5) & en[st["g_bus"]]
        slack = float((st["loads"] - st["renew"])[en].sum() - st["g_p"][live].sum())
        fixed: List[Candidate] = []
        for g in range(G):
            cur = 1.0 if st["g_on"][g] > 0.5 else 0.0
            fixed.append(Candidate(i=g, p="on", v=1.0 - cur, v_old=cur))
        for s in range(S):
            cur = 1.0 if st["sw_closed"][s] > 0.5 else 0.0
            fixed.append(Candidate(i=G + s, p="closed", v=1.0 - cur, v_old=cur))
        for k in range(P):
            cur = float(st["pst_shift"][k])
            for v in self._shift_values(cur):
                fixed.append(Candidate(i=G + S + k, p="shift", v=v, v_old=cur))
        relief = self._relief_steps(st)
        per_gen: Dict[int, List[float]] = {}
        for g in range(G):
            if live[g]:
                vals = self._pset_values(st, g, slack, relief.get(g))
                if vals:
                    per_gen[g] = [vals[j] for j in rng.permutation(len(vals))]
        budget = max(self._cand_budget - len(fixed), len(per_gen))
        chosen: Dict[int, List[float]] = {g: [] for g in per_gen}
        total = 0
        depth = 0
        while total < budget:
            progressed = False
            for g in sorted(per_gen):
                if depth < len(per_gen[g]) and total < budget:
                    chosen[g].append(per_gen[g][depth])
                    total += 1
                    progressed = True
            if not progressed:
                break
            depth += 1
        cands: List[Candidate] = []
        for g in range(G):
            for v in sorted(chosen.get(g, [])):
                cands.append(Candidate(i=g, p="p_set", v=float(v),
                                       v_old=float(st["g_p"][g])))
        by_elem: Dict[int, List[Candidate]] = {}
        for c in cands + fixed:
            by_elem.setdefault(c.i, []).append(c)
        order = {"p_set": 0, "on": 1, "closed": 2, "shift": 3}
        out: List[Candidate] = []
        for i in sorted(by_elem):
            out.extend(sorted(by_elem[i], key=lambda c: (order[c.p], c.v)))
        return out

    # ══ program / reference ══════════════════════════════════════════════════
    def program(self, spec: dict) -> DesignProgram:
        net = max(float(spec.get("net_load_mw") or
                        (sum(spec["loads"]) - sum(spec["renew"]))), 1.0)
        total_load = max(float(sum(float(x) for x in spec["loads"])), 1.0)
        # Violation units: 10% overload on the worst line ~ 1; one median
        # line limit of total excess MW ~ 10; 0.5% of net demand over a slack
        # limit ~ 1; 10% of the reserve requirement short ~ 1; 0.1% of net
        # demand shed ~ 1. Tight scales keep the smooth hinge from paying for
        # margin on a satisfied constraint: PST/switch moves are free, and at
        # scale 1.0 the gold policy kept nudging PSTs after reaching
        # feasibility to buy hinge margin (zero cost change).
        med_lim = float(np.median([float(l[3]) for l in spec["lines"]]))
        scales = {
            "line_loading": 0.1,
            "slack_import_mw": 0.005 * net,
            "slack_export_mw": 0.005 * net,
            "spinning_reserve_mw": 0.1 * max(float(spec["goals"]["minimum_spinning_reserve_mw"]), 1.0),
            "load_shed_mw": 0.001 * net,
            "overload_mw": 0.1 * med_lim,
        }
        prog = program_from_goals(
            spec["goals"],
            state_keys=_METRIC_KEYS,
            objective_key="total_cost",
            objective_ref=self.reference(spec),
            objective_sense="min",
            scales=scales,
            domain=self.name,
        )
        return replace(prog, validity_bounds=(
            ("line_loading", _LOADING_CAP),
            ("slack_import_mw", total_load),
            ("slack_export_mw", total_load),
            ("load_shed_mw", 0.5 * total_load),
        ))

    def reference(self, spec: dict) -> Optional[float]:
        ref = spec.get("ref_cost")
        if ref is None:
            st = self.load(spec)
            init = (st["sw_closed"].copy(), st["pst_shift"].copy(), st["g_on"] > 0.5)
            ref = self._reference_search(st, [init])
            if ref is None:
                return None
            spec["ref_cost"] = float(ref)
        return float(ref)

    def _lp_setup(self, st, sw_closed, cache: dict):
        key = tuple(int(x > 0.5) for x in sw_closed)
        if key in cache:
            return cache[key]
        n = int(st["n_bus"])
        fr, to, b = st["fr"], st["to"], st["b"]
        closed = _closed_mask(st["L"], st["sw_line"], sw_closed)
        if not _energized(n, fr[closed], to[closed]).all():
            cache[key] = None
            return None
        fa, ta, ba = fr[closed], to[closed], b[closed]
        B = _bmat(n, fa, ta, ba)
        X = np.zeros((n, n))
        X[1:, 1:] = np.linalg.inv(B[1:, 1:])
        PT = np.zeros((st["L"], n))
        PT[closed] = ba[:, None] * (X[fa] - X[ta])
        out = (closed, X, PT)
        cache[key] = out
        return out

    def _lp_cost(self, st, cfg, cache: dict) -> Optional[float]:
        sw_closed, shift, on = cfg
        setup = self._lp_setup(st, sw_closed, cache)
        if setup is None:
            return None
        closed, X, PT = setup
        n = int(st["n_bus"])
        fr, to, b = st["fr"], st["to"], st["b"]
        on = np.asarray(on, dtype=bool)
        on_idx = np.flatnonzero(on)
        if on_idx.size == 0:
            return None
        psi = np.zeros(st["L"])
        psi[st["pst_line"]] = np.asarray(shift, dtype=float) * _DEG
        bp = np.where(closed, b * psi, 0.0)
        rhs_s = -np.bincount(fr, bp, n) + np.bincount(to, bp, n)
        th = X @ rhs_s
        f_s = np.where(closed, b * (th[fr] - th[to] + psi), 0.0)
        f0 = PT @ (st["renew"] - st["loads"]) + f_s
        M = PT[:, st["g_bus"][on_idx]]
        Mc, f0c, lc = M[closed], f0[closed], st["lim"][closed]
        k = on_idx.size
        z2 = np.zeros((Mc.shape[0], 2))
        A_ub = np.vstack([np.hstack([Mc, z2]), np.hstack([-Mc, z2]),
                          np.concatenate([np.ones(k), [0.0, 0.0]])[None, :]])
        b_ub = np.concatenate([lc - f0c, lc + f0c,
                               [float(st["g_pmax"][on_idx].sum()) - st["reserve_req"]]])
        A_eq = np.concatenate([np.ones(k), [1.0, -1.0]])[None, :]
        net = float(st["loads"].sum() - st["renew"].sum())
        c = np.concatenate([st["g_cost"][on_idx], [st["import_price"], st["export_price"]]])
        bnds = ([(float(st["g_pmin"][g]), float(st["g_pmax"][g])) for g in on_idx]
                + [(0.0, st["imp_lim"]), (0.0, st["exp_lim"])])
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[net],
                          bounds=bnds, method="highs")
        except Exception:
            return None
        if not res.success:
            return None
        const = float(st["g_noload"][on].sum() + st["g_startup"][on & ~st["g_on_init"]].sum()
                      + st["pst_cost"] * np.abs(np.asarray(shift, dtype=float)).sum()
                      + self._switching_cost(st, sw_closed))
        val = float(res.fun) + const
        return val if math.isfinite(val) and val > 0.0 else None

    def _reference_search(self, st, configs, rounds: int = 3) -> Optional[float]:
        cache: dict = {}
        best_val, best_cfg = None, None
        for cfg in configs:
            cfg = (np.asarray(cfg[0], dtype=float), np.asarray(cfg[1], dtype=float),
                   np.asarray(cfg[2], dtype=bool))
            v = self._lp_cost(st, cfg, cache)
            if v is not None and (best_val is None or v < best_val):
                best_val, best_cfg = v, cfg
        if best_cfg is None:
            return None
        G, S, P = self._n(st)
        for _ in range(rounds):
            improved = None
            sw, sh, on = best_cfg
            neigh = []
            for s in range(S):
                t = sw.copy()
                t[s] = 1.0 - t[s]
                neigh.append((t, sh, on))
            for k in range(P):
                for d in (-5.0, 5.0):
                    t = sh.copy()
                    t[k] = _legal_shift(t[k] + d)
                    if t[k] != sh[k]:
                        neigh.append((sw, t, on))
            for g in range(G):
                t = on.copy()
                t[g] = not t[g]
                neigh.append((sw, sh, t))
            for cfg in neigh:
                v = self._lp_cost(st, cfg, cache)
                if v is not None and v < best_val - 1e-9 and (improved is None or v < improved[0]):
                    improved = (v, cfg)
            if improved is None:
                break
            best_val, best_cfg = improved
        return best_val

    # ══ feedback: hint / hint_move ═══════════════════════════════════════════
    def _sens_bus(self, st, net, l: int) -> np.ndarray:
        """d flow_l / d injection at each bus (withdrawn at bus 0), current
        topology; zero for dead buses."""
        n = int(st["n_bus"])
        act, en = net["act"], net["energized"]
        fr, to, b = st["fr"], st["to"], st["b"]
        idx = np.flatnonzero(en)
        idx = idx[idx != 0]
        out = np.zeros(n)
        if idx.size == 0 or not act[l]:
            return out
        B = _bmat(n, fr[act], to[act], b[act])
        rhs = np.zeros(n)
        i, j = int(fr[l]), int(to[l])
        rhs[i] += b[l]
        rhs[j] -= b[l]
        y = np.linalg.solve(B[np.ix_(idx, idx)], rhs[idx])   # B symmetric
        out[idx] = y
        return out

    def hint_move(self, st) -> Optional[Candidate]:
        """Operator's cheap rule: reconnect shed load; else redispatch the
        unit with the largest relieving shift factor on the worst line (one
        knob, v1-style 0.8/1.25 steps); else fix slack/reserve by redispatch
        or priority-list commitment; else trim the priciest unit."""
        try:
            G, S, P = self._n(st)
            net = self._network(st)
            m = self._metrics(st, st["loads"], st["renew"])
            p, pmin, pmax = st["g_p"], st["g_pmin"], st["g_pmax"]
            on = st["g_on"] > 0.5
            live = net["live"]
            en = net["energized"]
            eps = 1e-9
            up_ok = live & (p < pmax - eps * np.maximum(1.0, pmax))
            dn_ok = live & (p > pmin + eps * np.maximum(1.0, pmin))
            avg = st["g_cost"] + st["g_noload"] / st["g_pmax"]

            def mv(g, f):
                v = min(max(float(p[g]) * f, float(pmin[g])), float(pmax[g]))
                return Candidate(i=int(g), p="p_set", v=v, v_old=float(p[g]))

            if m["load_shed_mw"] > 0.0:
                for s in range(S):
                    if st["sw_closed"][s] < 0.5:
                        l = int(st["sw_line"][s])
                        if en[st["fr"][l]] != en[st["to"][l]]:
                            return Candidate(i=G + s, p="closed", v=1.0, v_old=0.0)
            if m["line_loading"] > 1.0:
                loading = np.where(net["act"], np.abs(net["flow"]) / st["lim"], 0.0)
                l = int(np.argmax(loading))
                sens = self._sens_bus(st, net, l)[st["g_bus"]]
                sign = 1.0 if net["flow"][l] >= 0.0 else -1.0
                for g in np.argsort(-np.abs(sens), kind="stable"):
                    g = int(g)
                    if abs(float(sens[g])) < 1e-9 or not live[g]:
                        continue
                    if float(sens[g]) * sign > 0.0:
                        if dn_ok[g]:
                            return mv(g, min(self.factors))
                    elif up_ok[g]:
                        return mv(g, max(self.factors))
            if m["slack_import_mw"] > st["imp_lim"]:
                for g in np.argsort(st["g_cost"], kind="stable"):
                    if up_ok[int(g)]:
                        return mv(int(g), max(self.factors))
            if (m["slack_import_mw"] > st["imp_lim"]
                    or m["spinning_reserve_mw"] < st["reserve_req"]):
                for g in np.argsort(avg, kind="stable"):
                    g = int(g)
                    if not on[g] and en[st["g_bus"][g]]:
                        return Candidate(i=g, p="on", v=1.0, v_old=0.0)
            if m["slack_export_mw"] > st["exp_lim"]:
                for g in np.argsort(-st["g_cost"], kind="stable"):
                    if dn_ok[int(g)]:
                        return mv(int(g), min(self.factors))
                for g in np.argsort(-avg, kind="stable"):
                    if on[int(g)]:
                        return Candidate(i=int(g), p="on", v=0.0, v_old=1.0)
            for g in np.argsort(-st["g_cost"], kind="stable"):
                if dn_ok[int(g)]:
                    return mv(int(g), 0.9)
            return None
        except Exception:
            return None

    def hint(self, st) -> Optional[Tuple[int, str, str]]:
        c = self.hint_move(st)
        if c is None:
            return None
        return (int(c.i), c.p, "up" if c.v > c.v_old else "down")

    # ══ observation ══════════════════════════════════════════════════════════
    @staticmethod
    def _fmt(x, sig: int = 4) -> str:
        x = float(x)
        if not math.isfinite(x):
            return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
        if x == 0.0:
            return "0"
        a = abs(x)
        if a >= 10.0 ** (sig - 1):
            return "%.0f" % x
        if a >= 1e-3:
            dec = max(0, sig - 1 - int(math.floor(math.log10(a))))
            s = "%.*f" % (dec, x)
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            if s in ("", "-", "0", "-0"):
                s = "%.*e" % (max(0, sig - 1), x)
            return s
        return "%.*e" % (max(0, sig - 1), x)

    @classmethod
    def _sfmt(cls, x, sig: int = 3) -> str:
        s = cls._fmt(x, sig)
        return s if (s.startswith("-") or s == "0") else "+" + s

    @staticmethod
    def _pct(x: float) -> str:
        """Loading fraction -> percent text (2 decimals within 0.5 of 100%)."""
        v = 100.0 * float(x)
        if 99.5 <= v < 100.5:
            return "%.2f%%" % v
        if 0.0 < v < 0.5:
            return "<1%"
        return "%.0f%%" % v

    @staticmethod
    def _dmw(x: float) -> str:
        """Signed MW change, compact; |x| < 0.05 MW prints as 0."""
        x = float(x)
        if not math.isfinite(x) or abs(x) < 0.05:
            return "0"
        return ("%+.1f" if abs(x) < 10.0 else "%+.0f") % x

    @staticmethod
    def _dsens(x: float) -> str:
        """Shift factor (MW per MW), 3 decimals; |x| < 0.0015 prints as 0."""
        x = float(x)
        if abs(x) < 0.0015:
            return "0"
        return ("%+.3f" % x).replace("0.", ".", 1)

    def _ptdf_obs(self, st, net) -> np.ndarray:
        """PT[l, k] = change of flow_l per +1 MW injected at bus k and withdrawn
        at slack bus 0, current topology (zero for dead buses / open lines)."""
        n, L = int(st["n_bus"]), int(st["L"])
        act, en = net["act"], net["energized"]
        fr, to, b = st["fr"], st["to"], st["b"]
        PT = np.zeros((L, n))
        idx = np.flatnonzero(en)
        idx = idx[idx != 0]
        if idx.size == 0 or not act.any():
            return PT
        B = _bmat(n, fr[act], to[act], b[act])
        X = np.zeros((n, n))
        X[np.ix_(idx, idx)] = np.linalg.inv(B[np.ix_(idx, idx)])
        PT[act] = b[act][:, None] * (X[fr[act]] - X[to[act]])
        return PT

    @staticmethod
    def _first_limit(flow, lim, d, ok) -> Tuple[float, int]:
        """Smallest x >= 0 at which a line in ok reaches |flow + x*d| = limit
        (lines in ok are at or below their limit); (inf, -1) if none."""
        dd = np.where(ok, d, 0.0)
        x = np.full(dd.shape, np.inf)
        pos, neg = dd > 1e-6, dd < -1e-6
        x[pos] = (lim[pos] - flow[pos]) / dd[pos]
        x[neg] = (lim[neg] + flow[neg]) / (-dd[neg])
        x = np.maximum(x, 0.0)
        if x.size == 0 or not np.isfinite(x).any():
            return math.inf, -1
        l = int(np.argmin(x))
        return float(x[l]), l

    def _after_txt(self, st, child, flow0, over0, watch, lim, m0,
                   n_new: int = 2) -> Tuple[str, str, dict]:
        """One change applied to child: (change in |flow| on the watched
        lines, whole-network result text, child network). A slack / reserve /
        load-shed limit is reported when the change breaks it, or alters it
        while it is already broken."""
        f, pct = self._fmt, self._pct
        nc = self._network(child)
        mc = self._metrics(child, child["loads"], child["renew"])
        ld = np.where(nc["act"], np.abs(nc["flow"]) / np.maximum(lim, 1e-12), 0.0)
        d = " ".join(self._dmw(abs(nc["flow"][l]) - abs(flow0[l])) for l in watch)
        w = int(np.argmax(ld)) if ld.size else 0
        if ld.size and ld[w] > 1.0:
            t = "-> L%d %s%s, %s MW over on %d" % (
                w, pct(ld[w]), "" if w in over0 else " (new)", f(mc["overload_mw"], 3),
                int(mc["n_overloaded"]))
        else:
            t = "-> none over (max L%d %s)" % (w, pct(ld[w]) if ld.size else "0%")
        new = [int(l) for l in np.argsort(-ld, kind="stable")
               if ld[l] > 1.0 and int(l) not in over0 and int(l) != w][:n_new]
        if new:
            t += "; new " + ", ".join("L%d %s" % (l, pct(ld[l])) for l in new)

        def moved(k):
            return abs(mc[k] - m0[k]) > 1e-6 * max(1.0, abs(m0[k]))
        if mc["slack_import_mw"] > st["imp_lim"] + 1e-6 and moved("slack_import_mw"):
            t += "; import %s > %s" % (f(mc["slack_import_mw"], 3), f(st["imp_lim"]))
        if mc["slack_export_mw"] > st["exp_lim"] + 1e-6 and moved("slack_export_mw"):
            t += "; export %s > %s" % (f(mc["slack_export_mw"], 3), f(st["exp_lim"]))
        if (mc["spinning_reserve_mw"] < st["reserve_req"] - 1e-6
                and moved("spinning_reserve_mw")):
            t += "; reserve %s < %s" % (f(mc["spinning_reserve_mw"], 3), f(st["reserve_req"]))
        if mc["load_shed_mw"] > 1e-9 and moved("load_shed_mw"):
            t += "; shed %s MW" % f(mc["load_shed_mw"], 3)
        return d, t, nc

    def _topology_txt(self, st, net) -> str:
        """Every line, compactly: runs of consecutive lines joining consecutive
        buses as one path, the rest one by one; open / de-energised marked."""
        fr, to = st["fr"], st["to"]
        L = int(st["L"])
        sw_of = {int(l): s for s, l in enumerate(st["sw_line"])}
        parts, l = [], 0
        while l < L:
            e = l
            if int(to[l]) == int(fr[l]) + 1:
                while (e + 1 < L and int(fr[e + 1]) == int(to[e])
                       and int(to[e + 1]) == int(fr[e + 1]) + 1):
                    e += 1
            if e - l >= 4:
                off = int(fr[l]) - l
                parts.append("L%d..L%d = %d-%d, %d-%d, ..., %d-%d (from-bus = L#%s)" % (
                    l, e, int(fr[l]), int(to[l]), int(fr[l + 1]), int(to[l + 1]),
                    int(fr[e]), int(to[e]), "" if off == 0 else " %+d" % off))
                l = e + 1
                continue
            parts.append("L%d %d-%d" % (l, int(fr[l]), int(to[l])))
            l += 1
        tags = []
        for l in range(L):
            if not net["closed"][l]:
                tags.append("L%d (SW%d open)" % (l, sw_of[l]))
            elif not net["act"][l]:
                tags.append("L%d (de-energised)" % l)
        return "; ".join(parts) + ". Out of service: " + (", ".join(tags) if tags else "none")

    def element_label(self, st, i) -> str:
        kind, k = self._kind(st, int(i))
        f = self._fmt
        if kind == "gen":
            return "[%d] generator G%d @ bus %d (%s, %s $/MWh, p_min %s, p_max %s MW)" % (
                int(i), k, int(st["g_bus"][k]), st["g_class"][k], f(st["g_cost"][k], 3),
                f(st["g_pmin"][k], 3), f(st["g_pmax"][k], 4))
        if kind == "switch":
            l = int(st["sw_line"][k])
            return "[%d] switchable line SW%d = L%d (bus %d-%d)" % (
                int(i), k, l, int(st["fr"][l]), int(st["to"][l]))
        l = int(st["pst_line"][k])
        return "[%d] phase shifter PST%d on line L%d (bus %d-%d)" % (
            int(i), k, l, int(st["fr"][l]), int(st["to"][l]))

    def describe(self, st, spec) -> str:
        f, sf, pct, dmw = self._fmt, self._sfmt, self._pct, self._dmw
        G, S, P = self._n(st)
        n = int(st["n_bus"])
        L = int(st["L"])
        fr, to, b, lim = st["fr"], st["to"], st["b"], st["lim"]
        loads, renew = st["loads"], st["renew"]
        gbus = st["g_bus"]
        cost, pmin, pmax = st["g_cost"], st["g_pmin"], st["g_pmax"]
        noload, startup = st["g_noload"], st["g_startup"]
        on = st["g_on"] > 0.5
        p = st["g_p"]
        areas = st["areas"]
        family = str(spec.get("family", "meshed"))
        roles = list(spec.get("area_roles", []))
        g = spec["goals"]
        imp_lim = float(g["maximum_slack_import_mw"])
        exp_lim = float(g["maximum_slack_export_mw"])
        r_req = float(g["minimum_spinning_reserve_mw"])
        imp_pr, exp_pr = float(st["import_price"]), float(st["export_price"])
        sw_of = {int(l): s for s, l in enumerate(st["sw_line"])}
        pst_of = {int(l): k for k, l in enumerate(st["pst_line"])}
        fam_txt = {
            "meshed": "one meshed area: a ring with cross-connecting lines",
            "two_area": "two areas joined by a few tie lines; area A (%s) and area B (%s)",
            "load_pocket": "a main area A and a load pocket B (%s / %s) fed over a few lines",
            "three_area": "three areas A, B, C (%s / %s / %s) joined in a ring by tie lines",
        }[family]
        role_txt = {"export": "generation-rich", "import": "load-heavy", "mixed": "mixed",
                    "main": "main system", "pocket": "load pocket"}
        if "%s" in fam_txt:
            fam_txt = fam_txt % tuple(role_txt.get(r, r) for r in roles)
        total_load = float(loads.sum())
        total_ren = float(renew.sum())
        tl = max(1.0, total_load)

        def snap(x: float) -> float:       # round-off of the 5-decimal inputs prints as 0
            return 0.0 if abs(float(x)) < max(1e-4, 1e-9 * tl) else float(x)
        out: List[str] = []
        A = out.append

        # ── PROBLEM (rules) ──
        A("PROBLEM")
        A("Unit commitment and DC power-flow dispatch on a %d-bus, %d-line network (%s; %d "
          "independent loops with every line in service). Loads %s MW on %d buses; "
          % (n, L, fam_txt, max(L - n + 1, 0), f(total_load), int(np.count_nonzero(loads)))
          + ("renewables %s MW on %d buses (free, fixed); net demand %s MW."
             % (f(total_ren), int(np.count_nonzero(renew)), f(total_load - total_ren))
             if total_ren > 0.0 else "no renewables; net demand %s MW." % f(total_load)))
        if self.stochastic:
            sig_net = math.sqrt(float(np.sum((st["sigma_load"] * loads) ** 2)
                                      + np.sum((st["sigma_renew"] * renew) ** 2)))
            A("Uncertainty: each scenario scales every bus load by an independent lognormal "
              "factor (mean 1, sigma %s) and every renewable by one (sigma %s); bus 0 takes "
              "the net error, 1 sigma = %s MW. Constraints must hold in 90%% of scenarios: "
              "keep the slack about %s MW (1.28 sigma) inside each slack limit. Everything "
              "below is the forecast (mean) case."
              % (f(st["sigma_load"], 3), f(st["sigma_renew"], 3), f(sig_net, 3),
                 f(1.28 * sig_net, 3)))
        A("Bus 0 is the slack, a tie to a neighbouring system covering any mismatch: import "
          "costs %s $/MWh (limit %s MW), exported surplus %s $/MWh (limit %s MW). Every MW "
          "of generation added or removed (%s) is balanced at bus 0, so moving output from "
          "one unit to another takes two changes (one up, one down)."
          % (f(imp_pr), f(imp_lim), f(exp_pr), f(exp_lim),
             "a DISPATCH change; COMMIT_UNIT adds p_min; DECOMMIT_UNIT removes the unit's "
             "output" if spelled_in_verbs(self)
             else "a p_set change; on=1 adds p_min; on=0 removes p_set"))
        A("Lossless DC power flow: flow = b*(theta_from - theta_to + shift), loading = "
          "|flow|/limit. p_set and shift effects are exact and linear at a fixed topology (-X "
          "gives the opposite of +X, two steps twice one, while no line reverses); switch and "
          "commitment changes are evaluated one by one below.")
        if spelled_in_verbs(self):
            # the same facts as the SET_PARAM block below, in the verbs the
            # policy is served (docstring "Verb surface"); the SET_PARAM block
            # again under the set_parameter spelling
            A("CHANGES: one per turn, one of DISPATCH(#, MW), COMMIT_UNIT(#), DECOMMIT_UNIT(#)"
              "%s%s; # = the number in brackets (not the G%s number); a value is the new "
              "ABSOLUTE setting, not a change."
              % (", CLOSE_LINE(#), OPEN_LINE(#)" if S else "",
                 ", SET_PHASE_SHIFT(#, deg)" if P else "",
                 (" / SW / L" if S else "") + (" / PST" if P else "")))
            A("* Generators [0..%d]: DISPATCH(#, MW) sets p_set, the output in [p_min, p_max] "
              "while the unit runs (clamped; not available while off), energy price c $/MWh. "
              "COMMIT_UNIT(#) starts the unit at exactly p_min and pays its no-load $/h; a "
              "unit off at the start also pays its start-up cost, a $/h-equivalent added in "
              "full while it is on (removed if decommitted again). DECOMMIT_UNIT(#): output "
              "0, no no-load; below p_min means decommitting." % (G - 1))
            if S:
                A("* Switchable lines [%d..%d]: CLOSE_LINE(#) puts the line in service, "
                  "OPEN_LINE(#) takes it out; %s $/h per switchable line whose status differs "
                  "from the start. A bus with no closed path to bus 0 is de-energised: load "
                  "shed (%s $/MWh, limit 0), renewables lost, generators tripped (no output or "
                  "reserve; no-load still paid while on)."
                  % (G, G + S - 1, f(st["sw_cost"]), f(st["voll"])))
            if P:
                A("* Phase shifters [%d..%d]: SET_PHASE_SHIFT(#, deg) sets the shift, the new "
                  "total in degrees (not a step), a multiple of 5 in [-30, +30]; + pushes more "
                  "power from->to through its own line and less through the parallel paths; "
                  "%s $/h per degree of |shift|."
                  % (G + S, G + S + P - 1, f(st["pst_cost"])))
        else:
            A("SET_PARAM(#, parameter, value) sets the new ABSOLUTE value; # = number in brackets.")
            A("* Generators [0..%d]: p_set = MW in [p_min, p_max] while on (clamped; not settable "
              "while off), energy price c $/MWh. on=1 starts the unit at exactly p_min and pays "
              "its no-load $/h; a unit off at the start also pays its start-up cost, a "
              "$/h-equivalent added in full while it is on (removed if switched off again). on=0: "
              "output 0, no no-load; below p_min means switching off." % (G - 1))
            if S:
                A("* Switchable lines [%d..%d]: closed 0/1 (1 = in service); %s $/h per switchable "
                  "line whose status differs from the start. A bus with no closed path to bus 0 is "
                  "de-energised: load shed (%s $/MWh, limit 0), renewables lost, generators "
                  "tripped (no output or reserve; no-load still paid while on)."
                  % (G, G + S - 1, f(st["sw_cost"]), f(st["voll"])))
            if P:
                A("* Phase shifters [%d..%d]: shift in degrees, multiples of 5 in [-30, +30]; + "
                  "pushes more power from->to through its own line and less through the parallel "
                  "paths; %s $/h per degree of |shift|." % (G + S, G + S + P - 1, f(st["pst_cost"])))
        A("CONSTRAINTS: every in-service line loading <= 100%%; overload_mw (sum over lines of "
          "the MW above their limits) = 0 - worst loading and total overload both count, so "
          "relieving ANY overloaded line helps; import <= %s MW; export <= %s MW; spinning "
          "reserve (sum over running, energised units of p_max - p_set) >= %s MW; load shed "
          "= 0." % (f(imp_lim), f(exp_lim), f(r_req)))
        A("OBJECTIVE: minimise TOTAL operating cost $/h, not only generation cost: energy "
          "(sum c*p_set) + no-load + start-up + phase shifters + switching + import + export "
          "+ load shed.")
        A("BUS LOADS (bus:MW): "
          + " ".join("%d:%s" % (k, f(loads[k], 3)) for k in range(n) if loads[k] != 0.0))
        if total_ren > 0.0:
            A("RENEWABLES (bus:MW): " + " ".join("%d:%s" % (k, f(renew[k], 3))
                                                  for k in range(n) if renew[k] != 0.0))
        if family != "meshed":
            spans = []
            for a in sorted(set(areas)):
                ks = [k for k in range(n) if areas[k] == a]
                spans.append("%s = buses %d-%d" % (a, min(ks), max(ks)))
            A("AREAS: " + "; ".join(spans))

        # ── solve once ──
        net = self._network(st)
        m = self._metrics(st, st["loads"], st["renew"])
        flow = np.where(np.abs(net["flow"]) < 1e-9 * tl, 0.0, net["flow"])
        act, en, live = net["act"], net["energized"], net["live"]
        loading = np.where(act, np.abs(flow) / np.maximum(lim, 1e-12), 0.0)
        slack = snap(net["slack"])
        imp, exp = max(slack, 0.0), max(-slack, 0.0)
        reserve = float(m["spinning_reserve_mw"])
        order = [int(x) for x in np.argsort(-loading, kind="stable")]
        over = [l for l in order if loading[l] > 1.0]
        over_set = set(over)
        k_watch = min(max(len(over), 3), 8, int(np.count_nonzero(act)))
        watch = [l for l in order[:k_watch] if act[l]]

        # ── STATE ──
        A("")
        rise = ("can rise %s MW before the import limit" % f(imp_lim - slack)
                if imp <= imp_lim else "import limit EXCEEDED by %s MW" % f(imp - imp_lim))
        fall = ("fall %s MW before the export limit" % f(exp_lim + slack)
                if exp <= exp_lim else "export limit EXCEEDED by %s MW" % f(exp - exp_lim))
        A("STATE: %d/%d units running, generation %s MW; slack %s MW (+ = import, - = export; "
          "it %s, %s; more generation lowers it); spinning reserve %s MW (need >= %s)."
          % (int(on.sum()), G, f(float(net["pg"].sum())), sf(slack, 4), rise, fall,
             f(reserve), f(r_req)))
        if slack > 0.0:
            mtxt = "c - %s (less import); -1 MW costs %s - c (more import)" % (
                f(imp_pr), f(imp_pr))
        elif slack < 0.0:
            mtxt = "c + %s (more export); -1 MW saves c + %s (less export)" % (
                f(exp_pr), f(exp_pr))
        else:
            mtxt = "c + %s (exported); -1 MW costs %s - c (imported)" % (f(exp_pr), f(imp_pr))
        A("Cost $/h: energy %s, no-load + start-up %s, phase shifters %s, switching %s, "
          "import %s, export %s, load shed %s; total %s. Now +1 MW from a unit costs %s, "
          "until the slack crosses 0."
          % (f(m["energy_cost"]), f(m["commitment_cost"]),
             f(st["pst_cost"] * float(np.abs(st["pst_shift"]).sum())),
             f(self._switching_cost(st, st["sw_closed"])), f(imp_pr * imp), f(exp_pr * exp),
             f(st["voll"] * m["load_shed_mw"]), f(m["total_cost"]), mtxt))
        if not en.all():
            dead = [k for k in range(n) if not en[k]]
            A("DE-ENERGISED buses (%d): %s; load shed %s MW."
              % (len(dead), " ".join(str(k) for k in dead), f(m["load_shed_mw"])))
        A("NETWORK (L# from-to): " + self._topology_txt(st, net))
        if family != "meshed":
            ties = [l for l in range(L) if areas[int(fr[l])] != areas[int(to[l])]]
            if ties:
                A("TIE LINES: " + "; ".join("L%d %d-%d %s-%s" % (
                    l, int(fr[l]), int(to[l]), areas[int(fr[l])], areas[int(to[l])])
                    for l in ties))

        def lrow(l, extra):
            return "L%d %d-%d %s/%s %s (%s)" % (
                l, int(fr[l]), int(to[l]), sf(flow[l], 3), f(lim[l], 3), pct(loading[l]), extra)
        if over:
            A("OVERLOADED (%d; flow/limit MW, + = from->to): " % len(over) + "; ".join(
                lrow(l, "+%s over" % f(abs(flow[l]) - lim[l], 3)) for l in over[:10])
              + ("; and %d more" % (len(over) - 10) if len(over) > 10 else ""))
        else:
            A("No line is overloaded.")
        near = [l for l in order if 0.9 <= loading[l] <= 1.0]
        if near:
            A("NEAR LIMIT (90-100%): " + "; ".join(
                lrow(l, "%s left" % f(lim[l] - abs(flow[l]), 3)) for l in near[:8])
              + ("; and %d more down to %s" % (len(near) - 8, pct(loading[near[-1]]))
                 if len(near) > 8 else ""))
        mid = [l for l in order if 0.75 <= loading[l] < 0.9]
        if mid:
            A("LOADED 75-90%% (%d lines, most loaded first; a>b = flow from bus a to b, a<b = "
              "from b to a): " % len(mid) + "; ".join(
                  "L%d %d%s%d %s" % (l, int(fr[l]), ">" if flow[l] > 0.0 else "<", int(to[l]),
                                     pct(loading[l])) for l in mid[:12])
              + ("; and %d more between 75%% and %s" % (len(mid) - 12, pct(loading[mid[11]]))
                 if len(mid) > 12 else ""))
        A("Every other in-service line is below 75% of its limit.")
        if not watch:
            return "\n".join(out)

        A("")
        A("EFFECTS (exact DC power flow). W = %s (%s). 'W' numbers = change in |flow| on "
          "these lines, in this order, MW (+ loads it more, - relieves it). '->' = the network "
          "after that one change: worst line, total MW over limits on N lines, lines newly "
          "over 100%%, slack/reserve/load-shed limits it breaks (or changes while broken)."
          % (" ".join("L%d" % l for l in watch),
             "every overloaded line" if len(watch) == len(over)
             else "the %d most overloaded of the %d overloaded lines" % (len(watch), len(over))
             if len(over) > len(watch)
             else "the overloaded lines, then the most loaded" if over
             else "the most loaded lines"))
        try:
            PT = self._ptdf_obs(st, net)
        except Exception:
            PT = None
        sgn = np.where(flow >= 0.0, 1.0, -1.0)
        ok = act & (loading <= 1.0)

        # ── generators ──
        merit = np.empty(G, dtype=int)
        merit[np.argsort(cost, kind="stable")] = np.arange(1, G + 1)
        A("GENERATORS: [#] name bus class c $/MWh (merit; 1 = cheapest of %d) p_min-p_max MW, "
          "no-load $/h, start-up if off at the start: status | W per +1 MW at its bus (an OFF "
          "unit starts at p_min) | p_set room alone before a NEW limit binds (a line not now "
          "over hits 100%%, p_min/p_max, slack or reserve limit) | on/off toggle ->" % G)
        for k in range(G):
            bk = int(gbus[k])
            row = "[%d] G%d bus %d%s %s c=%s (%d) %s-%s, no-load %s" % (
                k, k, bk, ("/" + areas[bk]) if family != "meshed" else "",
                st["g_class"][k], f(cost[k], 3), int(merit[k]), f(pmin[k]), f(pmax[k]),
                f(noload[k]))
            if not st["g_on_init"][k]:
                row += ", start-up %s" % f(startup[k])
            if on[k]:
                row += ": ON %s" % f(p[k])
                if not st["g_on_init"][k]:
                    row += " (off at start)"
                if not en[bk]:
                    row += ", TRIPPED (bus de-energised)"
            else:
                row += ": OFF" + (" (on at start)" if st["g_on_init"][k] else "")
                if not en[bk]:
                    row += ", bus de-energised"
            if en[bk]:
                try:
                    if PT is not None:
                        d = PT[:, bk]
                        row += " | W " + " ".join(self._dsens(sgn[l] * d[l]) for l in watch)
                        if on[k]:
                            ups = [(float(pmax[k] - p[k]), "p_max")]
                            dns = [(float(p[k] - pmin[k]), "p_min")]
                            if exp <= exp_lim + 1e-9:
                                ups.append((slack + exp_lim, "export limit"))
                            if reserve >= r_req - 1e-9:
                                ups.append((reserve - r_req, "reserve"))
                            if imp <= imp_lim + 1e-9:
                                dns.append((imp_lim - slack, "import limit"))
                            x, l = self._first_limit(flow, lim, d, ok)
                            if l >= 0:
                                ups.append((x, "L%d" % l))
                            x, l = self._first_limit(flow, lim, -d, ok)
                            if l >= 0:
                                dns.append((x, "L%d" % l))
                            u = min(ups, key=lambda t: t[0])
                            dn = min(dns, key=lambda t: t[0])
                            row += " | up %s (%s) / down %s (%s)" % (
                                f(max(u[0], 0.0), 3), u[1], f(max(dn[0], 0.0), 3), dn[1])
                except Exception:
                    pass
                try:
                    child = self.clone(st)
                    self.set(child, k, "on", 0.0 if on[k] else 1.0)
                    _, t, _ = self._after_txt(st, child, flow, over_set, watch, lim, m,
                                              0 if on[k] else 2)
                    row += " | %s %s" % ("stop" if on[k] else "start +%s" % f(pmin[k]), t)
                except Exception:
                    pass
            A(row)

        # ── switchable lines ──
        if S:
            A("SWITCHABLE LINES: [#] name line buses limit MW: status, flow | toggle: (for a "
              "close, the flow the line itself would carry) W change ->")
            for s in range(S):
                l = int(st["sw_line"][s])
                cl = st["sw_closed"][s] > 0.5
                init = bool(int(spec["switch_init"][s]))
                row = "[%d] SW%d L%d %d-%d limit %s: %s" % (
                    G + s, s, l, int(fr[l]), int(to[l]), f(lim[l], 3), "CLOSED" if cl else "OPEN")
                if cl != init:
                    row += " (%s at start, +%s $/h)" % ("closed" if init else "open",
                                                       f(st["sw_cost"]))
                if cl and act[l]:
                    row += " %s MW (%s)" % (sf(flow[l], 3), pct(loading[l]))
                try:
                    child = self.clone(st)
                    child["sw_closed"][s] = 0.0 if cl else 1.0
                    dW, t, nc = self._after_txt(st, child, flow, over_set, watch, lim, m)
                    ne = nc["energized"]
                    extra = ""
                    if ne.sum() < en.sum():
                        lost = en & ~ne
                        dead = [int(x) for x in np.flatnonzero(lost)]
                        trip = [kk for kk in range(G) if live[kk] and not ne[int(gbus[kk])]]
                        extra = "de-energises bus%s %s (%s MW load shed%s%s; slack then %s), " % (
                            "" if len(dead) == 1 else "es",
                            " ".join(str(x) for x in dead[:8])
                            + (" +%d more" % (len(dead) - 8) if len(dead) > 8 else ""),
                            f(float(loads[lost].sum()), 3),
                            "".join("; G%d %s MW tripped" % (kk, f(p[kk], 3)) for kk in trip),
                            ("; renewables %s MW lost" % f(float(renew[lost].sum()), 3))
                            if float(renew[lost].sum()) > 0.0 else "",
                            sf(snap(nc["slack"]), 4))
                    elif ne.sum() > en.sum():
                        extra = "re-energises %d bus%s (%s MW load restored), " % (
                            int(ne.sum() - en.sum()), "" if ne.sum() - en.sum() == 1 else "es",
                            f(float(loads[ne & ~en].sum()), 3))
                    if not cl and nc["act"][l]:
                        extra += "carries %s MW (%s), " % (
                            sf(nc["flow"][l], 3), pct(abs(nc["flow"][l]) / max(lim[l], 1e-12)))
                    row += " | %s: %sW %s %s" % ("open" if cl else "close", extra, dW, t)
                except Exception:
                    pass
                A(row)

        # ── phase shifters ──
        if P:
            A("PHASE SHIFTERS: [#] name line buses limit MW: shift deg (steps left), flow | per "
              "+5 deg: signed change of its own line's flow, W change | shift room alone before "
              "a NEW line limit binds (a line not now over reaching 100%) | +5 / -5 deg ->")
            for k in range(P):
                l = int(st["pst_line"][k])
                cur = float(st["pst_shift"][k])
                row = "[%d] PST%d L%d %d-%d limit %s: shift %s (%d steps down, %d up)" % (
                    G + S + k, k, l, int(fr[l]), int(to[l]), f(lim[l], 3),
                    sf(cur, 3) if cur != 0 else "0", int(round((cur + 30.0) / 5.0)),
                    int(round((30.0 - cur) / 5.0)))
                if not act[l]:
                    A(row + ", line out of service")
                    continue
                row += ", carries %s MW (%s)" % (sf(flow[l], 3), pct(loading[l]))
                try:
                    res = {}
                    dvec = None
                    for dstep in (5.0, -5.0):
                        nxt = _legal_shift(cur + dstep)
                        if nxt == cur:
                            continue
                        child = self.clone(st)
                        child["pst_shift"][k] = nxt
                        _, t, nc = self._after_txt(st, child, flow, over_set, watch, lim, m)
                        res[dstep] = t
                        if dvec is None:
                            dvec = (nc["flow"] - net["flow"]) * (1.0 if dstep > 0 else -1.0)
                    if dvec is not None:
                        row += " | per +5 deg: own %s, W %s" % (
                            dmw(dvec[l]), " ".join(dmw(sgn[w] * dvec[w]) for w in watch))
                        rng_txt = []
                        for sd, bound, bname in ((1.0, 30.0 - cur, "+30"), (-1.0, cur + 30.0, "-30")):
                            x, lx = self._first_limit(flow, lim, sd * dvec, ok)
                            if lx >= 0 and 5.0 * x < bound:
                                rng_txt.append("%s deg (L%d)" % (f(5.0 * x, 3), lx))
                            else:
                                rng_txt.append("%s deg (to %s)" % (f(bound, 3), bname))
                        row += " | range up %s / down %s" % tuple(rng_txt)
                    for dstep in (5.0, -5.0):
                        if dstep in res:
                            row += " | %s %s" % ("+5" if dstep > 0 else "-5", res[dstep])
                except Exception:
                    pass
                A(row)
        return "\n".join(out)


# ── module helpers ──────────────────────────────────────────────────────────

def _closed_mask(L: int, sw_line, sw_closed) -> np.ndarray:
    closed = np.ones(L, dtype=bool)
    sw_line = np.asarray(sw_line, dtype=int)
    if sw_line.size:
        closed[sw_line] = np.asarray(sw_closed, dtype=float) > 0.5
    return closed


def _legal_shift(v: float) -> float:
    v = min(max(float(v), -_SHIFT_MAX), _SHIFT_MAX)
    q = math.floor(v / _SHIFT_STEP + 0.5) * _SHIFT_STEP
    q = min(max(q, -_SHIFT_MAX), _SHIFT_MAX)
    return float(q) + 0.0


def _merit(on, pmin, pmax, cost, target) -> Optional[np.ndarray]:
    p = np.where(on, pmin, 0.0).astype(float)
    rem = target - float(p.sum())
    if rem < -1e-9:
        return None
    for g in np.argsort(cost, kind="stable"):
        if not on[g] or rem <= 1e-12:
            continue
        take = min(float(pmax[g] - p[g]), rem)
        p[g] += take
        rem -= take
    if rem > 1e-6:
        return None
    return p


def _merit_capped(on, pmin, pmax, cost, target) -> np.ndarray:
    """Merit order; if committed capacity is short, everything at p_max."""
    p = _merit(on, pmin, pmax, cost, target)
    if p is None:
        p = np.where(on, pmax, 0.0).astype(float)
    return p


def _spread(on, pmin, pmax, target) -> Optional[np.ndarray]:
    lo = float(pmin[on].sum())
    span = float((pmax - pmin)[on].sum())
    if span <= 0 or target < lo or target > lo + span:
        return None
    lam = (target - lo) / span
    return np.where(on, pmin + lam * (pmax - pmin), 0.0)


def _balance(p, on, pmin, pmax, target) -> np.ndarray:
    """Shift committed units proportionally within their ranges toward
    sum = target (as far as their ranges allow)."""
    p = np.array(p, dtype=float)
    gap = target - float(p.sum())
    if abs(gap) < 1e-12:
        return p
    room = np.where(on, (pmax - p) if gap > 0 else (p - pmin), 0.0)
    tot = float(room.sum())
    if tot <= 0:
        return p
    p = p + np.sign(gap) * room * min(1.0, abs(gap) / tot)
    return np.where(on, np.clip(p, pmin, pmax), 0.0)
