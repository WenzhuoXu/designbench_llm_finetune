"""Signal timing + lane allocation on a cell-transmission (CTM) road network,
with EXPLICIT time-stepped signals (v2: full action set).

v1 (dataxp/domains/traffic.py) averaged each signal over its cycle, so a stop
line discharged at ``capacity x green share`` every step and offsets could not
matter. Here every stop line discharges ONLY during its own effective green,
resolved per time step, so platoons form, offsets make or break green waves,
and the cycle length trades lost time against red waits and queue storage.

PHYSICS
  Network of signalised intersections, each with a MAIN approach (along the
  arterial / grid rows, west -> east) and a CROSS approach (side streets /
  grid columns, north -> south). Every approach is a link of L cells; the last
  cell is the stop line. Triangular fundamental diagram (Daganzo 1994), time
  step dt = 2.5 s, CFL = 1 (a free-flowing vehicle advances one cell per step,
  cell length ~35 m at 50 km/h). Per lane: saturation flow 0.5 veh/s
  (1.25 veh/step), jam storage 5 veh/cell, wave-speed ratio 0.5. Flow from
  cell j to its successor k:  y_j = min(S_j, R_k / through_j) with sending
  S_j = min(n_j, Q_j * green_j(t)) and receiving R_k = min(Q_k, 0.5 (N_k - n_k)).
  green_j(t) = 1 for interior cells; for a stop-line cell it is the fraction
  of [t, t + dt) that lies inside that approach's effective green window (exact
  periodic overlap, so any offset / split is resolved without rounding).
  At a stop line a fixed fraction of the discharge leaves the network (turns
  off into an unmodelled street); the rest enters the next link downstream
  (FIFO diverge: the downstream room limits the whole discharge).
  Network entries are fed from unbounded source queues (demand that cannot
  enter waits outside and counts as queued on that entry approach).
  Signals: two phases per cycle (MAIN, CROSS). Phase lengths g*C and (1-g)*C;
  each phase loses LOST_S = 7 s (yellow + all-red + start-up, lumped at the
  end of the phase), so effective greens are g*C - 7 and (1-g)*C - 7 s
  (floor 4 s). Some intersections have a reduced stop-line saturation flow
  (sat_factor 0.55-1, both approaches): the capacity-critical ones, whose
  flow ratios exceed what the initial cycle can serve at any split.
  The network starts in the free-flow state at nominal demand (every cell
  holds its stream's flow x dt), so there is no fill-up transient.

ELEMENTS (fixed list, index never changes): the intersections I0..I(n-1).
Every param applies to every intersection.

PARAMS (all set through SET_PARAM with float values)
  cycle        categorical, codes {60, 75, 90, 105, 120} = cycle length in s.
               set() rounds to the nearest code; the offset is then wrapped
               into [0, new cycle).
  green_frac   continuous in [0.2, 0.8]: MAIN phase share of the cycle (the
               CROSS phase gets the rest). Moves scale it (x0.8 .. x1.25).
  offset       circular additive, seconds in [0, cycle): time on the network
               clock (mod cycle) at which the FIRST phase of the cycle starts.
               Moves shift it by k * cycle/8, k in {-2, -1, +1, +2, +4},
               wrapping modulo the cycle.
  phase_order  toggle: 0 = MAIN phase first (MAIN green starts at the offset;
               changing green_frac moves the END of MAIN green), 1 = CROSS
               phase first (CROSS green starts at the offset, MAIN green starts
               (1-g)*C later and ends at the cycle boundary).
  lanes_main   categorical {1, 2, 3}: lanes on the MAIN link INTO this
               intersection; multiplies that link's saturation flow and its
               storage (all its cells). Every MAIN lane counts 1 against the
               network lane budget (constraint maximum_lanes_used). CROSS
               lanes are fixed by the problem.

METRICS (flat floats, all finite)
  mean_delay_s     objective: total stopped vehicle-seconds (incl. vehicles
                   held outside entries) / vehicles demanded over the horizon.
  max_queue_frac   max over approaches and time of (stopped vehicles on the
                   approach incl. entry backlog) / approach storage.
  throughput_frac  1 - (growth rate of the network's RESIDUAL queue in the
                   second half of the horizon) / (demand rate), clipped to
                   [0, 1]. The residual queue of an approach is its minimum
                   stopped stock over each of its own signal cycles (what is
                   left when its green ends); its growth is measured between
                   two windows of whole own cycles, centres 300 s apart. An
                   approach that clears every cycle reads 0 growth whatever
                   the upstream platoon pattern (intersections on different
                   cycles beat against each other with periods up to 840 s,
                   which a window MEAN of the queue mis-reads as growth); an
                   oversaturated one reads its arrival - capacity rate.
  lanes_used       total MAIN lanes (constraint vs the lane budget).
  max_density_err  max relative excursion of any cell above jam storage
                   (envelope; a correct CTM never leaves [0, N_jam]).

FAMILIES x SIZE LADDER (4 x 9 = 36 cells; v1 had 3 x 9)
  corridor    heavy arterial, decaying along the corridor as traffic turns
              off; sizes 4..24 intersections (an explicit-link arterial of 60
              would take longer to drive than the horizon).
  asymmetric  light arterial, bimodal side streets (half of them heavy).
  grid        rows x cols one-way grid (rows or columns heavy), blocks of
              4-10 cells; 2x2 .. 6x10 (4..60 intersections).
  downtown    same lattice with short blocks (3-5 cells): spillback-prone, the
              offsets decide whether platoons fit between stop lines.
  Every problem has 1-3 capacity-critical intersections (reduced saturation
  flow: their flow-ratio sum exceeds what the initial cycle can serve at ANY
  split, so only a longer cycle or a MAIN lane fixes them), a lane budget
  that is either slack (0-3 spare lanes) or already exceeded (35%: light
  links carry a 2nd lane that must be given back), and a per-problem stress
  level that grades the split error and the demand growth.
  Initial design = an existing plan: cycle 60 or 75 s everywhere,
  equisaturation splits from a stale count (per-intersection error), offset
  0 everywhere (no coordination), MAIN first. Demand is scaled until the
  start violates by at least a stress-dependent amount while the classical
  plan below (same lane budget) still satisfies every constraint with margin
  - so problems start infeasible and are provably solvable.

STOCHASTIC (traffic2_stoch): day-to-day demand variation - each entry's
demand is scaled, for the whole horizon, by an independent lognormal factor
(sigma 0.06, mean 1) drawn from np.random.default_rng(seed) only. (v1 used
per-interval factors, sigma 0.12 over 6 intervals - about the same spread of
the horizon-mean demand; with explicit queues those steps made the
second-half queue growth follow demand steps rather than capacity
shortfall. At sigma 0.10 the replicate spread of Phi left too few states
rankable: G2 measured 0.57 of problems, under the 0.70 bar.)

REFERENCE (reference_class "heuristic"): best feasible mean delay of a
classical plan evaluated at each common cycle (5 runs): lanes reallocated
greedily to minimise the worst flow-ratio sum (budget respected), Webster
equisaturation splits, free-flow progression offsets along MAIN (and down the
first column in grids). For the stochastic variant the plan must also satisfy
every constraint at the 1-in-10 high demand level.

SIMPLIFICATIONS (explicit)
  * No turning INTO the network: traffic only goes straight or leaves at a
    stop line (fixed fraction). Grids are one-way (W->E rows, N->S columns);
    the directed network is acyclic, so gridlock loops cannot form.
  * Two-phase signals, lost time lumped at the end of each phase, no
    pedestrian minimums, no actuation.
  * Lanes are a per-link multiplier of capacity and storage (no lane choice,
    no merge friction); a lane drop only acts through the receiving
    capacity of the narrower downstream link.
  * Reduced saturation flow at critical intersections is a fixed factor on
    both stop lines (no explicit pedestrian / parking model).
  * Deterministic fluid CTM (no microscopic randomness); the stochastic
    variant only draws one demand level per entry and run.
  * Throughput is a steady-state served share measured from queue growth, not
    a count of exits (long corridors would otherwise mix travel time into it).

Compatibility: python >= 3.8, numpy only, no file I/O, no prints.

Verb surface (docs/dataxp_verb_surface.md; owner's decision of 2026-10-23)
------------
The move set (candidate_moves), set(), Phi and the stored {i, p, v, v_old}
dicts are the v2 ones above and are untouched; only the spelling in prompts,
action lines and reply parsing changes (class attributes verb_surface and
verb_table; format_move / parse_move / grammar are the table-driven defaults
of dataxp.domain). The bijection, per parameter (i = intersection number):
  cycle v            <-> SET_CYCLE(i, v)                 [s], v in 60/75/90/105/120
  green_frac v       <-> SET_GREEN_SPLIT(i, v)           MAIN share of the cycle
  offset v           <-> SET_OFFSET(i, v)                [s] on the network clock
  phase_order 0 / 1  <-> SET_PHASE_ORDER(i, MAIN_FIRST / CROSS_FIRST)
  lanes_main v       <-> SET_LANES(i, v)                 lanes, 1 / 2 / 3
Judgment calls, recorded here as the surface rules ask:
  * The element argument of every verb is the intersection number SET_PARAM
    used (the number after I). SET_LANES names the intersection whose MAIN
    inbound link it re-stripes, not a link: the observation numbers
    intersections, not links.
  * cycle and lanes_main are categorical in the v2 sense (5 and 3 codes), but
    their stored values ARE the quantities (seconds, lanes), so they are
    spelled as those numbers through a Continuous codec: SET_CYCLE(3, 90),
    SET_LANES(3, 2). A Categorical codec indexes names by 0-based code and
    cannot carry 60..120 or 1..3, and a name table ("C90", "L2") would say
    nothing the number does not. Any number parses and set() rounds / clamps
    it exactly as before (a cycle of 100 rounds to 105, lanes 2.6 to 3), so
    the surface is one-to-one on the legal moves and admits the same values
    the SET_PARAM parser did; grammar() names the admitted values.
  * phase_order is a choice of which phase leads, not an on/off state, so it
    is a named argument (SET_PHASE_ORDER(i, CROSS_FIRST)) rather than a verb
    pair. The names are built from the observation's phase names MAIN and
    CROSS; the order column's M / X and the bare MAIN / CROSS are parse
    aliases, case ignored; format_move writes the canonical name only. A bare
    code (SET_PHASE_ORDER(3, 1)) is not accepted.
  * SET_OFFSET's value is the new absolute offset in seconds on the network
    clock, as SET_PARAM's value was, not a shift; set() still wraps it modulo
    the cycle.
  * Values print at 6 significant digits (fmt_arg), exactly as the SET_PARAM
    text did: cycles, lane counts and offsets (multiples of 1/8 s below
    120) are exact; a green split that repeated scaling has carried past 6
    digits reads back to within 1e-6 relative, as before.
  * SET_PARAM(...) text is no longer parsed on this domain: the grammar the
    policy is served lists the verbs only.
  * describe()'s "Tunable per intersection" paragraph is written in the verbs
    (observation text only; no generated file contains it), conditional on
    verb_surface; the per-intersection rows keep their cyc / g / off / order
    / lanes columns and the SET_PHASE_ORDER sentence ties MAIN_FIRST /
    CROSS_FIRST to the order column's M / X.
  * History and "change you are about to make" lines print the replaced value
    ("was 75", "order MAIN_FIRST -> CROSS_FIRST") for every verb: none is a
    toggle.

Set-parameter spelling (docs/dataxp_spellings.md; DATAXP_SPELLING=set_parameter)
-----------------------
The same moves through the one wrapper tool, generated from the verb table
by dataxp.domain (no override here): SET_CYCLE(3, 90) <-> SET_PARAM(3, cycle,
90), SET_GREEN_SPLIT(3, 0.55) <-> SET_PARAM(3, green_frac, 0.55),
SET_OFFSET(3, 12) <-> SET_PARAM(3, offset, 12), SET_PHASE_ORDER(3,
CROSS_FIRST) <-> SET_PARAM(3, phase_order, 1) (the stored code; the grammar
names the codes), SET_LANES(3, 2) <-> SET_PARAM(3, lanes_main, 2). Judgment
calls: the phrases are "Set a signal's cycle length / green split / offset /
phase order" and "Re-stripe a signal's MAIN lanes"; the "Tunable per
intersection" paragraph reverts to its SET_PARAM wording (the run2 text)
under this spelling (spelled_in_verbs); the history "was ..." prints the
stored phase_order code.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..domain import (ArgSpec, Candidate, Categorical, Continuous, Domain, ParamVerb,
                      spelled_in_verbs)
from ..potential import DesignProgram, program_from_goals

# ── fixed physics / discretisation ───────────────────────────────────────────
DT_S = 2.5                      # time step (s); every cycle code is a multiple
SAT_LANE = 0.5                  # saturation flow per lane (veh/s)
Q_LANE = SAT_LANE * DT_S        # per-lane capacity per cell boundary (veh/step)
N_LANE = 5.0                    # jam storage per lane per cell (veh)
DELTA = 0.5                     # wave-speed ratio w/v
LOST_S = 7.0                    # lost time per phase (s)
MIN_EFF_S = 4.0                 # floor on effective green (s)
CYCLES = (60.0, 75.0, 90.0, 105.0, 120.0)
G_MIN, G_MAX = 0.2, 0.8
LANES = (1.0, 2.0, 3.0)
OFFSET_DIV = 8                  # offset move granularity: cycle / 8
OFFSET_SHIFTS = (-2, -1, 1, 2, 4)
GREEN_FACTORS = (0.8, 0.9, 1.1, 1.25)
HORIZON_S = 900.0
N_INTERVALS = 6                 # demand array layout (levels constant in v2)
NOISE_SIGMA = 0.06              # daily level; ~v1's 0.12 averaged over 6 intervals
BIG = 1e12
DELAY_FLOOR_S = 1e-3
PARAMS = ("cycle", "green_frac", "offset", "phase_order", "lanes_main")
# verb-surface names of the phase_order codes (docstring "Verb surface")
ORDER_NAMES = ("MAIN_FIRST", "CROSS_FIRST")
_INT_ARG = ArgSpec("intersection", "int", "intersection number 0..n-1 (the number after I)")

_FAMILIES = ("corridor", "asymmetric", "grid", "downtown")
_CORR_SIZES = (4, 6, 8, 10, 12, 14, 16, 20, 24)
_GRID_DIMS = ((2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 6), (6, 6),
              (6, 8), (6, 10))

# candidate quotas when a network offers more moves than max_candidates
_QUOTA = {"cycle": 0.18, "green_frac": 0.22, "offset": 0.26,
          "phase_order": 0.12, "lanes_main": 0.22}


def _usable(cycle: float) -> float:
    """Max sum of flow ratios (MAIN + CROSS) one intersection can serve."""
    return 1.0 - 2.0 * LOST_S / float(cycle)


def _failure_metrics(lanes_used: float) -> dict:
    """Finite, maximally bad readout (outside the validity envelope)."""
    return {
        "mean_delay_s": 2.0 * HORIZON_S,
        "max_queue_frac": 10.0,
        "throughput_frac": 0.0,
        "lanes_used": float(lanes_used),
        "max_density_err": 1.0,
    }


def _nearest_cycle(v: float) -> float:
    return min(CYCLES, key=lambda c: (abs(c - v), c))


# ── topology ──────────────────────────────────────────────────────────────────
def _build_topo(spec: dict) -> dict:
    """Flat cell arrays for one problem. Approach 2i = MAIN into Ii, 2i+1 =
    CROSS into Ii. Entry approaches carry one extra leading SOURCE cell
    (unbounded storage) that holds demand waiting to enter."""
    n = int(spec["n_int"])
    A = 2 * n
    L = np.zeros(A, dtype=np.int64)          # real cells per approach
    feeder = np.full(A, -1, dtype=np.int64)  # approach whose stop line feeds a
    thru = np.ones(A, dtype=float)           # share of a's discharge that goes on
    rate = np.zeros(A, dtype=float)          # entry demand (veh/s), 0 if fed
    main_cells = [int(v) for v in spec["main_cells"]]
    cross_cells = [int(v) for v in spec["cross_cells"]]
    main_turn = [float(v) for v in spec["main_turn"]]
    cross_turn = [float(v) for v in spec["cross_turn"]]
    for i in range(n):
        L[2 * i] = main_cells[i]
        L[2 * i + 1] = cross_cells[i]
    if spec["layout"] == "grid":
        rows, cols = int(spec["rows"]), int(spec["cols"])
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                if c > 0:
                    feeder[2 * i] = 2 * (i - 1)
                else:
                    rate[2 * i] = float(spec["main_rate"][r])
                if r > 0:
                    feeder[2 * i + 1] = 2 * (i - cols) + 1
                else:
                    rate[2 * i + 1] = float(spec["cross_rate"][c])
                if c < cols - 1:
                    thru[2 * i] = 1.0 - main_turn[i]
                if r < rows - 1:
                    thru[2 * i + 1] = 1.0 - cross_turn[i]
    else:
        for i in range(n):
            if i > 0:
                feeder[2 * i] = 2 * (i - 1)
            else:
                rate[0] = float(spec["main_rate"][0])
            rate[2 * i + 1] = float(spec["cross_rate"][i])
            if i < n - 1:
                thru[2 * i] = 1.0 - main_turn[i]
    dest = np.full(A, -1, dtype=np.int64)
    for a in range(A):
        if feeder[a] >= 0:
            dest[int(feeder[a])] = a
    is_entry = feeder < 0
    # nominal flow arriving on every approach (veh/s), in feed order
    flow = np.zeros(A, dtype=float)
    order = _feed_order(feeder)
    for a in order:
        f = int(feeder[a])
        flow[a] = rate[a] if f < 0 else flow[f] * thru[f]

    ncell = L + is_entry.astype(np.int64)    # incl. source cell
    start = np.zeros(A, dtype=np.int64)
    start[1:] = np.cumsum(ncell)[:-1]
    Nc = int(ncell.sum())
    stop_idx = start + ncell - 1
    first_real = start + is_entry.astype(np.int64)
    src_cells = start[is_entry]
    entry_appr = np.nonzero(is_entry)[0]
    cell_appr = np.repeat(np.arange(A), ncell)
    is_src = np.zeros(Nc, dtype=bool)
    is_src[src_cells] = True

    nxt = np.arange(1, Nc + 1, dtype=np.int64)   # default: next cell
    pred = np.arange(-1, Nc - 1, dtype=np.int64)  # default: previous cell
    for a in range(A):
        d = int(dest[a])
        nxt[stop_idx[a]] = start[d] if d >= 0 else Nc   # Nc = sentinel
        f = int(feeder[a])
        pred[start[a]] = stop_idx[f] if f >= 0 else Nc
    thru_ext = np.ones(Nc + 1, dtype=float)
    thru_ext[stop_idx] = thru
    thru_ext[Nc] = 0.0
    rinv = np.ones(Nc, dtype=float)
    rinv[stop_idx] = 1.0 / np.maximum(thru, 1e-6)
    cell_flow = flow[cell_appr].copy()
    cell_flow[is_src] = 0.0
    steps = int(round(float(spec.get("horizon_s", HORIZON_S)) / DT_S))
    return {
        "n_int": n, "A": A, "Nc": Nc, "steps": steps,
        "L": L, "feeder": feeder, "dest": dest, "thru": thru, "rate": rate,
        "flow": flow, "is_entry": is_entry, "start": start,
        "stop_idx": stop_idx, "first_real": first_real,
        "src_cells": src_cells, "entry_appr": entry_appr,
        "entry_rate": rate[entry_appr].copy(),
        "cell_appr": cell_appr, "is_src": is_src, "real": ~is_src,
        "nxt": nxt, "pred": pred, "thru_ext": thru_ext, "rinv": rinv,
        "cell_flow": cell_flow,
        "lanes_x": np.asarray([float(v) for v in spec["cross_lanes"]]),
        "sat": np.asarray([float(v) for v in spec.get("sat_factor", [1.0] * n)]),
    }


def _feed_order(feeder: np.ndarray) -> List[int]:
    """Approaches ordered so every feeder precedes the approach it feeds."""
    A = len(feeder)
    done = np.zeros(A, dtype=bool)
    order: List[int] = []
    while len(order) < A:
        progressed = False
        for a in range(A):
            if done[a]:
                continue
            f = int(feeder[a])
            if f < 0 or done[f]:
                order.append(a)
                done[a] = True
                progressed = True
        if not progressed:      # cannot happen for the acyclic layouts here
            order.extend(int(a) for a in np.nonzero(~done)[0])
            break
    return order


def _green_matrix(steps: int, start: np.ndarray, glen: np.ndarray,
                  cyc: np.ndarray) -> np.ndarray:
    """F[t, a] = fraction of [t*dt, (t+1)*dt) inside approach a's periodic
    effective-green window [start + k*C, start + k*C + glen)."""
    tb = np.arange(steps + 1, dtype=float)[:, None] * DT_S
    x = tb - start[None, :]
    k = np.floor(x / cyc[None, :])
    r = x - k * cyc[None, :]
    G = k * glen[None, :] + np.minimum(r, glen[None, :])
    return np.clip(np.diff(G, axis=0) / DT_S, 0.0, 1.0)


class Traffic2Domain(Domain):
    """Explicit-signal CTM network (variants: traffic2, traffic2_stoch)."""
    # stochastic variants: phi_select compares candidates with a fresh
    # incumbent estimate (dataxp/ladder.py); no effect when deterministic
    rescore_incumbent = True

    name = "traffic2"
    params = PARAMS
    factors = GREEN_FACTORS
    reference_class = "heuristic"
    max_elements = 60
    max_candidates = 160

    # ── verb surface (docstring "Verb surface"; docs/dataxp_verb_surface.md) ──
    # Spelling only: every SET_PARAM move candidate_moves enumerates has one
    # verb text and parses back to the same move. format_move / parse_move /
    # grammar are the table-driven defaults of dataxp.domain.
    verb_surface = True
    verb_table = (
        ParamVerb("cycle", "SET_CYCLE", _INT_ARG,
                  Continuous("cycle", "s", "one of %s (another value rounds to the nearest); "
                                           "the new cycle length"
                             % " | ".join("%d" % c for c in CYCLES)),
                  doc="set the signal cycle length; the green split is kept (both effective "
                      "greens change) and the offset is wrapped modulo the new cycle.",
                  phrase="Set a signal's cycle length"),
        ParamVerb("green_frac", "SET_GREEN_SPLIT", _INT_ARG,
                  Continuous("split", "", "%g..%g, the new MAIN-phase share of the cycle "
                                          "(CROSS gets the rest; a value outside is set to "
                                          "the bound)" % (G_MIN, G_MAX)),
                  doc="split the cycle between the MAIN and CROSS phases.",
                  phrase="Set a signal's green split"),
        ParamVerb("offset", "SET_OFFSET", _INT_ARG,
                  Continuous("offset", "s", "0..cycle, the new network-clock time at which "
                                            "the first phase of the cycle starts (a time, "
                                            "not a shift; wraps modulo the cycle)"),
                  doc="retime the cycle against the network clock, to coordinate with the "
                      "neighbouring signals (green waves).",
                  phrase="Set a signal's offset"),
        ParamVerb("phase_order", "SET_PHASE_ORDER", _INT_ARG,
                  Categorical("order", ORDER_NAMES,
                              aliases=(("M", 0), ("X", 1), ("MAIN", 0), ("CROSS", 1)),
                              doc="MAIN_FIRST = MAIN green starts at the offset and CROSS "
                                  "split x cycle later (the rows' order M); CROSS_FIRST = "
                                  "CROSS green starts at the offset and MAIN (1 - split) x "
                                  "cycle later (order X)."),
                  doc="choose which phase starts at the offset; the other follows it.",
                  phrase="Set a signal's phase order"),
        ParamVerb("lanes_main", "SET_LANES", _INT_ARG,
                  Continuous("lanes", "lanes", "%d, %d or %d (a whole number; another value "
                                               "rounds and clamps), the new lane count of "
                                               "the MAIN link into the intersection"
                             % tuple(int(v) for v in LANES)),
                  doc="re-stripe the MAIN link into the intersection: its saturation flow "
                      "and storage scale with the lanes, and every MAIN lane counts 1 "
                      "against the lane budget.",
                  phrase="Re-stripe a signal's MAIN lanes"),
    )

    def __init__(self, stochastic: bool = False):
        super().__init__(stochastic)

    # ── problems ──────────────────────────────────────────────────────────────
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[index % len(_FAMILIES)]
        rung = (index // len(_FAMILIES)) % len(_CORR_SIZES)
        base = self._draw_structure(family, rung, rng)
        kappa = float(base["kappa0"])
        v_target = float(base["v_target"])
        best = None
        last = None
        for _attempt in range(12):
            spec = self._spec_from(base, kappa)
            heur_ok, ref, ref_cycle = self._heuristic(spec)
            v0 = self._initial_violation(spec)
            last = (spec, ref, ref_cycle)
            if heur_ok:
                best = (spec, ref, ref_cycle)
                if v0 >= v_target:
                    break               # infeasible start, provably solvable
                kappa *= 1.05           # (nearly) feasible start: load it up
            else:
                if best is not None:
                    break               # previous (easier) one stands
                kappa *= 0.93
        spec, ref, ref_cycle = best if best is not None else last
        spec["_ref_mean_delay_s"] = float(ref)
        spec["_ref_cycle"] = float(ref_cycle)
        spec["problem_id"] = "traffic2_%s_%04d" % (family, index)
        return spec

    def _draw_structure(self, family: str, rung: int,
                        rng: np.random.Generator) -> dict:
        """All random draws for one problem (consumed once, before calibration)."""
        C0 = float(rng.choice((60.0, 75.0), p=(0.65, 0.35)))
        lam0 = 0.5 - LOST_S / C0              # MAIN effective share at g = 0.5
        U0 = _usable(C0)
        goals = {
            "maximum_max_queue_frac": round(float(rng.uniform(0.85, 0.95)), 4),
            "minimum_throughput_frac": round(float(rng.uniform(0.955, 0.98)), 4),
        }
        stress = float(rng.uniform(0.0, 1.0))      # graded difficulty
        s = {"family": family, "C0": C0, "goals": goals, "stress": stress}
        if family in ("corridor", "asymmetric"):
            n = int(_CORR_SIZES[rung])
            s.update(layout="corridor", n_int=n)
            s["main_cells"] = ([int(rng.integers(5, 9))]
                               + [int(v) for v in rng.integers(4, 11, size=n - 1)])
            s["cross_cells"] = [int(v) for v in rng.integers(4, 9, size=n)]
            turn = rng.uniform(0.04, 0.16, size=n)
            s["main_turn"] = [round(float(v), 4) for v in turn]
            s["cross_turn"] = [0.0] * n
            s["cross_lanes"] = [1.0] * n
            keep = np.concatenate([[1.0], np.cumprod(1.0 - turn)[:-1]])
            if family == "corridor":
                ym0 = lam0 + float(rng.uniform(0.03, 0.14))
                ym = ym0 * keep
                yx = rng.uniform(0.10, 0.26, size=n)
            else:
                ym0 = float(rng.uniform(0.18, 0.30))
                ym = ym0 * keep
                heavy = rng.random(n) < 0.5
                if not heavy.any():
                    heavy[int(rng.integers(n))] = True
                yx = np.where(heavy, lam0 + rng.uniform(0.03, 0.12, size=n),
                              rng.uniform(0.05, 0.14, size=n))
            yx = np.clip(np.minimum(yx, U0 - 0.06 - ym), 0.05, 0.6)
            s["main_ratio"] = [float(ym0)]
            s["cross_ratio"] = [float(v) for v in yx]
            Y = ym + yx
        else:
            rows, cols = _GRID_DIMS[rung]
            n = rows * cols
            s.update(layout="grid", n_int=n, rows=rows, cols=cols)
            lo, hi = (4, 11) if family == "grid" else (3, 6)
            col_gap = [0] + [int(v) for v in rng.integers(lo, hi, size=cols - 1)]
            row_gap = [0] + [int(v) for v in rng.integers(lo, hi, size=rows - 1)]
            ent_m = [int(v) for v in rng.integers(5, 9, size=rows)]
            ent_x = [int(v) for v in rng.integers(5, 9, size=cols)]
            mc, xc = [], []
            for r in range(rows):
                for c in range(cols):
                    mc.append(ent_m[r] if c == 0 else col_gap[c])
                    xc.append(ent_x[c] if r == 0 else row_gap[r])
            s["main_cells"] = mc
            s["cross_cells"] = xc
            s["col_gap"] = col_gap
            s["row_gap"] = row_gap
            mt = rng.uniform(0.05, 0.18, size=n)
            xt = rng.uniform(0.05, 0.18, size=n)
            s["main_turn"] = [round(float(v), 4) for v in mt]
            s["cross_turn"] = [round(float(v), 4) for v in xt]
            s["cross_lanes"] = [1.0] * n
            rows_heavy = bool(rng.random() < 0.5)
            n_heavy_dir = rows if rows_heavy else cols
            n_light_dir = cols if rows_heavy else rows
            yh = lam0 + float(rng.uniform(0.02, 0.12))
            raw_h = rng.uniform(0.40, 0.72, size=n_heavy_dir)
            n_top = 1 if n_heavy_dir <= 3 else 2
            top = rng.choice(n_heavy_dir, size=n_top, replace=False)
            raw_h[top] = rng.uniform(0.92, 1.0, size=n_top)
            yl = float(rng.uniform(0.14, 0.26))
            raw_l = rng.uniform(0.6, 1.0, size=n_light_dir)
            heavy = [float(v) for v in yh * raw_h]
            light = [float(v) for v in yl * raw_l]
            s["rows_heavy"] = rows_heavy
            s["main_ratio"] = heavy if rows_heavy else light
            s["cross_ratio"] = light if rows_heavy else heavy
            mt2 = np.asarray(s["main_turn"]).reshape(rows, cols)
            xt2 = np.asarray(s["cross_turn"]).reshape(rows, cols)
            keep_m = np.concatenate([np.ones((rows, 1)),
                                     np.cumprod(1.0 - mt2, axis=1)[:, :-1]], axis=1)
            keep_x = np.concatenate([np.ones((1, cols)),
                                     np.cumprod(1.0 - xt2, axis=0)[:-1, :]], axis=0)
            ym = (np.asarray(s["main_ratio"])[:, None] * keep_m).ravel()
            yx = (np.asarray(s["cross_ratio"])[None, :] * keep_x).ravel()
            Y = ym + yx
        # capacity-critical intersections: a reduced saturation flow (narrow
        # lanes, parking, pedestrians) puts their flow-ratio sum above what
        # the initial cycle can serve but below what a longer one (or a lane)
        # can - a split alone cannot fix them.
        n = int(s["n_int"])
        crit_lo = U0 + 0.035
        crit_hi = max(crit_lo + 0.01, _usable(120.0) - 0.02)
        k_crit = int(rng.choice((1, 1, 2, 2, 3)))
        k_crit = min(k_crit, max(1, n // 2))
        sat = np.ones(n)
        if k_crit:
            w = np.maximum(Y, 1e-6) ** 6
            crit = rng.choice(n, size=k_crit, replace=False, p=w / w.sum())
            for i in crit:
                yt = float(rng.uniform(crit_lo, crit_hi))
                sat[int(i)] = float(np.clip(Y[int(i)] / yt, 0.55, 1.0))
        s["sat_factor"] = [round(float(v), 4) for v in sat]
        # existing timing plan: equisaturation splits for the cycle in place,
        # computed from a stale count (per-intersection error), no
        # coordination (offset 0 everywhere)
        ymf = ym / sat
        yxf = yx / sat
        eff = C0 - 2.0 * LOST_S
        g_eq = (eff * ymf / np.maximum(ymf + yxf, 1e-9) + LOST_S) / C0
        g0 = g_eq + rng.normal(0.0, 0.03 + 0.10 * stress, size=n)
        s["green0"] = [round(float(v), 2) for v in np.clip(g0, 0.25, 0.75)]
        # lane budget scenario
        n = int(s["n_int"])
        lanes = [1.0] * n
        if rng.random() < 0.35:
            # starts OVER budget: some light MAIN links carry a 2nd lane
            k2 = int(rng.integers(1, 3))
            s["over_budget"] = True
            s["over_k"] = k2
            s["over_by"] = int(rng.integers(1, k2 + 1))
        else:
            s["over_budget"] = False
            s["slack"] = int(rng.choice((0, 1, 1, 2, 2, 3)))
        s["lanes0"] = lanes
        # demand growth since the plan was timed, and the minimum initial
        # violation the calibration loads the problem up to
        s["kappa0"] = 0.97 + 0.25 * stress
        s["v_target"] = 0.05 + 0.5 * stress
        return s

    def _spec_from(self, s: dict, kappa: float) -> dict:
        """Materialise a JSON spec at demand scale kappa."""
        family = s["family"]
        n = int(s["n_int"])
        spec = {
            "problem_id": "",
            "family": family,
            "layout": s["layout"],
            "goals": dict(s["goals"]),
            "horizon_s": HORIZON_S,
            "dt_s": DT_S,
            "n_int": n,
            "main_cells": list(s["main_cells"]),
            "cross_cells": list(s["cross_cells"]),
            "main_turn": list(s["main_turn"]),
            "cross_turn": list(s["cross_turn"]),
            "cross_lanes": list(s["cross_lanes"]),
            "sat_factor": list(s["sat_factor"]),
            "demand_scale": round(float(kappa), 6),
        }
        if s["layout"] == "grid":
            spec.update(rows=int(s["rows"]), cols=int(s["cols"]),
                        col_gap=list(s["col_gap"]), row_gap=list(s["row_gap"]),
                        rows_heavy=bool(s["rows_heavy"]))
            spec["main_rate"] = [round(float(v) * SAT_LANE * kappa, 6)
                                 for v in s["main_ratio"]]
            spec["cross_rate"] = [round(float(v) * SAT_LANE * kappa, 6)
                                  for v in s["cross_ratio"]]
        else:
            spec["main_rate"] = [round(float(s["main_ratio"][0]) * SAT_LANE
                                       * kappa, 6)]
            spec["cross_rate"] = [round(float(v) * SAT_LANE * kappa, 6)
                                  for v in s["cross_ratio"]]
        # lanes / budget
        lanes = [float(v) for v in s["lanes0"]]
        if s.get("over_budget"):
            topo = _build_topo(dict(spec, cross_lanes=spec["cross_lanes"]))
            fm = topo["flow"][0::2]
            light = [int(i) for i in np.argsort(fm, kind="stable")[: int(s["over_k"])]]
            for i in light:
                lanes[i] = 2.0
            budget = int(round(sum(lanes))) - int(s["over_by"])
        else:
            budget = int(round(sum(lanes))) + int(s["slack"])
        spec["goals"]["maximum_lanes_used"] = float(budget)
        spec["initial"] = {
            "cycle": float(s["C0"]),
            "green_frac": list(s["green0"]),
            "offset": 0.0,
            "phase_order": 0.0,
            "lanes_main": lanes,
        }
        return spec

    # ── state ────────────────────────────────────────────────────────────────
    def load(self, spec: dict) -> dict:
        n = int(spec["n_int"])
        ini = spec["initial"]
        topo = _build_topo(spec)
        return {
            "spec": spec,
            "n_int": n,
            "cycle": np.full(n, float(ini["cycle"])),
            "green": self._init_green(ini, n),
            "offset": np.full(n, float(ini["offset"])),
            "phase": np.full(n, float(ini["phase_order"])),
            "lanes": np.asarray([float(v) for v in ini["lanes_main"]], dtype=float),
            "_topo": topo,            # read-only, shared across clones
            "_diag": None,
        }

    @staticmethod
    def _init_green(ini: dict, n: int) -> np.ndarray:
        g = ini["green_frac"]
        if isinstance(g, (list, tuple)):
            return np.asarray([float(v) for v in g], dtype=float)
        return np.full(n, float(g))

    def clone(self, st) -> dict:
        return {
            "spec": st["spec"],
            "n_int": st["n_int"],
            "cycle": np.array(st["cycle"], dtype=float, copy=True),
            "green": np.array(st["green"], dtype=float, copy=True),
            "offset": np.array(st["offset"], dtype=float, copy=True),
            "phase": np.array(st["phase"], dtype=float, copy=True),
            "lanes": np.array(st["lanes"], dtype=float, copy=True),
            "_topo": st["_topo"],
            "_diag": None,
        }

    # ── simulation ────────────────────────────────────────────────────────────
    def _signal_arrays(self, st) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-approach (green start s, effective green length, cycle)."""
        C = np.asarray(st["cycle"], dtype=float)
        g = np.asarray(st["green"], dtype=float)
        o = np.asarray(st["offset"], dtype=float)
        ph = np.asarray(st["phase"], dtype=float) > 0.5
        gm = np.maximum(g * C - LOST_S, MIN_EFF_S)
        gx = np.maximum((1.0 - g) * C - LOST_S, MIN_EFF_S)
        sm = np.where(ph, o + (1.0 - g) * C, o)
        sx = np.where(ph, o, o + g * C)
        A = 2 * len(C)
        s = np.empty(A)
        s[0::2] = sm
        s[1::2] = sx
        gl = np.empty(A)
        gl[0::2] = gm
        gl[1::2] = gx
        return s, gl, np.repeat(C, 2)

    @staticmethod
    def _high_day() -> float:
        """1-in-10 high demand factor of one entry (lognormal, mean 1)."""
        return math.exp(1.2816 * NOISE_SIGMA - 0.5 * NOISE_SIGMA * NOISE_SIGMA)

    def _simulate(self, st, seed: int, diag: bool = False,
                  nominal: bool = False, demand_mult: float = 1.0):
        topo = st["_topo"]
        A = int(topo["A"])
        Nc = int(topo["Nc"])
        steps = int(topo["steps"])
        lanes_a = np.empty(A)
        lanes_a[0::2] = np.asarray(st["lanes"], dtype=float)
        lanes_a[1::2] = topo["lanes_x"]
        lanes_c = lanes_a[topo["cell_appr"]]
        Q = Q_LANE * lanes_c
        N = N_LANE * lanes_c
        is_src = topo["is_src"]
        Q[is_src] = BIG
        N[is_src] = BIG
        DN = DELTA * N
        stop_idx = topo["stop_idx"]

        s_g, glen, cyc = self._signal_arrays(st)
        F = _green_matrix(steps, s_g, glen, cyc)            # (steps, A)
        Cs = F * (Q[stop_idx] * np.repeat(topo["sat"], 2))[None, :]

        base = topo["entry_rate"] * (DT_S * float(demand_mult))  # veh per step
        E = len(base)
        if self.stochastic and not nominal:
            # day-to-day demand variation: one lognormal level per entry for
            # the whole horizon (mean 1). Constant within a run, so the
            # throughput reading measures capacity shortfall, not the queue
            # level following a demand step.
            rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
            fac = rng.lognormal(0.0, NOISE_SIGMA, size=E)
            fac /= math.exp(0.5 * NOISE_SIGMA * NOISE_SIGMA)
            d_by_p = np.broadcast_to(base * fac, (N_INTERVALS, E)).copy()
        else:
            d_by_p = np.broadcast_to(base, (N_INTERVALS, E)).copy()
        interval_len = max(1, int(math.ceil(steps / float(N_INTERVALS))))
        counts = np.array([max(0, min(steps, (p + 1) * interval_len)
                               - p * interval_len) for p in range(N_INTERVALS)],
                          dtype=float)
        demanded = float((counts[:, None] * d_by_p).sum())

        n = np.minimum(topo["cell_flow"] * DT_S, 0.8 * N)   # free-flow warm start
        n[is_src] = 0.0
        cap = Q.copy()
        R_ext = np.full(Nc + 1, BIG)
        y_ext = np.zeros(Nc + 1)
        R = R_ext[:Nc]
        y = y_ext[:Nc]
        S = np.empty(Nc)
        tmp = np.empty(Nc)
        stp = np.empty(Nc)
        yt = np.empty(Nc + 1)
        Qh = np.empty((steps, A))
        nmax = n.copy()
        nxt = topo["nxt"]
        pred = topo["pred"]
        rinv = topo["rinv"]
        thru_ext = topo["thru_ext"]
        src_cells = topo["src_cells"]
        astart = topo["start"]

        if diag:
            first_real = topo["first_real"]
            aog_num = np.zeros(A)
            aog_den = np.zeros(A)
            dis = np.zeros(A)
            arr = np.zeros(A)
            blk = np.zeros(A)

        p_cur = 0
        d_cur = d_by_p[0]
        for t in range(steps):
            p = t // interval_len
            if p != p_cur and p < N_INTERVALS:
                p_cur = p
                d_cur = d_by_p[p]
            n[src_cells] += d_cur
            cap[stop_idx] = Cs[t]
            np.minimum(n, cap, out=S)
            np.multiply(n, -DELTA, out=tmp)
            tmp += DN
            np.minimum(tmp, Q, out=R)
            Rn = R_ext[nxt]
            Rn *= rinv
            np.minimum(S, Rn, out=y)
            np.subtract(n, y, out=stp)
            np.add.reduceat(stp, astart, out=Qh[t])
            np.multiply(y_ext, thru_ext, out=yt)
            inflow = yt[pred]
            np.add(stp, inflow, out=n)
            np.maximum(nmax, n, out=nmax)
            if diag:
                ys = y[stop_idx]
                dis += ys
                blk += ys < S[stop_idx] - 1e-9
                ins = inflow[stop_idx]
                tn = t + 1 if t + 1 < steps else t
                aog_num += ins * F[tn]
                aog_den += ins
                arr += inflow[first_real]

        # ── metrics ────────────────────────────────────────────────────────
        H = steps * DT_S
        B = Qh.sum(axis=1)
        total_delay = float(B.sum()) * DT_S
        mean_delay = max(total_delay / max(demanded, 1e-9), DELAY_FLOOR_S)

        L = topo["L"]
        storage = L * N_LANE * lanes_a
        qpeak = Qh.max(axis=0)
        qfrac = qpeak / storage

        # Steady-state served share from the growth of the RESIDUAL queue: per
        # approach, the minimum stopped stock over each of its own signal
        # cycles (the queue left when its green ends; zero for an approach
        # that clears every cycle), averaged over whole own cycles in a late
        # and an early window whose centres are K steps apart. Window MEANS of
        # the stock (the first version) mis-read intersections on different
        # cycles: the upstream platoon pattern beats against the own cycle with
        # periods up to 840 s, so the mean queue drifts within the horizon
        # without growing. On stable mixed-cycle networks that read as 0.94-1
        # (error up to 6% of demand either way, i.e. as large as the
        # constraint margin); the residual queue reads >= 0.993. Common-cycle
        # periodic states read exactly the same under both.
        P = np.maximum(1, np.round(cyc / DT_S).astype(np.int64))
        K = steps // 3
        W = steps // 4
        early_e = steps - K
        growth = np.zeros(A)
        for Pv in np.unique(P):
            Pv = int(Pv)
            cols = np.nonzero(P == Pv)[0]
            m_l = max(1, min(W // Pv, steps // Pv))
            m_e = max(1, min(W // Pv, early_e // Pv))
            if early_e < Pv:          # horizon too short to compare windows
                continue
            late_r = (Qh[steps - m_l * Pv:steps][:, cols]
                      .reshape(m_l, Pv, len(cols)).min(axis=1).mean(axis=0))
            early_r = (Qh[early_e - m_e * Pv:early_e][:, cols]
                       .reshape(m_e, Pv, len(cols)).min(axis=1).mean(axis=0))
            # centre-to-centre distance of the two windows, in steps
            sep = K + 0.5 * (m_e - m_l) * Pv
            growth[cols] = (late_r - early_r) / (max(sep, 1.0) * DT_S)
        d_rate = demanded / H
        thr = 1.0 - float(growth.sum()) / max(d_rate, 1e-9)
        thr = min(1.0, max(0.0, thr))

        real = topo["real"]
        dens_err = float(np.max((nmax[real] - N[real]) / N[real]))
        metrics = {
            "mean_delay_s": float(mean_delay),
            "max_queue_frac": float(qfrac.max()),
            "throughput_frac": float(thr),
            "lanes_used": float(np.sum(st["lanes"])),
            "max_density_err": max(0.0, dens_err),
        }
        out_diag = None
        if diag:
            out_diag = {
                "qfrac": qfrac,
                "qpeak": qpeak,
                "storage": storage,
                "growth": growth,
                "dly": Qh.sum(axis=0) * DT_S / np.maximum(dis, 1e-9),
                "dis_rate": dis / H,
                "arr_rate": arr / H,
                "aog": aog_num / np.maximum(aog_den, 1e-9),
                "blk_frac": blk / float(max(steps, 1)),
                "lanes_a": lanes_a,
                "glen": glen,
            }
        return metrics, out_diag

    def evaluate(self, st, seed: int = 0) -> dict:
        self.calls += 1
        try:
            metrics, _ = self._simulate(st, seed, diag=False)
        except Exception:
            metrics = _failure_metrics(float(np.sum(st["lanes"])))
        out = {}
        bad = _failure_metrics(float(np.sum(st["lanes"])))
        for k, v in metrics.items():
            fv = float(v)
            if not math.isfinite(fv):        # never let NaN/inf through
                fv = bad[k]
            out[k] = fv
        return out

    def _diag(self, st) -> dict:
        """Nominal (mean-demand) instrumented run, cached on the state."""
        d = st.get("_diag")
        if d is None:
            try:
                metrics, d = self._simulate(st, 0, diag=True, nominal=True)
                d["metrics"] = metrics
            except Exception:
                d = {}
            st["_diag"] = d
        return d

    # ── design variables ──────────────────────────────────────────────────────
    def n_elements(self, st) -> int:
        return int(st["n_int"])

    def get(self, st, i: int, p: str) -> Optional[float]:
        i = int(i)
        if not (0 <= i < int(st["n_int"])):
            return None
        if p == "cycle":
            return float(st["cycle"][i])
        if p == "green_frac":
            return float(st["green"][i])
        if p == "offset":
            return float(st["offset"][i])
        if p == "phase_order":
            return float(st["phase"][i])
        if p == "lanes_main":
            return float(st["lanes"][i])
        return None

    def set(self, st, i: int, p: str, v: float) -> None:
        i = int(i)
        if not (0 <= i < int(st["n_int"])):
            return
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v):
            return
        if p == "cycle":
            c = _nearest_cycle(v)
            st["cycle"][i] = c
            st["offset"][i] = float(st["offset"][i]) % c
        elif p == "green_frac":
            st["green"][i] = min(G_MAX, max(G_MIN, v))
        elif p == "offset":
            st["offset"][i] = v % float(st["cycle"][i])
        elif p == "phase_order":
            st["phase"][i] = 1.0 if v >= 0.5 else 0.0
        elif p == "lanes_main":
            st["lanes"][i] = float(min(LANES[-1], max(LANES[0], round(v))))
        else:
            return
        st["_diag"] = None            # invalidate cached diagnostics

    def bounds(self, st, p: str) -> Tuple[float, float]:
        if p == "cycle":
            return (CYCLES[0], CYCLES[-1])
        if p == "green_frac":
            return (G_MIN, G_MAX)
        if p == "offset":
            return (0.0, CYCLES[-1])
        if p == "phase_order":
            return (0.0, 1.0)
        if p == "lanes_main":
            return (LANES[0], LANES[-1])
        return (0.0, 0.0)

    # ── typed moves ────────────────────────────────────────────────────────────
    def candidate_values(self, st, i: int, p: str, cur: float) -> List[float]:
        i = int(i)
        if p == "cycle":
            return [c for c in CYCLES if abs(c - cur) > 1e-9]
        if p == "green_frac":
            out = []
            for f in GREEN_FACTORS:
                v = min(G_MAX, max(G_MIN, cur * f))
                if abs(v - cur) > 1e-9 and all(abs(v - u) > 1e-9 for u in out):
                    out.append(v)
            return out
        if p == "offset":
            C = float(st["cycle"][i])
            step = C / OFFSET_DIV
            out = []
            for k in OFFSET_SHIFTS:
                v = (cur + k * step) % C
                if abs(v - cur) > 1e-9 and all(abs(v - u) > 1e-9 for u in out):
                    out.append(v)
            return out
        if p == "phase_order":
            return [0.0 if cur >= 0.5 else 1.0]
        if p == "lanes_main":
            return [v for v in (cur - 1.0, cur + 1.0) if LANES[0] <= v <= LANES[-1]]
        return []

    def _main_start(self, C: float, g: float, o: float, ph: float) -> float:
        return (o + ((1.0 - g) * C if ph >= 0.5 else 0.0)) % C

    def candidate_moves(self, st, rng: np.random.Generator) -> List[Candidate]:
        n = int(st["n_int"])
        pools: Dict[str, List[Candidate]] = {p: [] for p in PARAMS}
        for i in range(n):
            C = float(st["cycle"][i])
            g = float(st["green"][i])
            o = float(st["offset"][i])
            ph = float(st["phase"][i])
            flip_start = self._main_start(C, g, o, 1.0 - ph)
            for p in PARAMS:
                cur = self.get(st, i, p)
                for v in self.candidate_values(st, i, p, cur):
                    if p == "offset":
                        # a shift that reproduces the phase-order flip exactly
                        # is the same timing plan twice: keep the flip only
                        ms = self._main_start(C, g, v, ph)
                        dd = abs(ms - flip_start)
                        if min(dd, C - dd) < 1e-6:
                            continue
                    pools[p].append(Candidate(i=i, p=p, v=float(v),
                                              v_old=float(cur)))
        total = sum(len(v) for v in pools.values())
        cap = int(self.max_candidates)
        if total <= cap:
            out = []
            for p in PARAMS:
                out.extend(pools[p])
            out.sort(key=lambda c: (c.i, PARAMS.index(c.p), c.v))
            return out
        # stratified: every move type keeps a quota; half of each quota goes
        # to the intersections the last nominal run shows most loaded, the
        # rest is drawn at random (so no type and no region is ever dropped)
        heat = self._heat(st)
        quota = {p: int(math.floor(_QUOTA[p] * cap)) for p in PARAMS}
        chosen: Dict[str, List[Candidate]] = {}
        rest: Dict[str, List[Candidate]] = {}
        spare = cap - sum(quota.values())
        for p in PARAMS:
            pool = pools[p]
            q = quota[p]
            if len(pool) <= q:
                chosen[p] = list(pool)
                rest[p] = []
                spare += q - len(pool)
                continue
            jitter = rng.random(len(pool))
            hot_order = sorted(range(len(pool)),
                               key=lambda k: (-heat[pool[k].i], jitter[k]))
            n_hot = q // 2
            take = set(hot_order[:n_hot])
            others = [k for k in range(len(pool)) if k not in take]
            pick = rng.choice(len(others), size=q - n_hot, replace=False)
            take.update(others[int(j)] for j in pick)
            chosen[p] = [pool[k] for k in sorted(take)]
            rest[p] = [pool[k] for k in range(len(pool)) if k not in take]
        leftovers = [c for p in PARAMS for c in rest[p]]
        if spare > 0 and leftovers:
            pick = rng.choice(len(leftovers), size=min(spare, len(leftovers)),
                              replace=False)
            extra = [leftovers[int(j)] for j in sorted(pick.tolist())]
        else:
            extra = []
        out = [c for p in PARAMS for c in chosen[p]] + extra
        out.sort(key=lambda c: (c.i, PARAMS.index(c.p), c.v))
        return out

    def _heat(self, st) -> np.ndarray:
        """Per-intersection load indicator from the nominal run: worst of
        (peak queue / storage, demand / capacity) over its two approaches."""
        n = int(st["n_int"])
        d = self._diag(st)
        if not d:
            return np.zeros(n)
        vc = self._vc(st)
        q = np.asarray(d["qfrac"])
        h = np.maximum(q, vc)
        return np.maximum(h[0::2], h[1::2])

    @staticmethod
    def _eff_lanes(lanes_main: np.ndarray, topo: dict) -> np.ndarray:
        """Per-approach discharge lanes: a stop line cannot discharge faster
        than the next link accepts (its own lanes, of which only the through
        share is used), so a lane drop downstream caps it."""
        A = int(topo["A"])
        lanes_a = np.empty(A)
        lanes_a[0::2] = lanes_main
        lanes_a[1::2] = topo["lanes_x"]
        dest = topo["dest"]
        has = dest >= 0
        eff = lanes_a.copy()
        eff[has] = np.minimum(lanes_a[has],
                              lanes_a[dest[has]] / np.maximum(topo["thru"][has], 1e-6))
        return eff

    def _vc(self, st) -> np.ndarray:
        """Nominal demand / capacity per approach (discharge lanes)."""
        topo = st["_topo"]
        s_g, glen, cyc = self._signal_arrays(st)
        lanes_a = self._eff_lanes(np.asarray(st["lanes"], dtype=float), topo)
        cap = lanes_a * SAT_LANE * np.repeat(topo["sat"], 2) * glen / cyc
        return topo["flow"] / np.maximum(cap, 1e-9)

    # ── program / reference ───────────────────────────────────────────────────
    def program(self, spec: dict) -> DesignProgram:
        prog = program_from_goals(
            spec.get("goals", {}),
            objective_key="mean_delay_s",
            objective_ref=self.reference(spec),
            objective_sense="min",
            # one lane over budget is a full unit of violation whatever the
            # network size (the default scale, the budget itself, would make
            # a 60-lane network's overrun nearly free)
            scales={"lanes_used": 1.0},
            domain=self.name,
        )
        horizon = float(spec.get("horizon_s", HORIZON_S))
        return dataclasses.replace(prog, validity_bounds=(
            ("max_density_err", 1e-3),
            ("throughput_frac", 1.001),
            ("mean_delay_s", horizon),
        ))

    def reference(self, spec: dict) -> Optional[float]:
        cached = spec.get("_ref_mean_delay_s")
        if cached is not None:
            try:
                c = float(cached)
                if math.isfinite(c) and c > 0.0:
                    return c
            except (TypeError, ValueError):
                pass
        _, ref, ref_cycle = self._heuristic(spec)
        spec["_ref_mean_delay_s"] = float(ref)
        spec["_ref_cycle"] = float(ref_cycle)
        return float(ref)

    @staticmethod
    def _meets(m: dict, goals: dict, q_margin: float = 0.0,
               t_margin: float = 0.0) -> bool:
        return (m["max_queue_frac"] <= float(goals["maximum_max_queue_frac"]) - q_margin
                and m["throughput_frac"] >= float(goals["minimum_throughput_frac"]) + t_margin
                and m["lanes_used"] <= float(goals["maximum_lanes_used"]) + 1e-9
                and m["max_density_err"] <= 1e-3)

    def _initial_violation(self, spec: dict) -> float:
        """Raw (unscaled, hard) total violation of the initial design."""
        st = self.load(spec)
        m, _ = self._simulate(st, 0, nominal=True)
        g = spec["goals"]
        v = max(0.0, (m["max_queue_frac"] - float(g["maximum_max_queue_frac"]))
                / float(g["maximum_max_queue_frac"]))
        v += max(0.0, (float(g["minimum_throughput_frac"]) - m["throughput_frac"])
                 / float(g["minimum_throughput_frac"]))
        v += max(0.0, m["lanes_used"] - float(g["maximum_lanes_used"]))
        return v

    def _flow_ratio_sum(self, lanes: np.ndarray, topo: dict) -> np.ndarray:
        """Y per intersection with discharge lanes (MAIN + CROSS flow ratios)."""
        eff = self._eff_lanes(lanes, topo)
        sat = np.repeat(topo["sat"], 2)
        y = topo["flow"] / np.maximum(eff * SAT_LANE * sat, 1e-9)
        return y[0::2] + y[1::2]

    def _heuristic_lanes(self, spec: dict, topo: dict) -> np.ndarray:
        """Greedy lane allocation: give back lanes where it raises the worst
        flow-ratio sum least; spend slack where it lowers it most."""
        lanes = np.asarray([float(v) for v in spec["initial"]["lanes_main"]])
        budget = int(round(float(spec["goals"]["maximum_lanes_used"])))
        n = len(lanes)

        def score(ln):
            Y = self._flow_ratio_sum(ln, topo)
            return (float(Y.max()), float(np.sort(Y)[::-1][: max(1, n // 4)].sum()))

        while lanes.sum() > budget + 1e-9:
            cand = [i for i in range(n) if lanes[i] > 1.0]
            if not cand:
                break
            best = None
            for i in cand:
                ln = lanes.copy()
                ln[i] -= 1.0
                sc = score(ln)
                if best is None or sc < best[0]:
                    best = (sc, i)
            lanes[best[1]] -= 1.0
        while lanes.sum() < budget - 1e-9:
            base = score(lanes)
            best = None
            for i in range(n):
                if lanes[i] >= LANES[-1]:
                    continue
                ln = lanes.copy()
                ln[i] += 1.0
                sc = score(ln)
                if best is None or sc < best[0]:
                    best = (sc, i)
            if best is None or best[0] >= base:
                break
            lanes[best[1]] += 1.0
        return lanes

    def _heuristic_state(self, spec: dict, C: float) -> dict:
        """Classical plan at common cycle C: Webster equisaturation splits,
        free-flow progression offsets, lane budget to the highest flow ratios."""
        st = self.load(spec)
        topo = st["_topo"]
        n = int(st["n_int"])
        lanes = self._heuristic_lanes(spec, topo)
        effl = self._eff_lanes(lanes, topo)
        ym = topo["flow"][0::2] / (effl[0::2] * SAT_LANE * topo["sat"])
        yx = topo["flow"][1::2] / (effl[1::2] * SAT_LANE * topo["sat"])
        eff = C - 2.0 * LOST_S
        gm = eff * ym / np.maximum(ym + yx, 1e-9)
        g = np.clip((gm + LOST_S) / C, G_MIN, G_MAX)
        off = np.zeros(n)
        Lm = np.asarray(spec["main_cells"], dtype=float) * DT_S
        Lx = np.asarray(spec["cross_cells"], dtype=float) * DT_S
        if spec["layout"] == "grid":
            rows, cols = int(spec["rows"]), int(spec["cols"])
            for r in range(rows):
                for c in range(cols):
                    i = r * cols + c
                    if c == 0:
                        if r == 0:
                            off[i] = 0.0
                        else:
                            up = i - cols
                            # CROSS green of i follows CROSS green of up by Lx
                            off[i] = off[up] + g[up] * C + Lx[i] - g[i] * C
                    else:
                        off[i] = off[i - 1] + Lm[i]
        else:
            for i in range(1, n):
                off[i] = off[i - 1] + Lm[i]
        st["cycle"][:] = C
        st["green"][:] = g
        st["offset"][:] = np.mod(off, C)
        st["phase"][:] = 0.0
        st["lanes"][:] = lanes
        return st

    def _heuristic(self, spec: dict) -> Tuple[bool, float, float]:
        goals = spec["goals"]
        best_feas = None
        best_any = None
        for C in CYCLES:
            st = self._heuristic_state(spec, C)
            try:
                m, _ = self._simulate(st, 0, nominal=True)
            except Exception:
                continue
            d = float(m["mean_delay_s"])
            if not math.isfinite(d) or d <= 0.0:
                continue
            if best_any is None or d < best_any[0]:
                best_any = (d, C)
            if self.stochastic:
                # chance-constrained variant: the plan must hold when every
                # entry runs at its 1-in-10 high demand level
                try:
                    m_hi, _ = self._simulate(st, 0, nominal=True,
                                             demand_mult=self._high_day())
                except Exception:
                    continue
            else:
                m_hi = m
            if self._meets(m_hi, goals, q_margin=0.03, t_margin=0.005):
                if best_feas is None or d < best_feas[0]:
                    best_feas = (d, C)
        if best_feas is not None:
            return True, max(best_feas[0], DELAY_FLOOR_S), best_feas[1]
        if best_any is not None:
            return False, max(best_any[0], DELAY_FLOOR_S), best_any[1]
        return False, 0.25 * HORIZON_S, CYCLES[0]

    # ── depth-0 feedback ──────────────────────────────────────────────────────
    def _worst_approach(self, st) -> Optional[int]:
        d = self._diag(st)
        if not d:
            return None
        q = np.asarray(d["qfrac"], dtype=float)
        vc = self._vc(st)
        key = np.maximum(q, vc)
        return int(np.argmax(key))

    def hint(self, st) -> Optional[Tuple[int, str, str]]:
        a = self._worst_approach(st)
        if a is None:
            return None
        # worst queue on MAIN -> more MAIN green ("up"); on CROSS -> less
        return (a // 2, "green_frac", "up" if a % 2 == 0 else "down")

    def hint_move(self, st) -> Optional[Candidate]:
        """Classic split-first feedback: move green toward the worst approach;
        when the split is already at its bound, lengthen that cycle."""
        h = self.hint(st)
        if h is None:
            return None
        i, _, direction = h
        g = float(st["green"][i])
        v = min(G_MAX, g * 1.1) if direction == "up" else max(G_MIN, g * 0.9)
        if abs(v - g) > 1e-9:
            return Candidate(i=i, p="green_frac", v=float(v), v_old=g)
        c = float(st["cycle"][i])
        k = CYCLES.index(_nearest_cycle(c))
        if k + 1 < len(CYCLES):
            return Candidate(i=i, p="cycle", v=float(CYCLES[k + 1]), v_old=c)
        return None

    # ── self-contained observation ────────────────────────────────────────────
    # describe() character budget: above it the per-intersection list is
    # condensed (full detail for the most loaded intersections, a short row
    # for other loaded ones, a grouped list for quiet ones).
    _OBS_CHARS = 11000
    _OBS_MIN_FULL = 3

    @staticmethod
    def _num(x, sig: int = 3) -> str:
        x = float(x)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if x == 0.0:
            return "0"
        mag = int(math.floor(math.log10(abs(x))))
        if mag >= sig - 1:
            return "%d" % int(round(x))
        return "%.*f" % (sig - 1 - mag, x)

    @staticmethod
    def _sec(x) -> str:
        """Seconds with at most one decimal."""
        x = float(x)
        if abs(x) < 0.05:
            return "0"
        s = "%.1f" % x
        return s[:-2] if s.endswith(".0") else s

    def _position(self, st, i: int) -> str:
        spec = st["spec"]
        n = int(st["n_int"])
        if spec["layout"] == "grid":
            r, c = divmod(int(i), int(spec["cols"]))
            return "grid r%dc%d" % (r, c)
        if i == 0:
            return "arterial start"
        if i == n - 1:
            return "arterial end"
        return "arterial"

    def _obs_disch(self, st, lanes_main=None) -> np.ndarray:
        """Per-approach discharge rate during green (veh/s), as the CTM step
        resolves it: min(own lanes x per-lane sat, next link's lanes x
        SAT_LANE / through share). Text only (the simulator is unchanged)."""
        topo = st["_topo"]
        A = int(topo["A"])
        lm = np.asarray(st["lanes"] if lanes_main is None else lanes_main,
                        dtype=float)
        la = np.empty(A)
        la[0::2] = lm
        la[1::2] = topo["lanes_x"]
        rate = la * SAT_LANE * np.repeat(topo["sat"], 2)
        dest = topo["dest"]
        has = dest >= 0
        rate[has] = np.minimum(rate[has], la[dest[has]] * SAT_LANE
                               / np.maximum(topo["thru"][has], 1e-6))
        return rate

    def _obs_vc(self, st, rate: np.ndarray) -> np.ndarray:
        _, glen, cyc = self._signal_arrays(st)
        return st["_topo"]["flow"] / np.maximum(rate * glen / cyc, 1e-9)

    def _obs_cap(self, st, rate: np.ndarray, i: int) -> dict:
        """Flow-ratio check of intersection i at its current cycle."""
        flow = st["_topo"]["flow"]
        C = float(st["cycle"][i])
        ym = float(flow[2 * i] / max(rate[2 * i], 1e-9))
        yx = float(flow[2 * i + 1] / max(rate[2 * i + 1], 1e-9))
        Y = ym + yx
        cmin = next((c for c in CYCLES if _usable(c) >= Y), None)
        gb = ((C - 2.0 * LOST_S) * ym / max(Y, 1e-9) + LOST_S) / C
        gb = min(G_MAX, max(G_MIN, gb))
        egm = max(gb * C - LOST_S, MIN_EFF_S)
        egx = max((1.0 - gb) * C - LOST_S, MIN_EFF_S)
        return {"ym": ym, "yx": yx, "Y": Y, "U": _usable(C), "cmin": cmin,
                "gb": gb, "vm": ym * C / egm, "vx": yx * C / egx}

    def _obs_win(self, start: float, length: float, C: float) -> str:
        s = float(start) % C
        e = s + float(length)
        if e > C + 1e-9:
            e -= C
        return "%s-%s" % (self._sec(s), self._sec(e))

    def _obs_arrive(self, st, a: int, s_g: np.ndarray, glen: np.ndarray) -> str:
        """When the upstream green's discharge reaches approach a's stop line."""
        topo = st["_topo"]
        fd = int(topo["feeder"][a])
        if fd < 0:
            return ""
        ca = int(round(float(st["cycle"][a // 2])))
        cf = int(round(float(st["cycle"][fd // 2])))
        P = ca * cf // math.gcd(ca, cf)
        if P != max(ca, cf):
            return (" arrives drifting (cycles %d/%d realign every %d s)"
                    % (cf, ca, P))
        T = float(topo["L"][a]) * DT_S
        wins = [self._obs_win(s_g[fd] + T + k * cf, glen[fd], P)
                for k in range(P // cf)]
        return " arrives %s%s" % (", ".join(wins),
                                  "" if P == ca else " (mod %d s)" % P)

    def _obs_lanes(self, st, i: int) -> str:
        """Y and MAIN v/c here, and the feeding MAIN approach's v/c, with one
        MAIN lane more / less at intersection i."""
        f = self._num
        topo = st["_topo"]
        a = 2 * i
        fd = int(topo["feeder"][a])
        lm = np.asarray(st["lanes"], dtype=float)
        out = []
        for dl in (1, -1):
            nl = float(lm[i]) + dl
            if not (LANES[0] <= nl <= LANES[-1]):
                continue
            ln = lm.copy()
            ln[i] = nl
            rate = self._obs_disch(st, ln)
            vc = self._obs_vc(st, rate)
            cap = self._obs_cap(st, rate, i)
            s = "lane%+d: Y %s v/c %s" % (dl, f(cap["Y"]), f(vc[a]))
            if dl < 0:
                s += " stor %g" % (topo["L"][a] * N_LANE * nl)
            if fd >= 0:
                s += ", I%d M v/c %s" % (fd // 2, f(vc[fd]))
            out.append(s)
        return "; ".join(out)

    def _obs_approach(self, st, d: dict, a: int, vc: np.ndarray,
                      rate: np.ndarray, dsh: np.ndarray, s_g: np.ndarray,
                      glen: np.ndarray, full: bool) -> str:
        f = self._num
        topo = st["_topo"]
        i = a // 2
        main = a % 2 == 0
        fd = int(topo["feeder"][a])
        ds = int(topo["dest"][a])
        flow = topo["flow"]
        tag = "M" if main else "X"
        if full:
            frm = "entry" if fd < 0 else "from I%d" % (fd // 2)
            to = "to exit" if ds < 0 else "to I%d" % (ds // 2)
            ln = float(st["lanes"][i]) if main else float(topo["lanes_x"][i])
            s = "%s %s %s: %d cells %ss stor %g" % (
                tag, frm, to, int(topo["L"][a]), self._sec(topo["L"][a] * DT_S),
                topo["L"][a] * N_LANE * ln)
            if not main and abs(ln - 1.0) > 1e-9:
                s += " lanes %d" % int(round(ln))
            s += " flow %s" % f(flow[a])
            if ds >= 0:
                s += " leave %s" % f(1.0 - topo["thru"][a])
            s += " disch %s" % f(rate[a])
            s += self._obs_arrive(st, a, s_g, glen)
            s += " | eg %s v/c %s" % (self._sec(glen[a]), f(vc[a]))
        else:
            s = "%s v/c %s" % (tag, f(vc[a]))
        if not d:
            return s
        s += " Q %s" % f(d["qfrac"][a])
        if full:
            s += " dly %s" % f(d["dly"][a])
            if fd >= 0:
                s += " AOG %s%%" % f(100.0 * d["aog"][a], 2)
        if full or dsh[a] >= 0.005:
            s += " dshare %d%%" % int(round(100.0 * dsh[a]))
        g = float(d["growth"][a])
        if abs(g) >= 1e-4:
            s += " grow %s" % f(g)
        if float(d["arr_rate"][a]) < 0.97 * float(flow[a]):
            s += " arr %s" % f(d["arr_rate"][a])
        if d["qfrac"][a] >= 0.95:
            s += (" FULL" if fd < 0 else " SPILLBACK onto I%d %s"
                  % (fd // 2, "M" if fd % 2 == 0 else "X"))
        if d["blk_frac"][a] > 0.0:
            s += " blocked %s%%" % f(100.0 * d["blk_frac"][a], 2)
            if ds >= 0:
                s += " by I%d %s" % (ds // 2, "M" if ds % 2 == 0 else "X")
        return s

    def _obs_head(self, st, i: int, cap: dict, s_g: np.ndarray,
                  glen: np.ndarray, skip: Optional[dict] = None) -> str:
        """Timing line of intersection i; with `skip`, a short row that leaves
        out fields equal to the network's common values."""
        f = self._num
        C = float(st["cycle"][i])
        vals = (("cyc", "%d" % int(round(C))), ("g", "%.6g" % st["green"][i]),
                ("off", "%.6g" % st["offset"][i]),
                ("order", "X" if st["phase"][i] >= 0.5 else "M"),
                ("lanes", "%d" % int(round(float(st["lanes"][i])))),
                ("sat", f(SAT_LANE * st["_topo"]["sat"][i])))
        parts = ["%s %s" % (k, v) for k, v in vals
                 if skip is None or k == "g" or skip.get(k) != v]
        pos = self._position(st, i)
        if skip is not None:
            pos = pos.replace("grid ", "")
        cmin = "%d" % cap["cmin"] if cap["cmin"] else "none"
        if skip is not None:
            # short row: usable / min-cycle only where Y exceeds usable
            s = "I%d [%s] %s | Y %s" % (i, pos, " ".join(parts), f(cap["Y"]))
            if cap["Y"] > cap["U"]:
                s += " > usable %s, min-cycle %s" % (f(cap["U"]), cmin)
            return s + ", g_bal %.3g" % cap["gb"]
        s = "I%d [%s] %s" % (i, pos, " ".join(parts))
        s += " | Mgreen %s Xgreen %s" % (
            self._obs_win(s_g[2 * i], glen[2 * i], C),
            self._obs_win(s_g[2 * i + 1], glen[2 * i + 1], C))
        s += " | Y %s = M %s + X %s, usable %s, min-cycle %s, g_bal %.3g" % (
            f(cap["Y"]), f(cap["ym"]), f(cap["yx"]), f(cap["U"]), cmin,
            cap["gb"])
        if abs(cap["vm"] - cap["vx"]) < 5e-3:
            s += " (v/c %s both)" % f(cap["vm"])
        else:
            s += " (v/c M %s X %s)" % (f(cap["vm"]), f(cap["vx"]))
        return s

    def describe(self, st, spec) -> str:
        f = self._num
        topo = st["_topo"]
        n = int(st["n_int"])
        fam = str(spec["family"])
        steps = int(topo["steps"])
        goals = spec.get("goals", {})
        budget = float(goals.get("maximum_lanes_used", 0.0))
        qlim = float(goals.get("maximum_max_queue_frac", 1.0))
        tlim = float(goals.get("minimum_throughput_frac", 1.0))
        bud = "%d" % int(round(budget))
        L = []
        # ── PROBLEM ──────────────────────────────────────────────────────────
        kind = {
            "corridor": "an arterial corridor (heavy arterial traffic)",
            "asymmetric": "an arterial corridor with a light arterial and "
                          "side streets that are each heavy or light",
            "grid": "a one-way street grid",
            "downtown": "a one-way downtown grid with short blocks",
        }[fam]
        L.append("PROBLEM: fixed-time signal timing and lane allocation for %d "
                 "signalised intersections on %s. Cell transmission traffic "
                 "model, %d s horizon = %d steps of %g s, %s demand; the "
                 "network starts in free flow at nominal demand."
                 % (n, kind, int(round(steps * DT_S)), steps, DT_S,
                    "randomly varying" if self.stochastic else "constant"))
        if spec["layout"] == "grid":
            rows, cols = int(spec["rows"]), int(spec["cols"])
            L.append("Topology: %d rows x %d columns, Ik with k = row*%d + col. "
                     "MAIN runs west->east along rows: the MAIN approach of "
                     "(r,c) comes from (r,c-1), the next MAIN intersection is "
                     "(r,c+1), column 0 MAIN links are entries. CROSS runs "
                     "north->south: the CROSS approach of (r,c) comes from "
                     "(r-1,c), row 0 CROSS links are entries. At every stop "
                     "line a fixed share of the discharge leaves the network, "
                     "the rest goes straight (no other turns)."
                     % (rows, cols, cols))
            L.append("Entry demand veh/s - MAIN rows r0..r%d: %s. CROSS "
                     "columns c0..c%d: %s."
                     % (rows - 1, " ".join(f(v) for v in spec["main_rate"]),
                        cols - 1, " ".join(f(v) for v in spec["cross_rate"])))
        else:
            L.append("Topology: MAIN chains I0 -> I1 -> ... -> I%d (the next "
                     "MAIN intersection of Ik is I(k+1)); MAIN traffic enters "
                     "before I0 at %s veh/s and at every stop line a fixed "
                     "share leaves the arterial. Every CROSS approach is its "
                     "own network entry and leaves after crossing."
                     % (n - 1, f(spec["main_rate"][0])))
        L.append("Links: a link of L cells takes L x %g s to drive in free "
                 "flow; per lane it carries at most %g veh/s and stores %g veh "
                 "per cell (storage = L x %g x lanes veh). In its effective "
                 "green a stop line discharges at disch = min(own lanes x sat, "
                 "next link's lanes x %g / (1 - leave)) veh/s. So a MAIN "
                 "lane added at Ik "
                 "alone raises Ik's MAIN discharge only up to the next MAIN "
                 "link's cap (fully where MAIN exits after Ik), and a lane "
                 "removed at Ik also lowers the cap of the MAIN approach "
                 "feeding Ik and cuts Ik's MAIN storage. A full link blocks "
                 "the stop line feeding it (spillback); demand that cannot "
                 "enter at an entry waits outside and counts in that entry "
                 "approach's queue, which can then exceed storage."
                 % (DT_S, SAT_LANE, N_LANE, N_LANE, SAT_LANE))
        L.append("Signals: two phases per cycle (MAIN, CROSS); each loses %g s "
                 "at its end (yellow + all-red), so effective green eg MAIN = "
                 "g x cycle - %g s, CROSS = (1 - g) x cycle - %g s (floor %g "
                 "s). Capacity = disch x eg / cycle veh/s; v/c = nominal flow "
                 "/ capacity." % (LOST_S, LOST_S, LOST_S, MIN_EFF_S))
        L.append("Flow ratio y = nominal flow / disch; Y = yM + yX. Phase "
                 "changes cost %g s per cycle, so an intersection can serve "
                 "its demand at some split only if Y < usable = 1 - %g/cycle: "
                 "%s. The split g_bal = ((cycle - %g) x yM / Y + %g) / cycle "
                 "gives both approaches v/c = Y / usable. No cycle serves Y "
                 "above %s; only more discharge lanes lower it."
                 % (2 * LOST_S, 2 * LOST_S,
                    ", ".join("%s at %d" % (f(_usable(c)), int(c))
                              for c in CYCLES),
                    2 * LOST_S, LOST_S, f(_usable(CYCLES[-1]))))
        if spelled_in_verbs(self):
            # the same facts as the SET_PARAM block below, in the verbs the
            # policy is served (docstring "Verb surface"); the SET_PARAM block
            # again under the set_parameter spelling
            L.append("Tunable per intersection, one change per turn, written as "
                     "SET_CYCLE(i, cycle), SET_GREEN_SPLIT(i, split), SET_OFFSET(i, "
                     "offset), SET_PHASE_ORDER(i, order) or SET_LANES(i, lanes); i "
                     "is the bare number after I (I3 -> 3); a value is the new "
                     "setting, not a change, and any value in range is accepted:")
            L.append("  SET_CYCLE(i, cycle): 60, 75, 90, 105 or 120 s (others round "
                     "to the nearest). Longer cycles lose less time to phase "
                     "changes but give longer reds: a red of R s stores about flow "
                     "x R veh on an approach, and beyond its storage the queue "
                     "spills back into the upstream stop line (at an entry it waits "
                     "outside). A new cycle keeps the green split (both eg change) "
                     "and the offset mod the new cycle. Intersections on different "
                     "cycles realign only every LCM of the two (60/120: 120 s, a "
                     "fixed relation; 60/90: 180; 60/75: 300; up to 840 for "
                     "105/120), so offsets cannot coordinate them. One move per "
                     "intersection.")
            L.append("  SET_GREEN_SPLIT(i, split): [%g, %g], MAIN phase share of the "
                     "cycle (CROSS gets the rest); usual steps x0.8 / x0.9 / x1.1 / "
                     "x1.25, or write a value such as g_bal directly."
                     % (G_MIN, G_MAX))
            L.append("  SET_OFFSET(i, offset): seconds in [0, cycle), wraps: "
                     "network-clock time (mod cycle) at which the first phase "
                     "starts; usual steps are multiples of cycle/8. Green wave on a "
                     "link: the downstream green should start about the link's "
                     "free-flow time after the upstream green starts, earlier by "
                     "(standing queue / disch) s; AOG shows how well that holds "
                     "now.")
            L.append("  SET_PHASE_ORDER(i, order): MAIN_FIRST (order M in the rows: "
                     "MAIN green starts at the offset, CROSS g x cycle later) or "
                     "CROSS_FIRST (order X: CROSS green starts at the offset, MAIN "
                     "(1 - g) x cycle later). Switching it shifts both greens by "
                     "(1 - g) x cycle, the same timing as that offset shift; it "
                     "also decides which end of MAIN green a later SET_GREEN_SPLIT "
                     "moves: the end under MAIN_FIRST, the start under CROSS_FIRST "
                     "(MAIN then ends where the next cycle starts).")
            L.append("  SET_LANES(i, lanes): 1, 2 or 3 lanes on the MAIN link INTO "
                     "the intersection; scales that link's saturation flow and "
                     "storage. CROSS lanes are fixed.")
        else:
            L.append("Tunable per intersection, SET_PARAM(i, param, value); any "
                     "value in range is accepted:")
            L.append("  cycle: 60, 75, 90, 105 or 120 s (others round to the "
                     "nearest). Longer cycles lose less time to phase changes but "
                     "give longer reds: a red of R s stores about flow x R veh on "
                     "an approach, and beyond its storage the queue spills back "
                     "into the upstream stop line (at an entry it waits outside). "
                     "A new cycle keeps green_frac (both eg change) and the offset "
                     "mod the new cycle. Intersections on different cycles "
                     "realign only every LCM of the two (60/120: 120 s, a fixed "
                     "relation; 60/90: 180; 60/75: 300; up to 840 for 105/120), so "
                     "offsets cannot coordinate them. One move per intersection.")
            L.append("  green_frac: [%g, %g], MAIN phase share of the cycle (CROSS "
                     "gets the rest); usual steps x0.8 / x0.9 / x1.1 / x1.25, or "
                     "write a value such as g_bal directly." % (G_MIN, G_MAX))
            L.append("  offset: seconds in [0, cycle), wraps: network-clock time "
                     "(mod cycle) at which the first phase starts; usual steps "
                     "are multiples of cycle/8. Green wave on a link: the "
                     "downstream green should start about the link's free-flow "
                     "time after the upstream green starts, earlier by (standing "
                     "queue / disch) s; AOG shows how well that holds now.")
            L.append("  phase_order: 0 = MAIN first (MAIN green starts at the "
                     "offset, CROSS g x cycle later), 1 = CROSS first (CROSS green "
                     "starts at the offset, MAIN (1 - g) x cycle later). Flipping "
                     "it shifts both greens by (1 - g) x cycle, the same timing as "
                     "that offset shift; it also decides which end of MAIN green a "
                     "later green_frac change moves: the end under order 0, the "
                     "start under order 1 (MAIN then ends where the next cycle "
                     "starts).")
            L.append("  lanes_main: 1, 2 or 3 lanes on the MAIN link INTO the "
                     "intersection; scales that link's saturation flow and "
                     "storage. CROSS lanes are fixed.")
        L.append("Cost: timing changes are free; every MAIN lane counts 1 "
                 "against the lane budget (lanes_used <= %s)." % bud)
        L.append("Violation weights: each lane over budget = 1.0; "
                 "max_queue_frac (value - limit) / limit; throughput_frac "
                 "(limit - value) / limit. One lane over budget weighs as much "
                 "as a queue at twice its limit, or about 100 times a 0.01 "
                 "throughput shortfall. The goal is first to satisfy every "
                 "constraint (a smaller weighted violation is better), then "
                 "the lowest mean_delay_s.")
        L.append("Measures: mean_delay_s (objective) = stopped "
                 "vehicle-seconds (incl. waiting outside entries) per vehicle "
                 "demanded; max_queue_frac = worst over approaches and time of "
                 "stopped vehicles / storage; throughput_frac = 1 - (sum of "
                 "grow) / network demand, grow = growth rate, over the second "
                 "half of the horizon, of an approach's residual queue "
                 "(vehicles still stopped when its green ends; 0 if it clears "
                 "every cycle); max_density_err = numerical validity check (0 "
                 "when correct), not tunable.")
        if self.stochastic:
            hi = math.exp(-0.5 * NOISE_SIGMA ** 2 + 1.2816 * NOISE_SIGMA)
            L.append("Demand noise: on any given day each entry's demand is "
                     "its nominal value times an independent random lognormal "
                     "factor (sigma %g, mean 1, constant over the horizon); a "
                     "1-in-10 high day is about x%s. The constraints must "
                     "hold on 9 days in 10. Everything below is a nominal run "
                     "at mean demand." % (NOISE_SIGMA, self._num(hi)))
        # ── NOW: network summary ─────────────────────────────────────────────
        d = self._diag(st)
        rate = self._obs_disch(st)
        vc = self._obs_vc(st, rate)
        s_g, glen, _cyc = self._signal_arrays(st)
        caps = [self._obs_cap(st, rate, i) for i in range(n)]
        used = float(np.sum(st["lanes"]))
        A = 2 * n

        def an(a):
            return "I%d %s" % (a // 2, "M" if a % 2 == 0 else "X")

        L.append("")
        if d:
            m = d["metrics"]
            D = float(np.sum(topo["entry_rate"]))
            gsum = float(np.sum(d["growth"]))
            vl = max(0.0, used - budget)
            vq = max(0.0, (m["max_queue_frac"] - qlim) / qlim)
            vt = max(0.0, (tlim - m["throughput_frac"]) / tlim)
            qa = int(np.argmax(d["qfrac"]))
            L.append("NOW%s: lanes used %d of %s; mean_delay_s %s; "
                     "max_queue_frac %s at %s (limit %s); throughput_frac %.4f "
                     "(limit %.4f): network demand D = %s veh/s, the limit "
                     "allows sum of grow <= %s veh/s, now %s. Weighted "
                     "violation %s = lanes %s + queue %s + throughput %s."
                     % (" (nominal run at mean demand)" if self.stochastic else "",
                        int(round(used)), bud, f(m["mean_delay_s"]),
                        f(m["max_queue_frac"]), an(qa), f(qlim),
                        m["throughput_frac"], tlim, f(D), f((1.0 - tlim) * D),
                        f(gsum), f(vl + vq + vt), f(vl), f(vq), f(vt)))
            gr = np.asarray(d["growth"], dtype=float)
            order = [int(a) for a in np.argsort(-gr, kind="stable")
                     if gr[a] >= 1e-4]
            if order:
                L.append("Growing residual queues (veh/s, largest first): %s%s."
                         % (", ".join("%s %s" % (an(a), f(gr[a]))
                                      for a in order[:8]),
                            "" if len(order) <= 8 else
                            ", +%d more totalling %s"
                            % (len(order) - 8, f(float(gr[order[8:]].sum())))))
            else:
                L.append("Growing residual queues: none.")
            qf = np.asarray(d["qfrac"], dtype=float)
            order = [int(a) for a in np.argsort(-qf, kind="stable")
                     if qf[a] >= 0.8 * qlim]
            if order:
                L.append("Queues above 0.8 x limit (Q, largest first): %s%s."
                         % (", ".join("%s %s" % (an(a), f(qf[a]))
                                      for a in order[:8]),
                            "" if len(order) <= 8 else
                            ", +%d more" % (len(order) - 8)))
        else:
            L.append("NOW: lanes used %d of %s; (evaluation failed)"
                     % (int(round(used)), bud))
        over = [i for i in range(n) if caps[i]["Y"] > caps[i]["U"]]
        over.sort(key=lambda i: -(caps[i]["Y"] - caps[i]["U"]))
        L.append("Y above usable at the current cycle: %s."
                 % (", ".join("I%d %s > %s" % (i, f(caps[i]["Y"]),
                                               f(caps[i]["U"]))
                              for i in over) if over else "none"))
        # ── per intersection ─────────────────────────────────────────────────
        L.append("")
        L.append("PER INTERSECTION: cyc g off order (M / X = MAIN / CROSS "
                 "first) lanes (MAIN) sat (per-lane saturation flow veh/s; "
                 "below %g = constrained, both approaches); Mgreen / Xgreen = "
                 "effective-green windows, s on the network clock mod cycle "
                 "(a-b with b < a wraps); Y, usable, min-cycle = shortest "
                 "cycle with usable >= Y, g_bal and the v/c it gives. Per "
                 "approach M / X: from / to = upstream / downstream "
                 "intersection (or entry / exit); cells, free-flow s, storage "
                 "veh (CROSS links have 1 lane unless shown); flow = nominal "
                 "veh/s; leave = share leaving here (all if to exit); disch "
                 "veh/s; arrives = when the upstream green's discharge "
                 "reaches this stop line (its window + free-flow s); eg s; "
                 "v/c; Q = peak stopped queue / storage; dly = mean stopped s "
                 "per discharged vehicle; AOG = share arriving in green; "
                 "dshare = share of the network's stopped vehicle-seconds; "
                 "grow (if nonzero) veh/s; arr = actual arrivals veh/s if "
                 "below flow (upstream short of capacity, blocked or "
                 "backlogged; v/c uses flow, the load once upstream is fixed); "
                 "blocked = share of steps the named next link cut the "
                 "discharge (queue at its entrance, or fewer lanes); FULL = Q "
                 ">= 0.95 at an entry (excess waits outside); SPILLBACK = Q >= "
                 "0.95 on a fed link, blocking the named upstream stop line; "
                 "lane+1 / lane-1 = Y and MAIN v/c here and v/c of the "
                 "feeding MAIN approach after a 1-lane change here." % SAT_LANE)
        if d:
            dd = np.asarray(d["dly"]) * np.asarray(d["dis_rate"])
            dsh = dd / max(float(dd.sum()), 1e-12)
        else:
            dsh = np.zeros(A)

        def full_block(i):
            rows = [self._obs_head(st, i, caps[i], s_g, glen)]
            ln = self._obs_lanes(st, i)
            for a in (2 * i, 2 * i + 1):
                s = "  " + self._obs_approach(st, d, a, vc, rate, dsh, s_g,
                                              glen, True)
                if a == 2 * i and ln:
                    s += " | " + ln
                rows.append(s)
            return "\n".join(rows)

        blocks = [full_block(i) for i in range(n)]
        fixed = sum(len(s) + 1 for s in L)
        if fixed + sum(len(b) + 1 for b in blocks) <= self._OBS_CHARS:
            L.extend(blocks)
            return "\n".join(L)
        # ── condensed listing for large networks ─────────────────────────────
        def common(vals):
            u, c = np.unique(np.asarray(vals, dtype=float), return_counts=True)
            return float(u[int(np.argmax(c))])

        skip = {"cyc": "%d" % int(round(common(st["cycle"]))),
                "off": "%.6g" % common(st["offset"]),
                "order": "X" if common(st["phase"]) >= 0.5 else "M",
                "lanes": "%d" % int(round(common(st["lanes"]))),
                "sat": f(SAT_LANE * common(topo["sat"]))}
        qf = np.asarray(d["qfrac"]) if d else np.zeros(A)
        gr = np.asarray(d["growth"]) if d else np.zeros(A)
        bk = np.asarray(d["blk_frac"]) if d else np.zeros(A)

        def rank_key(i):
            a, b = 2 * i, 2 * i + 1
            hot = (max(gr[a], gr[b]) >= 1e-4 or max(qf[a], qf[b]) >= 0.8 * qlim
                   or caps[i]["Y"] > caps[i]["U"])
            lv = max(vc[a], vc[b], qf[a] / qlim, qf[b] / qlim,
                     caps[i]["Y"] / caps[i]["U"])
            return (0 if hot else 1, -lv, -(dsh[a] + dsh[b]), i)

        def quiet(i):
            a, b = 2 * i, 2 * i + 1
            return (max(vc[a], vc[b]) < 0.85
                    and max(qf[a], qf[b]) < 0.5 * qlim
                    and max(abs(gr[a]), abs(gr[b])) < 1e-4
                    and max(bk[a], bk[b]) <= 0.0
                    and caps[i]["Y"] < 0.9 * caps[i]["U"]
                    and float(st["lanes"][i]) <= LANES[0])

        def short_row(i):
            s = self._obs_head(st, i, caps[i], s_g, glen, skip)
            for a in (2 * i, 2 * i + 1):
                s += " | " + self._obs_approach(st, d, a, vc, rate, dsh, s_g,
                                                glen, False)
            if float(st["lanes"][i]) > LANES[0]:
                s += " | " + self._obs_lanes(st, i).split("; ")[-1]
            return s

        def quiet_tok(i):
            s = self._obs_head(st, i, caps[i], s_g, glen, skip)
            s = s.split(" | ")[0]
            return s[:s.index(" [")] + s[s.index("] ") + 1:]

        rank = sorted(range(n), key=rank_key)
        rows_s = {i: short_row(i) for i in range(n)}
        toks = {i: quiet_tok(i) for i in range(n)}
        is_q = {i: quiet(i) for i in range(n)}

        def listing(k, m=None):
            # k full blocks (rank order), then short rows for up to m of the
            # next loaded intersections, then everything else grouped
            fs = set(rank[:k])
            loaded = [i for i in rank if i not in fs and not is_q[i]]
            sr = set(loaded if m is None else loaded[:m])
            out = ["Network of %d: full detail for the %d most loaded "
                   "intersections, then short rows (other loaded) and a "
                   "grouped list (the rest); these omit cyc / off / order / "
                   "lanes / sat equal to the common %s / %s / %s / %s / %s."
                   % (n, len(fs), skip["cyc"], skip["off"], skip["order"],
                      skip["lanes"], skip["sat"]),
                   "MOST LOADED (index order):"]
            out.extend(blocks[i] for i in sorted(fs))
            if sr:
                out.append("OTHER LOADED (Y, g_bal; per approach v/c, Q, "
                           "dshare, flags; lane-1 where MAIN has 2+ lanes):")
                out.extend(rows_s[i] for i in sorted(sr))
            qs = [i for i in range(n) if i not in fs and i not in sr]
            if qs:
                ia = [a for i in qs for a in (2 * i, 2 * i + 1)]
                allq = all(is_q[i] for i in qs)
                tk = []
                for i in qs:
                    t = toks[i]
                    if not is_q[i]:
                        a2 = (2 * i, 2 * i + 1)
                        t += " (v/c %s Q %s)" % (f(max(vc[x] for x in a2)),
                                                 f(max(qf[x] for x in a2)))
                    tk.append(t)
                out.append("%s (every approach v/c < 0.85, Q < 0.5 x limit, "
                           "no growth, not blocked, Y < 0.9 x usable, 1 MAIN "
                           "lane%s; together max v/c %s, max Q %s, dshare "
                           "%d%%): %s."
                           % ("QUIET" if allq else "REST",
                              "" if allq else ", except those showing their "
                              "worst v/c and Q",
                              f(max(vc[a] for a in ia)),
                              f(max(qf[a] for a in ia)),
                              int(round(100.0 * sum(dsh[a] for a in ia))),
                              "; ".join(tk)))
            return out

        def fits(k, m=None):
            return fixed + len("\n".join(listing(k, m))) <= self._OBS_CHARS

        k = min(n, self._OBS_MIN_FULL)
        m = None
        if fits(k):
            while k < n and fits(k + 1):
                k += 1
        else:
            m = len([i for i in rank[k:] if not is_q[i]])
            while m > 0 and not fits(k, m):
                m -= 1
        L.extend(listing(k, m))
        return "\n".join(L)

    def element_label(self, st, i) -> str:
        i = int(i)
        if not (0 <= i < int(st["n_int"])):
            return "intersection %d" % i
        flow = st["_topo"]["flow"]
        return ("intersection I%d (%s; flow main %s / cross %s veh/s)"
                % (i, self._position(st, i), self._num(flow[2 * i]),
                   self._num(flow[2 * i + 1])))
