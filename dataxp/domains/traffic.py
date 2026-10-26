"""Signal timing on a cell-transmission-model (CTM) road network.

A social/operations decision problem: n signalised intersections, each with a
MAIN and a CROSS approach. Every approach is a short chain of CTM cells with
the standard triangular fundamental diagram (Daganzo 1994): per time step
``dt`` the flow between cell j and j+1 is

    y = min(n_j, Q, delta * (N_jam - n_{j+1}))            [veh/step]

with per-cell jam storage ``N_jam``, capacity ``Q = q_cap * dt`` and
backward/forward wave-speed ratio ``delta = w/v``. CFL = 1 (a free-flowing
vehicle advances exactly one cell per step), so per-step queueing delay is
(vehicles present - vehicles that advanced) * dt.

The signal at each intersection runs a fixed cycle with per-intersection green
fraction g for the MAIN direction; the cross street gets (1 - g), and each
phase loses a fixed lost-time fraction. The stop-line (last) cell of an
approach discharges at capacity ``Q * g_eff``. Because this macroscopic model
averages over the cycle, the classic uniform (Webster first-term) signal delay
``0.5 * C_cycle * (1 - g_eff)^2`` seconds is charged analytically to every
vehicle crossing a stop line - so *every* green split influences the delay
metric, saturated or not.

Families:
  corridor    an arterial line; the main approaches chain into each other,
              every cross street is an independent boundary inflow.
  grid        an m x k lattice; main approaches chain along rows (entry at the
              west boundary), cross approaches chain along columns (entry at
              the north boundary). No turning movements.
  asymmetric  corridor topology with bimodal per-intersection loads (half the
              cross streets nearly idle, half near or above saturation).

ELEMENTS are intersections; the single tunable PARAM is ``green_frac`` in
[0.15, 0.85]. Initial design: g = 0.5 everywhere, with demands drawn so that
most problems start infeasible (main-street queues starve throughput).

METRICS (flat floats):
  mean_delay_s     objective: (CTM queueing delay + uniform signal delay) per
                   vehicle served.
  max_queue_frac   max over approach cells and time of n / N_jam (spillback
                   proxy).
  throughput_frac  vehicles that left the network / vehicles demanded.
  total_stops      supply-blocked vehicle-steps per served vehicle (cheap
                   stop proxy).
  max_density_err  max relative excursion of any cell outside [0, N_jam]
                   (envelope: a correct CTM never leaves it).

Stochastic mode: seeded piecewise demand profiles - the horizon is split into
intervals and each entry's demand is scaled by an independent lognormal factor
(mean 1) drawn from ``np.random.default_rng(seed)`` only.

Reference: mean delay under the classic PROPORTIONAL split heuristic
(green ratio = demand ratio, clipped to bounds) -> reference_class
"heuristic".

Compatibility: python >= 3.8, numpy only, no file I/O, no prints.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Tuple

import numpy as np

from ..domain import Domain
from ..potential import DesignProgram, program_from_goals

# ── fixed physics / discretisation ───────────────────────────────────────────
DT_S = 2.0                      # time step (s)
CELLS = 3                       # CTM cells per approach (stop line last)
CAP_VPS = 1.0                   # saturation flow per approach (veh/s)
Q_STEP = CAP_VPS * DT_S         # capacity per cell boundary (veh/step)
N_JAM = 8.0                     # jam storage per cell (veh)
DELTA = 0.5                     # wave-speed ratio w/v
LOST = 0.05                     # lost-time fraction per phase
CYCLE_S = 90.0                  # signal cycle (s), for the uniform-delay term
G_MIN, G_MAX = 0.15, 0.85       # green_frac bounds
GE_MIN = 0.02                   # floor on effective green (never fully starved)
USABLE = CAP_VPS * (1.0 - 2.0 * LOST)   # jointly serviceable demand (veh/s)
DELAY_FLOOR_S = 1e-3            # keeps the objective strictly positive
N_INTERVALS = 6                 # stochastic piecewise-demand intervals
NOISE_SIGMA = 0.12              # lognormal sigma of demand factors

_FAMILIES = ("corridor", "grid", "asymmetric")
_SIZE_LADDER = (4, 6, 9, 12, 16, 24, 36, 48, 60)
_GRID_DIMS = {4: (2, 2), 6: (2, 3), 9: (3, 3), 12: (3, 4), 16: (4, 4),
              24: (4, 6), 36: (6, 6), 48: (6, 8), 60: (6, 10)}

_HINT_NONE = "none"             # cache sentinel: hint computed, none exists


def _failure_metrics() -> dict:
    """Maximally infeasible readout for a diverged/failed simulation."""
    return {
        "mean_delay_s": float("inf"),
        "max_queue_frac": float("inf"),
        "throughput_frac": 0.0,
        "total_stops": float("inf"),
        "max_density_err": float("inf"),
    }


class TrafficDomain(Domain):
    """CTM network signal-timing domain (variants: traffic, traffic_stoch)."""

    name = "traffic"
    params = ("green_frac",)
    reference_class = "heuristic"

    def __init__(self, stochastic: bool = False):
        super().__init__(stochastic)

    # ── problems ──────────────────────────────────────────────────────────────
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[index % len(_FAMILIES)]
        rung = _SIZE_LADDER[(index // len(_FAMILIES)) % len(_SIZE_LADDER)]
        # larger networks get a shorter horizon so one evaluate stays cheap
        horizon_cap = max(900.0, 54000.0 / float(rung))
        horizon_s = float(min(rng.uniform(900.0, 1800.0), horizon_cap))
        goals = {
            "maximum_max_queue_frac": round(float(rng.uniform(0.90, 0.96)), 4),
            "minimum_throughput_frac": round(float(rng.uniform(0.86, 0.94)), 4),
        }
        spec = {
            "problem_id": "traffic_%s_%04d" % (family, index),
            "family": family,
            "goals": goals,
            "horizon_s": round(horizon_s, 1),
            "dt_s": DT_S,
            "initial_green": 0.5,
        }
        # Demand calibration. Writing D = d_main + d_cross at one intersection,
        # the approach capacities under green split g are (g - LOST)*CAP and
        # (1 - g - LOST)*CAP, so:
        #   * SOME g serves both directions  <=>  D <= USABLE      (solvable)
        #   * g = 0.5 starves the heavy one  <=>  d_heavy > USABLE/2 (infeasible start)
        # Drawing rho = D/USABLE and the heavy direction's share so that
        # share*rho > 0.5 while rho < 1 makes every problem provably start
        # infeasible AND provably admit a green window that fixes it. Without
        # this coupling the set was 100% feasible at birth and produced no
        # restoration data at all.
        rho = float(rng.uniform(0.72, 0.90))
        share = float(rng.uniform(max(0.58, 0.56 / rho), 0.86))
        heavy_d = share * rho * USABLE          # starved at g = 0.5
        light_d = (1.0 - share) * rho * USABLE
        spec["_rho"] = round(rho, 4)
        spec["_share"] = round(share, 4)

        if family == "grid":
            rows, cols = _GRID_DIMS.get(rung, (2, max(2, rung // 2)))
            rows_heavy = bool(rng.random() < 0.5)   # rows or columns starved
            row_raw = rng.uniform(0.55, 1.0, size=rows)
            col_raw = rng.uniform(0.55, 1.0, size=cols)
            row_raw[int(rng.integers(rows))] = 1.0  # worst row/col hits the target
            col_raw[int(rng.integers(cols))] = 1.0
            row_d = row_raw * ((heavy_d if rows_heavy else light_d))
            col_d = col_raw * ((light_d if rows_heavy else heavy_d))
            spec.update({
                "rows": int(rows),
                "cols": int(cols),
                "n_int": int(rows * cols),
                "rows_heavy": rows_heavy,
                "row_demand": [round(float(v), 6) for v in row_d],
                "col_demand": [round(float(v), 6) for v in col_d],
            })
        else:
            n_int = int(rung)
            if family == "corridor":
                # arterial: the main (through) direction is the starved one at
                # every intersection, so the fix is uniformly "more main green".
                d_main = heavy_d
                cross = light_d * rng.uniform(0.55, 1.0, size=n_int)
                cross[int(rng.integers(n_int))] = light_d
            else:
                # asymmetric: a light arterial with a bimodal set of cross
                # demands - some intersections are cross-starved, so the
                # corrective direction differs along the same corridor.
                d_main = light_d
                is_heavy = rng.random(n_int) < 0.5
                if not is_heavy.any():
                    is_heavy[int(rng.integers(n_int))] = True
                cross = np.where(
                    is_heavy,
                    heavy_d * rng.uniform(0.92, 1.0, size=n_int),
                    light_d * rng.uniform(0.35, 0.70, size=n_int))
            # never exceed the joint capacity: keep every intersection solvable
            cross = np.minimum(cross, 0.94 * USABLE - d_main)
            cross = np.maximum(cross, 0.03)
            spec.update({
                "n_int": n_int,
                "main_demand": round(float(d_main), 6),
                "cross_demand": [round(float(v), 6) for v in cross],
            })
        return spec

    def load(self, spec: dict) -> dict:
        n_int = int(spec["n_int"])
        A = 2 * n_int                       # approach 2i = main_i, 2i+1 = cross_i
        feeder = np.full(A, -1, dtype=np.int64)
        demand = np.zeros(A, dtype=float)   # boundary demand rate (veh/s)
        main_flow = np.zeros(n_int, dtype=float)
        cross_flow = np.zeros(n_int, dtype=float)
        if str(spec["family"]) == "grid":
            rows, cols = int(spec["rows"]), int(spec["cols"])
            row_d = [float(v) for v in spec["row_demand"]]
            col_d = [float(v) for v in spec["col_demand"]]
            for r in range(rows):
                for c in range(cols):
                    i = r * cols + c
                    main_flow[i] = row_d[r]
                    cross_flow[i] = col_d[c]
                    if c > 0:
                        feeder[2 * i] = 2 * (i - 1)
                    else:
                        demand[2 * i] = row_d[r]
                    if r > 0:
                        feeder[2 * i + 1] = 2 * (i - cols) + 1
                    else:
                        demand[2 * i + 1] = col_d[c]
        else:
            d_main = float(spec["main_demand"])
            cross = [float(v) for v in spec["cross_demand"]]
            for i in range(n_int):
                main_flow[i] = d_main
                cross_flow[i] = cross[i]
                if i > 0:
                    feeder[2 * i] = 2 * (i - 1)
                else:
                    demand[2 * i] = d_main
                demand[2 * i + 1] = cross[i]
        dest = np.full(A, -1, dtype=np.int64)
        for a in range(A):
            f = int(feeder[a])
            if f >= 0:
                dest[f] = a
        topo = {
            "A": A,
            "fed_idx": np.nonzero(feeder >= 0)[0],
            "feed_src": feeder[feeder >= 0],
            "dest_safe": np.maximum(dest, 0),
            "exit_idx": np.nonzero(dest < 0)[0],
            "demand": demand,
            "steps": int(round(float(spec["horizon_s"]) / DT_S)),
            "main_flow": main_flow,
            "cross_flow": cross_flow,
        }
        return {
            "spec": spec,
            "n_int": n_int,
            "greens": np.full(n_int, float(spec.get("initial_green", 0.5))),
            "_topo": topo,            # read-only, shared across clones
            "_hint": None,
        }

    def clone(self, st) -> dict:
        return {
            "spec": st["spec"],
            "n_int": st["n_int"],
            "greens": np.array(st["greens"], dtype=float, copy=True),
            "_topo": st["_topo"],
            "_hint": None,            # reset cached diagnostics
        }

    # ── simulation ────────────────────────────────────────────────────────────
    def _simulate(self, st, seed: int, want_diag: bool = False,
                  deterministic: bool = False):
        topo = st["_topo"]
        A = int(topo["A"])
        steps = int(topo["steps"])
        g = np.asarray(st["greens"], dtype=float)

        ge = np.empty(A, dtype=float)                 # effective green fraction
        ge[0::2] = np.maximum(g - LOST, GE_MIN)
        ge[1::2] = np.maximum(1.0 - g - LOST, GE_MIN)
        qg = ge * Q_STEP                              # stop-line capacity
        w_sig = 0.5 * CYCLE_S * (1.0 - ge) ** 2       # uniform delay (s/veh)

        base = topo["demand"] * DT_S                  # veh per step
        if self.stochastic and not deterministic:
            rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
            fac = rng.lognormal(0.0, NOISE_SIGMA, size=(N_INTERVALS, A))
            fac /= math.exp(0.5 * NOISE_SIGMA * NOISE_SIGMA)  # E[factor] = 1
            d_by_p = base[None, :] * fac
        else:
            d_by_p = np.broadcast_to(base, (N_INTERVALS, A))
        dsum_by_p = d_by_p.sum(axis=1)
        interval_len = max(1, int(math.ceil(steps / float(N_INTERVALS))))

        dest_s = topo["dest_safe"]
        exit_idx = topo["exit_idx"]
        fed_idx = topo["fed_idx"]
        feed_src = topo["feed_src"]

        n0 = np.zeros(A)
        n1 = np.zeros(A)
        n2 = np.zeros(A)
        src_q = np.zeros(A)                 # boundary source queues (veh)
        out_by_appr = np.zeros(A)           # cumulative stop-line discharge
        # Running per-approach max of the TOTAL waiting queue. This must
        # include src_q: the in-network cells saturate at a level set by the
        # green split alone (inflow is capped by the receiving term), so
        # oversaturated demand piles up in the source queue and is otherwise
        # invisible. Measuring cells only makes the queue constraint a phantom
        # that no demand can bind.
        occ = np.zeros(A)

        delay_veh_steps = 0.0
        blocked = 0.0
        demanded = 0.0
        mx = 0.0
        mn = 0.0
        d_cur = d_by_p[0]
        d_sum = float(dsum_by_p[0])
        p_prev = 0

        for t in range(steps):
            p = t // interval_len
            if p != p_prev:
                if p >= N_INTERVALS:
                    p = N_INTERVALS - 1
                d_cur = d_by_p[p]
                d_sum = float(dsum_by_p[p])
                p_prev = p
            src_q += d_cur
            demanded += d_sum
            present = float(n0.sum() + n1.sum() + n2.sum() + src_q.sum())

            # sending / receiving (triangular fundamental diagram)
            s0 = np.minimum(n0, Q_STEP)
            y0 = np.minimum(s0, DELTA * (N_JAM - n1))
            s1 = np.minimum(n1, Q_STEP)
            y1 = np.minimum(s1, DELTA * (N_JAM - n2))
            s2 = np.minimum(n2, qg)                       # signal-capped
            recv = DELTA * (N_JAM - n0[dest_s])
            recv[exit_idx] = Q_STEP                       # exits discharge freely
            y2 = np.minimum(s2, recv)
            inflow = np.minimum(np.minimum(src_q, Q_STEP),
                                DELTA * (N_JAM - n0))
            src_q -= inflow

            n0 += inflow - y0
            n0[fed_idx] += y2[feed_src]
            n1 += y0 - y1
            n2 += y1 - y2
            out_by_appr += y2

            y_moved = float(y0.sum() + y1.sum() + y2.sum())
            moved = y_moved + float(inflow.sum())
            delay_veh_steps += present - moved
            blocked += float(s0.sum() + s1.sum() + s2.sum()) - y_moved

            step_max = max(float(n0.max()), float(n1.max()), float(n2.max()))
            if step_max > mx:
                mx = step_max
            step_min = min(float(n0.min()), float(n1.min()), float(n2.min()))
            if step_min < mn:
                mn = step_min
            np.maximum(occ, n0 + n1 + n2 + src_q, out=occ)

        served = float(out_by_appr[exit_idx].sum())
        total_delay_s = (delay_veh_steps * DT_S
                         + float((out_by_appr * w_sig).sum()))
        if served > 1e-9:
            mean_delay = max(total_delay_s / served, DELAY_FLOOR_S)
            stops = blocked / served
        else:
            mean_delay = float("inf")
            stops = float("inf")
        # Queue as a fraction of an approach's physical storage (all its cells).
        # > 1 means the queue has outgrown the approach: spillback.
        storage = CELLS * N_JAM
        metrics = {
            "mean_delay_s": mean_delay,
            "max_queue_frac": float(occ.max()) / storage,
            "throughput_frac": (served / demanded) if demanded > 0.0 else 0.0,
            "total_stops": stops,
            # CTM validity is about the CELLS only (src_q is not a cell).
            "max_density_err": max(0.0, (mx - N_JAM) / N_JAM, -mn / N_JAM),
        }
        diag = None
        if want_diag:
            diag = {"worst_approach": int(np.argmax(occ))}
        return metrics, diag

    def evaluate(self, st, seed: int = 0) -> dict:
        self.calls += 1
        try:
            metrics, _ = self._simulate(st, seed, want_diag=False)
        except Exception:
            metrics = _failure_metrics()
        out = {}
        for k, v in metrics.items():
            fv = float(v)
            if math.isnan(fv):   # never let a NaN through silently
                fv = 0.0 if k == "throughput_frac" else float("inf")
            out[k] = fv
        return out

    # ── design variables ──────────────────────────────────────────────────────
    def n_elements(self, st) -> int:
        return int(st["n_int"])

    def get(self, st, i: int, p: str) -> Optional[float]:
        if p != "green_frac" or not (0 <= int(i) < int(st["n_int"])):
            return None
        return float(st["greens"][int(i)])

    def set(self, st, i: int, p: str, v: float) -> None:
        if p != "green_frac" or not (0 <= int(i) < int(st["n_int"])):
            return
        st["greens"][int(i)] = min(G_MAX, max(G_MIN, float(v)))
        st["_hint"] = None            # invalidate cached diagnostics

    def bounds(self, st, p: str) -> Tuple[float, float]:
        return (G_MIN, G_MAX)

    # ── program / reference ───────────────────────────────────────────────────
    def program(self, spec: dict) -> DesignProgram:
        prog = program_from_goals(
            spec.get("goals", {}),
            objective_key="mean_delay_s",
            objective_ref=self.reference(spec),
            objective_sense="min",
            domain=self.name,
        )
        horizon = float(spec.get("horizon_s", 1800.0))
        # Envelope: a correct CTM keeps every cell inside [0, N_jam]; vehicles
        # out can never exceed vehicles demanded; and a mean delay at or past
        # the horizon means (near-)nobody was served - the simulator's answer
        # is semantically meaningless there, whatever Phi says.
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
        st = self.load(spec)
        topo = st["_topo"]
        tot = topo["main_flow"] + topo["cross_flow"]
        share = topo["main_flow"] / np.maximum(tot, 1e-9)
        st["greens"] = np.clip(share, G_MIN, G_MAX)   # proportional split rule
        try:
            metrics, _ = self._simulate(st, 0, want_diag=False,
                                        deterministic=True)
            ref = float(metrics["mean_delay_s"])
        except Exception:
            ref = float("inf")
        if not math.isfinite(ref) or ref <= 0.0:
            # heuristic itself gridlocked: fall back to a horizon-scale delay
            ref = 0.25 * float(spec.get("horizon_s", 1800.0))
        ref = max(ref, DELAY_FLOOR_S)
        spec["_ref_mean_delay_s"] = float(ref)
        return float(ref)

    # ── depth-0 feedback hint ─────────────────────────────────────────────────
    def hint(self, st) -> Optional[Tuple[int, str, str]]:
        cached = st.get("_hint")
        if cached is not None:
            return None if cached == _HINT_NONE else cached
        try:
            _, diag = self._simulate(st, 0, want_diag=True,
                                     deterministic=True)
        except Exception:
            st["_hint"] = _HINT_NONE
            return None
        a = int(diag["worst_approach"])
        # worst queue on the main approach -> give it more green ("up");
        # on the cross approach -> give main less green ("down").
        h = (a // 2, "green_frac", "up" if a % 2 == 0 else "down")
        st["_hint"] = h
        return h

    # ── self-contained observation (additive; used by the CoT renderer) ───────
    @staticmethod
    def _obs_num(x, sig: int = 3) -> str:
        """Compact significant-figure formatting (a nonzero value never prints
        as a bare 0)."""
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

    def _obs_position(self, st, i: int) -> str:
        """Position of intersection i in the network ("grid r1c2", "arterial
        start", "arterial", "arterial end")."""
        spec = st["spec"]
        n = int(st["n_int"])
        i = int(i)
        if str(spec["family"]) == "grid":
            r, c = divmod(i, int(spec["cols"]))
            return "grid r%dc%d" % (r, c)
        if i == 0:
            return "arterial start"
        if i == n - 1:
            return "arterial end"
        return "arterial"

    def _obs_stats(self, st) -> dict:
        """Per-approach nominal replay of _simulate (constant mean demand, no
        noise). Read-only: same update equations, extra accumulators only."""
        topo = st["_topo"]
        A = int(topo["A"])
        steps = int(topo["steps"])
        g = np.array(st["greens"], dtype=float, copy=True)
        ge = np.empty(A, dtype=float)
        ge[0::2] = np.maximum(g - LOST, GE_MIN)
        ge[1::2] = np.maximum(1.0 - g - LOST, GE_MIN)
        qg = ge * Q_STEP
        w_sig = 0.5 * CYCLE_S * (1.0 - ge) ** 2
        d = np.asarray(topo["demand"], dtype=float) * DT_S
        dest_s = topo["dest_safe"]
        exit_idx = topo["exit_idx"]
        fed_idx = topo["fed_idx"]
        feed_src = topo["feed_src"]

        n0 = np.zeros(A)
        n1 = np.zeros(A)
        n2 = np.zeros(A)
        src_q = np.zeros(A)
        out = np.zeros(A)            # stop-line discharge (veh)
        arr = np.zeros(A)            # vehicles arriving at the approach (veh)
        occ = np.zeros(A)            # peak total queue incl. entry backlog
        wait = np.zeros(A)           # queueing delay (veh-steps)
        blk = np.zeros(A)            # steps with stop line blocked downstream
        for _ in range(steps):
            src_q += d
            arr += d
            pres = n0 + n1 + n2 + src_q
            s0 = np.minimum(n0, Q_STEP)
            y0 = np.minimum(s0, DELTA * (N_JAM - n1))
            s1 = np.minimum(n1, Q_STEP)
            y1 = np.minimum(s1, DELTA * (N_JAM - n2))
            s2 = np.minimum(n2, qg)
            recv = DELTA * (N_JAM - n0[dest_s])
            recv[exit_idx] = Q_STEP
            y2 = np.minimum(s2, recv)
            inflow = np.minimum(np.minimum(src_q, Q_STEP),
                                DELTA * (N_JAM - n0))
            src_q -= inflow
            n0 += inflow - y0
            n0[fed_idx] += y2[feed_src]
            arr[fed_idx] += y2[feed_src]
            n1 += y0 - y1
            n2 += y1 - y2
            out += y2
            wait += pres - (y0 + y1 + y2 + inflow)
            blk += (recv < s2 - 1e-9)
            np.maximum(occ, n0 + n1 + n2 + src_q, out=occ)
        horizon = max(steps * DT_S, DT_S)
        tot_delay = wait * DT_S + out * w_sig
        return {
            "ge": ge,
            "w_sig": w_sig,
            "arr_rate": arr / horizon,
            "out_rate": out / horizon,
            "q_frac": occ / (CELLS * N_JAM),
            "q_veh": occ,
            "held_end": src_q.copy(),
            "dly": tot_delay / np.maximum(out, 1e-9),
            "dly_tot": tot_delay,
            "blk_frac": blk / float(max(steps, 1)),
            "steps": steps,
        }

    def _obs_dem(self, st) -> Tuple[np.ndarray, np.ndarray]:
        topo = st["_topo"]
        return (np.asarray(topo["main_flow"], dtype=float),
                np.asarray(topo["cross_flow"], dtype=float))

    def describe(self, st, spec) -> str:
        f = self._obs_num
        n = int(st["n_int"])
        fam = str(spec["family"])
        steps = int(st["_topo"]["steps"])
        dm, dx = self._obs_dem(st)
        L = []
        # ── PROBLEM ──────────────────────────────────────────────────────────
        horizon_txt = ("cell transmission traffic model, %d s horizon = %d "
                       "steps of %g s, %s arrival rates"
                       % (int(round(steps * DT_S)), steps, DT_S,
                          "noisy" if self.stochastic else "constant"))
        if fam == "grid":
            rows, cols = int(spec["rows"]), int(spec["cols"])
            L.append("PROBLEM: fixed-time signal timing, %d x %d grid of %d "
                     "signalised intersections (%s)."
                     % (rows, cols, n, horizon_txt))
            L.append("Topology: I = row*%d + col. MAIN runs west->east along "
                     "each row: the main approach of (r,c) is fed only by the "
                     "main discharge of (r,c-1); column 0 is the row's network "
                     "entry; exits after column %d. CROSS runs north->south "
                     "along each column: the cross approach of (r,c) is fed by "
                     "the cross discharge of (r-1,c); row 0 is the entry; "
                     "exits after row %d. No turns." % (cols, cols - 1, rows - 1))
            L.append("Demand veh/s - MAIN by row r0..r%d: %s. CROSS by column "
                     "c0..c%d: %s."
                     % (rows - 1, " ".join(f(v) for v in spec["row_demand"]),
                        cols - 1, " ".join(f(v) for v in spec["col_demand"])))
        else:
            kind = ("an arterial corridor" if fam == "corridor" else
                    "an asymmetric corridor (light arterial; side streets "
                    "bimodal - some heavy, some nearly idle)")
            L.append("PROBLEM: fixed-time signal timing on %s of %d "
                     "signalised intersections (%s)." % (kind, n, horizon_txt))
            L.append("Topology: MAIN chains along the arterial I0 -> I1 -> "
                     "... -> I%d: the main approach of I(k) is fed only by the "
                     "main discharge of I(k-1); main traffic enters at I0 (%s "
                     "veh/s) and exits after I%d. Each CROSS street is its own "
                     "network entry and exits after crossing. No turns."
                     % (n - 1, f(dm[0]), n - 1))
        L.append("Approach: %d cells x %g veh jam storage = %g veh; "
                 "saturation flow %g veh/s; wave-speed ratio %g. An approach "
                 "whose entry cell is (nearly) full cannot accept the upstream "
                 "discharge and blocks the upstream stop line (spillback); at "
                 "a network entry the excess waits outside but counts in that "
                 "approach's queue, which can then exceed storage."
                 % (CELLS, N_JAM, CELLS * N_JAM, CAP_VPS, DELTA))
        L.append("Signal: %g s cycle, two phases (MAIN green, CROSS green). "
                 "Tunable per intersection: green_frac g in [%g, %g] "
                 "(dimensionless) = MAIN-direction share of the cycle; the "
                 "cross street gets 1 - g. Each phase loses %g of the cycle: "
                 "effective green MAIN = g - %g, CROSS = %g - g (floor %g). An "
                 "approach discharges at most effective green x %g veh/s, so "
                 "it is oversaturated when demand exceeds that (v/c > 1). "
                 "Raising g helps MAIN and hurts CROSS at that intersection. "
                 "Each vehicle crossing a stop line also pays uniform signal "
                 "delay 0.5 x %g s x (1 - effective green)^2."
                 % (CYCLE_S, G_MIN, G_MAX, LOST, LOST, 1.0 - LOST, GE_MIN,
                    CAP_VPS, CYCLE_S))
        if self.stochastic:
            hi = math.exp(-0.5 * NOISE_SIGMA ** 2 + 1.2816 * NOISE_SIGMA)
            L.append("Demand noise: the horizon is split into %d equal "
                     "intervals; each entry's demand is scaled per interval by "
                     "an independent random lognormal factor (sigma %g, mean "
                     "1). A 1-in-10 high interval has factor ~%s, so an "
                     "approach with nominal v/c above ~%s is oversaturated "
                     "during it. RESULTS are a nominal evaluation at mean "
                     "demand." % (N_INTERVALS, NOISE_SIGMA, f(hi),
                                  f(1.0 / hi, 2)))
        # ── DESIGN ───────────────────────────────────────────────────────────
        S = self._obs_stats(st)
        ge = S["ge"]
        L.append("")
        if fam == "grid":
            cols = int(spec["cols"])
            L.append("DESIGN (one line per intersection Ik; rRcC = grid row R, "
                     "column C; g = green_frac; dem = nominal demand "
                     "MAIN/CROSS, veh/s; cap = capacity MAIN/CROSS, veh/s. "
                     "The MAIN approach of Ik is fed by "
                     "I(k-1) and its CROSS approach by I(k-%d); column-0 MAIN "
                     "and row-0 CROSS approaches are network entries):" % cols)
        else:
            L.append("DESIGN (one line per intersection Ik; g = green_frac; "
                     "dem = nominal demand MAIN/CROSS, veh/s; eff green and "
                     "cap = effective green and capacity MAIN/CROSS, veh/s. The MAIN "
                     "approach of Ik is fed by I(k-1), I0 MAIN is the arterial "
                     "entry; every CROSS approach is a network entry):")
        g = np.asarray(st["greens"], dtype=float)
        for i in range(n):
            if fam == "grid":
                pos = " r%dc%d" % divmod(i, cols)
            elif i == 0:
                pos = " [arterial start]"
            elif i == n - 1:
                pos = " [arterial end]"
            else:
                pos = ""
            # effective green and capacity printed, not left to the reader:
            # traces otherwise did the lost-time arithmetic themselves
            L.append("I%d%s g=%s dem %s/%s cap %s/%s"
                     % (i, pos, "%.5g" % g[i], f(dm[i]), f(dx[i]),
                        f(ge[2 * i] * CAP_VPS), f(ge[2 * i + 1] * CAP_VPS)))
        # ── RESULTS ──────────────────────────────────────────────────────────
        L.append("")
        L.append("RESULTS (%sper approach, M = MAIN, X = CROSS; v/c = nominal "
                 "demand / capacity (an approach fed by an upstream "
                 "intersection receives at most that one's discharge, so its "
                 "actual load can be lower); Q = peak queue as a MULTIPLE of the %g-veh storage (a ratio, not vehicles) "
                 "(>1 = SPILLBACK: the excess is held outside the network "
                 "entry); dly = mean delay per discharged vehicle, s (queueing "
                 "+ signal); blocked P%% of time by Ik = during P%% of the time "
                 "steps this stop line's discharge was cut (not stopped) because "
                 "the same-direction approach at Ik had too little room left):"
                 % ("nominal evaluation, " if self.stochastic else "",
                    CELLS * N_JAM))
        dest = st["_topo"]["dest_safe"]
        for i in range(n):
            parts = []
            for tag, a, dem in (("M", 2 * i, dm[i]), ("X", 2 * i + 1, dx[i])):
                s = "%s v/c %s Q %sx dly %s" % (
                    tag, f(dem / max(ge[a] * CAP_VPS, 1e-12)),
                    f(S["q_frac"][a]), f(S["dly"][a]))
                if S["q_frac"][a] > 1.0:
                    s += " SPILLBACK"
                if S["blk_frac"][a] > 0.0:
                    s += " blocked %s%% of time by I%d" % (
                        f(100.0 * S["blk_frac"][a], 2), int(dest[a]) // 2)
                parts.append(s)
            L.append("I%d %s | %s" % (i, parts[0], parts[1]))
        return "\n".join(L)

    def element_label(self, st, i) -> str:
        i = int(i)
        if not (0 <= i < int(st["n_int"])):
            return "intersection %d" % i
        dm, dx = self._obs_dem(st)
        return ("intersection %d (%s; demand main %s / cross %s veh/s)"
                % (i, self._obs_position(st, i), self._obs_num(dm[i]),
                   self._obs_num(dx[i])))
