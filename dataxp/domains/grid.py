"""DC power-flow economic dispatch on synthetic grids (dataxp domain).

Model
-----
Connected synthetic network with n_bus in [14, 120] and three families:
"radial" (random tree + 0-2 chords), "meshed" (ring + ~0.25-0.5n chords),
"congested" (ring + chords, deliberately tight thermal limits). Every line
carries a susceptance b and a thermal limit; loads are fixed per bus; G in
[3, 12] generators sit on non-slack buses with linear cost c_g and
p_g in [0, p_max_g]. Bus 0 is the slack and absorbs any imbalance at a
penalty price.  DC power flow: solve B_red * theta = P_injection (theta_0=0),
line flow = b_ij (theta_i - theta_j).

Metrics: total_cost = sum c_g p_g + slack_price * |slack| (objective, min),
max_line_loading = max |flow|/limit (goal <= 1.0; also emitted under the
constraint-resolved alias key "line_loading"), slack_abs_mw = |sum(load) -
sum(p)| (goal <= a few MW), n_overloaded.

Elements are the generators, the single tunable parameter is "p_set" (MW);
set() clamps to the per-generator [0, p_max_g]. The reference optimum is the
exact DC-OPF LP (scipy.optimize.linprog, HiGHS; PTDF flow constraints, slack
linearised via t >= |slack|), so reference_class = "exact" ("heuristic" for
the stochastic variant, where the LP is solved at nominal load). Stochastic
mode multiplies every load by a per-seed lognormal factor exp(sigma z -
sigma^2/2) (mean 1, family-dependent sigma in 0.05-0.15) drawn from
np.random.default_rng(seed) only. The initial dispatch serves the whole load
from the cheapest generators (plus a small floor on the others so
multiplicative moves stay live), which overloads lines: an infeasible start.
Validity envelope: max_line_loading <= 10 (beyond that the DC linearisation
means nothing) and slack_abs_mw <= nominal total load (a dispatch leaning
entirely on the slack is not a dispatch).

Hint: PTDF sensitivity of the most-loaded line to each generator's injection;
the largest-|sensitivity| generator movable in the relieving direction.

numpy + scipy + stdlib only; no I/O, no prints, no global randomness.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from ..domain import Domain
from ..potential import DesignProgram, program_from_goals

_FAMILIES = ("radial", "meshed", "congested")
_SIZE_LADDER = (14, 20, 28, 40, 57, 72, 96, 118)
# "line_loading" is the alias program_from_goals resolves "maximum_line_loading"
# to; it always equals max_line_loading.
_METRIC_KEYS = ("total_cost", "line_loading", "max_line_loading",
                "slack_abs_mw", "n_overloaded")
_LOADING_CAP = 10.0           # validity envelope on max_line_loading
_MAKE_ATTEMPTS = 64


def _laplacian_reduced(n: int, li: np.ndarray, lj: np.ndarray,
                       b: np.ndarray) -> np.ndarray:
    """Susceptance-weighted Laplacian with the slack row/column removed."""
    B = np.zeros((n, n))
    np.add.at(B, (li, li), b)
    np.add.at(B, (lj, lj), b)
    np.add.at(B, (li, lj), -b)
    np.add.at(B, (lj, li), -b)
    return np.ascontiguousarray(B[1:, 1:])


def _dc_flows(li: np.ndarray, lj: np.ndarray, b: np.ndarray,
              B_red: np.ndarray, inj: np.ndarray) -> np.ndarray:
    """Line flows for the full injection vector inj (bus 0 = slack, theta_0=0)."""
    theta = np.zeros(inj.shape[0])
    theta[1:] = np.linalg.solve(B_red, inj[1:])
    return b * (theta[li] - theta[lj])


class GridDomain(Domain):
    """DC-OPF economic dispatch: rebalance generation to clear line overloads
    and keep the slack injection near zero, at minimum cost."""

    name = "grid"
    params = ("p_set",)
    # 2.0/0.5 keep near-zero generators reachable under multiplicative moves;
    # 0.98/1.02 allow fine balancing against the tight slack budget.
    factors = (0.5, 0.8, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.25, 2.0)
    reference_class = "exact"

    def __init__(self, stochastic: bool = False):
        super().__init__(stochastic)
        if self.stochastic:
            # nominal-load LP is only a heuristic reference under load noise
            self.reference_class = "heuristic"

    # ── problems ─────────────────────────────────────────────────────────────
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[index % len(_FAMILIES)]
        allow_easy = bool(rng.random() < 0.10)   # ~10% may start feasible
        fallback = None
        for _ in range(_MAKE_ATTEMPTS):
            try:
                spec = self._draw_spec(index, family, rng)
                st = self.load(spec)
                ref = self._reference_lp(st)
                if ref is None:
                    continue                      # LP infeasible: regenerate
                spec["ref_cost"] = float(ref)
                m0 = self._metrics(st, st["loads"])
                s_lim = float(spec["goals"]["maximum_slack_abs_mw"])
                infeasible0 = (m0["max_line_loading"] > 1.0 + 1e-9
                               or m0["slack_abs_mw"] > s_lim + 1e-9)
                spec["initially_infeasible_nominal"] = bool(infeasible0)
                if infeasible0 or allow_easy:
                    return spec
                if fallback is None:
                    fallback = spec
            except Exception:
                continue
        if fallback is not None:
            return fallback
        raise RuntimeError("grid.make_problem: no LP-feasible problem in %d "
                           "attempts (index=%d)" % (_MAKE_ATTEMPTS, index))

    def _draw_spec(self, index: int, family: str,
                   rng: np.random.Generator) -> dict:
        base = int(_SIZE_LADDER[int(rng.integers(len(_SIZE_LADDER)))])
        n = int(np.clip(base + int(rng.integers(-2, 3)), 14, 120))

        edges = set()
        ei, ej = [], []

        def add_edge(a: int, c: int) -> bool:
            if a == c:
                return False
            key = (min(a, c), max(a, c))
            if key in edges:
                return False
            edges.add(key)
            ei.append(key[0])
            ej.append(key[1])
            return True

        if family == "radial":
            for k in range(1, n):
                add_edge(int(rng.integers(max(0, k - 6), k)), k)
            n_chords = int(rng.integers(0, 3))
        elif family == "meshed":
            for k in range(n):
                add_edge(k, (k + 1) % n)
            n_chords = int(round(n * float(rng.uniform(0.25, 0.50))))
        else:  # congested
            for k in range(n):
                add_edge(k, (k + 1) % n)
            n_chords = int(round(n * float(rng.uniform(0.10, 0.30))))
        added, attempts = 0, 0
        while added < n_chords and attempts < 8 * n_chords + 16:
            attempts += 1
            if add_edge(int(rng.integers(0, n)), int(rng.integers(0, n))):
                added += 1

        li = np.asarray(ei, dtype=int)
        lj = np.asarray(ej, dtype=int)
        n_lines = int(li.shape[0])
        b = rng.uniform(4.0, 25.0, n_lines)

        # loads (nominal, MW); slack bus carries none
        load_scale = float(rng.uniform(15.0, 45.0))
        loads = np.zeros(n)
        for k in range(1, n):
            if rng.random() < 0.75:
                loads[k] = load_scale * float(rng.uniform(0.3, 1.0))
        if int(np.count_nonzero(loads)) < 3:
            for k in (1, 2, 3):
                loads[k] = load_scale * float(rng.uniform(0.3, 1.0))
        total_load = float(loads.sum())

        # generators on distinct non-slack buses
        G = int(rng.integers(3, 13))
        gen_bus = np.sort(rng.choice(np.arange(1, n), size=G,
                                     replace=False)).astype(int)
        cost = rng.uniform(10.0, 100.0, G)
        frac = rng.uniform(0.5, 1.5, G)
        margin = float(rng.uniform(1.35, 1.9))
        pmax = frac / float(frac.sum()) * margin * total_load
        slack_price = float(rng.uniform(3.0, 6.0)) * float(cost.max())

        # initial dispatch: small floor on every unit, remainder greedily on
        # the cheapest -> concentrated flows, balanced total (slack ~ 0)
        p0 = rng.uniform(0.02, 0.08, G) * pmax
        residual = total_load - float(p0.sum())
        if residual < 0.0:
            p0 *= total_load / float(p0.sum())
            residual = 0.0
        for g in np.argsort(cost):
            take = min(float(pmax[g] - p0[g]), residual)
            p0[g] += take
            residual -= take
            if residual <= 1e-12:
                break

        # thermal limits calibrated to a proportional 'spread' dispatch, which
        # is therefore feasible by construction (the LP always has a point)
        headroom = float({"radial": rng.uniform(1.5, 2.2),
                          "meshed": rng.uniform(1.25, 1.8),
                          "congested": rng.uniform(1.02, 1.22)}[family])
        p_spread = pmax * (total_load / float(pmax.sum()))
        inj = -loads.copy()
        np.add.at(inj, gen_bus, p_spread)
        f_spread = np.abs(_dc_flows(li, lj, b, _laplacian_reduced(n, li, lj, b),
                                    inj))
        floor = 0.05 * float(f_spread.mean()) + 0.5
        limit = np.maximum(headroom * f_spread, floor)

        # Blend the greedy dispatch back toward the (feasible) spread dispatch
        # until the worst line sits at a TARGET overload. Starting at pure
        # greedy put the worst line at ~480% of its limit with 8 lines over,
        # which no budget of single-generator multiplicative moves can undo:
        # phi_select ground 4.82 -> 2.42 over 20 turns and still solved 0/6 at
        # 40 turns. Targeting the initial overload keeps every problem both
        # infeasible at birth and reachable, and the target's spread IS the
        # difficulty axis.
        p_greedy = p0.copy()
        target = float(rng.uniform(1.15, 2.30))

        def _loading(p_try):
            inj_t = -loads.copy()
            np.add.at(inj_t, gen_bus, p_try)
            f_t = np.abs(_dc_flows(li, lj, b,
                                   _laplacian_reduced(n, li, lj, b), inj_t))
            return float(np.max(f_t / np.maximum(limit, 1e-9)))

        lo_w, hi_w = 0.0, 1.0
        if _loading(p_greedy) > target:
            for _ in range(24):                     # bisect on the blend weight
                mid = 0.5 * (lo_w + hi_w)
                if _loading((1.0 - mid) * p_spread + mid * p_greedy) > target:
                    hi_w = mid
                else:
                    lo_w = mid
            w = hi_w                                # stay just ABOVE the target
            p0 = (1.0 - w) * p_spread + w * p_greedy
        p0 = np.clip(p0, 0.0, pmax)

        s_lim = round(max(1.5, total_load * float(rng.uniform(0.010, 0.030))), 4)
        sigma_lo, sigma_hi = {"radial": (0.05, 0.08), "meshed": (0.07, 0.11),
                              "congested": (0.10, 0.15)}[family]
        sigma = round(float(rng.uniform(sigma_lo, sigma_hi)), 4)

        return {
            "problem_id": "grid_%04d" % int(index),
            "family": family,
            "n_bus": n,
            "n_lines": n_lines,
            "lines": [[int(li[l]), int(lj[l]), round(float(b[l]), 5),
                       round(float(limit[l]), 5)] for l in range(n_lines)],
            "loads": [round(float(x), 5) for x in loads],
            "gen_bus": [int(x) for x in gen_bus],
            "gen_cost": [round(float(x), 5) for x in cost],
            "gen_pmax": [round(float(x), 5) for x in pmax],
            "p_init": [round(float(x), 5) for x in p0],
            "slack_price": round(slack_price, 4),
            "sigma_load": sigma,
            "total_load_mw": round(total_load, 5),
            "goals": {"maximum_line_loading": 1.0,
                      "maximum_slack_abs_mw": s_lim},
        }

    # ── state ────────────────────────────────────────────────────────────────
    def load(self, spec: dict):
        lines = spec["lines"]
        li = np.asarray([int(l[0]) for l in lines], dtype=int)
        lj = np.asarray([int(l[1]) for l in lines], dtype=int)
        b = np.asarray([float(l[2]) for l in lines], dtype=float)
        limit = np.asarray([float(l[3]) for l in lines], dtype=float)
        n = int(spec["n_bus"])
        gen_pmax = np.asarray(spec["gen_pmax"], dtype=float)
        return {
            "n_bus": n,
            "line_i": li,
            "line_j": lj,
            "line_b": b,
            "line_limit": limit,
            "loads": np.asarray(spec["loads"], dtype=float),
            "gen_bus": np.asarray(spec["gen_bus"], dtype=int),
            "gen_cost": np.asarray(spec["gen_cost"], dtype=float),
            "gen_pmax": gen_pmax,
            "B_red": _laplacian_reduced(n, li, lj, b),
            "p": np.clip(np.asarray(spec["p_init"], dtype=float), 0.0, gen_pmax),
            "slack_price": float(spec["slack_price"]),
            "sigma_load": float(spec.get("sigma_load", 0.08)),
            "s_lim": float(spec["goals"]["maximum_slack_abs_mw"]),
        }

    def clone(self, st):
        # "p" is the only array set() mutates; topology/loads are shared
        # read-only. There is no evaluation cache to reset.
        new = dict(st)
        new["p"] = np.array(st["p"], dtype=float)
        return new

    # ── simulation ───────────────────────────────────────────────────────────
    def evaluate(self, st, seed: int = 0) -> dict:
        try:
            loads = st["loads"]
            if self.stochastic:
                rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
                sigma = float(st["sigma_load"])
                z = rng.standard_normal(loads.shape[0])
                loads = loads * np.exp(sigma * z - 0.5 * sigma * sigma)
            return self._metrics(st, loads)
        except Exception:
            return self._failure_metrics(st)

    def _metrics(self, st, loads: np.ndarray) -> dict:
        try:
            p = st["p"]
            if not (np.all(np.isfinite(p)) and np.all(np.isfinite(loads))):
                return self._failure_metrics(st)
            inj = -np.asarray(loads, dtype=float)
            np.add.at(inj, st["gen_bus"], p)
            flow = _dc_flows(st["line_i"], st["line_j"], st["line_b"],
                             st["B_red"], inj)
            loading = np.abs(flow) / st["line_limit"]
            slack_abs = abs(float(np.sum(loads) - np.sum(p)))
            cost = float(np.dot(st["gen_cost"], p)
                         + st["slack_price"] * slack_abs)
            max_loading = float(np.max(loading)) if loading.size else 0.0
            if not (math.isfinite(cost) and math.isfinite(max_loading)
                    and math.isfinite(slack_abs)):
                return self._failure_metrics(st)
            return {
                "total_cost": cost,
                "line_loading": max_loading,     # constraint alias
                "max_line_loading": max_loading,
                "slack_abs_mw": slack_abs,
                "n_overloaded": float(np.count_nonzero(loading > 1.0)),
            }
        except Exception:
            return self._failure_metrics(st)

    def _failure_metrics(self, st) -> dict:
        try:
            n_lines = int(np.asarray(st["line_b"]).shape[0])
        except Exception:
            n_lines = 0
        inf = float("inf")
        return {"total_cost": inf, "line_loading": inf,
                "max_line_loading": inf, "slack_abs_mw": inf,
                "n_overloaded": float(n_lines)}

    # ── design variables ─────────────────────────────────────────────────────
    def n_elements(self, st) -> int:
        return int(st["gen_cost"].shape[0])

    def get(self, st, i: int, p: str) -> Optional[float]:
        if p != "p_set":
            return None
        return float(st["p"][int(i)])

    def set(self, st, i: int, p: str, v: float) -> None:
        if p != "p_set":
            raise ValueError("grid domain has no parameter %r" % (p,))
        i = int(i)
        v = float(v)
        if not math.isfinite(v):
            v = 0.0
        st["p"][i] = min(max(v, 0.0), float(st["gen_pmax"][i]))

    def bounds(self, st, p: str) -> Tuple[float, float]:
        if p != "p_set":
            raise ValueError("grid domain has no parameter %r" % (p,))
        return (0.0, float(np.max(st["gen_pmax"])))

    # ── program / reference ──────────────────────────────────────────────────
    def program(self, spec: dict) -> DesignProgram:
        prog = program_from_goals(
            spec["goals"],
            state_keys=_METRIC_KEYS,
            objective_key="total_cost",
            objective_ref=self.reference(spec),
            objective_sense="min",
            domain=self.name,
        )
        total_load = max(float(sum(float(x) for x in spec["loads"])), 1.0)
        return replace(prog, validity_bounds=(
            ("max_line_loading", _LOADING_CAP),
            ("slack_abs_mw", total_load),
        ))

    def reference(self, spec: dict) -> Optional[float]:
        ref = spec.get("ref_cost")
        if ref is None:
            ref = self._reference_lp(self.load(spec))
            if ref is None:
                return None
            spec["ref_cost"] = float(ref)   # deterministic; cache in spec
        return float(ref)

    def _reference_lp(self, st) -> Optional[float]:
        """Exact DC-OPF optimum: min c.p + slack_price*t  s.t. PTDF line-flow
        limits, t >= |total_load - sum p|, 0 <= p <= p_max, 0 <= t <= s_lim."""
        try:
            n = int(st["n_bus"])
            li, lj, b = st["line_i"], st["line_j"], st["line_b"]
            n_lines = int(b.shape[0])
            G = int(st["gen_cost"].shape[0])
            Bf = np.zeros((n_lines, n - 1))
            rows = np.arange(n_lines)
            mi = li > 0
            Bf[rows[mi], li[mi] - 1] += b[mi]
            mj = lj > 0
            Bf[rows[mj], lj[mj] - 1] -= b[mj]
            PTDF = np.linalg.solve(st["B_red"], Bf.T).T      # B_red symmetric
            M = PTDF[:, st["gen_bus"] - 1]                   # gens never at slack
            f0 = PTDF @ st["loads"][1:]                      # load withdrawals
            total_load = float(np.sum(st["loads"]))
            limit = st["line_limit"]
            c = np.concatenate([st["gen_cost"], [st["slack_price"]]])
            A_ub = np.vstack([
                np.hstack([M, np.zeros((n_lines, 1))]),      # flow <= limit
                np.hstack([-M, np.zeros((n_lines, 1))]),     # -flow <= limit
                np.concatenate([-np.ones(G), [-1.0]]),       # t >= load - sum p
                np.concatenate([np.ones(G), [-1.0]]),        # t >= sum p - load
            ])
            b_ub = np.concatenate([limit + f0, limit - f0,
                                   [-total_load, total_load]])
            var_bounds = ([(0.0, float(pm)) for pm in st["gen_pmax"]]
                          + [(0.0, float(st["s_lim"]))])
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=var_bounds,
                          method="highs")
            if not res.success:
                return None
            val = float(res.fun)
            return val if (math.isfinite(val) and val > 0.0) else None
        except Exception:
            return None

    # ── depth-0 feedback hint ────────────────────────────────────────────────
    def hint(self, st) -> Optional[Tuple[int, str, str]]:
        try:
            p = st["p"]
            pmax = st["gen_pmax"]
            loads = st["loads"]
            if not (np.all(np.isfinite(p)) and np.all(np.isfinite(loads))):
                return None
            inj = -loads.copy()
            np.add.at(inj, st["gen_bus"], p)
            flow = _dc_flows(st["line_i"], st["line_j"], st["line_b"],
                             st["B_red"], inj)
            loading = np.abs(flow) / st["line_limit"]
            can_up = p < pmax - 1e-9 * np.maximum(1.0, pmax)
            can_dn = p > 1e-9

            if loading.size and float(np.max(loading)) > 1.0:
                l = int(np.argmax(loading))
                rhs = np.zeros(int(st["n_bus"]) - 1)
                i, j = int(st["line_i"][l]), int(st["line_j"][l])
                bl = float(st["line_b"][l])
                if i > 0:
                    rhs[i - 1] += bl
                if j > 0:
                    rhs[j - 1] -= bl
                y = np.linalg.solve(st["B_red"], rhs)   # d flow_l / d P_bus
                sens = y[st["gen_bus"] - 1]
                sign = 1.0 if flow[l] >= 0.0 else -1.0
                for g in np.argsort(-np.abs(sens)):
                    g = int(g)
                    if abs(float(sens[g])) < 1e-9:
                        break
                    if float(sens[g]) * sign > 0.0:     # raising p_g worsens it
                        if can_dn[g]:
                            return (g, "p_set", "down")
                    elif can_up[g]:
                        return (g, "p_set", "up")

            slack = float(np.sum(loads) - np.sum(p))
            if abs(slack) > float(st["s_lim"]):
                if slack > 0.0:                          # under-generation
                    for g in np.argsort(st["gen_cost"]):
                        if can_up[int(g)]:
                            return (int(g), "p_set", "up")
                else:                                    # over-generation
                    for g in np.argsort(-st["gen_cost"]):
                        if can_dn[int(g)]:
                            return (int(g), "p_set", "down")
                return None

            # feasible: cost pressure blames the priciest running unit
            for g in np.argsort(-st["gen_cost"]):
                if p[int(g)] > 1e-6 and can_dn[int(g)]:
                    return (int(g), "p_set", "down")
            return None
        except Exception:
            return None

    # ── self-contained observation (additive; read-only on st) ────────────────
    @staticmethod
    def _fmt(x, sig: int = 4) -> str:
        """Compact significant-figure formatting; never rounds a nonzero small
        magnitude to '0'."""
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
        return "%.*e" % (max(0, sig - 1), x)          # sig significant digits

    @classmethod
    def _sfmt(cls, x, sig: int = 3) -> str:
        """Signed variant of _fmt (explicit '+' on positive values)."""
        s = cls._fmt(x, sig)
        return s if (s.startswith("-") or s == "0") else "+" + s

    def _nominal_flows(self, st) -> np.ndarray:
        """Line flows (MW, + = from line_i to line_j) at nominal loads and the
        current dispatch. Pure recomputation; st is not touched."""
        inj = -np.asarray(st["loads"], dtype=float).copy()
        np.add.at(inj, st["gen_bus"], np.asarray(st["p"], dtype=float))
        return _dc_flows(st["line_i"], st["line_j"], st["line_b"],
                         st["B_red"], inj)

    def _gen_ptdf(self, st) -> np.ndarray:
        """(n_lines, G) matrix: d flow_l / d p_g for 1 MW injected at the
        generator's bus and withdrawn at the slack bus 0."""
        n = int(st["n_bus"])
        li, lj, b = st["line_i"], st["line_j"], st["line_b"]
        n_lines = int(b.shape[0])
        Bf = np.zeros((n_lines, n - 1))
        rows = np.arange(n_lines)
        mi = li > 0
        Bf[rows[mi], li[mi] - 1] += b[mi]
        mj = lj > 0
        Bf[rows[mj], lj[mj] - 1] -= b[mj]
        ptdf = np.linalg.solve(st["B_red"], Bf.T).T
        return ptdf[:, np.asarray(st["gen_bus"], dtype=int) - 1]

    def element_label(self, st, i) -> str:
        i = int(i)
        return "generator %d @ bus %d (%s $/MWh, p_max %s MW)" % (
            i, int(st["gen_bus"][i]), self._fmt(st["gen_cost"][i], 3),
            self._fmt(st["gen_pmax"][i], 4))

    def describe(self, st, spec) -> str:
        f, sf = self._fmt, self._sfmt
        n = int(st["n_bus"])
        li = np.asarray(st["line_i"], dtype=int)
        lj = np.asarray(st["line_j"], dtype=int)
        b = np.asarray(st["line_b"], dtype=float)
        lim = np.asarray(st["line_limit"], dtype=float)
        loads = np.asarray(st["loads"], dtype=float)
        gbus = np.asarray(st["gen_bus"], dtype=int)
        cost = np.asarray(st["gen_cost"], dtype=float)
        pmax = np.asarray(st["gen_pmax"], dtype=float)
        p = np.array(st["p"], dtype=float)
        G = int(cost.shape[0])
        L = int(b.shape[0])
        family = str(spec.get("family", "synthetic"))
        fam_txt = {"radial": "radial: a spanning tree plus loop-closing lines",
                   "meshed": "meshed: a ring plus many cross-connecting lines",
                   "congested": "meshed ring with cross lines and tight thermal "
                                "limits"}.get(family, family)
        # connected by construction, so independent loops = lines - buses + 1
        fam_txt += "; %d independent loops" % max(L - n + 1, 0)
        total_load = float(loads.sum())
        n_load_bus = int(np.count_nonzero(loads))
        merit = np.empty(G, dtype=int)
        merit[np.argsort(cost, kind="stable")] = np.arange(1, G + 1)

        out = []
        out.append("PROBLEM")
        out.append("DC power-flow economic dispatch on a %d-bus, %d-line "
                   "transmission network (%s)." % (n, L, fam_txt))
        out.append("Fixed bus loads total %s MW on %d buses (listed below). "
                   "%d generators with linear energy cost c ($/MWh)."
                   % (f(total_load), n_load_bus, G))
        if self.stochastic:
            out.append("Loads are uncertain: each scenario scales every bus load "
                       "by an independent lognormal factor (mean 1, sigma %s). "
                       "All per-line and per-generator results below are a "
                       "nominal (mean-load) evaluation."
                       % f(st["sigma_load"], 3))
        out.append("Tunable parameter, one per generator: p_set = real-power "
                   "output setpoint in MW, bounded 0 <= p_set <= that "
                   "generator's p_max (values outside are clamped).")
        out.append("Bus 0 is the slack bus (no generator%s): it absorbs "
                   "the imbalance (total load - total p_set), billed at %s "
                   "$/MWh on its magnitude (the most expensive generator costs "
                   "%s $/MWh)." % ("" if loads[0] != 0.0 else ", no load",
                                   f(st["slack_price"]), f(float(cost.max()), 3)))
        out.append("Physics: bus angles solve B*theta = P_injection (generation "
                   "minus load at each bus) with theta_0 = 0; line flow = b*(theta_from - theta_to) MW, where b is the "
                   "line susceptance (MW/rad); + flow runs from->to. Loading = "
                   "|flow| / thermal limit; above 100% the line is overloaded.")
        out.append("Raising one generator by 1 MW is balanced at slack bus 0; "
                   "shifting 1 MW from generator h to g changes a line's flow by "
                   "(effect of g) - (effect of h) with no net imbalance (effects "
                   "listed under RESULTS).")
        load_txt = " ".join("%d:%s" % (k, f(loads[k], 3))
                            for k in range(n) if loads[k] != 0.0)
        out.append("BUS LOADS (bus:MW; unlisted buses have no load): " + load_txt)

        out.append("")
        out.append("DESIGN (generator dispatch; G<i> = generator i; merit = "
                   "position in cost order, 1 = cheapest)")
        for g in range(G):
            tag = ""
            if p[g] >= pmax[g] - 1e-9 * max(1.0, pmax[g]):
                tag = " AT MAX"
            elif p[g] <= 1e-9:
                tag = " AT ZERO"
            out.append("G%d @bus %d: c=%s $/MWh (merit %d/%d), p_max=%s MW, "
                       "p_set=%s MW (%s%% of max)%s"
                       % (g, int(gbus[g]), f(cost[g], 3), int(merit[g]), G,
                          f(pmax[g]), f(p[g]),
                          f(100.0 * p[g] / max(pmax[g], 1e-12), 3), tag))
        tot_p = float(p.sum())
        gap = total_load - tot_p
        if abs(gap) < 1e-9 * max(1.0, total_load):
            gap = 0.0                                   # summation round-off
        out.append("Total p_set %s MW vs total %sload %s MW; slack bus 0 injects "
                   "the difference (%s MW; + = slack covers a generation "
                   "shortfall, - = slack absorbs surplus generation)."
                   % (f(tot_p), "nominal " if self.stochastic else "",
                      f(total_load), sf(gap, 4)))

        flow = self._nominal_flows(st)
        # linear-solve round-off (e.g. 1e-14 MW on a dead-end line) is not flow
        flow = np.where(np.abs(flow) < 1e-9 * max(1.0, total_load), 0.0, flow)
        loading = np.abs(flow) / np.maximum(lim, 1e-12)
        out.append("")
        out.append("RESULTS (DC power flow at the current dispatch%s)"
                   % (", nominal loads" if self.stochastic else ""))
        out.append("LINES: L# from-to | b = susceptance (MW/rad) | f = flow/"
                   "thermal limit (MW; + = from->to) | loading % of limit")
        for l in range(L):
            out.append("L%d %d-%d b=%s f=%s/%s %s%%%s"
                       % (l, int(li[l]), int(lj[l]), f(b[l], 3),
                          sf(flow[l], 3), f(lim[l], 3),
                          f(100.0 * loading[l], 3),
                          " OVER" if loading[l] > 1.0 else ""))

        order = [int(x) for x in np.argsort(-loading, kind="stable")]
        over = [l for l in order if loading[l] > 1.0]
        n_show = 8
        if over:
            txt = "; ".join("L%d %d-%d %s%% (+%s MW over)"
                            % (l, int(li[l]), int(lj[l]),
                               f(100.0 * loading[l], 3),
                               f(abs(flow[l]) - lim[l], 3))
                            for l in over[:n_show])
            if len(over) > n_show:
                txt += "; and %d more flagged OVER above" % (len(over) - n_show)
            out.append("Overloaded lines (%d), worst first: %s" % (len(over), txt))
        else:
            out.append("No line is overloaded. Most loaded: "
                       + "; ".join("L%d %d-%d %s%%" % (l, int(li[l]), int(lj[l]),
                                                       f(100.0 * loading[l], 3))
                                   for l in order[:3]))
        # `over` is a prefix of `order`, so this is the k most loaded lines:
        # up to 5 overloaded ones, topped up to at least 3 with the next most
        # loaded (so a move that would overload a near-limit line is visible)
        k_watch = min(max(min(len(over), 5), 3), L)
        watch = order[:k_watch]
        n_watch_over = sum(1 for l in watch if loading[l] > 1.0)
        S = None
        if watch:
            try:
                S = self._gen_ptdf(st)
                S = np.where(np.abs(S) < 1e-9, 0.0, S)
            except Exception:
                S = None
        if S is not None:
            out.append("GENERATOR EFFECTS on the %d most loaded line%s (%d of "
                       "them overloaded): change in each line's |flow| (MW) per "
                       "+1 MW of p_set, balanced at slack bus 0 (+ loads the "
                       "line more, - relieves it); headroom = p_max - p_set:"
                       % (len(watch), "" if len(watch) == 1 else "s",
                          n_watch_over))
            for g in range(G):
                parts = []
                for l in watch:
                    sgn = 1.0 if flow[l] >= 0.0 else -1.0
                    parts.append("L%d %s" % (l, sf(sgn * S[l, g], 3)))
                out.append("G%d @bus %d: %s; headroom %s MW"
                           % (g, int(gbus[g]), ", ".join(parts),
                              f(max(pmax[g] - p[g], 0.0))))
        return "\n".join(out)
