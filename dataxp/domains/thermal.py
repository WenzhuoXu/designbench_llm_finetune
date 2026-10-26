"""Steady-state lumped thermal network: heat-sink allocation.

Model
-----
n nodes (10..100), node i dissipates power P_i [W]. Fixed inter-node
conduction edges with conductances w_ij [W/K] (weighted graph Laplacian L),
plus a per-node sink conductance g_i [W/K] to ambient - the design variable.
Steady state temperature rise over ambient solves

    (L + diag(g)) T = P            (SPD for g > 0; one dense numpy solve)

Cooling cost is superlinear in sink size: cost = sum_i k_i * g_i**1.3.

Families: "grid" (r x c plate), "stack" (1-D chip stack, strong vertical
conduction, die at the bottom), "random" (random recursive tree + chords).

Constraints come from the goals dict (maximum_max_temp_rise always,
maximum_mean_temp_rise sometimes); the objective is to minimise cost.
The reference optimum is the cost of the UNIFORM allocation obtained by
bisecting a single global g level until the constraints are just met
("heuristic": the true non-uniform optimum is cheaper).

Initial designs are built from the uniform reference level with a few
DEFICIENT nodes (undersized sinks on high-power nodes), so most problems
start infeasible with graded severity, and feasibility is restorable by a
handful of single-parameter moves.

Stochastic mode: per-seed power draw P_i * lognormal(sigma), mean-one, via
np.random.default_rng(seed) only - the chance constraint on peak temperature
is applied upstream by quantile aggregation.

numpy + stdlib only; no file I/O, no prints.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from ..domain import Domain
from ..potential import DesignProgram, program_from_goals

_FAMILIES = ("grid", "stack", "random")

_GRID_SHAPES = ((3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (10, 10))
_STACK_SIZES = (10, 16, 24, 40, 64, 100)
_RANDOM_SIZES = (12, 20, 32, 48, 72, 100)

# family-dependent peak temperature-rise limits [K] and power-noise sigma
_LIMIT_RANGE = {"grid": (40.0, 70.0), "stack": (30.0, 60.0), "random": (45.0, 80.0)}
_SIGMA_RANGE = {"grid": (0.05, 0.08), "stack": (0.07, 0.12), "random": (0.10, 0.16)}

G_MIN = 1e-4          # [W/K] smallest sink (essentially bare node)
G_MAX = 2.0           # [W/K] largest sink that fits
COST_EXP = 1.3        # superlinear sink cost exponent
TEMP_CAP = 500.0      # [K] validity envelope on any temperature rise

_FAIL_METRICS = {
    "max_temp_rise": float("inf"),
    "mean_temp_rise": float("inf"),
    "cost": float("inf"),
    "hotspot_node": -1.0,
    "temp_negativity": float("inf"),
}


def _laplacian(n: int, edges, weights) -> np.ndarray:
    """Dense weighted graph Laplacian from an edge list."""
    L = np.zeros((n, n), dtype=float)
    for (u, v), w in zip(edges, weights):
        u = int(u)
        v = int(v)
        w = float(w)
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    return L


def _solve_temps(L: np.ndarray, g: np.ndarray, P: np.ndarray) -> np.ndarray:
    A = L + np.diag(g)
    return np.linalg.solve(A, P)


def _constraint_scale(L, g, P, goals) -> float:
    """max over temperature constraints of value/limit for a uniform-noise-free
    solve (monotone decreasing in every g_i); >1 means infeasible."""
    T = _solve_temps(L, g, P)
    s = float(np.max(T)) / float(goals["maximum_max_temp_rise"])
    mean_lim = goals.get("maximum_mean_temp_rise")
    if mean_lim is not None:
        s = max(s, float(np.mean(T)) / float(mean_lim))
    return s


def _bisect_uniform_level(L, P, goals, g_min: float, g_max: float,
                          target: float = 1.0, iters: int = 64) -> float:
    """Uniform sink level whose worst constraint ratio equals ``target``.

    The ratio is monotone decreasing in the level; g_max is guaranteed to be
    on the feasible side by construction of make_problem (power is scaled so
    the fully-sized design has margin).
    """
    n = P.shape[0]
    lo, hi = g_min, g_max
    if _constraint_scale(L, np.full(n, hi), P, goals) > target:
        return hi          # cannot do better than everything at max size
    if _constraint_scale(L, np.full(n, lo), P, goals) <= target:
        return lo          # already fine at the minimum (degenerate problem)
    for _ in range(iters):
        mid = math.sqrt(lo * hi)     # log-scale bisection over 4+ decades
        if _constraint_scale(L, np.full(n, mid), P, goals) > target:
            lo = mid
        else:
            hi = mid
    return hi


class ThermalDomain(Domain):
    """Heat-sink allocation on a lumped thermal conduction network."""

    name = "thermal"
    params = ("g_sink",)
    # wide factors: deficient sinks start up to ~50x undersized and must be
    # restorable in a few single-parameter moves
    factors = (0.6, 0.8, 1.25, 1.8, 3.2)
    reference_class = "heuristic"

    def __init__(self, stochastic: bool = False):
        super().__init__(stochastic)

    # ── problem generation ──────────────────────────────────────────────────
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[index % len(_FAMILIES)]

        if family == "grid":
            rows, cols = _GRID_SHAPES[int(rng.integers(len(_GRID_SHAPES)))]
            n = rows * cols
            edges = []
            for r in range(rows):
                for c in range(cols):
                    i = r * cols + c
                    if c + 1 < cols:
                        edges.append((i, i + 1))
                    if r + 1 < rows:
                        edges.append((i, i + cols))
            w0 = rng.uniform(0.08, 0.4)
        elif family == "stack":
            n = int(_STACK_SIZES[int(rng.integers(len(_STACK_SIZES)))])
            edges = [(i, i + 1) for i in range(n - 1)]
            w0 = rng.uniform(0.5, 2.0)
        else:  # random: recursive tree + chords (always connected)
            n = int(_RANDOM_SIZES[int(rng.integers(len(_RANDOM_SIZES)))])
            edges = [(int(rng.integers(0, i)), i) for i in range(1, n)]
            have = {(min(u, v), max(u, v)) for u, v in edges}
            n_extra = int(round(0.4 * n))
            attempts = 0
            while n_extra > 0 and attempts < 20 * n:
                attempts += 1
                u = int(rng.integers(0, n))
                v = int(rng.integers(0, n))
                key = (min(u, v), max(u, v))
                if u != v and key not in have:
                    have.add(key)
                    edges.append(key)
                    n_extra -= 1
            w0 = rng.uniform(0.05, 0.3)

        m = len(edges)
        weights = w0 * np.exp(rng.normal(0.0, 0.25, size=m))

        # powers: lognormal base + concentrated hotspots
        if family == "stack":
            decay = rng.uniform(3.0, max(4.0, n / 3.0))
            P_raw = np.exp(rng.normal(0.0, 0.35, size=n)) * np.exp(-np.arange(n) / decay)
            P_raw[0] *= rng.uniform(5.0, 15.0)          # the die
        else:
            P_raw = np.exp(rng.normal(0.0, 0.6, size=n))
            n_hot = max(1, n // 16)
            hot = rng.choice(n, size=n_hot, replace=False)
            P_raw[hot] *= rng.uniform(4.0, 12.0, size=n_hot)

        k_cost = rng.uniform(0.5, 2.0, size=n)

        # limits, then scale power so the fully-sized design has real margin
        limit = float(rng.uniform(*_LIMIT_RANGE[family]))
        L = _laplacian(n, edges, weights)
        T_full = _solve_temps(L, np.full(n, G_MAX), P_raw)
        beta = rng.uniform(0.15, 0.45)
        P = P_raw * (beta * limit / float(np.max(T_full)))

        goals = {"maximum_max_temp_rise": limit}
        if rng.random() < 0.4:
            mean_at_full = float(np.mean(T_full)) * beta * limit / float(np.max(T_full))
            goals["maximum_mean_temp_rise"] = float(
                max(limit * rng.uniform(0.4, 0.75), 1.35 * mean_at_full))

        # uniform reference allocation (heuristic optimum)
        g_star = _bisect_uniform_level(L, P, goals, G_MIN, G_MAX)
        ref_cost = float(np.sum(k_cost * np.power(g_star, COST_EXP)))

        # initial design: reference-scale sinks with a few deficient nodes
        u = rng.random()
        feas_lo = 1.25 if self.stochastic else 1.06
        # A few already-feasible problems are worth keeping (they test pure
        # objective optimisation with no restoration phase), but they carry no
        # repair decisions, so the share stays small: at 18% the set sat at 54%
        # infeasible, close enough to G1's 50% floor to fail on a resample.
        if u < 0.08:
            klass, n_def = "feasible", 0
            g0 = g_star * rng.uniform(feas_lo, feas_lo + 0.35, size=n)
        elif u < 0.42:
            klass = "near"
            n_def = 1 + int(rng.random() < 0.35)
            g0 = g_star * rng.uniform(0.98, 1.35, size=n)
            deficit = rng.uniform(0.65, 0.92, size=n_def)
        else:
            klass = "deep"
            n_def_hi = 4 if n <= 48 else 3
            n_def = 1 + int(rng.integers(0, n_def_hi))
            g0 = g_star * rng.uniform(0.98, 1.35, size=n)
            deficit = np.exp(rng.uniform(math.log(0.02), math.log(0.35), size=n_def))
        def_nodes = []
        if n_def > 0:
            pool = np.argsort(-P)[: max(4, n_def + 3)]
            def_nodes = rng.choice(pool, size=n_def, replace=False)
            g0[def_nodes] = g_star * deficit
        g0 = np.clip(g0, G_MIN, G_MAX)

        sigma = float(rng.uniform(*_SIGMA_RANGE[family]))

        return {
            "family": family,
            "n": int(n),
            "edges": [[int(a), int(b)] for a, b in edges],
            "conductances": [float(x) for x in weights],
            "powers": [float(x) for x in P],
            "k_cost": [float(x) for x in k_cost],
            "g0": [float(x) for x in g0],
            "g_min": float(G_MIN),
            "g_max": float(G_MAX),
            "sigma": sigma,
            "goals": {k: float(v) for k, v in goals.items()},
            "ref_cost": ref_cost,
            "g_uniform_ref": float(g_star),
            "difficulty_class": klass,
            "deficient_nodes": [int(i) for i in def_nodes],
        }

    # ── state ────────────────────────────────────────────────────────────────
    def load(self, spec: dict):
        n = int(spec["n"])
        L = _laplacian(n, spec["edges"], spec["conductances"])
        return {
            "n": n,
            "L": L,                                           # never mutated
            "P": np.asarray(spec["powers"], dtype=float),     # never mutated
            "k": np.asarray(spec["k_cost"], dtype=float),     # never mutated
            "g": np.asarray(spec["g0"], dtype=float).copy(),  # the design
            "g_min": float(spec["g_min"]),
            "g_max": float(spec["g_max"]),
            "sigma": float(spec.get("sigma", 0.1)),
            "_hint": None,                                    # eval cache
        }

    def clone(self, st):
        out = dict(st)
        out["g"] = st["g"].copy()   # the only array set() mutates
        out["_hint"] = None         # reset evaluation cache
        return out

    # ── simulation ───────────────────────────────────────────────────────────
    def evaluate(self, st, seed: int = 0) -> dict:
        self.calls += 1
        try:
            g = st["g"]
            P = st["P"]
            if self.stochastic:
                sig = st["sigma"]
                r = np.random.default_rng(seed)
                P = P * np.exp(r.normal(-0.5 * sig * sig, sig, size=P.shape[0]))
            T = _solve_temps(st["L"], g, P)
            if not np.all(np.isfinite(T)):
                return dict(_FAIL_METRICS)
            cost = float(np.sum(st["k"] * np.power(g, COST_EXP)))
            if not math.isfinite(cost):
                return dict(_FAIL_METRICS)
            return {
                "max_temp_rise": float(np.max(T)),
                "mean_temp_rise": float(np.mean(T)),
                "cost": cost,
                "hotspot_node": float(int(np.argmax(T))),
                "temp_negativity": float(max(0.0, -float(np.min(T)))),
            }
        except Exception:
            # singular/diverged solve: maximally infeasible, never raise
            return dict(_FAIL_METRICS)

    # ── design variables ─────────────────────────────────────────────────────
    def n_elements(self, st) -> int:
        return int(st["n"])

    def get(self, st, i: int, p: str) -> Optional[float]:
        if p != "g_sink" or not (0 <= i < st["n"]):
            return None
        return float(st["g"][i])

    def set(self, st, i: int, p: str, v: float) -> None:
        if p != "g_sink" or not (0 <= i < st["n"]):
            return
        st["g"][i] = min(st["g_max"], max(st["g_min"], float(v)))
        st["_hint"] = None          # invalidate cache

    def bounds(self, st, p: str) -> Tuple[float, float]:
        return (st["g_min"], st["g_max"])

    # ── program / reference ──────────────────────────────────────────────────
    def program(self, spec: dict) -> DesignProgram:
        prog = program_from_goals(
            spec["goals"],
            objective_key="cost",
            objective_ref=self.reference(spec),
            objective_sense="min",
            domain=self.name,
        )
        # envelope: any temperature rise outside [0, 500] K is broken physics
        # (temp_negativity > 0 flags a negative rise); cost is capped at what
        # the fully-sized allocation can possibly reach.
        cost_cap = 1.05 * float(np.sum(
            np.asarray(spec["k_cost"], dtype=float)
            * float(spec["g_max"]) ** COST_EXP))
        return replace(prog, validity_bounds=(
            ("max_temp_rise", TEMP_CAP),
            ("mean_temp_rise", TEMP_CAP),
            ("temp_negativity", 1e-6),
            ("cost", cost_cap),
        ))

    def reference(self, spec: dict) -> Optional[float]:
        ref = spec.get("ref_cost")
        if ref is None:      # not produced by make_problem: recompute + cache
            n = int(spec["n"])
            L = _laplacian(n, spec["edges"], spec["conductances"])
            P = np.asarray(spec["powers"], dtype=float)
            k = np.asarray(spec["k_cost"], dtype=float)
            g_star = _bisect_uniform_level(L, P, spec["goals"],
                                           float(spec["g_min"]),
                                           float(spec["g_max"]))
            ref = float(np.sum(k * np.power(g_star, COST_EXP)))
            spec["ref_cost"] = ref
        return float(ref)

    # ── depth-0 diagnostic ────────────────────────────────────────────────────
    def hint(self, st) -> Optional[Tuple[int, str, str]]:
        """Hottest node under nominal power wants a bigger sink."""
        cached = st.get("_hint")
        if cached is None:
            try:
                T = _solve_temps(st["L"], st["g"], st["P"])
                if not np.all(np.isfinite(T)):
                    return None
                cached = int(np.argmax(T))
            except Exception:
                return None
            st["_hint"] = cached
        return (cached, "g_sink", "up")

    # ── observation for the policy (additive; never touches the design) ─────
    def describe(self, st, spec) -> str:
        """Self-contained plain-text observation of the current design.

        PROBLEM (fixed setup and physics), DESIGN (one line per node: role,
        fixed power and cost coefficient, current g_sink, link conductances),
        RESULTS (per-node temperature rise from one nominal steady-state solve,
        the nodes over the limit, and the heat paths around them). Read-only
        on ``st``; deterministic.
        """
        n = int(st["n"])
        L = st["L"]
        P = np.asarray(st["P"], dtype=float)
        g = np.asarray(st["g"], dtype=float)
        k = np.asarray(st["k"], dtype=float)
        lay = _obs_layout(L)
        fam = lay[0]
        adj = _obs_adjacency(L)
        n_links = sum(len(a) for a in adj) // 2
        goals = (spec or {}).get("goals") or {}
        t_lim = goals.get("maximum_max_temp_rise")
        t_lim = float(t_lim) if t_lim is not None else None
        m_lim = goals.get("maximum_mean_temp_rise")
        m_lim = float(m_lim) if m_lim is not None else None
        T = _obs_temps(st)

        # ── PROBLEM ──
        out = ["PROBLEM"]
        head = "Heat-sink sizing on a steady-state lumped thermal network: "
        if fam == "grid":
            rows, cols = lay[1], lay[2]
            out.append(head + (
                "a flat plate discretised as a %d x %d grid, %d nodes (index = "
                "row*%d + col; rXcY = row X, col Y) joined by %d conduction links "
                "to their up/down/left/right neighbours." % (rows, cols, n, cols, n_links)))
        elif fam == "stack":
            out.append(head + (
                "a 1-D chip stack of %d layers (n0 = the die at the bottom, n%d = "
                "top layer), %d conduction links, each between a layer and the one "
                "directly above it." % (n, n - 1, n_links)))
        else:
            deg = [len(a) for a in adj]
            out.append(head + (
                "an irregular network (random tree plus cross-links) of %d nodes "
                "and %d conduction links, node degree %d-%d."
                % (n, n_links, min(deg), max(deg))))
        ws = [w for a in adj for _, w in a]
        out.append(
            "Node i dissipates a fixed power P_i [W] (total %s W, per node %s to %s W). "
            "Linked nodes i,j exchange w_ij*(T_i - T_j) [W] through a fixed link "
            "conductance w_ij [W/K] (%s to %s W/K). Node i also rejects g_i*T_i [W] "
            "to ambient through its own heat sink."
            % (_obs_num(P.sum()), _obs_num(P.min()), _obs_num(P.max()),
               _obs_num(min(ws) if ws else 0.0), _obs_num(max(ws) if ws else 0.0)))
        out.append(
            "T_i = steady temperature rise of node i above ambient [K], set by the "
            "heat balance at every node: P_i + sum_j w_ij*(T_j - T_i) = g_i*T_i, "
            "i.e. (L + diag(g)) T = P with L the link-conductance Laplacian.")
        out.append(
            "Tunable parameter, one per node: g_sink (g_i) = conductance of node "
            "i's heat sink to ambient [W/K], bounds [%s, %s]. A bigger sink cools "
            "its own node most and its linked neighbours less, and pulls heat in "
            "from hotter linked nodes." % (_obs_num(st["g_min"]), _obs_num(st["g_max"])))
        out.append(
            "Cooling cost = sum_i k_i*g_i^%s with a fixed per-node cost coefficient "
            "k_i (%s to %s here). Superlinear: doubling one sink multiplies its cost "
            "by %s, and the marginal cost d(cost)/d(g_i) = %s*k_i*g_i^%s grows with "
            "sink size, so a few oversized sinks cost more than several moderate ones."
            % (_obs_num(COST_EXP), _obs_num(k.min()), _obs_num(k.max()),
               _obs_num(2.0 ** COST_EXP), _obs_num(COST_EXP), _obs_num(COST_EXP - 1.0)))
        # limits at 5 significant figures so they match the constraint status
        # the caller prints (3 decimals) and a node just over the limit never
        # reads as "equal to" a rounded limit
        lim_s = _obs_num(t_lim, 5) if t_lim is not None else None
        lim_txt = []
        if t_lim is not None:
            lim_txt.append("every node T_i <= %s K" % lim_s)
        if m_lim is not None:
            lim_txt.append("average of T_i over all %d nodes <= %s K" % (n, _obs_num(m_lim, 5)))
        if lim_txt:
            out.append("Temperature limits: " + "; ".join(lim_txt) + ".")
        # stochastic: 1-in-10 high value of the mean-one lognormal power factor
        # exp(-sigma^2/2 + z90*sigma), rounded as printed so the reader can
        # reproduce every "at xh" figure below from the printed numbers
        hi = None
        if self.stochastic:
            sig = float(st.get("sigma", 0.0))
            hi = float(_obs_num(math.exp(-0.5 * sig * sig + 1.2815515655446004 * sig)))
            out.append(
                "Power is uncertain: every evaluation multiplies each P_i by an "
                "independent mean-one lognormal factor (sigma %s; its 1-in-10 high "
                "value is x%s). Each limit is judged on a high (90th-percentile) "
                "quantile over several such draws, so it can be violated while "
                "every nominal T_i is under it. T is linear in the powers: with "
                "every P_i scaled by x%s, every T_i scales by x%s. The per-node "
                "RESULTS below are a nominal evaluation at the mean powers P_i."
                % (_obs_num(sig), _obs_num(hi), _obs_num(hi), _obs_num(hi)))

        # ── DESIGN ──
        out.append("")
        if fam == "grid":
            cols = lay[2]
            out.append("DESIGN (per node: position, fixed power P [W], cost "
                       "coefficient k, current g = g_sink [W/K]; after '|': "
                       "conductance w [W/K] of the link to the right (R, node+1) "
                       "and lower (D, node+%d) neighbour)" % cols)
        elif fam == "stack":
            out.append("DESIGN (per layer: fixed power P [W], cost coefficient k, "
                       "current g = g_sink [W/K]; after '|': conductance w [W/K] "
                       "of the link up to the next layer)")
        else:
            # each link once, on its lower-numbered end (as the grid lists only
            # R/D): listing both ends doubled the longest observation
            out.append("DESIGN (per node: fixed power P [W], cost coefficient k, "
                       "current g = g_sink [W/K]; after '|': each link to a "
                       "higher-numbered neighbour as node:w, w in W/K - every "
                       "link is listed once, on its lower-numbered end)")
        for i in range(n):
            if fam == "grid":
                cols = lay[2]
                r, c = divmod(i, cols)
                role = " r%dc%d" % (r, c)
                links = []
                if c + 1 < cols:
                    links.append("R %s" % _obs_num(-L[i, i + 1]))
                if i + cols < n:
                    links.append("D %s" % _obs_num(-L[i, i + cols]))
            elif fam == "stack":
                role = " die" if i == 0 else (" top" if i == n - 1 else "")
                links = ["up %s" % _obs_num(-L[i, i + 1])] if i + 1 < n else []
            else:
                role = ""
                links = ["%d:%s" % (j, _obs_num(w)) for j, w in adj[i] if j > i]
            out.append("n%d%s P=%s k=%s g=%s%s" % (
                i, role, _obs_num(P[i]), _obs_num(k[i]), _obs_num(g[i], 4),
                (" | " + " ".join(links)) if links else ""))

        # ── RESULTS ──
        out.append("")
        if T is None:
            out.append("RESULTS: the steady-state solve failed for this design "
                       "(singular or non-finite temperatures).")
            return "\n".join(out)
        hot = int(np.argmax(T))
        risk_on = hi is not None and t_lim is not None
        out.append("RESULTS (%s steady-state solve; T = temperature rise [K]%s%s)" % (
            "nominal mean-power" if self.stochastic else "deterministic",
            "; AT RISK = under the limit at the mean powers but over it with "
            "every power at x%s" % _obs_num(hi) if risk_on else "",
            "; up = heat conducted from the layer into the one above [W], "
            "negative = downward" if fam == "stack" else ""))
        for i in range(n):
            txt = "n%d T=%s" % (i, _obs_num(T[i]))
            if fam == "stack" and i + 1 < n:
                txt += " up=%s" % _obs_num(-L[i, i + 1] * (T[i] - T[i + 1]))
            if t_lim is not None and T[i] > t_lim:
                txt += " OVER +%s" % _obs_num(T[i] - t_lim)
            elif risk_on and T[i] * hi > t_lim:
                txt += " AT RISK"
            if i == hot:
                txt += " HOTSPOT"
            out.append(txt)

        over, risk = [], []
        if t_lim is not None:
            order = np.argsort(-T, kind="stable").tolist()
            over = [i for i in order if T[i] > t_lim]
            if risk_on:
                risk = [i for i in order if T[i] <= t_lim and T[i] * hi > t_lim]
        at_mean = " at the mean powers" if self.stochastic else ""
        if over:
            out.append("Over the %s K limit%s (%d node%s): %s" % (
                lim_s, at_mean, len(over), "" if len(over) == 1 else "s",
                ", ".join("n%d +%s K" % (i, _obs_num(T[i] - t_lim)) for i in over)))
            # ranked, so "the only other warm node" is read off, not guessed
            under = [i for i in order if T[i] <= t_lim][:3]
            if under:
                out.append("Warmest nodes under the limit: %s." % ", ".join(
                    "n%d %s K (%s K below)" % (i, _obs_num(T[i]), _obs_num(t_lim - T[i]))
                    for i in under))
            focus = (over + risk)[:3]
        else:
            if t_lim is not None:
                top = np.argsort(-T, kind="stable").tolist()[:3]
                out.append("%so node is over the %s K limit; closest: %s." % (
                    "At the mean powers n" if self.stochastic else "N",
                    lim_s, ", ".join(
                        "n%d %s K (%s K below%s)" % (
                            i, _obs_num(T[i]), _obs_num(t_lim - T[i]),
                            "; %s K at x%s" % (_obs_num(T[i] * hi), _obs_num(hi))
                            if risk_on else "")
                        for i in top)))
            focus = risk[:3] or [hot]
        if risk:
            shown = risk[:5]
            out.append("AT RISK at x%s power (%d node%s): %s%s" % (
                _obs_num(hi), len(risk), "" if len(risk) == 1 else "s",
                ", ".join("n%d %s K -> %s K" % (i, _obs_num(T[i]), _obs_num(T[i] * hi))
                          for i in shown),
                ", + %d more" % (len(risk) - len(shown)) if len(risk) > len(shown) else ""))
        out.append("Hotspot: n%d (T=%s K)." % (hot, _obs_num(T[hot])))
        out.append("Heat paths (-> heat conducted out over a link, <- heat "
                   "conducted in; with each linked node's state and where it "
                   "passes the heat on):")
        for r, i in enumerate(focus):
            out.extend(_obs_heat_paths(i, P, g, T, adj, hot, t_lim, ring2=(r < 2),
                                       at_risk=(i in risk)))
        return "\n".join(out)

    def element_label(self, st, i) -> str:
        """Short role label for node i, e.g. 'node 47 (r5c7, P=39.1 W, hotspot)'."""
        i = int(i)
        n = int(st["n"])
        if not (0 <= i < n):
            return "node %d" % i
        lay = _obs_layout(st["L"])
        adj = _obs_adjacency(st["L"])
        fam = lay[0]
        if fam == "grid":
            r, c = divmod(i, lay[2])
            role = "r%dc%d" % (r, c)
        elif fam == "stack":
            role = ("die, bottom layer" if i == 0 else
                    ("top layer %d" % i if i == n - 1 else "layer %d" % i))
        else:
            role = "degree %d" % len(adj[i])
        bits = [role, "P=%s W" % _obs_num(st["P"][i])]
        T = _obs_temps(st)
        if T is not None:
            hot = int(np.argmax(T))
            if i == hot:
                bits.append("hotspot")
            elif any(j == hot for j, _ in adj[i]):
                bits.append("linked to hotspot n%d" % hot)
        return "node %d (%s)" % (i, ", ".join(bits))


# ── observation helpers (read-only; used only by describe/element_label) ─────
def _obs_num(x, sig: int = 3) -> str:
    """``sig`` significant figures, so a small nonzero value never prints as 0:
    fixed point down to 1e-4 (0.000132 -> '0.000132'), compact exponent below
    that (1.75e-9, e.g. the negligible powers at the top of a tall stack)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    if x == 0.0:
        return "0"
    if abs(x) < 1e-4:
        mant, ex = ("%.*e" % (max(sig - 1, 0), x)).split("e")
        if "." in mant:
            mant = mant.rstrip("0").rstrip(".")
        return "%se%d" % (mant, int(ex))
    mag = int(math.floor(math.log10(abs(x))))
    dec = max(0, sig - 1 - mag)
    s = "%.*f" % (dec, x)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _obs_layout(L: np.ndarray):
    """Infer the family from the link pattern: ('grid', rows, cols),
    ('stack', n, 1) or ('random', n, 0)."""
    n = int(L.shape[0])
    iu, ju = np.nonzero(np.triu(L, 1))
    links = set(zip(iu.tolist(), ju.tolist()))
    if n > 1 and links == {(i, i + 1) for i in range(n - 1)}:
        return ("stack", n, 1)
    for rows, cols in _GRID_SHAPES:
        if rows * cols != n:
            continue
        lat = set()
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                if c + 1 < cols:
                    lat.add((i, i + 1))
                if r + 1 < rows:
                    lat.add((i, i + cols))
        if links == lat:
            return ("grid", rows, cols)
    return ("random", n, 0)


def _obs_adjacency(L: np.ndarray):
    """Per node, list of (neighbour, w_ij [W/K]) in neighbour order."""
    n = int(L.shape[0])
    adj = []
    for i in range(n):
        js = np.nonzero(L[i])[0].tolist()
        adj.append([(int(j), float(-L[i, j])) for j in js if j != i])
    return adj


def _obs_temps(st) -> Optional[np.ndarray]:
    """Nominal (mean-power) steady-state temperature rises; None on failure.
    Identical to what evaluate() solves in the deterministic variant."""
    try:
        T = _solve_temps(st["L"], np.asarray(st["g"], dtype=float),
                         np.asarray(st["P"], dtype=float))
    except Exception:
        return None
    if not np.all(np.isfinite(T)):
        return None
    return T


def _obs_heat_paths(i: int, P, g, T, adj, hot: int, t_lim, ring2: bool,
                    max_onward: int = 4, at_risk: bool = False):
    """Where node i's heat goes: its own sink and each link, with the linked
    node's state (the neighbours whose sinks can pull heat away from i) and,
    for ring2, where each receiving neighbour passes that heat on."""
    tags = []
    if i == hot:
        tags.append("hotspot")
    if t_lim is not None and T[i] > t_lim:
        tags.append("OVER")
    elif at_risk:
        tags.append("AT RISK")
    q = g[i] * T[i]
    inflow = sum(max(0.0, w * (T[j] - T[i])) for j, w in adj[i])
    shed = P[i] + inflow
    lines = ["n%d [%s]: P=%s W, g=%s, T=%s K; own sink removes %s W (%s%% of the "
             "%s W it must shed)" % (
                 i, ", ".join(tags) if tags else "node", _obs_num(P[i]),
                 _obs_num(g[i], 4), _obs_num(T[i]), _obs_num(q),
                 _obs_num(100.0 * q / shed if shed > 0 else 0.0, 2), _obs_num(shed))]
    order = sorted(adj[i], key=lambda jw: (-(jw[1] * (T[i] - T[jw[0]])), jw[0]))
    for j, w in order:
        f = w * (T[i] - T[j])
        txt = "  %s n%d %s W over w=%s | n%d: T=%s P=%s g=%s, own sink %s W" % (
            "->" if f >= 0 else "<-", j, _obs_num(abs(f)), _obs_num(w), j,
            _obs_num(T[j]), _obs_num(P[j]), _obs_num(g[j], 4), _obs_num(g[j] * T[j]))
        if ring2 and f > 0:
            onward = sorted(((kk, wk * (T[j] - T[kk])) for kk, wk in adj[j]
                             if kk != i and T[j] > T[kk]),
                            key=lambda x: (-x[1], x[0]))
            if onward:
                shown = onward[:max_onward]
                txt += "; passes on " + ", ".join(
                    "-> n%d %s W (T=%s g=%s)" % (kk, _obs_num(fk), _obs_num(T[kk]),
                                                _obs_num(g[kk], 4))
                    for kk, fk in shown)
                if len(onward) > len(shown):
                    txt += ", + %d smaller" % (len(onward) - len(shown))
            else:
                txt += "; passes none on"
        lines.append(txt)
    return lines
