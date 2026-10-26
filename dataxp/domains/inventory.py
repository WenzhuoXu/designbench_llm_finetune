"""Multi-echelon serial inventory control domain (intrinsically stochastic).

Model
-----
A serial supply chain of N stages (stage 0 faces the customer, stage N-1
orders from an infinite-supply source). Periodic review, installation
base-stock (order-up-to) policy: each period every stage orders up to its
base-stock level S_i; the upstream stage ships what its on-hand allows, the
source always ships in full. Shipments to stage i arrive after a fixed lead
time L_i in {1..4}. Customer demand at stage 0 is family-dependent (poisson /
negative binomial / bursty regime-switching poisson), drawn from a seeded
generator. Unmet customer demand backorders.

Horizon T ~ 300 periods with a warmup; metrics are averaged over the scored
window: fill rate at the customer (service_level), average per-period cost
split into holding (h_i per unit on hand, rising downstream) and backorder
(b per unit of customer demand NOT served in its arrival period - a per-unit
shortfall penalty; charging per unit-period of backlog would make the
objective a near-critically-loaded random walk whose time average has
divergent across-seed variance and Phi could not rank candidates). One
evaluate() averages a few independent demand paths derived from its seed.

Design variables: one base_stock level per stage in [0, base_stock_max].
Goals: minimum_service_level (family-dependent 0.90-0.98) and
maximum_holding_cost (a budget calibrated to the newsvendor heuristic, tight
enough that blanket over-stocking is infeasible). Objective: minimise
total_cost. The initial design carries NO safety stock (base stock = mean
lead-time demand), which starts most problems infeasible on service.

numpy + stdlib only; one evaluate() is a few ms at N=20, T=300.
"""

from __future__ import annotations

import math
from dataclasses import replace
from statistics import NormalDist

import numpy as np

from ..domain import Domain, stable_seed
from ..potential import program_from_goals

_FAMILIES = ("short_chain", "long_chain", "bursty_demand")

_METRIC_KEYS = ("service_level", "total_cost", "holding_cost", "backorder_frac")

# Returned on any numerical failure: reads as maximally infeasible, never NaN.
_FAIL_METRICS = {
    "service_level": 0.0,
    "total_cost": float("inf"),
    "holding_cost": float("inf"),
    "backorder_frac": 1.0,
}

_REF_SEEDS = 5          # fixed replicate seeds behind reference()/budget calibration
_HORIZON = 300
_WARMUP = 60


class InventoryDomain(Domain):
    """Serial multi-echelon base-stock inventory control."""

    name = "inventory"
    params = ("base_stock",)
    reference_class = "heuristic"   # newsvendor-style per-stage base stock

    def __init__(self, stochastic: bool = False):
        super().__init__(stochastic)

    # ── demand model ─────────────────────────────────────────────────────────
    @staticmethod
    def _demand_stats(dem: dict):
        """Stationary (mean, std) of per-period demand."""
        kind = dem["kind"]
        mu = float(dem["mu"])
        if kind == "poisson":
            return mu, math.sqrt(mu)
        if kind == "negbin":
            r = float(dem["r"])
            return mu, math.sqrt(mu + mu * mu / r)
        if kind == "bursty":
            m = float(dem["burst_mult"])
            pe = float(dem["p_enter"])
            px = float(dem["p_exit"])
            pi_b = pe / (pe + px)          # stationary burst probability
            mu_b = mu * m
            mean = (1.0 - pi_b) * mu + pi_b * mu_b
            var = ((1.0 - pi_b) * mu + pi_b * mu_b
                   + (1.0 - pi_b) * (mu - mean) ** 2
                   + pi_b * (mu_b - mean) ** 2)
            return mean, math.sqrt(var)
        raise ValueError("unknown demand kind %r" % kind)

    @staticmethod
    def _gen_demand(dem: dict, T: int, rng: np.random.Generator) -> np.ndarray:
        kind = dem["kind"]
        mu = float(dem["mu"])
        if kind == "poisson":
            return rng.poisson(mu, T).astype(np.float64)
        if kind == "negbin":
            r = float(dem["r"])
            p = r / (r + mu)
            return rng.negative_binomial(r, p, T).astype(np.float64)
        if kind == "bursty":
            m = float(dem["burst_mult"])
            pe = float(dem["p_enter"])
            px = float(dem["p_exit"])
            u = rng.random(T)
            lam = np.empty(T, dtype=np.float64)
            mu_b = mu * m
            burst = False
            for t in range(T):
                if burst:
                    if u[t] < px:
                        burst = False
                else:
                    if u[t] < pe:
                        burst = True
                lam[t] = mu_b if burst else mu
            return rng.poisson(lam).astype(np.float64)
        raise ValueError("unknown demand kind %r" % kind)

    # ── core simulator ───────────────────────────────────────────────────────
    def _simulate(self, spec: dict, base_stock, rng: np.random.Generator,
                  collect_diag: bool = False):
        """One seeded episode of the chain. Returns (metrics, diag_or_None)."""
        n = int(spec["n_stages"])
        T = int(spec["horizon"])
        warm = int(spec["warmup"])
        L = np.asarray(spec["lead_times"], dtype=np.int64)
        h = np.asarray(spec["holding_costs"], dtype=np.float64)
        b = float(spec["backorder_cost"])
        S = np.asarray(base_stock, dtype=np.float64)

        demand = self._gen_demand(spec["demand"], T, rng)

        l_max = int(L.max())
        arrivals = np.zeros((n, T + l_max + 1), dtype=np.float64)
        on_hand = S.copy()                 # start at base stock, empty pipelines
        intransit = np.zeros(n, dtype=np.float64)
        owed = np.zeros(n, dtype=np.float64)   # owed[i]: stage i owes downstream;
        owed_up = np.zeros(n, dtype=np.float64)  # owed[0] = customer backorders
        rows = np.arange(n - 1)            # shipping destinations 0..n-2
        lead_dest = L[: n - 1]

        hold_sum = 0.0
        bo_sum = 0.0
        demand_tot = 0.0
        filled_tot = 0.0
        bo_periods = 0
        n_scored = max(T - warm, 1)
        stockouts = np.zeros(n, dtype=np.int64)
        hold_per_stage = np.zeros(n, dtype=np.float64)

        for t in range(T):
            # 1. arrivals
            arr_t = arrivals[:, t]
            on_hand += arr_t
            intransit -= arr_t

            # 2. customer demand at stage 0 (serve backlog first, then today)
            d = float(demand[t])
            avail = float(on_hand[0])
            bo0 = float(owed[0])
            serve_bo = bo0 if bo0 < avail else avail
            avail -= serve_bo
            bo0 -= serve_bo
            serve_d = d if d < avail else avail
            avail -= serve_d
            bo0 += d - serve_d
            on_hand[0] = avail
            owed[0] = bo0

            # 3. order-up-to: IP_i = on_hand + in-transit + on-order backlog
            #    at my supplier - what I owe downstream
            owed_up[: n - 1] = owed[1:]
            owed_up[n - 1] = 0.0
            q = S - (on_hand + intransit + owed_up - owed)
            np.maximum(q, 0.0, out=q)
            if n > 1:
                owed[1:] += q[: n - 1]
            q_top = float(q[n - 1])
            if q_top > 0.0:                # infinite source ships in full
                intransit[n - 1] += q_top
                arrivals[n - 1, t + L[n - 1]] += q_top

            # 4. shipping i -> i-1, capped by on-hand at i
            if n > 1:
                ship = np.minimum(on_hand[1:], owed[1:])
                on_hand[1:] -= ship
                owed[1:] -= ship
                intransit[: n - 1] += ship
                arrivals[rows, t + lead_dest] += ship

            # 5. accounting over the scored window
            if t >= warm:
                hold_vec = h * on_hand
                hold_sum += float(hold_vec.sum())
                bo_sum += b * float(owed[0])
                demand_tot += d
                filled_tot += serve_d
                if owed[0] > 1e-9:
                    bo_periods += 1
                if collect_diag:
                    hold_per_stage += hold_vec
                    stockouts += (owed > 1e-9)

        holding_cost = hold_sum / n_scored
        backorder_cost = bo_sum / n_scored
        service = filled_tot / demand_tot if demand_tot > 0.0 else 1.0
        metrics = {
            "service_level": float(service),
            "total_cost": float(holding_cost + backorder_cost),
            "holding_cost": float(holding_cost),
            "backorder_frac": float(bo_periods / n_scored),
        }
        diag = None
        if collect_diag:
            diag = {"stockouts": stockouts,
                    "hold_per_stage": hold_per_stage / n_scored}
        return metrics, diag

    # ── problems ─────────────────────────────────────────────────────────────
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[index % len(_FAMILIES)]
        if family == "short_chain":
            n = int(rng.integers(2, 6))
            demand = {"kind": "poisson", "mu": float(rng.uniform(4.0, 20.0))}
            target = round(float(rng.uniform(0.90, 0.98)), 3)
            leads = [int(x) for x in rng.integers(1, 4, size=n)]
        elif family == "long_chain":
            n = int(rng.integers(8, 21))
            demand = {"kind": "negbin",
                      "mu": float(rng.uniform(3.0, 12.0)),
                      "r": float(rng.uniform(2.0, 8.0))}
            target = round(float(rng.uniform(0.90, 0.95)), 3)
            leads = [int(x) for x in rng.integers(1, 5, size=n)]
        else:  # bursty_demand
            n = int(rng.integers(4, 13))
            demand = {"kind": "bursty",
                      "mu": float(rng.uniform(2.0, 8.0)),
                      "burst_mult": float(rng.uniform(2.5, 5.0)),
                      "p_enter": float(rng.uniform(0.03, 0.08)),
                      "p_exit": float(rng.uniform(0.2, 0.4))}
            target = round(float(rng.uniform(0.90, 0.96)), 3)
            leads = [int(x) for x in rng.integers(1, 5, size=n)]

        # holding cost rises downstream (stage 0 = finished goods = highest)
        h_top = float(rng.uniform(0.2, 1.0))
        growth = rng.uniform(1.05, 1.3, size=max(n - 1, 1))
        h_rev = [h_top]
        for g in growth[: n - 1]:
            h_rev.append(h_rev[-1] * float(g))
        holding_costs = [float(x) for x in h_rev[::-1]]
        backorder_cost = holding_costs[0] * float(rng.uniform(5.0, 20.0))

        mean_d, std_d = self._demand_stats(demand)
        lt = np.asarray(leads, dtype=np.float64) + 1.0   # lead + review period
        init_bs = mean_d * lt                            # NO safety stock
        z = NormalDist().inv_cdf(min(max(target, 0.5), 0.999))
        heur_bs = mean_d * lt + z * std_d * np.sqrt(lt)
        base_stock_max = float(np.ceil(
            4.0 * (mean_d * float(lt.max()) + 6.0 * std_d * math.sqrt(float(lt.max())))))

        demand_seed = int(rng.integers(0, 2 ** 31 - 1))
        spec = {
            "family": family,
            "n_stages": n,
            "lead_times": leads,
            "holding_costs": holding_costs,
            "backorder_cost": float(backorder_cost),
            "demand": demand,
            "horizon": _HORIZON,
            "warmup": _WARMUP,
            "demand_seed": demand_seed,
            "initial_base_stock": [float(x) for x in init_bs],
            "heuristic_base_stock": [float(x) for x in heur_bs],
            "base_stock_max": base_stock_max,
        }
        # calibrate the holding budget against the heuristic design (fixed
        # seeds -> deterministic given the rng draws above)
        ref_cost, heur_hold = self._heuristic_cost(spec, heur_bs)
        budget = float(heur_hold * float(rng.uniform(1.25, 1.9)))
        spec["goals"] = {
            "minimum_service_level": float(target),
            "maximum_holding_cost": budget,
        }
        spec["reference_cost"] = float(max(ref_cost, 1e-6))
        return spec

    def _heuristic_cost(self, spec: dict, base_stock):
        costs, holds = [], []
        for k in range(_REF_SEEDS):
            rng = np.random.default_rng(
                [int(spec["demand_seed"]), 7919 + 104729 * k])
            m, _ = self._simulate(spec, base_stock, rng)
            costs.append(m["total_cost"])
            holds.append(m["holding_cost"])
        return float(np.mean(costs)), float(np.mean(holds))

    # ── state ────────────────────────────────────────────────────────────────
    def load(self, spec: dict):
        return {
            "spec": spec,
            "base_stock": np.asarray(spec["initial_base_stock"],
                                     dtype=np.float64).copy(),
            "_diag": None,
        }

    def clone(self, st):
        return {"spec": st["spec"],
                "base_stock": st["base_stock"].copy(),
                "_diag": None}

    # ── simulation ───────────────────────────────────────────────────────────
    def evaluate(self, st, seed: int = 0) -> dict:
        self.calls += 1
        spec = st["spec"]
        try:
            if self.stochastic:
                rng = np.random.default_rng(
                    [int(spec["demand_seed"]), int(seed) & 0xFFFFFFFF])
            else:
                rng = np.random.default_rng(int(spec["demand_seed"]))
            metrics, _ = self._simulate(spec, st["base_stock"], rng)
        except Exception:
            return dict(_FAIL_METRICS)
        for v in metrics.values():
            if not math.isfinite(v):
                return dict(_FAIL_METRICS)
        return metrics

    # ── design variables ─────────────────────────────────────────────────────
    def n_elements(self, st) -> int:
        return int(st["spec"]["n_stages"])

    def get(self, st, i: int, p: str):
        if p != "base_stock" or not (0 <= i < st["base_stock"].shape[0]):
            return None
        return float(st["base_stock"][i])

    def set(self, st, i: int, p: str, v: float) -> None:
        if p != "base_stock" or not (0 <= i < st["base_stock"].shape[0]):
            return
        lo, hi = self.bounds(st, p)
        st["base_stock"][i] = min(hi, max(lo, float(v)))
        st["_diag"] = None            # invalidate the diagnostic cache

    def bounds(self, st, p: str):
        return (0.0, float(st["spec"]["base_stock_max"]))

    # ── program / reference ──────────────────────────────────────────────────
    def program(self, spec: dict):
        prog = program_from_goals(
            spec["goals"],
            state_keys=_METRIC_KEYS,
            objective_key="total_cost",
            objective_ref=self.reference(spec),
            objective_sense="min",
            domain=self.name,
        )
        # Envelope: a fill rate is a fraction, costs are positive and finite.
        # Any state outside this is simulator nonsense whatever the goals say.
        return replace(prog, validity_bounds=(
            ("service_level", 1.0 + 1e-9),
            ("backorder_frac", 1.0 + 1e-9),
            ("holding_cost", 1e12),
            ("total_cost", 1e12),
        ))

    def reference(self, spec: dict):
        cached = spec.get("reference_cost")
        try:
            if cached is not None and math.isfinite(float(cached)) \
                    and float(cached) > 0.0:
                return float(cached)
        except (TypeError, ValueError):
            pass
        goals = spec.get("goals", {})
        target = float(goals.get("minimum_service_level", 0.95))
        mean_d, std_d = self._demand_stats(spec["demand"])
        lt = np.asarray(spec["lead_times"], dtype=np.float64) + 1.0
        z = NormalDist().inv_cdf(min(max(target, 0.5), 0.999))
        heur_bs = mean_d * lt + z * std_d * np.sqrt(lt)
        ref, _ = self._heuristic_cost(spec, heur_bs)
        spec["reference_cost"] = float(max(ref, 1e-6))
        return spec["reference_cost"]

    # ── depth-0 feedback hint ────────────────────────────────────────────────
    def hint(self, st):
        spec = st["spec"]
        diag = st.get("_diag")
        if diag is None:
            try:
                rng = np.random.default_rng(
                    [int(spec["demand_seed"]),
                     stable_seed(spec.get("problem_id", ""), "hint")])
                metrics, d = self._simulate(spec, st["base_stock"], rng,
                                            collect_diag=True)
            except Exception:
                return None
            diag = {"metrics": metrics,
                    "stockouts": d["stockouts"],
                    "hold_per_stage": d["hold_per_stage"]}
            st["_diag"] = diag
        goals = spec.get("goals", {})
        target = float(goals.get("minimum_service_level", 0.0))
        budget = float(goals.get("maximum_holding_cost", float("inf")))
        m = diag["metrics"]
        if m["service_level"] < target:
            # service binds: raise stock where shortfalls concentrate
            return (int(np.argmax(diag["stockouts"])), "base_stock", "up")
        if m["holding_cost"] > budget:
            # budget binds: cut stock where holding money is spent
            return (int(np.argmax(diag["hold_per_stage"])), "base_stock", "down")
        # feasible on the diagnostic run: trim the costliest stage
        return (int(np.argmax(diag["hold_per_stage"])), "base_stock", "down")

    # ── self-contained observation (additive; read-only on the state) ────────
    @staticmethod
    def _fmt(x, sig: int = 4) -> str:
        """Compact number: `sig` significant figures, never rounds a nonzero
        small magnitude to 0, no exponent for ordinary magnitudes."""
        try:
            x = float(x)
        except (TypeError, ValueError):
            return str(x)
        if not math.isfinite(x):
            return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
        if x == 0.0:
            return "0"
        ax = abs(x)
        if ax >= 10 ** sig:
            return "%d" % int(round(x))
        if ax < 1e-3:
            return "%.*e" % (max(sig - 2, 1), x)
        digits = max(sig - 1 - int(math.floor(math.log10(ax))), 0)
        s = "%.*f" % (digits, x)
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    @staticmethod
    def _pct(x) -> str:
        x = 100.0 * float(x)
        if x == 0.0:
            return "0%"
        if abs(x) < 0.1:
            return "%.2g%%" % x
        if 99.95 <= x < 100.0:          # never print a shortfall as "100.0%"
            return "%.2f%%" % min(x, 99.99)
        return "%.1f%%" % x

    def _nominal_detail(self, spec: dict, base_stock):
        """Per-stage diagnostics for the nominal run: the same demand stream
        evaluate(st, seed=0) draws, replayed step for step as _simulate does
        (that method is left untouched; this is an independent read-only
        recomputation). Returns per-stage arrays plus the run's metrics."""
        if self.stochastic:
            rng = np.random.default_rng([int(spec["demand_seed"]), 0])
        else:
            rng = np.random.default_rng(int(spec["demand_seed"]))
        n = int(spec["n_stages"])
        T = int(spec["horizon"])
        warm = int(spec["warmup"])
        L = np.asarray(spec["lead_times"], dtype=np.int64)
        h = np.asarray(spec["holding_costs"], dtype=np.float64)
        b = float(spec["backorder_cost"])
        S = np.asarray(base_stock, dtype=np.float64).copy()

        demand = self._gen_demand(spec["demand"], T, rng)

        l_max = int(L.max())
        arrivals = np.zeros((n, T + l_max + 1), dtype=np.float64)
        on_hand = S.copy()
        intransit = np.zeros(n, dtype=np.float64)
        owed = np.zeros(n, dtype=np.float64)
        owed_up = np.zeros(n, dtype=np.float64)
        rows = np.arange(n - 1)
        lead_dest = L[: n - 1]

        hold_sum = 0.0
        bo_sum = 0.0
        demand_tot = 0.0
        filled_tot = 0.0
        bo_periods = 0
        n_scored = max(T - warm, 1)

        oh_sum = np.zeros(n)        # end-of-period on-hand
        owed_sum = np.zeros(n)      # end-of-period backlog owed downstream
        so_cnt = np.zeros(n)        # periods ending with unfilled backlog
        req_sum = np.zeros(n)       # new requests (customer demand / orders)
        fill_sum = np.zeros(n)      # new requests shipped in the same period

        for t in range(T):
            arr_t = arrivals[:, t]
            on_hand += arr_t
            intransit -= arr_t

            d = float(demand[t])
            avail = float(on_hand[0])
            bo0 = float(owed[0])
            serve_bo = bo0 if bo0 < avail else avail
            avail -= serve_bo
            bo0 -= serve_bo
            serve_d = d if d < avail else avail
            avail -= serve_d
            bo0 += d - serve_d
            on_hand[0] = avail
            owed[0] = bo0

            owed_up[: n - 1] = owed[1:]
            owed_up[n - 1] = 0.0
            q = S - (on_hand + intransit + owed_up - owed)
            np.maximum(q, 0.0, out=q)
            prev_owed = owed[1:].copy()
            if n > 1:
                owed[1:] += q[: n - 1]
            q_top = float(q[n - 1])
            if q_top > 0.0:
                intransit[n - 1] += q_top
                arrivals[n - 1, t + L[n - 1]] += q_top

            ship = None
            if n > 1:
                ship = np.minimum(on_hand[1:], owed[1:])
                on_hand[1:] -= ship
                owed[1:] -= ship
                intransit[: n - 1] += ship
                arrivals[rows, t + lead_dest] += ship

            if t >= warm:
                hold_vec = h * on_hand
                hold_sum += float(hold_vec.sum())
                bo_sum += b * float(owed[0])
                demand_tot += d
                filled_tot += serve_d
                if owed[0] > 1e-9:
                    bo_periods += 1
                oh_sum += on_hand
                owed_sum += owed
                so_cnt += (owed > 1e-9)
                req_sum[0] += d
                fill_sum[0] += serve_d
                if n > 1:
                    req_sum[1:] += q[: n - 1]
                    fill_sum[1:] += np.maximum(ship - prev_owed, 0.0)

        holding_cost = hold_sum / n_scored
        backorder_cost = bo_sum / n_scored
        service = filled_tot / demand_tot if demand_tot > 0.0 else 1.0
        metrics = {
            "service_level": float(service),
            "total_cost": float(holding_cost + backorder_cost),
            "holding_cost": float(holding_cost),
            "backorder_frac": float(bo_periods / n_scored),
        }
        fill = np.ones(n)
        pos = req_sum > 0.0
        fill[pos] = fill_sum[pos] / req_sum[pos]
        return {
            "metrics": metrics,
            "on_hand": oh_sum / n_scored,
            "backlog": owed_sum / n_scored,
            "stockout_frac": so_cnt / n_scored,
            "fill": fill,
            "request_rate": req_sum / n_scored,
            "holding": h * (oh_sum / n_scored),
            "n_scored": n_scored,
        }

    def _demand_text(self, dem: dict) -> str:
        kind = dem.get("kind")
        mean_d, std_d = self._demand_stats(dem)
        f = self._fmt
        if kind == "poisson":
            core = "Poisson, i.i.d. per period, rate %s units/period" % f(dem["mu"])
        elif kind == "negbin":
            core = ("negative binomial (over-dispersed Poisson), i.i.d. per period, "
                    "mean %s units/period, dispersion r=%s (smaller r = more variable)"
                    % (f(dem["mu"]), f(dem["r"])))
        elif kind == "bursty":
            pe = float(dem["p_enter"])
            px = float(dem["p_exit"])
            mu_b = float(dem["mu"]) * float(dem["burst_mult"])
            core = ("regime-switching Poisson: normal rate %s units/period; rate "
                    "during a burst %s (x%s), entered w.p. %s per period, left w.p. %s per "
                    "period (bursts last ~%s periods, ~%s of the time); demand is "
                    "autocorrelated, so a burst drains stock for several periods "
                    "in a row"
                    % (f(dem["mu"]), f(mu_b), f(dem["burst_mult"], 3), f(pe, 3),
                       f(px, 3), f(1.0 / px, 3), self._pct(pe / (pe + px))))
        else:
            core = str(kind)
        return "%s; long-run mean %s, sd %s units/period" % (core, f(mean_d), f(std_d))

    @staticmethod
    def _stage_role(i: int, n: int) -> str:
        if n == 1:
            return "customer-facing, orders from external source"
        if i == 0:
            return "customer-facing, supplied by stage 1"
        if i == n - 1:
            return "most upstream, feeds stage %d, orders from external source" % (i - 1)
        return "feeds stage %d, supplied by stage %d" % (i - 1, i + 1)

    def describe(self, st, spec) -> str:
        """Plain-text observation of the current design: problem setup, the
        per-stage design, and per-stage results of a nominal seed-0 run."""
        if spec is None:
            spec = st["spec"]
        f = self._fmt
        pct = self._pct
        n = int(spec["n_stages"])
        L = [int(x) for x in spec["lead_times"]]
        h = [float(x) for x in spec["holding_costs"]]
        b = float(spec["backorder_cost"])
        S = np.asarray(st["base_stock"], dtype=np.float64).copy()
        lo, hi = 0.0, float(spec["base_stock_max"])
        mean_d, std_d = self._demand_stats(spec["demand"])
        T = int(spec["horizon"])
        warm = int(spec["warmup"])
        fam = str(spec.get("family", "?")).replace("_", " ")

        out = ["PROBLEM"]
        out.append(
            "Serial %d-stage supply chain (%s family), periodic review, one "
            "base-stock (order-up-to) level per stage. Stages are listed "
            "customer-facing FIRST: stage 0 sells to the customer (finished "
            "goods); stage i orders from stage i+1; stage %d orders from an "
            "unlimited external source that always ships in full."
            % (n, fam, n - 1))
        out.append(
            "Each period: (1) shipments due arrive; (2) stage 0 serves its "
            "customer backlog first, then today's demand; unmet demand is "
            "backordered, never lost; (3) every stage orders so that its inventory "
            "position (on-hand + inbound in-transit + units its supplier still owes "
            "it - units it owes downstream) returns to its base stock S_i; (4) each "
            "supplier ships min(on-hand, owed) and keeps the rest as backlog; a "
            "shipment to stage i arrives L_i periods later; (5) costs are charged "
            "on end-of-period stock.")
        out.append("Customer demand at stage 0: %s. Orders pass this demand up the "
                   "chain one-for-one." % self._demand_text(spec["demand"]))
        out.append(
            "Costs ($ = generic cost unit): holding h_i $/unit/period on stage i's "
            "end-of-period on-hand (h rises downstream); backorder %s $ per unit of "
            "customer backlog outstanding at the end of each period (%s x h_0). "
            "Only stage 0 can backorder the customer; an upstream stockout delays "
            "the downstream stage's replenishment (its effective lead time grows "
            "beyond L)."
            % (f(b), f(b / h[0], 3) if h[0] > 0 else "n/a"))
        if self.stochastic:
            judged = ("Demand is random, so the constraint metrics are judged "
                      "conservatively over several independent demand draws: "
                      "service_level on a low (unlucky) draw, holding_cost on a "
                      "high one; total_cost and backorder_frac are draw averages. "
                      "A typical single path therefore usually reads better than "
                      "the constraint status.")
        else:
            judged = ("Demand follows one fixed seeded path, and the constraint "
                      "metrics are measured on that same path.")
        out.append(
            "Horizon %d periods; stock starts at S_i with empty pipelines; the "
            "first %d periods are warmup; results are averages over the last %d. "
            "Metric meanings: service_level = share of customer demand shipped in "
            "the period it arrives (stage 0's fill); backorder_frac = share of "
            "periods ending with customer backlog (stage 0's stockout share); "
            "holding_cost = chain-wide holding $/period; total_cost = holding_cost "
            "+ backorder $/period. %s" % (T, warm, max(T - warm, 1), judged))
        out.append(
            "Tunable parameter per stage: base_stock S_i (units), the order-up-to "
            "level of stage i's inventory position, bounds [%s, %s]. Raising S_i "
            "adds on-hand stock at stage i (cost h_i per unit per period) and "
            "covers more of the demand that arrives before a replenishment can "
            "land (the protection interval). Stage 0 reorders in the same period "
            "it sees customer demand, so it covers L_0 periods; an upstream stage "
            "i receives a downstream order only after placing its own order that "
            "period, so it covers L_i + 1 periods; any wait for a backlogged "
            "supplier lengthens the interval further. S_i at the mean demand "
            "over the interval (zero safety stock) stocks out in roughly half "
            "the periods; upstream stock protects the customer only indirectly, "
            "by cutting the downstream stage's supply wait." % (f(lo), f(hi)))

        out.append("")
        out.append("DESIGN (u = units of stock, per = periods, $ = cost unit, sd = "
                   "standard deviation. stage [role]: base stock S; lead time L "
                   "periods; holding h; protection interval P periods and the demand "
                   "over it; safety stock = S - that mean, also in sd units%s)"
                   % ("; the sd treats periods as independent, so bursts make the "
                      "real spread larger" if spec["demand"].get("kind") == "bursty"
                      else ""))
        for i in range(n):
            cyc = self._protection_periods(i, L[i])
            m_c = mean_d * cyc
            s_c = std_d * math.sqrt(cyc)
            ss = float(S[i]) - m_c
            k = ss / s_c if s_c > 0 else 0.0
            out.append(
                "stage %d [%s]: S=%s u; L=%d; h=%s $/u/per; P=%d, demand mean %s "
                "sd %s u; safety %s%s u (%s%s sd)"
                % (i, self._stage_role(i, n), f(S[i]), L[i], f(h[i]), cyc,
                   f(m_c), f(s_c), "+" if ss >= 0 else "", f(ss, 3),
                   "+" if k >= 0 else "", f(k, 3)))

        out.append("")
        try:
            det = self._nominal_detail(spec, S)
        except Exception:
            det = None
        if det is None or not all(math.isfinite(v) for v in det["metrics"].values()):
            out.append("RESULTS: simulation failed for this design.")
            return "\n".join(out)
        if self.stochastic:
            hdr = ("RESULTS - per-stage NOMINAL evaluation: one demand path (seed 0), "
                   "means over the %d scored periods. Demand is stochastic, so this "
                   "is one sample, not the conservative multi-draw constraint values "
                   "(a single path can be luckier or unluckier, and usually looks "
                   "better); read it for WHERE shortfalls and holding concentrate. ")
        else:
            hdr = ("RESULTS - per-stage evaluation on the fixed demand path, means "
                   "over the %d scored periods. ")
        out.append(
            (hdr % det["n_scored"])
            + "on-hand and backlog are end-of-period units; stockout = "
            "share of periods ending with unfilled backlog; fill = share of new "
            "requests shipped in the period they arrive; owes = mean backlog to "
            "the downstream party and the mean wait it causes them (periods, "
            "Little's law); supply wait = the mean wait this stage suffers from "
            "its own supplier; effective protection = P + that wait.")
        hold_tot = float(np.sum(det["holding"]))
        waits = []
        for i in range(n):
            rate = float(det["request_rate"][i])
            waits.append(float(det["backlog"][i]) / rate if rate > 1e-12 else 0.0)
        for i in range(n):
            who = "customer" if i == 0 else "stage %d" % (i - 1)
            share = det["holding"][i] / hold_tot if hold_tot > 0 else 0.0
            p_i = self._protection_periods(i, L[i])
            if i + 1 < n:
                sup_wait = waits[i + 1]
                sup_txt = ("supply wait %s per -> effective protection %s per"
                           % (f(sup_wait, 3), f(p_i + sup_wait, 3)))
            else:
                sup_txt = ("external supply, no wait -> effective protection %d per"
                           % p_i)
            out.append(
                "stage %d: on-hand %s u; holding %s $/per (%s of chain); stockout "
                "%s; fill %s of %s requests; owes %s %s u (wait %s per); %s"
                % (i, f(det["on_hand"][i]), f(det["holding"][i]), pct(share),
                   pct(det["stockout_frac"][i]), pct(det["fill"][i]), who, who,
                   f(det["backlog"][i]), f(waits[i], 3), sup_txt))
        return "\n".join(out)

    @staticmethod
    def _protection_periods(i: int, lead: int) -> int:
        """Periods of demand S_i must cover under this simulator's timing:
        stage 0 sees customer demand before it reorders (L_0); an upstream
        stage sees a downstream order after it has already ordered that
        period, so replenishment lags one extra period (L_i + 1)."""
        return int(lead) if i == 0 else int(lead) + 1

    def element_label(self, st, i) -> str:
        """Short inline role label for stage i."""
        spec = st["spec"]
        n = int(spec["n_stages"])
        i = int(i)
        if not (0 <= i < n):
            return "stage %d (out of range)" % i
        L = int(spec["lead_times"][i])
        h = float(spec["holding_costs"][i])
        if i == 0:
            role = "customer-facing"
        elif i == n - 1:
            role = "most upstream, feeds stage %d" % (i - 1)
        else:
            role = "feeds stage %d" % (i - 1)
        return "stage %d (%s, L=%d, h=%s)" % (i, role, L, self._fmt(h, 3))
