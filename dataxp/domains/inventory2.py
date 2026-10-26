"""Multi-echelon inventory NETWORK with a full action set (dataxp v2 domain).

Extends dataxp/domains/inventory.py (v1, one base-stock knob per stage, left
untouched) to a network whose STRUCTURE is designed too: review periods,
which intermediate stocking points hold stock, and which transport legs are
expedited.

Physics (always stochastic, like v1)
------------------------------------
A tree of N stages. Stages are numbered customer-facing FIRST: the leaves
(indices 0..n_customer-1) face external customer demand; every other stage
supplies one or more downstream stages; the root (last index) orders from an
unlimited external source that always ships in full. Serial chains are the
special case of one leaf. Each stage j has an inbound transport leg from its
physical parent with an integer transit time L_j (periods).

Periodic-review installation (R, S) policy, one period at a time:
  1. shipments due this period arrive;
  2. each customer-facing stage serves its customer backlog first, then
     today's demand; unmet demand is backordered (never lost);
  3. each ACTIVE stage whose review falls in this period ((t + j) mod R_j == 0)
     orders q = max(0, S_j - IP_j), IP = on-hand + in-transit + units its
     supplier still owes it - units it owes downstream. A supplier sees a
     downstream order in the period after it is placed (same timing as v1);
  4. every supplier ships min(on-hand, total owed); when short it rations in
     proportion to what each downstream stage is owed (one-period batch
     proportional rationing, a simplification); a shipment arrives after the
     route's transit time; the external source ships in full immediately;
  5. holding is charged on end-of-period on-hand.

Topology moves: an optional stocking point may be switched OFF. An inactive
stage is a cross-dock pass-through: it holds no stock, places no orders and
pays no ordering or holding cost; the stages it supplies order directly from
the nearest ACTIVE ancestor (or the external source) and their route transit
time becomes the sum of the legs along the path (their own leg plus every
inactive stage's inbound leg). Every on/off combination is a valid network
(pass-through always reconnects upstream), so there is no invalid topology;
a numerical failure returns finite, maximally bad metrics instead.

Demand: per customer-facing stage, family dependent: Poisson, negative
binomial (over-dispersed), or bursty regime-switching Poisson (shared burst
parameters, independent regimes per stage). Each evaluate() simulates
PATHS=20 independent demand paths vectorised side by side (seeded from
(demand_seed, seed)) over HORIZON=160 periods, scores the last 128 (WARMUP
32), and pools them (many short paths: per-evaluation service noise is about
a quarter of a single long path's at similar cost, since the cost is
per-period Python overhead). The pipelines start filled at mean flow and
on-hand at S - mean pipeline (approximate steady state), so a short warmup
suffices.

Planner convention for structural moves (set()): a review, expedite or
active change that alters any stage's protection interval P (route transit +
R - 1, +1 for non-customer stages) shifts that stage's base_stock by
mean_flow x (P_new - P_old), i.e. its SAFETY STOCK (S - mean demand over P)
is kept, the way a planning system re-derives order-up-to levels when lead
times or review cycles change. This touches the changed stage and, for
active/expedite on a pass-through or a stocking point, the stages whose
supply route runs through it. Without it every route-shortening move
inflates on-hand at fixed S and only pays off after a follow-up cut, which a
one-step chooser cannot see (measured on 24 problems: phi_select reached
feasibility on 58% without it, 79% with it).

Elements and params (element = stage; indices never change)
----------------------------------------------------------
  base_stock  continuous, units, [0, base_stock_max]; legalised to 0.1 units.
              The order-up-to level S_j. Dormant (kept, no effect) while the
              stage is inactive. Re-derived by structural moves (see above).
  review      categorical, codes {1, 2, 4} = review period in periods (order
              every period / every 2nd / every 4th). Other values round to the
              nearest code (<1.5 -> 1, <3 -> 2, else 4). Each review of an
              active stage costs the order cost K_j, so ordering costs K_j/R_j
              per period; a longer period needs R_j - 1 extra periods of cover.
  active      toggle {0, 1}, only on the problem's fixed pool of OPTIONAL
              stocking points (non-customer-facing stages); get() returns None
              on every other stage (customer-facing and mandatory stages are
              always active).
  expedite    toggle {0, 1}, every stage (its inbound leg): leg transit
              L -> ceil(L/2) at a premium e_j $ per unit moved over the leg.

Metrics ($ = generic cost unit, per period over the scored window)
  service_level   share of customer demand served in the period it arrives,
                  pooled over every customer-facing stage and path
  holding_cost    sum_j h_j * mean end-of-period on-hand_j
  order_cost      sum over active j of K_j / R_j
  expedite_cost   sum over expedited legs of e_j * mean flow through the leg
  backorder_cost  b * units of customer demand not served on arrival
  total_cost      holding + order + expedite + backorder   (objective, min)
Goals: minimum_service_level (0.90-0.97) and maximum_holding_cost (a budget
calibrated per problem, see below). Validity envelope: service in [0, 1],
every cost <= 1e9.

Problems: four families x size ladder (n = 3..20 stages, v1 had 2..20):
serial_short (3-6, Poisson), serial_long (8-20, negative binomial),
distribution (plant -> 1 DC with 3-6 stores, or 2-3 DCs with 2-4 stores
each; 5-16 stages), bursty_network (serial 4-8, or plant -> 1-3 DCs with 2-4
/ 2-3 stores; 4-13 stages; bursty demand). Transit times: customer legs 2-4,
intermediate legs 2-5, the root's leg 3-6 periods. The
initial design carries little or no safety stock, long review periods at the
customer-facing stages, a random half of the optional stocking points off,
and almost no expediting, so it usually starts infeasible on service.

Budget and heuristic (reference_class "heuristic"): a seeded coarse grid over
structure (active pool {as given, all on, all off}) x review {all 1,
EOQ-rounded per stage} x expedite {none, customer legs, all legs}, with base
stock from the textbook (R, S) formula S = mu*P + z*sd*sqrt(P)
(customer-facing stages at safety factor z, upstream stages at 0.6 z) and z
scanned upward until the service target (+0.005) is met on EVERY one of 4
fixed calibration seeds; a design's holding is its highest over those seeds
(at least as strict as the evaluator's chance constraint, so the grid
designs are feasible as judged). The same scan on the GIVEN structure
(initial review / active / expedite, base stock only; best of the upstream
rules 0.6 z, 0.3 z, 0) gives the knob-only holding H_knob. With H_min = the
lowest holding any grid design needs to meet service, the budget is
H_min + u (H_knob - H_min), u ~ U(0.3, 0.7): the formula knob-only repair
of the given structure is over budget by construction, while restructured
designs keep slack (budget >= 1.05 H_min, so the H_min design is a
feasible witness). When restructuring buys little (H_knob missing or within
1.25 H_min) the budget is (1.05 + 0.25 u) H_min, capped at 0.95 H_knob
but never below 1.05 H_min. The heuristic cost
(objective normaliser for obj/ref) is the mean total cost of the cheapest
grid design meeting both goals with 2% holding slack.

Simplifications (explicit): no pipeline holding cost; ordering cost is
charged per review epoch (orders are placed at virtually every review at
these demand rates); the expedite premium is charged on the analytic mean
flow through the leg (not the simulated shipments), so it is deterministic;
proportional rationing; no capacity limits; unit-level demand but continuous
(fractional) shipments; review phases are fixed per stage ((t + j) mod R).

numpy + stdlib only; one evaluate() is ~3-8 ms at N=20 (20 paths x 160
periods vectorised).

Verb surface (docs/dataxp_verb_surface.md; owner's decision of 2026-10-23)
------------
The move set (candidate_moves), set() with its re-planning rule, Phi and the
stored {i, p, v, v_old} dicts are the v2 ones above and are untouched; only
the spelling in prompts, action lines and reply parsing changes (class
attributes verb_surface and verb_table; format_move / parse_move / grammar are
the table-driven defaults of dataxp.domain). The bijection, per parameter:
  base_stock v      <-> SET_BASE_STOCK(n, v)                          [units]
  review c          <-> SET_REVIEW_PERIOD(n, c)        c = 1 / 2 / 4  [periods]
  active 1 / 0      <-> OPEN_STOCKING_POINT(n) / CLOSE_STOCKING_POINT(n)
  expedite 1 / 0    <-> ENABLE_EXPEDITING(n) / DISABLE_EXPEDITING(n)
Judgment calls, recorded here as the surface rules ask:
  * n is the stage number, the element index SET_PARAM used (0..N-1, the
    number after "stage" on the DESIGN line).
  * The review period is spelled as its length in periods (SET_REVIEW_PERIOD
    (n, 2)), a Continuous with the unit "periods", not a Categorical by name:
    the codes 1 / 2 / 4 ARE the counts the field speaks in ("review every 2
    periods"), and a Categorical's names are indexed by code 0..n-1, which
    cannot hold {1, 2, 4} without placeholder entries. The grammar lists the
    three admissible values; parse_move accepts any number and set() rounds
    it to the nearest code exactly as SET_PARAM(n, review, 2.9) did (legal
    moves only ever carry 1, 2 or 4, so the surface is one-to-one on them).
    The mechanism's Continuous prints its argument as "float" in grammar();
    integral values are written bare ("2"), so action lines read as counts.
  * SET_BASE_STOCK's value is the new absolute order-up-to level in units
    (unit in grammar() only), legalised to 0.1 and clamped by set() as
    before. On a closed stage it is never offered; written anyway it parses
    and set() stores the dormant value, as SET_PARAM did.
  * OPEN / CLOSE_STOCKING_POINT on a stage that is not in the optional pool
    parses and set() ignores it (get() is None there), as SET_PARAM did;
    legal_moves never offers it. ENABLE / DISABLE_EXPEDITING act on the
    stage's inbound leg, a pass-through's included, as the parameter did.
  * The automatic re-planning of base stock on a structural change is set()'s
    behaviour, not a verb; the verb docs and the describe() paragraph say it
    happens, as the SET_PARAM text did.
  * SET_PARAM(...) text is no longer parsed on this domain: the grammar the
    policy is served lists the verbs only.
  * describe()'s "Parameters" paragraph is written in the verbs under
    verb_surface (observation text only; no generated file contains it);
    every other observation line (the DESIGN previews' "switch on" /
    "un-expedite", RESULTS) and element_label are unchanged, so history lines
    read e.g. "SET_BASE_STOCK(3, 52.3) (stage 3 (customer-facing, leg 3 per);
    was 46.6)", "SET_REVIEW_PERIOD(0, 2) (...; was 4)"; the toggle pairs
    need no "was".

Set-parameter spelling (docs/dataxp_spellings.md; DATAXP_SPELLING=set_parameter)
-----------------------
The same moves through the one wrapper tool, generated from the verb table
by dataxp.domain (no override here): SET_BASE_STOCK(3, 52.3) <-> SET_PARAM(3,
base_stock, 52.3), SET_REVIEW_PERIOD(0, 2) <-> SET_PARAM(0, review, 2),
OPEN_STOCKING_POINT(4) / CLOSE_STOCKING_POINT(4) <-> SET_PARAM(4, active, 1 /
0), ENABLE_EXPEDITING(2) / DISABLE_EXPEDITING(2) <-> SET_PARAM(2, expedite,
1 / 0). Judgment calls: the phrases are "Set a stage's base stock", "Set a
stage's review period", "Open a stocking point" / "Close a stocking point",
"Enable expediting" / "Disable expediting"; the "Parameters" paragraph
reverts to its SET_PARAM wording (the run2 text) under this spelling
(spelled_in_verbs).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from ..domain import (ArgSpec, Candidate, Continuous, Domain, ParamVerb, Toggle, spelled_in_verbs,
                      stable_seed)
from ..potential import program_from_goals

_FAMILIES = ("serial_short", "serial_long", "distribution", "bursty_network")
_REVIEW_CODES = (1, 2, 4)
_PARAMS = ("base_stock", "review", "active", "expedite")

_METRIC_KEYS = ("service_level", "total_cost", "holding_cost", "order_cost",
                "expedite_cost", "backorder_cost")

_FAIL_VALUE = 1e10        # beyond the validity envelope (1e9): invalid, finite
_FAIL_METRICS = {
    "service_level": 0.0,
    "total_cost": _FAIL_VALUE,
    "holding_cost": _FAIL_VALUE,
    "order_cost": _FAIL_VALUE,
    "expedite_cost": _FAIL_VALUE,
    "backorder_cost": _FAIL_VALUE,
}

_HORIZON = 160
_WARMUP = 32
_PATHS = 20
# Calibration (budget + heuristic): every design is judged on 4 fixed seeds by
# its WORST service and WORST holding, with a small service margin, so the
# heuristic design is feasible under the evaluator's chance constraint (10% /
# 90% replicate quantiles). Judging by the 2-seed MEAN (+0.01) left the
# heuristic design infeasible under the evaluator on ~1/4 of problems.
_CAL_SEEDS = 4
_CAL_MARGIN = 0.005
_REF_HOLD_SLACK = 0.02
_Z_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5)
# upstream safety factor = rule x customer-facing safety factor z. Grid
# (witness) designs use the robust 0.6 rule only: with ~0 upstream safety the
# service of a long chain varies so much between seeds that a design passing
# 4 calibration seeds often fails the evaluator (5 of 8 serial_long problems).
# The knob-only proxy takes the best of three rules, since one rule alone
# overstates what a knob-only repair needs on long chains.
_UP_RULES_GRID = (0.6,)
_UP_RULES_KNOB = (0.6, 0.3, 0.0)
# base_stock move ladder in priority order: ("x", f) scales S by f; ("sd", k)
# adds k standard deviations of demand over the stage's protection interval
_BS_LADDER = (("x", 0.8), ("x", 1.25), ("sd", 1.0), ("sd", -1.0),
              ("x", 0.9), ("x", 1.1), ("sd", 0.5), ("sd", -0.5),
              ("x", 0.65), ("x", 1.5), ("sd", 2.0), ("sd", -2.0),
              ("x", 0.95), ("x", 1.05), ("x", 0.5), ("x", 2.0))
_CAND_TARGET = 90          # aim for ~90 moves per turn on small networks
_CAND_MAX = 160

# ── verb surface (docstring "Verb surface"; docs/dataxp_verb_surface.md) ─────
_STAGE_ARG = ArgSpec("stage", "int", "stage number 0..N-1 (the number after 'stage' on its "
                                     "DESIGN line)")
_LEVEL = Continuous("level", "units",
                    "0..the base-stock bound the problem states, the new order-up-to level S "
                    "(legalised to 0.1 units; a value outside is set to the bound)")
_PERIOD = Continuous("period", "periods",
                     "%s: review and order every period / every 2nd / every 4th (any other "
                     "number rounds to the nearest of the three)"
                     % ", ".join(str(c) for c in _REVIEW_CODES))


def _round_review(v: float) -> int:
    if v < 1.5:
        return 1
    if v < 3.0:
        return 2
    return 4


class Inventory2Domain(Domain):
    """Multi-echelon (R, S) inventory network with structural moves."""
    # stochastic variants: phi_select compares candidates with a fresh
    # incumbent estimate (dataxp/ladder.py); no effect when deterministic
    rescore_incumbent = True

    name = "inventory2"
    params = _PARAMS
    factors = (0.8, 0.9, 1.1, 1.25)
    reference_class = "heuristic"
    max_elements = 24
    max_candidates = _CAND_MAX
    rebase = True       # structural moves keep safety stock (see set())

    # ── verb surface (docstring "Verb surface"; docs/dataxp_verb_surface.md) ──
    # Spelling only: every SET_PARAM move candidate_moves enumerates has one
    # verb text and parses back to the same move. format_move / parse_move /
    # grammar are the table-driven defaults of dataxp.domain.
    verb_surface = True
    verb_table = (
        ParamVerb("base_stock", "SET_BASE_STOCK", _STAGE_ARG, _LEVEL,
                  doc="set a stage's order-up-to level S (its safety stock moves by the same "
                      "amount); dormant on a closed (pass-through) stage.",
                  phrase="Set a stage's base stock"),
        ParamVerb("review", "SET_REVIEW_PERIOD", _STAGE_ARG, _PERIOD,
                  doc="set a stage's review period R: ordering costs K/R per period, and the "
                      "stage's base stock is re-planned to the new protection interval so its "
                      "safety stock is kept.",
                  phrase="Set a stage's review period"),
        ParamVerb("active", "OPEN_STOCKING_POINT", _STAGE_ARG,
                  Toggle("CLOSE_STOCKING_POINT",
                         off_doc="close an optional stocking point: it becomes a cross-dock "
                                 "pass-through (no stock, no orders, no holding or ordering "
                                 "cost); the stages below order from the nearest open stage "
                                 "above over the longer route, their base stock re-planned to "
                                 "keep their safety stock.",
                         off_phrase="Close a stocking point"),
                  doc="open an optional stocking point (a stage marked optional in DESIGN): it "
                      "holds stock at its stored base stock and reviews every R periods; the "
                      "stages below get the shorter route, their base stock re-planned to keep "
                      "their safety stock.",
                  phrase="Open a stocking point"),
        ParamVerb("expedite", "ENABLE_EXPEDITING", _STAGE_ARG,
                  Toggle("DISABLE_EXPEDITING",
                         off_doc="stop expediting a stage's inbound leg: its transit returns to "
                                 "L periods and the premium stops; base stock along the route "
                                 "is re-planned to keep safety stock.",
                         off_phrase="Disable expediting"),
                  doc="expedite a stage's inbound transport leg (a pass-through's too): transit "
                      "L -> ceil(L/2) periods at e $ per unit of mean flow over the leg; base "
                      "stock along the route is re-planned to keep safety stock.",
                  phrase="Enable expediting"),
    )

    def __init__(self, stochastic: bool = True):
        super().__init__(stochastic)

    # ── demand ───────────────────────────────────────────────────────────────
    @staticmethod
    def _leaf_stats(dem: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Stationary per-period (mean, sd) of each customer-facing stage."""
        mu = np.asarray(dem["mu"], dtype=np.float64)
        kind = dem["kind"]
        if kind == "poisson":
            return mu, np.sqrt(mu)
        if kind == "negbin":
            r = float(dem["r"])
            return mu, np.sqrt(mu + mu * mu / r)
        if kind == "bursty":
            m = float(dem["burst_mult"])
            pe = float(dem["p_enter"])
            px = float(dem["p_exit"])
            pi_b = pe / (pe + px)
            mu_b = mu * m
            mean = (1.0 - pi_b) * mu + pi_b * mu_b
            var = ((1.0 - pi_b) * mu + pi_b * mu_b
                   + (1.0 - pi_b) * (mu - mean) ** 2 + pi_b * (mu_b - mean) ** 2)
            return mean, np.sqrt(var)
        raise ValueError("unknown demand kind %r" % kind)

    @staticmethod
    def _gen_demand(dem: dict, T: int, P: int, rng: np.random.Generator) -> np.ndarray:
        """Demand array (T, n_customer, P)."""
        mu = np.asarray(dem["mu"], dtype=np.float64)
        nl = mu.shape[0]
        kind = dem["kind"]
        if kind == "poisson":
            return rng.poisson(np.broadcast_to(mu[None, :, None], (T, nl, P))
                               ).astype(np.float64)
        if kind == "negbin":
            r = float(dem["r"])
            p = r / (r + mu)
            return rng.negative_binomial(
                r, np.broadcast_to(p[None, :, None], (T, nl, P))).astype(np.float64)
        if kind == "bursty":
            m = float(dem["burst_mult"])
            pe = float(dem["p_enter"])
            px = float(dem["p_exit"])
            pi_b = pe / (pe + px)
            u = rng.random((T, nl, P))
            burst = rng.random((nl, P)) < pi_b
            lam = np.empty((T, nl, P), dtype=np.float64)
            base = np.broadcast_to(mu[:, None], (nl, P))
            hi = base * m
            for t in range(T):
                burst = np.where(burst, u[t] >= px, u[t] < pe)
                lam[t] = np.where(burst, hi, base)
            return rng.poisson(lam).astype(np.float64)
        raise ValueError("unknown demand kind %r" % kind)

    # ── structure ────────────────────────────────────────────────────────────
    @staticmethod
    def _leg_lead(L: int, exp: bool) -> int:
        return int((L + 1) // 2) if exp else int(L)

    def _routes(self, spec: dict, active, expedite):
        """For EVERY stage j: (supplier, route transit) it would have if it
        were active - supplier is the nearest active proper ancestor (-1 =
        external source), transit = its own leg plus every inactive
        ancestor's leg on the way."""
        n = int(spec["n_stages"])
        par = spec["parent"]
        L = spec["lead_times"]
        leg = [self._leg_lead(int(L[j]), bool(expedite[j])) for j in range(n)]
        sup, lead = [], []
        for j in range(n):
            tot = leg[j]
            k = int(par[j])
            while k != -1 and not active[k]:
                tot += leg[k]
                k = int(par[k])
            sup.append(k)
            lead.append(tot)
        return sup, lead, leg

    @staticmethod
    def _protection(spec: dict, j: int, lead: int, R: int) -> int:
        """Periods of demand S_j must cover: route transit + R - 1, plus one
        for a non-customer stage (it sees downstream orders a period late)."""
        extra = 0 if j < int(spec["n_customer"]) else 1
        return int(lead) + int(R) - 1 + extra

    # ── core simulator ───────────────────────────────────────────────────────
    def _simulate(self, spec: dict, S, R, active, expedite,
                  rng: np.random.Generator, diag: bool = False):
        n = int(spec["n_stages"])
        nl = int(spec["n_customer"])
        T = int(spec["horizon"])
        warm = int(spec["warmup"])
        P = int(spec["paths"])
        h = np.asarray(spec["holding_costs"], dtype=np.float64)
        K = np.asarray(spec["order_costs"], dtype=np.float64)
        e = np.asarray(spec["expedite_premiums"], dtype=np.float64)
        b = float(spec["backorder_cost"])
        flow = np.asarray(spec["flow_mean"], dtype=np.float64)

        sup, lead, _ = self._routes(spec, active, expedite)
        act = [j for j in range(n) if active[j]]
        pos = {j: k for k, j in enumerate(act)}
        m = len(act)
        eff_par = np.array([pos[sup[j]] if sup[j] != -1 else m for j in act],
                           dtype=np.int64)
        Leff = np.array([lead[j] for j in act], dtype=np.int64)
        Sa = np.array([float(S[j]) for j in act], dtype=np.float64)[:, None]
        Ra = np.array([int(R[j]) for j in act], dtype=np.int64)
        off = np.array([j % int(R[j]) for j in act], dtype=np.int64)
        ha = h[act]
        mua = flow[act]

        D = self._gen_demand(spec["demand"], T, P, rng)          # (T, nl, P)

        rev = (((np.arange(T)[:, None] + off[None, :]) % Ra[None, :]) == 0)
        rev = rev.astype(np.float64)[:, :, None]                   # (T, m, 1)
        # supplier incidence: C[s, k] = 1 if stage k orders from s (row m =
        # the external source)
        C = np.zeros((m + 1, m), dtype=np.float64)
        C[eff_par, np.arange(m)] = 1.0
        Cs = C[nl:]                    # supplier rows: internal stages + source
        Ci = C[nl:m]                   # internal stages only
        par_s = eff_par - nl           # supplier row of each stage within Cs

        l_max = int(Leff.max())
        arrivals = np.zeros((T + l_max + 1, m, P), dtype=np.float64)
        for k in range(m):
            arrivals[: int(Leff[k]), k, :] = mua[k]
        # inv rows: customer-facing stages hold NET inventory (on-hand minus
        # customer backlog, may be negative); internal stages hold on-hand
        # (never negative: they ship at most what they have); last row is the
        # external source (infinite stock)
        inv = np.zeros((m + 1, P), dtype=np.float64)
        inv[m] = np.inf
        invm = inv[:m]
        invl = inv[:nl]
        invs = inv[nl:]
        pipe0 = (mua * Leff)[:, None]
        invm[:] = np.maximum(Sa - pipe0, 0.0)
        ip = invm + pipe0              # inventory position, tracked incrementally
        ipl = ip[:nl]
        ipi = ip[nl:]
        pend = np.zeros((m, P), dtype=np.float64)   # owed TO stage k by its supplier
        kk = np.arange(m)

        oh_sum = np.zeros((m, P), dtype=np.float64)
        short_sum = np.zeros((nl, P), dtype=np.float64)
        if diag:
            owed_sum = np.zeros((m, P))
            so_cnt = np.zeros((m, P))
            req_sum = np.zeros((m, P))
            pend_sum = np.zeros((m, P))
            q_sum = np.zeros((m, P))

        for t in range(T):
            invm += arrivals[t]                       # 1. arrivals
            d = D[t]                                  # 2. customers: backlog first
            sd = np.minimum(d, np.maximum(invl, 0.0))  #    served on arrival
            sh = d - sd
            invl -= d
            ipl -= d
            q = Sa - ip                               # 3. order up to S at reviews
            np.maximum(q, 0.0, out=q)
            q *= rev[t]
            ip += q
            cq = Ci @ q                               #    seen upstream next period
            ipi -= cq
            pend += q
            cs = Cs @ pend                            # 4. ship, ration pro rata
            ship = np.minimum(invs, cs)
            ratio = ship / np.maximum(cs, 1e-12)
            invs -= ship
            alloc = pend * ratio[par_s]
            pend -= alloc
            arrivals[t + Leff, kk] += alloc
            if t >= warm:                             # 5. accounting
                oh_sum += np.maximum(invm, 0.0)
                short_sum += sh
                if diag:
                    owed = np.zeros((m, P))
                    owed[:nl] = np.maximum(-invl, 0.0)
                    owed[nl:] = Ci @ pend
                    owed_sum += owed
                    so_cnt += owed > 1e-9
                    req = np.zeros((m, P))
                    req[:nl] = d
                    req[nl:] = cq
                    req_sum += req
                    pend_sum += pend
                    q_sum += q

        n_sc = max(T - warm, 1)
        denom = float(n_sc * P)
        holding = float((ha[:, None] * oh_sum).sum() / denom)
        short_tot = float(short_sum.sum())
        dem_tot = float(D[warm:].sum())
        service = 1.0 - short_tot / dem_tot if dem_tot > 0.0 else 1.0
        backorder = b * short_tot / denom
        order = float(sum(K[j] / float(R[j]) for j in act))
        expedite_cost = float(sum(e[j] * flow[j] for j in range(n) if expedite[j]))
        metrics = {
            "service_level": float(service),
            "total_cost": float(holding + order + expedite_cost + backorder),
            "holding_cost": holding,
            "order_cost": order,
            "expedite_cost": expedite_cost,
            "backorder_cost": float(backorder),
        }
        if not diag:
            return metrics, None
        leaf_dem = D[warm:].sum(axis=(0, 2))
        leaf_short = short_sum.sum(axis=1)
        det = {
            "act": act,
            "sup": sup,
            "lead": lead,
            "on_hand": oh_sum.sum(axis=1) / denom,
            "owed": owed_sum.sum(axis=1) / denom,
            "stockout": so_cnt.sum(axis=1) / denom,
            "req": req_sum.sum(axis=1) / denom,
            "pend": pend_sum.sum(axis=1) / denom,
            "q": q_sum.sum(axis=1) / denom,
            "leaf_fill": np.where(leaf_dem > 0, 1.0 - leaf_short / np.maximum(leaf_dem, 1e-12), 1.0),
            "holding": ha * (oh_sum.sum(axis=1) / denom),
        }
        return metrics, det

    def _run(self, spec, S, R, active, expedite, seed: int, diag: bool = False):
        if self.stochastic:
            rng = np.random.default_rng([int(spec["demand_seed"]), int(seed) & 0xFFFFFFFF])
        else:
            rng = np.random.default_rng([int(spec["demand_seed"]), 0])
        return self._simulate(spec, S, R, active, expedite, rng, diag=diag)

    # ── problems ─────────────────────────────────────────────────────────────
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[index % len(_FAMILIES)]
        parent: List[int] = []
        lead: List[int] = []
        if family in ("serial_short", "serial_long") or (
                family == "bursty_network" and rng.random() < 0.5):
            if family == "serial_short":
                n = int(rng.integers(3, 7))
            elif family == "serial_long":
                n = int(rng.integers(8, 21))
            else:
                n = int(rng.integers(4, 9))
            nl = 1
            parent = [j + 1 for j in range(n - 1)] + [-1]
            lead = [int(rng.integers(2, 5))] + [int(rng.integers(2, 6))
                                                 for _ in range(n - 2)]
            if n >= 2:
                lead.append(int(rng.integers(3, 7)))
            topo = "serial"
        else:
            if family == "distribution":
                n_dc = int(rng.integers(1, 4))
                per = ([int(rng.integers(3, 7))] if n_dc == 1 else
                       [int(rng.integers(2, 5)) for _ in range(n_dc)])
            else:
                n_dc = int(rng.integers(1, 4))
                per = ([int(rng.integers(2, 5))] if n_dc == 1 else
                       [int(rng.integers(2, 4)) for _ in range(n_dc)])
            nl = int(sum(per))
            n = nl + n_dc + 1
            root = n - 1
            for d_i, cnt in enumerate(per):
                for _ in range(cnt):
                    parent.append(nl + d_i)
            for _ in range(n_dc):
                parent.append(root)
            parent.append(-1)
            lead = ([int(rng.integers(2, 5)) for _ in range(nl)]
                    + [int(rng.integers(2, 6)) for _ in range(n_dc)]
                    + [int(rng.integers(3, 7))])
            topo = "tree"

        # demand
        if family == "serial_short":
            demand = {"kind": "poisson",
                      "mu": [float(rng.uniform(4.0, 20.0)) for _ in range(nl)]}
            target = round(float(rng.uniform(0.90, 0.97)), 3)
        elif family == "serial_long":
            demand = {"kind": "negbin",
                      "mu": [float(rng.uniform(3.0, 12.0)) for _ in range(nl)],
                      "r": float(rng.uniform(2.0, 8.0))}
            target = round(float(rng.uniform(0.90, 0.95)), 3)
        elif family == "distribution":
            if rng.random() < 0.5:
                demand = {"kind": "poisson",
                          "mu": [float(rng.uniform(2.0, 10.0)) for _ in range(nl)]}
            else:
                demand = {"kind": "negbin",
                          "mu": [float(rng.uniform(2.0, 10.0)) for _ in range(nl)],
                          "r": float(rng.uniform(2.0, 8.0))}
            target = round(float(rng.uniform(0.90, 0.97)), 3)
        else:
            demand = {"kind": "bursty",
                      "mu": [float(rng.uniform(2.0, 8.0)) for _ in range(nl)],
                      "burst_mult": float(rng.uniform(2.5, 4.5)),
                      "p_enter": float(rng.uniform(0.03, 0.08)),
                      "p_exit": float(rng.uniform(0.2, 0.4))}
            target = round(float(rng.uniform(0.90, 0.95)), 3)

        # flow through every stage (sum of the customer demand below it)
        mu_l, sd_l = self._leaf_stats(demand)
        flow_mean = np.zeros(n)
        flow_var = np.zeros(n)
        for leaf in range(nl):
            k = leaf
            while k != -1:
                flow_mean[k] += mu_l[leaf]
                flow_var[k] += sd_l[leaf] ** 2
                k = parent[k]
        flow_sd = np.sqrt(flow_var)

        # costs: holding rises downstream
        depth = np.zeros(n, dtype=np.int64)
        for j in range(n):
            k, dd = j, 0
            while parent[k] != -1:
                k = parent[k]
                dd += 1
            depth[j] = dd
        h = np.zeros(n)
        h_root = float(rng.uniform(0.2, 0.8))
        step = (rng.uniform(1.05, 1.3, size=n) if topo == "serial"
                else rng.uniform(1.25, 1.7, size=n))
        order_top = sorted(range(n), key=lambda j: depth[j])
        for j in order_top:
            h[j] = h_root if parent[j] == -1 else h[parent[j]] * float(step[j])
        r_star = np.exp(rng.uniform(math.log(0.9), math.log(4.5), size=n))
        K = r_star ** 2 * h * flow_mean / 2.0
        e_prem = h * rng.uniform(0.15, 1.2, size=n)
        b = float(np.mean(h[:nl]) * rng.uniform(6.0, 25.0))

        # optional stocking points (fixed pool): non-customer stages
        optional = []
        for j in range(nl, n):
            p_opt = 0.85 if (topo == "tree" and parent[j] != -1) else 0.7
            if rng.random() < p_opt:
                optional.append(j)
        if not optional:
            optional.append(int(rng.integers(nl, n)))
        optional.sort()

        # initial structure: long reviews at the customer, half the optional
        # points off, almost no expediting
        R0 = []
        for j in range(n):
            if j < nl:
                R0.append(int(rng.choice([2, 4], p=[0.35, 0.65])))
            else:
                R0.append(int(rng.choice([1, 2, 4], p=[0.3, 0.3, 0.4])))
        active0 = [True] * n
        for j in optional:
            active0[j] = bool(rng.random() < 0.5)
        exp0 = [bool(rng.random() < 0.08) for _ in range(n)]

        # base_stock bound: generous multiple of the longest possible cover
        full = []
        for j in range(n):
            tot, k = 0, j
            while k != -1:
                tot += int(lead[k])
                k = parent[k]
            full.append(tot + 4)
        bs_max = float(np.ceil(3.0 * max(flow_mean[j] * full[j]
                                         + 4.0 * flow_sd[j] * math.sqrt(full[j])
                                         for j in range(n))))

        spec = {
            "family": family,
            "topology": topo,
            "n_stages": int(n),
            "n_customer": int(nl),
            "parent": [int(x) for x in parent],
            "lead_times": [int(x) for x in lead],
            "holding_costs": [float(x) for x in h],
            "order_costs": [float(x) for x in K],
            "expedite_premiums": [float(x) for x in e_prem],
            "backorder_cost": float(b),
            "optional": [int(x) for x in optional],
            "demand": demand,
            "flow_mean": [float(x) for x in flow_mean],
            "flow_sd": [float(x) for x in flow_sd],
            "horizon": _HORIZON,
            "warmup": _WARMUP,
            "paths": _PATHS,
            "demand_seed": int(rng.integers(0, 2 ** 31 - 1)),
            "base_stock_max": bs_max,
        }
        # initial base stock: little or no safety stock where active; a
        # zero-safety value kept dormant where inactive
        kappa = float(rng.uniform(-1.0, 0.3))
        kappa_up = float(rng.uniform(-0.3, 0.5))
        _, lead_r, _ = self._routes(spec, active0, exp0)
        S0 = []
        for j in range(n):
            Pj = self._protection(spec, j, lead_r[j], R0[j])
            if active0[j]:
                k_j = (kappa if j < nl else kappa_up) + float(rng.uniform(-0.25, 0.25))
            else:
                k_j = 0.0
            v = flow_mean[j] * Pj + k_j * flow_sd[j] * math.sqrt(Pj)
            S0.append(self._legal_bs(max(v, 0.0), bs_max))
        spec["initial"] = {
            "base_stock": S0,
            "review": [int(x) for x in R0],
            "active": [1 if a else 0 for a in active0],
            "expedite": [1 if x else 0 for x in exp0],
        }

        cal = self._calibrate(spec, target, r_star)
        # budget between the best structured design and the knob-only repair
        # of the given structure: the latter stays over budget by
        # construction, the former keeps slack. When restructuring buys
        # little (knob-only within 1.25x of the best grid design), the budget
        # sits 12-23% above the best grid design but, where that leaves room,
        # still 5% below the knob-only repair.
        u = float(rng.uniform(0.3, 0.7))
        h_min = float(cal["holding_min"])
        h_knob = cal["knob_only_holding"]
        if h_knob is not None and h_knob > 1.25 * h_min:
            budget = h_min + u * (float(h_knob) - h_min)
        else:
            budget = h_min * (1.05 + 0.25 * u)
            if h_knob is not None:
                budget = max(1.05 * h_min, min(budget, 0.95 * float(h_knob)))
        # heuristic = cheapest grid design meeting both goals, with 2% holding
        # slack so it is not a design sitting exactly on the budget (the
        # holding_min design always qualifies: budget >= 1.05 holding_min)
        ref = None
        for row in cal["rows"]:
            if (row["service"] >= target + _CAL_MARGIN
                    and row["holding"] <= (1.0 - _REF_HOLD_SLACK) * budget):
                if ref is None or row["cost"] < ref["cost"]:
                    ref = row
        if ref is None:
            ref = min(cal["rows"], key=lambda r: r["holding"])
        spec["goals"] = {
            "minimum_service_level": float(target),
            "maximum_holding_cost": budget,
        }
        spec["reference_cost"] = float(max(ref["cost"], 1e-6))
        spec["calibration"] = {
            "holding_min": float(cal["holding_min"]),
            "knob_only_holding": (float(cal["knob_only_holding"])
                                  if cal["knob_only_holding"] is not None else None),
            "heuristic_config": ref["config"],
        }
        return spec

    @staticmethod
    def _legal_bs(v: float, hi: float) -> float:
        v = min(hi, max(0.0, float(v)))
        return float(round(v, 1))

    def _formula_S(self, spec, R, active, expedite, z_leaf, z_up):
        n = int(spec["n_stages"])
        nl = int(spec["n_customer"])
        _, lead, _ = self._routes(spec, active, expedite)
        out = []
        for j in range(n):
            Pj = self._protection(spec, j, lead[j], R[j])
            z = z_leaf if j < nl else z_up
            v = spec["flow_mean"][j] * Pj + z * spec["flow_sd"][j] * math.sqrt(Pj)
            out.append(self._legal_bs(max(v, 0.0), float(spec["base_stock_max"])))
        return out

    def _cal_eval(self, spec, S, R, active, expedite, svc_floor=None):
        """Calibration metrics over _CAL_SEEDS fixed seeds, judged at least as
        strictly as the evaluator's chance constraint (which takes the 10% /
        90% quantile over its replicates): the LOWEST service and the HIGHEST
        holding seen, and the mean total cost. With svc_floor the scan stops
        at the first seed whose service is below it (the design cannot pass);
        'complete' says whether every seed ran."""
        svc, hold, cost = [], [], []
        for k in range(_CAL_SEEDS):
            rng = np.random.default_rng([int(spec["demand_seed"]), 7919 + 104729 * k])
            m, _ = self._simulate(spec, S, R, active, expedite, rng)
            svc.append(m["service_level"])
            hold.append(m["holding_cost"])
            cost.append(m["total_cost"])
            if svc_floor is not None and m["service_level"] < svc_floor:
                break
        return {"service_level": float(min(svc)), "holding_cost": float(max(hold)),
                "total_cost": float(sum(cost) / len(cost)),
                "complete": len(svc) == _CAL_SEEDS}

    def _calibrate(self, spec, target, r_star):
        n = int(spec["n_stages"])
        nl = int(spec["n_customer"])
        init = spec["initial"]
        a_init = [bool(x) for x in init["active"]]
        opt = set(spec["optional"])
        a_on = [True] * n
        a_off = [(j not in opt) for j in range(n)]
        actives = []
        for name, a in (("given", a_init), ("all_on", a_on), ("all_off", a_off)):
            if a not in [x for _, x in actives]:
                actives.append((name, a))
        rev_all1 = [1] * n
        rev_eoq = []
        for j in range(n):
            best = min(_REVIEW_CODES, key=lambda c: abs(math.log(c) - math.log(r_star[j])))
            rev_eoq.append(int(best))
        reviews = [("all1", rev_all1)]
        if rev_eoq != rev_all1:
            reviews.append(("eoq", rev_eoq))
        exps = [("none", [False] * n),
                ("customer", [j < nl for j in range(n)]),
                ("all", [True] * n)]
        rows = []
        seen = []          # every scanned grid point (fallback only)

        floor = target + _CAL_MARGIN

        def scan(R, active, expedite, tag, up):
            # z: safety factor of the customer-facing stages; upstream stages
            # carry up * z (the upstream safety rule)
            found = 0
            best_hold = None
            tag = tag + ",up=%g" % up
            for z in _Z_GRID:
                S = self._formula_S(spec, R, active, expedite, z, up * z)
                m = self._cal_eval(spec, S, R, active, expedite, svc_floor=floor)
                if not tag.startswith("knob_only"):
                    seen.append({"config": tag + ",z=%g" % z,
                                 "service": m["service_level"],
                                 "holding": m["holding_cost"],
                                 "cost": m["total_cost"]})
                ok = m["complete"] and m["service_level"] >= floor
                if ok:
                    rows.append({"config": tag + ",z=%g" % z,
                                 "service": m["service_level"],
                                 "holding": m["holding_cost"],
                                 "cost": m["total_cost"]})
                    if best_hold is None or m["holding_cost"] < best_hold:
                        best_hold = m["holding_cost"]
                    found += 1
                    if found >= 2:
                        break
            return best_hold

        for aname, a in actives:
            for rname, R in reviews:
                for ename, ex in exps:
                    for up in _UP_RULES_GRID:
                        scan(R, a, ex, "active=%s,review=%s,expedite=%s"
                             % (aname, rname, ename), up)
        # knob-only: the GIVEN structure, base stock only, best of several
        # upstream safety rules (one rule alone overstates what a knob-only
        # repair needs on long chains, where little upstream safety is best)
        knob = None
        for up in _UP_RULES_KNOB:
            kh = scan([int(x) for x in init["review"]], a_init,
                      [bool(x) for x in init["expedite"]], "knob_only", up)
            if kh is not None and (knob is None or kh < knob):
                knob = kh
        grid = [r for r in rows if not r["config"].startswith("knob_only")]
        if not grid:
            # no grid design met the target even at the largest z: anchor
            # on the best-serving design scanned (never observed in practice)
            best = max(seen, key=lambda r: (r["service"], -r["holding"]))
            grid = [dict(best, config="fallback:" + best["config"])]
        return {"rows": grid,
                "holding_min": float(min(r["holding"] for r in grid)),
                "knob_only_holding": knob}

    # ── state ────────────────────────────────────────────────────────────────
    def load(self, spec: dict):
        init = spec["initial"]
        return {
            "spec": spec,
            "S": np.asarray(init["base_stock"], dtype=np.float64).copy(),
            "R": [int(x) for x in init["review"]],
            "active": [bool(x) for x in init["active"]],
            "expedite": [bool(x) for x in init["expedite"]],
            "_diag": None,
        }

    def clone(self, st):
        return {"spec": st["spec"], "S": st["S"].copy(), "R": list(st["R"]),
                "active": list(st["active"]), "expedite": list(st["expedite"]),
                "_diag": None}

    # ── simulation ───────────────────────────────────────────────────────────
    def evaluate(self, st, seed: int = 0) -> dict:
        self.calls += 1
        spec = st["spec"]
        try:
            metrics, _ = self._run(spec, st["S"], st["R"], st["active"],
                                   st["expedite"], seed)
        except Exception:
            return dict(_FAIL_METRICS)
        for v in metrics.values():
            if not math.isfinite(v):
                return dict(_FAIL_METRICS)
        return metrics

    # ── design variables ─────────────────────────────────────────────────────
    def n_elements(self, st) -> int:
        return int(st["spec"]["n_stages"])

    def _is_optional(self, st, i: int) -> bool:
        return int(i) in st["spec"]["optional"]

    def get(self, st, i: int, p: str) -> Optional[float]:
        n = int(st["spec"]["n_stages"])
        if not (0 <= int(i) < n):
            return None
        i = int(i)
        if p == "base_stock":
            return float(st["S"][i])
        if p == "review":
            return float(st["R"][i])
        if p == "expedite":
            return 1.0 if st["expedite"][i] else 0.0
        if p == "active":
            if not self._is_optional(st, i):
                return None
            return 1.0 if st["active"][i] else 0.0
        return None

    def _legal(self, st, i: int, p: str, v: float) -> Optional[float]:
        """The value set() would store for (i, p, v), or None if (i, p) is
        not a tunable of that stage."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        if p == "base_stock":
            return self._legal_bs(v, float(st["spec"]["base_stock_max"]))
        if p == "review":
            return float(_round_review(v))
        if p == "expedite":
            return 1.0 if v >= 0.5 else 0.0
        if p == "active":
            if not self._is_optional(st, i):
                return None
            return 1.0 if v >= 0.5 else 0.0
        return None

    def _cover_periods(self, st) -> List[int]:
        """Protection interval of every stage (as if active) under st."""
        spec = st["spec"]
        _, lead, _ = self._routes(spec, st["active"], st["expedite"])
        return [self._protection(spec, j, lead[j], st["R"][j])
                for j in range(int(spec["n_stages"]))]

    def set(self, st, i: int, p: str, v: float) -> None:
        n = int(st["spec"]["n_stages"])
        if not (0 <= int(i) < n):
            return
        i = int(i)
        lv = self._legal(st, i, p, v)
        if lv is None:
            return
        if p == "base_stock":
            st["S"][i] = lv
            st["_diag"] = None
            return
        cur = self.get(st, i, p)
        if cur is not None and abs(cur - lv) < 1e-12:
            return
        old_P = self._cover_periods(st) if self.rebase else None
        if p == "review":
            st["R"][i] = int(lv)
        elif p == "expedite":
            st["expedite"][i] = lv >= 0.5
        elif p == "active":
            st["active"][i] = lv >= 0.5
        if self.rebase:
            # planner convention: a structural change keeps every stage's
            # safety stock (S - mean demand over its protection interval) and
            # re-derives the order-up-to level of each stage whose interval moved
            new_P = self._cover_periods(st)
            mu = st["spec"]["flow_mean"]
            hi = float(st["spec"]["base_stock_max"])
            for j in range(n):
                if new_P[j] != old_P[j]:
                    st["S"][j] = self._legal_bs(
                        float(st["S"][j]) + float(mu[j]) * (new_P[j] - old_P[j]), hi)
        st["_diag"] = None

    def bounds(self, st, p: str):
        if p == "base_stock":
            return (0.0, float(st["spec"]["base_stock_max"]))
        if p == "review":
            return (1.0, 4.0)
        return (0.0, 1.0)

    # ── typed move set ───────────────────────────────────────────────────────
    def candidate_moves(self, st, rng: np.random.Generator) -> List[Candidate]:
        """Every stage, every applicable move type, every turn:
        structural moves (other review codes, expedite flip, active flip on
        the optional pool) are always all offered; base_stock gets a scale
        ladder sized so the turn has ~40-160 moves (trimmed evenly per stage,
        never to fewer than two values per active stage)."""
        spec = st["spec"]
        n = int(spec["n_stages"])
        struct: List[Candidate] = []
        bs_lists = {}
        _, lead, _ = self._routes(spec, st["active"], st["expedite"])
        for i in range(n):
            if st["active"][i]:
                cur_r = int(st["R"][i])
                for c in _REVIEW_CODES:
                    if c != cur_r:
                        struct.append(Candidate(i=i, p="review", v=float(c),
                                                v_old=float(cur_r)))
            cur_e = 1.0 if st["expedite"][i] else 0.0
            struct.append(Candidate(i=i, p="expedite", v=1.0 - cur_e, v_old=cur_e))
            if self._is_optional(st, i):
                cur_a = 1.0 if st["active"][i] else 0.0
                struct.append(Candidate(i=i, p="active", v=1.0 - cur_a, v_old=cur_a))
            if st["active"][i]:
                cur = float(st["S"][i])
                Pj = self._protection(spec, i, lead[i], st["R"][i])
                mean_cov = float(spec["flow_mean"][i]) * Pj
                sd_cov = float(spec["flow_sd"][i]) * math.sqrt(Pj)
                vals = []
                for kind, amt in _BS_LADDER:
                    raw = cur * amt if kind == "x" else cur + amt * sd_cov
                    v = self._legal_bs(raw, float(spec["base_stock_max"]))
                    if abs(v - cur) > 1e-9 and v not in vals:
                        vals.append(v)
                if cur < 0.5 * mean_cov:
                    for f in (1.0, 0.5):
                        v = self._legal_bs(mean_cov * f, float(spec["base_stock_max"]))
                        if abs(v - cur) > 1e-9 and v not in vals:
                            vals.insert(0, v)
                bs_lists[i] = vals
        n_act = max(len(bs_lists), 1)
        per = int(round((_CAND_TARGET - len(struct)) / n_act))
        per = max(4, min(len(_BS_LADDER), per))
        while len(struct) + per * n_act > _CAND_MAX and per > 2:
            per -= 1
        out: List[Candidate] = []
        for i in range(n):
            for v in bs_lists.get(i, [])[:per]:
                out.append(Candidate(i=i, p="base_stock", v=float(v),
                                     v_old=float(st["S"][i])))
            out.extend(c for c in struct if c.i == i)
        if len(out) > _CAND_MAX:
            # last resort (only for networks far beyond the ladder): drop
            # base_stock values at random, never a structural move
            bs_idx = [k for k, c in enumerate(out) if c.p == "base_stock"]
            drop = set(rng.choice(len(bs_idx), size=len(out) - _CAND_MAX,
                                  replace=False).tolist())
            out = [c for k, c in enumerate(out)
                   if not (c.p == "base_stock" and bs_idx.index(k) in drop)]
        return out

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
        return replace(prog, validity_bounds=(
            ("service_level", 1.0 + 1e-9),
            ("holding_cost", 1e9),
            ("order_cost", 1e9),
            ("expedite_cost", 1e9),
            ("backorder_cost", 1e9),
            ("total_cost", 1e9),
        ))

    def reference(self, spec: dict):
        cached = spec.get("reference_cost")
        try:
            if cached is not None and math.isfinite(float(cached)) and float(cached) > 0:
                return float(cached)
        except (TypeError, ValueError):
            pass
        return None

    # ── diagnostics / greedy ─────────────────────────────────────────────────
    def _diag(self, st):
        d = st.get("_diag")
        if d is not None:
            return d
        spec = st["spec"]
        try:
            m, det = self._run(spec, st["S"], st["R"], st["active"], st["expedite"],
                               stable_seed(spec.get("problem_id", ""), "hint"),
                               diag=True)
        except Exception:
            return None
        d = {"metrics": m, "det": det}
        st["_diag"] = d
        return d

    def hint(self, st) -> Optional[Tuple[int, str, str]]:
        """One-knob diagnostic: service short -> raise base stock at the
        customer stage with the worst fill, or at its supplier when that
        stage mostly waits on supply; holding over budget (or feasible) ->
        cut base stock where the most holding money is spent."""
        d = self._diag(st)
        if d is None:
            return None
        spec = st["spec"]
        nl = int(spec["n_customer"])
        goals = spec.get("goals", {})
        target = float(goals.get("minimum_service_level", 0.0))
        budget = float(goals.get("maximum_holding_cost", float("inf")))
        m, det = d["metrics"], d["det"]
        act = det["act"]
        pos = {j: k for k, j in enumerate(act)}
        if m["service_level"] < target:
            worst = int(np.argmin(det["leaf_fill"]))
            k = pos[worst]
            wait = det["pend"][k] / det["q"][k] if det["q"][k] > 1e-12 else 0.0
            sup = det["sup"][worst]
            if sup != -1 and wait > 0.5:
                return (int(sup), "base_stock", "up")
            return (worst, "base_stock", "up")
        k = int(np.argmax(det["holding"]))
        return (int(act[k]), "base_stock", "down")

    def hint_move(self, st) -> Optional[Candidate]:
        h = self.hint(st)
        if h is None:
            return None
        i, p, direction = h
        spec = st["spec"]
        cur = float(st["S"][i])
        if direction == "up":
            v = cur * 1.25
            if cur < 1.0:
                _, lead, _ = self._routes(spec, st["active"], st["expedite"])
                v = spec["flow_mean"][i] * self._protection(spec, i, lead[i], st["R"][i])
        else:
            v = cur * 0.8
        v = self._legal_bs(v, float(spec["base_stock_max"]))
        if abs(v - cur) <= 1e-9:
            return None
        return Candidate(i=int(i), p="base_stock", v=float(v), v_old=cur)

    # ── observation ──────────────────────────────────────────────────────────
    # describe() is a pure function of the state: it re-simulates the design on
    # fixed demand seeds (never the evaluator's), so the same state always
    # prints the same text. It never changes the state.
    _OBS_SEEDS = 8        # results: means over demand seeds 0..7
    _SENS_SEEDS = 4       # base_stock sensitivities: common seeds 0..3

    @staticmethod
    def _fmt(x, sig: int = 4) -> str:
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
        if 99.95 <= x < 100.0:
            return "%.2f%%" % min(x, 99.99)
        return "%.1f%%" % x

    def _sg(self, x, sig: int = 3) -> str:
        """Signed short number (+1.2, -0.034, 0)."""
        s = self._fmt(x, sig)
        return s if s.startswith("-") or s == "0" else "+" + s

    @staticmethod
    def _children(spec, j):
        return [k for k, p in enumerate(spec["parent"]) if int(p) == int(j)]

    @staticmethod
    def _span(idx: List[int]) -> str:
        if not idx:
            return "-"
        idx = sorted(idx)
        if len(idx) > 2 and idx[-1] - idx[0] == len(idx) - 1:
            return "%d-%d" % (idx[0], idx[-1])
        return ",".join(str(x) for x in idx)

    def _burst_var(self, spec) -> List[float]:
        """Per stage: summed variance of the burst-modulated demand RATE of the
        customer-facing stages below it (0 unless demand is bursty). Regimes
        persist (lag-k autocorrelation rho^k), so it adds covariance between
        periods that the per-period sd leaves out."""
        n = int(spec["n_stages"])
        out = [0.0] * n
        dem = spec["demand"]
        if dem["kind"] != "bursty":
            return out
        pe, px, m = float(dem["p_enter"]), float(dem["p_exit"]), float(dem["burst_mult"])
        pi_b = pe / (pe + px)
        par = spec["parent"]
        for leaf in range(int(spec["n_customer"])):
            mu = float(dem["mu"][leaf])
            v = pi_b * (1.0 - pi_b) * (mu * (m - 1.0)) ** 2
            k = leaf
            while k != -1:
                out[k] += v
                k = int(par[k])
        return out

    def _sd_over(self, spec, j: int, X: float, bvar=None) -> float:
        """sd of the demand through stage j over X periods; for bursty demand
        Var = X sd^2 + 2 Var(rate) sum_{k=1}^{X-1} (X - k) rho^k,
        rho = 1 - p_enter - p_exit (closed form, smooth in X)."""
        X = float(X)
        if X <= 1e-9:
            return 0.0
        sd1 = float(spec["flow_sd"][j])
        var = X * sd1 * sd1
        dem = spec["demand"]
        if dem["kind"] == "bursty":
            if bvar is None:
                bvar = self._burst_var(spec)
            rho = 1.0 - float(dem["p_enter"]) - float(dem["p_exit"])
            if bvar[j] > 0.0 and 0.0 < rho < 1.0:
                acc = rho * (X * (1.0 - rho) - 1.0 + rho ** X) / (1.0 - rho) ** 2
                var += 2.0 * bvar[j] * max(acc, 0.0)
        return math.sqrt(max(var, 0.0))

    @staticmethod
    def _norm_loss(k: float) -> float:
        phi = math.exp(-0.5 * k * k) / math.sqrt(2.0 * math.pi)
        tail = 0.5 * math.erfc(k / math.sqrt(2.0))
        return phi - k * tail

    def _fill_est(self, spec, j, S, C, R, bvar) -> float:
        """Two-term normal-loss fill of a customer-facing stage at effective
        cover C (periods) and review R: 1 - [E(D_C - S)+ - E(D_{C-R} - S)+]
        / (mean x R)."""
        mu = float(spec["flow_mean"][j])
        if mu <= 0.0:
            return 1.0
        S = float(S)

        def short(X):
            if X <= 1e-9:
                return max(0.0, -S)
            s = self._sd_over(spec, j, X, bvar)
            if s <= 1e-12:
                return max(0.0, mu * X - S)
            return s * self._norm_loss((S - mu * X) / s)

        sh = short(C) - short(C - R)
        return min(1.0, max(0.0, 1.0 - sh / (mu * float(R))))

    def _S_for_fill(self, spec, j, target, C, R, bvar) -> Optional[float]:
        lo, hi = 0.0, float(spec["base_stock_max"])
        if self._fill_est(spec, j, hi, C, R, bvar) < target:
            return None
        if self._fill_est(spec, j, lo, C, R, bvar) >= target:
            return 0.0
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if self._fill_est(spec, j, mid, C, R, bvar) >= target:
                hi = mid
            else:
                lo = mid
        return hi

    def _preview(self, st, j: int, p: str, v: float):
        """Deterministic consequences of one structural move (applied to a
        copy; st is untouched): [(stage, transit0, transit1, P0, P1, S0, S1)]
        for every stage active after the move whose route, P or S changes
        (the toggled stage itself always, when switched on)."""
        spec = st["spec"]
        n = int(spec["n_stages"])
        child = self.clone(st)
        self.set(child, j, p, v)
        P0 = self._cover_periods(st)
        P1 = self._cover_periods(child)
        _, l0, _ = self._routes(spec, st["active"], st["expedite"])
        sup1, l1, _ = self._routes(spec, child["active"], child["expedite"])
        rows = []
        for k in range(n):
            if not child["active"][k]:
                continue
            s0, s1 = float(st["S"][k]), float(child["S"][k])
            if (P0[k] != P1[k] or l0[k] != l1[k] or abs(s0 - s1) > 1e-9
                    or (k == j and p == "active")):
                rows.append((k, l0[k], l1[k], P0[k], P1[k], s0, s1))
        return rows, sup1

    def _safety_sd(self, spec, k, S, P, bvar) -> float:
        sdv = self._sd_over(spec, k, P, bvar)
        return (float(S) - float(spec["flow_mean"][k]) * P) / sdv if sdv > 0 else 0.0

    def _fmt_shift(self, spec, j, rows, bvar) -> str:
        """Post-move values of every affected stage: 'S 46.6 (+1.2 sd)', the
        sd figure being its safety in sd (stage j itself unnamed; old values
        are on each stage's own DESIGN line; P moves with the transit)."""
        f, g = self._fmt, self._sg
        parts = []
        for (k, t0, t1, p0, p1, s0, s1) in rows:
            sd_txt = "%s sd" % g(self._safety_sd(spec, k, s1, p1, bvar), 2)
            txt = ("S %s (%s)" % (f(s1), sd_txt) if abs(s0 - s1) > 1e-9
                   else "safety %s" % sd_txt)
            if t0 != t1 and k != j:
                txt = "transit %d, %s" % (t1, txt)
            parts.append(txt if k == j else "stage %d %s" % (k, txt))
        return "; ".join(parts) if parts else "no stage re-planned"

    def _role(self, spec, j: int, active) -> str:
        nl = int(spec["n_customer"])
        par = int(spec["parent"][j])
        tree = spec.get("topology", "serial") != "serial"
        parts = []
        if j < nl:
            parts.append("store" if tree else "customer-facing")
        elif par == -1:
            parts.append("plant" if tree else "most upstream")
        elif tree:
            parts.append("distribution center")
        kids = self._children(spec, j)
        if tree and kids:
            parts.append("supplies %s" % self._span(kids))
        if tree and j >= nl:
            parts.append("flow %s" % self._fmt(spec["flow_mean"][j], 3))
        if j in spec["optional"]:
            parts.append("optional, %s" % ("on" if active[j] else "PASS-THROUGH"))
        elif j >= nl:
            parts.append("always active")
        return "; ".join(parts)

    @staticmethod
    def _src_txt(spec, j: int, sup, active) -> str:
        """'14 via 11': the stage j orders from, and the pass-throughs on the
        way (physical route)."""
        via = []
        k = int(spec["parent"][j])
        while k != -1 and k != sup[j]:
            via.append(k)
            k = int(spec["parent"][k])
        src = "source" if sup[j] == -1 else "%d" % sup[j]
        return src + (" via %s" % ",".join(str(x) for x in via) if via else "")

    def _demand_text(self, spec) -> str:
        dem = spec["demand"]
        f = self._fmt
        mu, sd = self._leaf_stats(dem)
        kind = dem["kind"]
        nl = int(spec["n_customer"])
        if kind == "poisson":
            core = "Poisson, i.i.d. per period"
        elif kind == "negbin":
            core = ("negative binomial (over-dispersed), i.i.d. per period, "
                    "dispersion r=%s (smaller = more variable)" % f(dem["r"], 3))
        else:
            pe, px = float(dem["p_enter"]), float(dem["p_exit"])
            core = ("regime-switching Poisson, independent per stage: rate x%s in a "
                    "burst, entered w.p. %s, left w.p. %s per period (bursts last "
                    "~%s periods, ~%s of the time); bursts drain stock several "
                    "periods in a row, and every sd over P below includes this "
                    "persistence (1.3-1.7x the independent-period value)"
                    % (f(dem["burst_mult"], 3), f(pe, 3), f(px, 3), f(1.0 / px, 3),
                       self._pct(pe / (pe + px))))
        if nl == 1:
            per = "mean %s, sd %s per period" % (f(mu[0]), f(sd[0]))
        else:
            per = "mean, sd per period by store: " + "; ".join(
                "%d: %s, %s" % (j, f(mu[j], 3), f(sd[j], 3)) for j in range(nl))
        return "%s; %s." % (core, per)

    def _obs_runs(self, spec, S, R, active, expd, seeds, diag):
        runs = []
        for s in seeds:
            m, det = self._run(spec, S, R, active, expd, s, diag=diag)
            if not all(math.isfinite(v) for v in m.values()):
                raise ValueError("non-finite metrics")
            runs.append((m, det))
        return runs

    def describe(self, st, spec=None) -> str:
        if spec is None:
            spec = st["spec"]
        f, g, pct = self._fmt, self._sg, self._pct
        n = int(spec["n_stages"])
        nl = int(spec["n_customer"])
        L = [int(x) for x in spec["lead_times"]]
        h = [float(x) for x in spec["holding_costs"]]
        K = [float(x) for x in spec["order_costs"]]
        e = [float(x) for x in spec["expedite_premiums"]]
        mu = [float(x) for x in spec["flow_mean"]]
        b = float(spec["backorder_cost"])
        T = int(spec["horizon"])
        warm = int(spec["warmup"])
        n_paths = int(spec["paths"])
        goals = spec.get("goals", {})
        target = float(goals.get("minimum_service_level", float("nan")))
        budget = float(goals.get("maximum_holding_cost", float("nan")))
        fam = str(spec.get("family", "?")).replace("_", " ")
        tree = spec.get("topology", "serial") != "serial"
        S = st["S"]
        R = [int(x) for x in st["R"]]
        active = [bool(x) for x in st["active"]]
        expd = [bool(x) for x in st["expedite"]]
        bvar = self._burst_var(spec)
        st = dict(st, spec=spec)

        out = ["PROBLEM"]
        if tree:
            out.append(
                "Tree-shaped distribution network (%s family), %d stages: stores 0-%d "
                "sell to customers, every other stage supplies the stages listed on "
                "its line, plant %d orders from an unlimited external source that "
                "ships in full. Each active stage runs a periodic-review base-stock "
                "(order-up-to) policy." % (fam, n, nl - 1, n - 1))
        else:
            out.append(
                "Serial inventory network (%s family), %d stages: stage 0 sells to "
                "the customer, stage j supplies stage j-1, stage %d orders from an "
                "unlimited external source that ships in full. Each active stage "
                "runs a periodic-review base-stock (order-up-to) policy."
                % (fam, n, n - 1))
        out.append(
            "Each period: (1) due shipments arrive; (2) customer-facing stages serve "
            "backlog, then today's demand; unmet demand is backordered; (3) each "
            "active stage reviewing now ((t + j) mod R = 0) orders S - IP, IP = "
            "on-hand + in transit + owed to it - owed by it; (4) each supplier ships "
            "min(on-hand, owed), pro rata when short, this period's orders included, "
            "but counts an order in its own IP (and reorders for it) from the next "
            "period; shipments arrive after the route transit time; (5) holding is "
            "charged on end-of-period on-hand. Stock in transit is free: a shorter "
            "route saves holding only via the safety stock it lets you remove.")
        out.append("Customer demand: %s" % self._demand_text(spec))
        if spelled_in_verbs(self):
            # the same facts as the SET_PARAM paragraph below, in the verbs the
            # policy is served (docstring "Verb surface"); the SET_PARAM
            # paragraph again under the set_parameter spelling
            out.append(
                "Operations (one change per turn; stage = the bare stage number):\n"
                "- SET_BASE_STOCK(stage, level) (u, [0, %s], step 0.1): the order-up-to "
                "level S. It covers demand over P = route transit + R - 1 (+1 at a supplying "
                "stage) plus any supply wait; safety = S - mean demand over P. A unit more at "
                "a customer-facing stage adds ~1 u on-hand there; at a supplying stage it "
                "mostly cuts its downstream stages' supply wait. Fill rule for a "
                "customer-facing stage (normal approximation; error ~0.03, more for bursty "
                "demand at low fill): C = effective cover, sd_X = sd over X periods, k_X = "
                "(S - mean x X)/sd_X; shortfall per review ~ sd_C G(k_C) - sd_(C-R) "
                "G(k_(C-R)); fill ~ 1 - shortfall/(mean x R); normal loss G(-1)=1.08, "
                "G(-0.5)=0.70, G(0)=0.40, G(0.5)=0.20, G(1)=0.083, G(1.5)=0.029, "
                "G(2)=0.0085, G(2.5)=0.002. Dormant on a pass-through.\n"
                "- SET_REVIEW_PERIOD(stage, period) (1, 2 or 4 periods; other numbers round "
                "to the nearest): K per review, so K/R $/per ordering. A longer R lengthens "
                "P; on-hand rises ~mean flow x (R_new - R_old)/2 (cycle stock, partly at "
                "neighbouring stages). R* = sqrt(2K / (h x mean flow)) balances ordering "
                "against cycle stock: the period nearest R* is cheapest if the holding "
                "budget has room, a shorter one if not.\n"
                "- OPEN_STOCKING_POINT(stage) / CLOSE_STOCKING_POINT(stage) (optional "
                "stocking points only; closed = cross-dock pass-through: no stock, orders, "
                "holding or K; the stages below order from the nearest open stage above, its "
                "leg added to their route). Opening stocks it to its stored base stock "
                "(safety shown, usually ~0) for K/R; the stages below get a shorter route, "
                "base stock cut to match (same safety u). With ~0 safety the new point "
                "stocks out often and can LOWER service until its base stock is raised. "
                "Closing saves its holding and K/R; the stages below get a longer route, "
                "base stock raised to match (same u, fewer sd), and stop waiting on its "
                "stockouts.\n"
                "- ENABLE_EXPEDITING(stage) / DISABLE_EXPEDITING(stage) (the stage's inbound "
                "leg): L -> ceil(L/2) for e $ per unit of mean flow over the leg "
                "(pass-through legs too).\n"
                "Re-planning rule (automatic): a review-period, stocking-point or expediting "
                "change shifts the base stock of every stage whose P it alters by mean flow "
                "x (P_new - P_old), keeping its safety in units; DESIGN previews the results."
                % f(spec["base_stock_max"]))
        else:
            out.append(
            "Parameters (one change per turn, SET_PARAM(stage, param, value)):\n"
            "- base_stock (u, [0, %s], step 0.1): order-up-to level S. It covers "
            "demand over P = route transit + R - 1 (+1 at a supplying stage) plus any "
            "supply wait; safety = S - mean demand over P. A unit more at a "
            "customer-facing stage adds ~1 u on-hand there; at a supplying stage it "
            "mostly cuts its downstream stages' supply wait. Fill rule for a "
            "customer-facing stage (normal approximation; error ~0.03, more for "
            "bursty demand at low fill): C = effective cover, sd_X = sd over X "
            "periods, k_X = (S - mean x X)/sd_X; shortfall per review ~ sd_C G(k_C) "
            "- sd_(C-R) G(k_(C-R)); fill ~ 1 - shortfall/(mean x R); normal loss "
            "G(-1)=1.08, G(-0.5)=0.70, G(0)=0.40, G(0.5)=0.20, G(1)=0.083, "
            "G(1.5)=0.029, G(2)=0.0085, G(2.5)=0.002. Dormant on a pass-through.\n"
            "- review (codes 1, 2, 4; others round to the nearest): K per review, so "
            "K/R $/per ordering. A longer R lengthens P; on-hand rises ~mean flow x (R_new - "
            "R_old)/2 (cycle stock, partly at neighbouring stages). R* = sqrt(2K / "
            "(h x mean flow)) balances ordering against cycle stock: the code nearest "
            "R* is cheapest if the holding budget has room, a shorter one if not.\n"
            "- active (optional stocking points; 0 = cross-dock pass-through: no "
            "stock, orders, holding or K; the stages below order from the nearest "
            "active stage above, its leg added to their route). Switching on stocks "
            "it to its stored base_stock (safety shown, usually ~0) for K/R; the "
            "stages below get a shorter route, base_stock cut to match (same safety "
            "u). With ~0 safety the new point stocks out often and can LOWER service "
            "until its base_stock is raised. Switching off saves its holding and K/R; "
            "the stages below get a longer route, base_stock raised to match (same "
            "u, fewer sd), and stop waiting on its stockouts.\n"
            "- expedite (inbound leg): L -> ceil(L/2) for e $ per unit of mean flow "
            "over the leg (pass-through legs too).\n"
            "Re-planning rule (automatic): a review, active or expedite change shifts "
            "the base_stock of every stage whose P it alters by mean flow x (P_new - "
            "P_old), keeping its safety in units; DESIGN previews the results."
            % f(spec["base_stock_max"]))
        out.append(
            "Costs, $/per: holding = sum of h x mean end-of-period on-hand and order "
            "= sum of K/R over active stages; expedite = sum of e x mean flow over "
            "expedited legs; backorder = %s $ per unit of demand not served on "
            "arrival; total = all four. Goals: service_level (demand served on "
            "arrival, pooled over customer-facing stages) >= %s, holding_cost <= %s; "
            "minimise total_cost." % (f(b), f(target), f(budget)))
        if self.stochastic:
            out.append(
                "An evaluation = %d demand paths x %d periods from filled pipelines, "
                "last %d scored. Goals are judged on 4-24 independent evaluations: "
                "service_level at their 10th percentile (with 4: 0.7 x lowest + 0.3 x "
                "2nd lowest), ~0.8-1.3 sd below its mean, holding_cost at the 90th; "
                "keep ~1.3 sd of margin (sd in RESULTS). Constraint-status values are "
                "such judged values, taken on other seeds when the design was picked "
                "(rows mix quantiles and means); a before -> after pair of them "
                "carries ~1 sd of noise." % (n_paths, T, max(T - warm, 1)))
        else:
            out.append("An evaluation = %d fixed demand paths x %d periods, last %d "
                       "scored." % (n_paths, T, max(T - warm, 1)))

        sup, lead, leg = self._routes(spec, active, expd)
        cover = [self._protection(spec, j, lead[j], R[j]) for j in range(n)]
        try:
            seeds = list(range(self._OBS_SEEDS if self.stochastic else 1))
            runs = self._obs_runs(spec, S, R, active, expd, seeds, True)
        except Exception:
            runs = None
        res = None
        if runs:
            act = runs[0][1]["act"]

            def mean_det(key):
                return np.mean([r[1][key] for r in runs], axis=0)

            pend, qq = mean_det("pend"), mean_det("q")
            res = {"pos": {jj: k for k, jj in enumerate(act)},
                   "on_hand": mean_det("on_hand"), "holding": mean_det("holding"),
                   "owed": mean_det("owed"), "stockout": mean_det("stockout"),
                   "fill": mean_det("leaf_fill"),
                   "wait": [float(pend[k]) / float(qq[k]) if qq[k] > 1e-12 else 0.0
                            for k in range(len(act))]}

        # ── DESIGN ──
        out.append("")
        out.append(
            "DESIGN (u = units, per = periods, $ = $/per; P = protection interval, "
            "sd = sd of demand over P, safety = S - mean demand over P; a supplying "
            "stage sees downstream ORDERS, each ~R_d periods of demand, so its demand "
            "is lumpier than this sd. Move previews show values after the move.)")
        if not tree:
            out.append("Flow through every stage: mean %s, sd %s u/per."
                       % (f(mu[0]), f(spec["flow_sd"][0])))
        for j in range(n):
            head = "stage %d [%s]" % (j, self._role(spec, j, active))
            Rs = math.sqrt(2.0 * K[j] / (h[j] * mu[j])) if h[j] * mu[j] > 0 else float("nan")
            erows, _ = self._preview(st, j, "expedite", 0.0 if expd[j] else 1.0)
            ecost = e[j] * mu[j]
            if expd[j]:
                exp_txt = ("leg %d expedited to %d (%s); un-expedite saves %s: %s"
                           % (L[j], leg[j], f(ecost, 3), f(ecost, 3),
                              self._fmt_shift(spec, j, erows, bvar)))
            else:
                exp_txt = ("leg %d, expedite -> %d for +%s: %s"
                           % (L[j], self._leg_lead(L[j], True), f(ecost, 3),
                              self._fmt_shift(spec, j, erows, bvar)))
            if not active[j]:
                below = self._routed_through(spec, j, active, sup)
                on_rows, sup_on = self._preview(st, j, "active", 1.0)
                own = [r for r in on_rows if r[0] == j]
                Pj = own[0][4] if own else cover[j]
                sdj = self._sd_over(spec, j, Pj, bvar)
                ss = float(S[j]) - mu[j] * Pj
                src = self._src_txt(spec, j, sup_on, active)
                out.append(
                    "%s: on the route of %s; stored S %s, R %d (K %s, R* %s); h %s; "
                    "%s; switch on (+%s ordering): from %s, transit %d, P %d, sd %s, "
                    "safety %s u (%s sd); %s"
                    % (head, self._span(below), f(S[j]), R[j], f(K[j], 3), f(Rs, 2),
                       f(h[j], 3), exp_txt, f(K[j] / R[j], 3), src, lead[j], Pj,
                       f(sdj, 3), g(ss), g(ss / sdj if sdj > 0 else 0.0, 2),
                       self._fmt_shift(spec, j, [r for r in on_rows if r[0] != j], bvar)))
                continue
            Pj = cover[j]
            sdj = self._sd_over(spec, j, Pj, bvar)
            ss = float(S[j]) - mu[j] * Pj
            src = self._src_txt(spec, j, sup, active)
            line = ("%s: from %s, transit %d; S %s, P %d, sd %s, safety %s u (%s sd); "
                    "h %s; R %d (K %s, R* %s); %s"
                    % (head, src, lead[j], f(S[j]), Pj, f(sdj, 3), g(ss),
                       g(ss / sdj if sdj > 0 else 0.0, 2), f(h[j], 3), R[j],
                       f(K[j], 3), f(Rs, 2), exp_txt))
            if j in spec["optional"]:
                off_rows, _ = self._preview(st, j, "active", 0.0)
                hj = (float(res["holding"][res["pos"][j]]) if res and j in res["pos"]
                      else float("nan"))
                line += ("; switch off (saves %s holding + %s ordering): %s"
                         % (f(hj, 3), f(K[j] / R[j], 3),
                            self._fmt_shift(spec, j, [r for r in off_rows if r[0] != j],
                                            bvar)))
            out.append(line)

        # ── RESULTS ──
        out.append("")
        if res is None:
            out.append("RESULTS: simulation failed for this design.")
            return "\n".join(out)
        ns = len(runs)
        ms = [r[0] for r in runs]
        sv = np.array([m["service_level"] for m in ms])
        hv = np.array([m["holding_cost"] for m in ms])

        def q(vals, qq):
            s = sorted(float(x) for x in vals)
            if len(s) == 1:
                return s[0]
            p = qq * (len(s) - 1)
            lo = int(math.floor(p))
            hi = min(lo + 1, len(s) - 1)
            return s[lo] * (1.0 - (p - lo)) + s[hi] * (p - lo)

        def mean(key):
            return float(np.mean([m[key] for m in ms]))

        sens_seeds = list(range(min(self._SENS_SEEDS, ns)))
        if self.stochastic and ns > 1:
            sv_sd, hv_sd = float(np.std(sv, ddof=1)), float(np.std(hv, ddof=1))
            out.append(
                "RESULTS - means of %d evaluations on fixed demand seeds 0-%d; one "
                "evaluation scatters by the sd shown, so differences under ~2 sd are "
                "noise. service_level %.3f (sd %.3f, 10th percentile %.3f); "
                "holding_cost %s (sd %s = %s, 90th percentile %s); order_cost %s, "
                "expedite_cost %s, backorder_cost %s, total_cost %s."
                % (ns, ns - 1, float(sv.mean()), sv_sd, q(sv, 0.1), f(hv.mean()),
                   f(hv_sd, 3), pct(hv_sd / hv.mean() if hv.mean() > 0 else 0.0),
                   f(q(hv, 0.9)), f(mean("order_cost")), f(mean("expedite_cost")),
                   f(mean("backorder_cost")), f(mean("total_cost"))))
        else:
            out.append(
                "RESULTS - one evaluation. service_level %.3f; holding_cost %s, "
                "order_cost %s, expedite_cost %s, backorder_cost %s, total_cost %s."
                % (float(sv.mean()), f(hv.mean()), f(mean("order_cost")),
                   f(mean("expedite_cost")), f(mean("backorder_cost")),
                   f(mean("total_cost"))))
        out.append(
            "Per stage: on-hand u; holding $ (share); stockout = share of periods "
            "ending owing units; supply wait = mean periods its orders wait at its "
            "supplier, cover = P + it; fill = demand served on arrival (share of "
            "network unserved); rule = fill rule at this cover, now and the S for "
            "%s; S +/-1 sd = change in service_level / holding_cost if only this S "
            "moves one sd over P (%s, not noise)."
            % (pct(target), "same seeds 0-%d" % (len(sens_seeds) - 1)
               if len(sens_seeds) > 1 else "same seed"))
        base_sv = np.mean([runs[s][0]["service_level"] for s in sens_seeds])
        base_hv = np.mean([runs[s][0]["holding_cost"] for s in sens_seeds])
        hold_tot = float(np.sum(res["holding"]))
        unserved = {j: max(0.0, 1.0 - float(res["fill"][j])) * mu[j] for j in range(nl)}
        un_tot = sum(unserved.values())
        hi_bs = float(spec["base_stock_max"])
        for j in range(n):
            if j not in res["pos"]:
                out.append("stage %d: pass-through, holds nothing (its leg adds %d per "
                           "to the route of %s)"
                           % (j, leg[j], self._span(self._routed_through(spec, j, active, sup))))
                continue
            k = res["pos"][j]
            share = float(res["holding"][k]) / hold_tot if hold_tot > 0 else 0.0
            w = float(res["wait"][k])
            Pj = cover[j]
            if sup[j] == -1:
                sw = "no supply wait, cover %d" % Pj
            else:
                sw = ("supply wait %s, cover %s" % (f(w, 2), f(Pj + w, 3)) if w >= 0.005
                      else "supply wait ~0, cover %d" % Pj)
            txt = ("stage %d: on-hand %s, holding %s (%s), stockout %s, %s"
                   % (j, f(res["on_hand"][k], 3), f(res["holding"][k], 3), pct(share),
                      pct(res["stockout"][k]), sw))
            if j < nl:
                C = Pj + (w if sup[j] != -1 else 0.0)
                now = self._fill_est(spec, j, S[j], C, R[j], bvar)
                need = self._S_for_fill(spec, j, target, C, R[j], bvar)
                txt += ("; fill %s (%s); rule %s, S %s"
                        % (pct(res["fill"][j]),
                           pct(unserved[j] / un_tot if un_tot > 0 else 0.0), pct(now),
                           f(need, 3) if need is not None else "> bound"))
            sdj = self._sd_over(spec, j, Pj, bvar)
            sens = []
            for sgn in (1.0, -1.0):
                S2 = np.array(S, dtype=np.float64).copy()
                S2[j] = self._legal_bs(float(S[j]) + sgn * sdj, hi_bs)
                if abs(S2[j] - float(S[j])) < 1e-9:
                    sens.append(None)
                    continue
                try:
                    rr = self._obs_runs(spec, S2, R, active, expd, sens_seeds, False)
                except Exception:
                    sens.append(None)
                    continue
                sens.append((float(np.mean([m["service_level"] for m, _ in rr]) - base_sv),
                             float(np.mean([m["holding_cost"] for m, _ in rr]) - base_hv)))
            sv_t = "/".join(("%+.3f" % d[0] if abs(d[0]) >= 5e-4 else "0") if d
                            else "bound" for d in sens)
            hv_t = "/".join(g(d[1]) if d else "bound" for d in sens)
            txt += "; S +/-1 sd: service %s, holding %s" % (sv_t, hv_t)
            out.append(txt)
        return "\n".join(out)

    @staticmethod
    def _routed_through(spec, j: int, active, sup) -> List[int]:
        """Active stages whose supply route includes pass-through j's leg:
        below j, ordering from the same supplier j would order from."""
        par = spec["parent"]
        out = []
        for k in range(int(spec["n_stages"])):
            if not active[k] or sup[k] != sup[j]:
                continue
            a = int(par[k])
            while a != -1 and a != j:
                a = int(par[a])
            if a == j:
                out.append(k)
        return out

    def element_label(self, st, i) -> str:
        spec = st["spec"]
        n = int(spec["n_stages"])
        i = int(i)
        if not (0 <= i < n):
            return "stage %d (out of range)" % i
        nl = int(spec["n_customer"])
        par = int(spec["parent"][i])
        if i < nl:
            role = "customer-facing"
        elif par == -1:
            role = "most upstream"
        else:
            role = ("intermediate" if spec.get("topology") == "serial"
                    else "distribution center")
        kids = self._children(spec, i)
        bits = [role]
        if kids:
            bits.append("feeds %s" % self._span(kids))
        if i in spec["optional"]:
            bits.append("optional, %s" % ("on" if st["active"][i] else "off"))
        bits.append("leg %d per" % int(spec["lead_times"][i]))
        return "stage %d (%s)" % (i, ", ".join(bits))
