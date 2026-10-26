"""Age-structured SEIR epidemic control: a social decision problem domain.

MODEL. K population groups (sizes N_g), a contact matrix C[g,h] (daily
contacts a member of g has with members of h; assortative + random mixing),
and S/E/I/R compartments per group plus a cumulative-death ledger D_g.
Fixed-step discrete-time chain over t_days days at steps_per_day steps/day
using exact exponential transition probabilities, so flows can never exceed
their source compartment and the integrator cannot go negative or blow up:

    lambda_g = beta * sum_h Ceff[g,h] * I_h / N_h
    Ceff[g,h] = C[g,h] * (1 - cr_g) * (1 - cr_h)
    p(rate)   = 1 - exp(-rate * dt)   for infection (lambda_g), E->I (sigma),
                I->out (gamma), R->S waning (omega), S->R vaccination (v_g,
                capped by a per-group dose supply)
    deaths    = IFR_g fraction of the I->out flow (elderly >> young)

beta is calibrated per problem from a family-specific R0 via the
next-generation matrix NGM[g,h] = (beta/gamma) * C[g,h] * N_g/N_h:
beta = R0 * gamma / spectral_radius(C * N_g/N_h).

DESIGN VARIABLES (elements = the K groups): contact_reduction in [0, 0.9]
scales group g's row AND column of C; vacc_rate in [0, 0.02]/day moves S->R
until the group's dose supply is spent. The initial design is near-zero
intervention, so the epidemic runs (almost) free and most problems start
infeasible.

METRICS: peak_infected_frac (max_t sum_g I_g / N), total_deaths_per_1k,
attack_rate, cost (objective: superlinear closure cost
w1 * sum_g cr_g^2 * (N_g/N) * t_days plus dose cost w2 * doses_per_capita,
plus a small fixed base so cost stays positive), doses_per_capita,
pop_conservation_err (envelope invariant), peak_day, peak_group,
final_infected_frac, mean_contact_reduction.

STOCHASTIC MODE: the same chain with integer compartments and per-step
binomial draws from np.random.default_rng(seed) only; group populations are
small (1.5k-8k per group) so demographic noise is real.

REFERENCE (heuristic): bisect a uniform contact-reduction level (no
vaccination) to the minimal level meeting both goals; reference = its cost.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from ..domain import Domain
from ..potential import DesignProgram, program_from_goals

_FAMILIES = ("flu_like", "covid_like", "measles_like", "covid_waning")
_K_LADDER = (2, 3, 4, 6, 8, 10, 12, 16)
_CR_MAX = 0.9
_VACC_MAX = 0.02
_BASE_COST = 0.01


class SEIRDomain(Domain):
    """Age-structured SEIR epidemic control; see module docstring."""

    name = "seir"
    params = ("contact_reduction", "vacc_rate")
    # Interventions span orders of magnitude from the near-zero start, so the
    # move ladder is coarser than the framework default.
    factors = (0.5, 0.7, 1.5, 2.2, 3.0)
    reference_class = "heuristic"

    def __init__(self, stochastic: bool = False):
        super().__init__(stochastic)

    # -- problems --------------------------------------------------------------
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[int(rng.integers(len(_FAMILIES)))]
        k = int(_K_LADDER[index % len(_K_LADDER)])
        n_pop = rng.integers(1500, 8001, size=k)

        if family == "flu_like":
            r0 = float(rng.uniform(1.4, 1.9))
            latent = float(rng.uniform(1.5, 2.5))
            infectious = float(rng.uniform(3.0, 5.0))
            waning = 0.0
        elif family in ("covid_like", "covid_waning"):
            r0 = float(rng.uniform(2.5, 3.8))
            latent = float(rng.uniform(3.0, 5.0))
            infectious = float(rng.uniform(5.0, 8.0))
            waning = (1.0 / float(rng.uniform(60.0, 120.0))
                      if family == "covid_waning" else 0.0)
        else:  # measles_like
            r0 = float(rng.uniform(6.0, 10.0))
            latent = float(rng.uniform(8.0, 12.0))
            infectious = float(rng.uniform(5.0, 8.0))
            waning = 0.0

        # Assortative + random mixing; younger groups are more active.
        age = np.arange(k, dtype=float) / max(k - 1, 1)
        activity = rng.uniform(10.0, 18.0, size=k) * (1.0 - 0.45 * age)
        assort = float(rng.uniform(0.2, 0.7))
        weight = activity * n_pop
        weight = weight / weight.sum()
        contact = activity[:, None] * (assort * np.eye(k)
                                       + (1.0 - assort) * weight[None, :])

        # IFR ladder: elderly orders of magnitude above the young.
        ifr0 = math.exp(rng.uniform(math.log(2e-5), math.log(2e-4)))
        ifr_ratio = math.exp(rng.uniform(math.log(50.0), math.log(400.0)))
        ifr = np.clip(ifr0 * np.power(ifr_ratio, age), 0.0, 0.15)

        # Seed >= 8 infected per seeded group: stochastic extinction of the
        # whole outbreak stays rare (<~7% even at R0=1.4) while the initial
        # prevalence stays well under the smallest peak cap (0.004).
        n_seed_groups = 1 if k < 4 else int(rng.integers(1, 3))
        seed_groups = rng.choice(k, size=n_seed_groups, replace=False)
        seed_frac = float(rng.uniform(5e-4, 1.5e-3))
        i0 = np.zeros(k, dtype=np.int64)
        for g in seed_groups:
            i0[int(g)] = max(8, int(round(seed_frac * float(n_pop[int(g)]))))

        spec = {
            "problem_id": "seir_%04d" % index,
            "family": family,
            "K": k,
            "N": [int(x) for x in n_pop],
            "contact_matrix": [[float(c) for c in row] for row in contact],
            "r0": r0,
            "latent_days": latent,
            "infectious_days": infectious,
            "waning_per_day": float(waning),
            "ifr": [float(x) for x in ifr],
            "I0": [int(x) for x in i0],
            "supply_frac": float(rng.uniform(0.2, 0.8)),
            "w_closure": float(rng.uniform(0.7, 1.5)),
            "w_dose": float(rng.uniform(4.0, 12.0)),
            "cr0": float(rng.uniform(0.005, 0.04)),
            "v0": float(rng.uniform(1e-4, 1e-3)),
            "t_days": 180.0,
            "steps_per_day": 2,
        }

        # Calibrate the goals against the (near-free) initial design: caps are
        # u (resp. v) times the initial peak (resp. deaths), so u, v < 1 means
        # an infeasible start. Difficulty is drawn from a three-tier mixture:
        # the evaluator normalises the initial violation by its own scale, so
        # only a graded tail of near-feasible and feasible starts gives the
        # problem SET a real spread of the (normalised) difficulty proxy.
        core = self._core(spec)
        metrics, _ = self._simulate(core, np.full(k, spec["cr0"]),
                                    np.full(k, spec["v0"]),
                                    stochastic=False, seed=0)
        tier = float(rng.uniform())
        if tier < 0.20:      # easy: feasible or barely-infeasible start
            u = float(rng.uniform(0.95, 1.50))
            v = float(rng.uniform(0.90, 1.60))
        elif tier < 0.42:    # medium: clearly infeasible, near the tau scale
            u = math.exp(rng.uniform(math.log(0.35), math.log(0.95)))
            v = math.exp(rng.uniform(math.log(0.40), math.log(1.10)))
        else:                # hard: the epidemic must be beaten down a lot
            u = math.exp(rng.uniform(math.log(0.08), math.log(0.45)))
            v = math.exp(rng.uniform(math.log(0.15), math.log(0.80)))
        peak_cap = min(max(metrics["peak_infected_frac"] * u, 0.004), 0.5)
        death_cap = min(max(metrics["total_deaths_per_1k"] * v, 0.02), 500.0)
        spec["goals"] = {
            "maximum_peak_infected_frac": float(peak_cap),
            "maximum_total_deaths_per_1k": float(death_cap),
        }
        return spec

    def load(self, spec: dict):
        core = self._core(spec)
        return {
            "core": core,  # read-only, shared between clones
            "cr": np.full(core["k"], float(spec.get("cr0", 0.02))),
            "vacc": np.full(core["k"], float(spec.get("v0", 5e-4))),
            "_cache": {},
            "_hint": None,
        }

    def clone(self, st):
        return {
            "core": st["core"],
            "cr": np.array(st["cr"], dtype=float, copy=True),
            "vacc": np.array(st["vacc"], dtype=float, copy=True),
            "_cache": {},
            "_hint": None,
        }

    # -- simulation ------------------------------------------------------------
    def evaluate(self, st, seed: int = 0) -> dict:
        self.calls += 1
        key = int(seed) if self.stochastic else 0
        cache = st.get("_cache")
        if cache is not None and key in cache:
            return dict(cache[key])
        try:
            metrics, _ = self._simulate(st["core"], st["cr"], st["vacc"],
                                        stochastic=self.stochastic,
                                        seed=int(seed))
            for value in metrics.values():
                if not math.isfinite(value):
                    metrics = self._failure_metrics(st)
                    break
        except Exception:
            metrics = self._failure_metrics(st)
        if cache is not None:
            cache[key] = dict(metrics)
        return dict(metrics)

    def _failure_metrics(self, st) -> dict:
        """Read as maximally infeasible: inf objective (=> invalid state) and
        every constrained/envelope metric at its worst."""
        try:
            mean_cr = float(np.mean(np.clip(np.asarray(st["cr"], dtype=float),
                                            0.0, _CR_MAX)))
        except Exception:
            mean_cr = 0.0
        return {
            "cost": float("inf"),
            "peak_infected_frac": 1.0,
            "total_deaths_per_1k": 1000.0,
            "attack_rate": 1.0,
            "doses_per_capita": 0.0,
            "mean_contact_reduction": mean_cr,
            "pop_conservation_err": 1.0,
            "peak_day": 0.0,
            "peak_group": 0.0,
            "final_infected_frac": 1.0,
        }

    def _core(self, spec: dict) -> dict:
        """Parse the JSON spec into numpy arrays and calibrate beta from R0."""
        n_pop = np.asarray(spec["N"], dtype=float)
        contact = np.asarray(spec["contact_matrix"], dtype=float)
        gamma = 1.0 / float(spec["infectious_days"])
        ngm_shape = contact * (n_pop[:, None] / n_pop[None, :])
        try:
            rho = float(np.max(np.abs(np.linalg.eigvals(ngm_shape))))
        except Exception:
            rho = float(np.max(ngm_shape.sum(axis=1)))
        return {
            "k": int(spec["K"]),
            "n_pop": n_pop,
            "n_int": np.asarray(spec["N"], dtype=np.int64),
            "n_total": float(n_pop.sum()),
            "contact": contact,
            "beta": float(spec["r0"]) * gamma / max(rho, 1e-9),
            "sigma": 1.0 / float(spec["latent_days"]),
            "gamma": gamma,
            "waning": float(spec.get("waning_per_day", 0.0)),
            "ifr": np.asarray(spec["ifr"], dtype=float),
            "i0": np.asarray(spec["I0"], dtype=np.int64),
            "doses_cap": np.maximum(
                np.round(float(spec["supply_frac"]) * n_pop), 0.0),
            "w_closure": float(spec["w_closure"]),
            "w_dose": float(spec["w_dose"]),
            "t_days": float(spec.get("t_days", 180.0)),
            "steps_per_day": int(spec.get("steps_per_day", 2)),
        }

    def _simulate(self, core: dict, cr, vacc, *, stochastic: bool, seed: int):
        k = core["k"]
        n_pop = core["n_pop"]
        n_total = core["n_total"]
        dt = 1.0 / core["steps_per_day"]
        n_steps = int(round(core["t_days"] * core["steps_per_day"]))

        cr = np.clip(np.asarray(cr, dtype=float), 0.0, _CR_MAX)
        vacc = np.clip(np.asarray(vacc, dtype=float), 0.0, _VACC_MAX)

        keep = 1.0 - cr
        foi_matrix = (core["beta"] * core["contact"]
                      * keep[:, None] * keep[None, :]) / n_pop[None, :]
        p_ei = -math.expm1(-core["sigma"] * dt)
        p_ir = -math.expm1(-core["gamma"] * dt)
        p_wane = -math.expm1(-core["waning"] * dt)
        p_vacc = -np.expm1(-vacc * dt)
        ifr = core["ifr"]
        waning_on = core["waning"] > 0.0
        vacc_on = bool(np.any(p_vacc > 0.0))

        if stochastic:
            rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
            i_c = core["i0"].copy()
            s_c = core["n_int"] - i_c
            e_c = np.zeros(k, dtype=np.int64)
            r_c = np.zeros(k, dtype=np.int64)
            d_c = np.zeros(k, dtype=np.int64)
            doses_cap = core["doses_cap"].astype(np.int64)
            doses = np.zeros(k, dtype=np.int64)
        else:
            rng = None
            i_c = core["i0"].astype(float)
            s_c = n_pop - i_c
            e_c = np.zeros(k)
            r_c = np.zeros(k)
            d_c = np.zeros(k)
            doses_cap = core["doses_cap"].astype(float)
            doses = np.zeros(k)

        cum_inf = float(i_c.sum())
        peak_tot = float(i_c.sum())
        peak_i = i_c.astype(float).copy()
        peak_step = 0

        for t in range(n_steps):
            p_inf = -np.expm1(-(foi_matrix @ i_c) * dt)
            if stochastic:
                new_e = rng.binomial(s_c, p_inf)
                new_v = (np.minimum(rng.binomial(s_c - new_e, p_vacc),
                                    doses_cap - doses) if vacc_on else 0)
                out_e = rng.binomial(e_c, p_ei)
                out_i = rng.binomial(i_c, p_ir)
                new_d = rng.binomial(out_i, ifr)
                wane = rng.binomial(r_c, p_wane) if waning_on else 0
            else:
                new_e = s_c * p_inf
                new_v = (np.minimum((s_c - new_e) * p_vacc,
                                    doses_cap - doses) if vacc_on else 0.0)
                out_e = e_c * p_ei
                out_i = i_c * p_ir
                new_d = out_i * ifr
                wane = r_c * p_wane if waning_on else 0.0
            s_c = s_c - new_e - new_v + wane
            e_c = e_c + new_e - out_e
            i_c = i_c + out_e - out_i
            r_c = r_c + out_i - new_d + new_v - wane
            d_c = d_c + new_d
            doses = doses + new_v
            cum_inf += float(np.sum(new_e))
            tot_i = float(i_c.sum())
            if tot_i > peak_tot:
                peak_tot = tot_i
                peak_i = i_c.astype(float).copy()
                peak_step = t + 1

        frac = n_pop / n_total
        closure_cost = (core["w_closure"] * float(np.sum(cr * cr * frac))
                        * core["t_days"])
        doses_pc = float(np.sum(doses)) / n_total
        cost = _BASE_COST + closure_cost + core["w_dose"] * doses_pc
        conserv = (float(np.sum(s_c) + np.sum(e_c) + np.sum(i_c)
                         + np.sum(r_c) + np.sum(d_c)) - n_total) / n_total
        metrics = {
            "cost": float(cost),
            "peak_infected_frac": float(peak_tot / n_total),
            "total_deaths_per_1k": float(1000.0 * float(np.sum(d_c)) / n_total),
            "attack_rate": float(cum_inf / n_total),
            "doses_per_capita": float(doses_pc),
            "mean_contact_reduction": float(np.mean(cr)),
            "pop_conservation_err": float(conserv),
            "peak_day": float(peak_step * dt),
            "peak_group": float(int(np.argmax(peak_i))),
            "final_infected_frac": float((float(np.sum(e_c))
                                          + float(np.sum(i_c))) / n_total),
        }
        aux = {"peak_share": peak_i / max(peak_tot, 1e-12)}
        return metrics, aux

    # -- design variables --------------------------------------------------------
    def n_elements(self, st) -> int:
        return int(st["core"]["k"])

    @staticmethod
    def _array(st, p: str):
        if p == "contact_reduction":
            return st["cr"]
        if p == "vacc_rate":
            return st["vacc"]
        return None

    def get(self, st, i: int, p: str) -> Optional[float]:
        arr = self._array(st, p)
        if arr is None or not (0 <= int(i) < arr.shape[0]):
            return None
        return float(arr[int(i)])

    def set(self, st, i: int, p: str, v: float) -> None:
        arr = self._array(st, p)
        if arr is None or not (0 <= int(i) < arr.shape[0]):
            return
        lo, hi = self.bounds(st, p)
        arr[int(i)] = min(hi, max(lo, float(v)))
        st["_cache"] = {}
        st["_hint"] = None

    def bounds(self, st, p: str) -> Tuple[float, float]:
        if p == "contact_reduction":
            return (0.0, _CR_MAX)
        if p == "vacc_rate":
            return (0.0, _VACC_MAX)
        raise ValueError("unknown param %r" % p)

    # -- program / reference -------------------------------------------------------
    def program(self, spec: dict) -> DesignProgram:
        prog = program_from_goals(
            spec["goals"],
            objective_key="cost",
            objective_ref=self._lockdown_cost(spec),
            objective_sense="min",
            domain=self.name,
        )
        # Envelope (red-team): no physically meaningful trajectory can show
        # prevalence or attack above 1, and the chain conserves people exactly;
        # a state outside these is integrator/exploit territory whatever the
        # goals say. cost > 0 is enforced by objective_must_be_positive.
        return replace(prog, validity_bounds=(
            ("peak_infected_frac", 1.0),
            ("attack_rate", 1.0),
            ("pop_conservation_err", 1e-6),
        ))

    @staticmethod
    def _lockdown_cost(spec: dict) -> float:
        """Cost of the full lockdown (every group at max reduction): the
        objective_ref, so obj/ref lands in (0, 1] for sane designs."""
        return (_BASE_COST + float(spec["w_closure"]) * _CR_MAX * _CR_MAX
                * float(spec.get("t_days", 180.0)))

    def reference(self, spec: dict) -> Optional[float]:
        """Heuristic: bisect a UNIFORM contact-reduction level (no vaccination)
        to the minimal level meeting both goals under the deterministic chain;
        reference = its cost. Falls back to the full-lockdown cost when even
        max uniform reduction does not meet the goals."""
        cached = spec.get("_reference_cost")
        if cached is not None:
            return float(cached)
        core = self._core(spec)
        k = core["k"]
        peak_cap = float(spec["goals"]["maximum_peak_infected_frac"])
        death_cap = float(spec["goals"]["maximum_total_deaths_per_1k"])
        no_vacc = np.zeros(k)

        def ok(level: float) -> bool:
            m, _ = self._simulate(core, np.full(k, level), no_vacc,
                                  stochastic=False, seed=0)
            return (m["peak_infected_frac"] <= peak_cap
                    and m["total_deaths_per_1k"] <= death_cap)

        if ok(0.0):
            level = 0.0
        elif not ok(_CR_MAX):
            level = _CR_MAX
        else:
            lo, hi = 0.0, _CR_MAX
            for _ in range(22):
                mid = 0.5 * (lo + hi)
                if ok(mid):
                    hi = mid
                else:
                    lo = mid
            level = hi
        cost = (_BASE_COST + float(spec["w_closure"]) * level * level
                * float(spec.get("t_days", 180.0)))
        spec["_reference_cost"] = float(cost)
        return float(cost)

    # -- depth-0 feedback hint -------------------------------------------------------
    def hint(self, st) -> Optional[Tuple[int, str, str]]:
        """Blame the group carrying the largest share of infections at the
        epidemic peak (cheap deterministic diagnostic): reduce its contacts;
        if its reduction is already maxed, vaccinate it."""
        cached = st.get("_hint")
        if cached is not None:
            return cached if cached != () else None
        try:
            _, aux = self._simulate(st["core"], st["cr"], st["vacc"],
                                    stochastic=False, seed=0)
        except Exception:
            st["_hint"] = ()
            return None
        order = np.argsort(-aux["peak_share"])
        for g in order:
            if float(st["cr"][int(g)]) < _CR_MAX * (1.0 - 1e-9):
                st["_hint"] = (int(g), "contact_reduction", "up")
                return st["_hint"]
        for g in order:
            if float(st["vacc"][int(g)]) < _VACC_MAX * (1.0 - 1e-9):
                st["_hint"] = (int(g), "vacc_rate", "up")
                return st["_hint"]
        st["_hint"] = ()
        return None

    # -- observation for the policy (additive; read-only on the state) -----------
    def describe(self, st, spec) -> str:
        """Self-contained plain-text observation of the CURRENT design:
        PROBLEM (setup, groups, contacts, cost law, parameter meaning/bounds),
        DESIGN (per-group interventions and their current transmission
        effect) and RESULTS (per-group outcomes of a deterministic nominal
        evaluation). Never mutates ``st``."""
        core = st["core"]
        k = int(core["k"])
        n_pop = np.asarray(core["n_pop"], dtype=float)
        n_total = float(core["n_total"])
        frac = n_pop / n_total
        contact = np.asarray(core["contact"], dtype=float)
        ifr = np.asarray(core["ifr"], dtype=float)
        i0 = np.asarray(core["i0"], dtype=float)
        doses_cap = np.asarray(core["doses_cap"], dtype=float)
        t_days = float(core["t_days"])
        spd = int(core["steps_per_day"])
        cr = np.clip(np.array(st["cr"], dtype=float, copy=True), 0.0, _CR_MAX)
        vacc = np.clip(np.array(st["vacc"], dtype=float, copy=True), 0.0,
                       _VACC_MAX)
        num = self._obs_num
        pct = self._obs_pct
        family = str(spec.get("family", "unknown"))
        fam_r0 = {"flu_like": "1.4-1.9", "covid_like": "2.5-3.8",
                  "covid_waning": "2.5-3.8", "measles_like": "6-10"}
        waning = float(core["waning"])
        w_c = float(core["w_closure"])
        w_d = float(core["w_dose"])
        supply_frac = float(spec.get("supply_frac",
                                     float(np.sum(doses_cap)) / n_total))

        L = []
        # ---------------------------------------------------------------- PROBLEM
        L.append("PROBLEM")
        L.append("Age-structured SEIR epidemic control, family %s (R0 range %s), "
                 "K=%d population groups ordered by age (g0 youngest ... g%d "
                 "oldest), total population %d, horizon %s days (%d steps/day)."
                 % (family, fam_r0.get(family, "?"), k, k - 1,
                    int(round(n_total)), num(t_days), spd))
        imm = ("permanent" if waning <= 0.0 else
               "wanes at %s/day (mean %s days) for recovered AND vaccinated, "
               "who return to susceptible" % (num(waning), num(1.0 / waning)))
        L.append("Disease: R0 %s (with no intervention, fully susceptible), "
                 "latent period %s d, infectious period %s d, immunity %s."
                 % (num(float(spec.get("r0", float("nan")))),
                    num(1.0 / float(core["sigma"])),
                    num(1.0 / float(core["gamma"])), imm))
        L.append("Transmission: new infections/day in g = beta*S_g*sum_h "
                 "C[g,h]*(1-cr_g)*(1-cr_h)*I_h/N_h, beta %s per contact; "
                 "deaths = IFR_g x people leaving I in g."
                 % num(float(core["beta"])))
        seeds = ", ".join("%d in g%d" % (int(round(i0[g])), g)
                          for g in range(k) if i0[g] > 0)
        L.append("Day 0: infectious %s; everyone else susceptible."
                 % (seeds if seeds else "none"))
        L.append("Tunable per group g:")
        L.append("  contact_reduction cr_g in [0, %s]: fraction of g's contacts "
                 "removed; scales g's row AND column of C, so a g-h link keeps "
                 "(1-cr_g)(1-cr_h)." % num(_CR_MAX))
        L.append("  vacc_rate v_g in [0, %s] per day: fraction of g's remaining "
                 "susceptibles vaccinated per day (S->R, immune); stops when g's "
                 "dose supply (%s of g's population) is used up."
                 % (num(_VACC_MAX), pct(supply_frac)))
        L.append("Cost (objective, minimise) = %s + %s*%s*sum_g (N_g/N)*cr_g^2 "
                 "+ %s*(doses used)/N; i.e. closure cost of g = coef_g*cr_g^2, "
                 "each dose costs %s."
                 % (num(_BASE_COST), num(w_c), num(t_days), num(w_d),
                    num(w_d / n_total)))
        rowsum = contact.sum(axis=1)
        L.append("GROUPS (g role: size, IFR, dose supply, contacts/day "
                 "[own-group share], top cross-group contacts/day, closure coef):")
        for g in range(k):
            others = [h for h in range(k) if h != g]
            others.sort(key=lambda h: (-contact[g, h], h))
            cross = ", ".join("g%d %s" % (h, num(contact[g, h]))
                              for h in others[:2])
            own = contact[g, g] / rowsum[g] if rowsum[g] > 0 else 0.0
            L.append("  g%d %s: N %d (%s), IFR %s, supply %d, %s c/d [%s own]%s, "
                     "coef %s"
                     % (g, self._obs_role(g, k), int(round(n_pop[g])),
                        pct(frac[g]), num(ifr[g]), int(round(doses_cap[g])),
                        num(rowsum[g]), pct(own),
                        ("; cross " + cross) if cross else "",
                        num(w_c * t_days * frac[g])))

        # ----------------------------------------------------------------- DESIGN
        keep = 1.0 - cr
        foi = (float(core["beta"]) * contact * keep[:, None] * keep[None, :]
               / n_pop[None, :])
        ngm = n_pop[:, None] * foi / float(core["gamma"])
        try:
            r_eff = float(np.max(np.abs(np.linalg.eigvals(ngm))))
        except Exception:
            r_eff = float("nan")
        onward = ngm.sum(axis=0)
        L.append("DESIGN")
        L.append("Reproduction number at day 0 under current contact reductions "
                 "(before any vaccination or immunity): %s. Per group: cr, v, "
                 "closure cost now, and how many people ONE infectious member of "
                 "g infects over its illness in a fully susceptible population "
                 "(top recipient groups):" % num(r_eff))
        for g in range(k):
            order = sorted(range(k), key=lambda h: (-ngm[h, g], h))[:3]
            tgt = ", ".join("g%d %s" % (h, num(ngm[h, g])) for h in order)
            L.append("  g%d %s: contact_reduction %s, vacc_rate %s/d | closure "
                     "cost %s | infects %s (%s)"
                     % (g, self._obs_role(g, k), num(cr[g]), num(vacc[g]),
                        num(w_c * t_days * frac[g] * cr[g] * cr[g]),
                        num(onward[g]), tgt))

        # ---------------------------------------------------------------- RESULTS
        det = self._obs_detail(core, cr, vacc)
        L.append("RESULTS")
        if self.stochastic:
            L.append("Per-group results are a NOMINAL deterministic (mean-field) "
                     "evaluation of the current design; each stochastic replicate "
                     "uses integer people and binomial draws, so realised values "
                     "scatter around these.")
        else:
            L.append("Per-group results of the deterministic simulation of the "
                     "current design.")
        deaths = det["deaths"]
        d_tot = float(np.sum(deaths))
        src = det["src"]
        caused = src.sum(axis=0)
        caused_tot = float(np.sum(caused))
        # Infections g's cases caused, weighted by the recipients' IFR: the
        # share of the death burden that traces back to g as a source.
        caused_d = (src * ifr[:, None]).sum(axis=0)
        caused_d_tot = float(np.sum(caused_d))
        attack_def = ("attack (cumulative infections incl. day-0 seeds / group "
                      "size%s)"
                      % ("; reinfections after waning count, so it can exceed "
                         "100%" if waning > 0.0 else ""))
        L.append("Total infected peaks on day %s. Per group: own peak infected "
                 "(share of group, day), share of all infected at that overall "
                 "peak, %s, deaths (per 1k of TOTAL population, share of "
                 "deaths), doses used/supply, who infected g (share of its "
                 "infections by source group), share of all infections caused "
                 "by g's cases, plain / weighted by the IFR of the people they "
                 "infected (= share of the death burden traced to g as source):"
                 % (num(det["peak_day"]), attack_def))
        for g in range(k):
            got = src[g]
            got_tot = float(np.sum(got))
            if got_tot > 0:
                order = sorted(range(k), key=lambda h: (-got[h], h))[:3]
                by = ", ".join("g%d %s" % (h, pct(got[h] / got_tot))
                               for h in order if got[h] > 0)
            else:
                by = "none"
            if det["vacc_on"]:
                dose_txt = "%d/%d" % (int(round(det["doses"][g])),
                                      int(round(doses_cap[g])))
                if math.isfinite(det["dose_out_day"][g]):
                    dose_txt += " (used up day %s)" % num(det["dose_out_day"][g])
            else:
                dose_txt = "0/%d" % int(round(doses_cap[g]))
            L.append("  g%d: peak %s d%s | %s of peak | attack %s | deaths %s/1k "
                     "(%s) | doses %s | infected by %s | causes %s / %s"
                     % (g, pct(det["own_peak"][g]), num(det["own_peak_day"][g]),
                        pct(det["peak_share"][g]), pct(det["attack"][g]),
                        num(1000.0 * deaths[g] / n_total),
                        pct(deaths[g] / d_tot) if d_tot > 0 else "0%",
                        dose_txt, by,
                        pct(caused[g] / caused_tot) if caused_tot > 0 else "0%",
                        (pct(caused_d[g] / caused_d_tot) if caused_d_tot > 0
                         else "0%")))
        return "\n".join(L)

    def element_label(self, st, i) -> str:
        """Short role label for group i, e.g. 'group 3 (elderly, 14% of pop,
        IFR 0.012)'."""
        core = st["core"]
        k = int(core["k"])
        g = int(i)
        if not (0 <= g < k):
            return "group %d" % g
        frac = float(core["n_pop"][g]) / float(core["n_total"])
        return "group %d (%s, %s of pop, IFR %s)" % (
            g, self._obs_role(g, k), self._obs_pct(frac),
            self._obs_num(float(core["ifr"][g])))

    # -- observation helpers (new, private; no effect on simulation) -------------
    @staticmethod
    def _obs_role(g: int, k: int) -> str:
        a = float(g) / max(k - 1, 1)
        if a < 0.15:
            return "young"
        if a < 0.4:
            return "young adult"
        if a < 0.6:
            return "middle-aged"
        if a < 0.85:
            return "older adult"
        return "elderly"

    @staticmethod
    def _obs_num(x, sig: int = 3) -> str:
        """Significant-figure formatting that never prints a nonzero as 0."""
        x = float(x)
        if x != x:
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if x == 0.0:
            return "0"
        if abs(x) >= 10.0 ** sig:
            return "%.0f" % x
        s = "%.*g" % (sig, x)
        if "e+" in s:
            return "%.0f" % x
        return s.replace("e-0", "e-")

    @classmethod
    def _obs_pct(cls, f) -> str:
        return cls._obs_num(100.0 * float(f)) + "%"

    def _obs_detail(self, core: dict, cr, vacc) -> dict:
        """Deterministic chain identical to _simulate(stochastic=False), with
        per-group bookkeeping added (own peaks, attack, deaths, doses, and who
        infected whom). Pure: touches no state."""
        k = core["k"]
        n_pop = core["n_pop"]
        dt = 1.0 / core["steps_per_day"]
        n_steps = int(round(core["t_days"] * core["steps_per_day"]))
        cr = np.clip(np.asarray(cr, dtype=float), 0.0, _CR_MAX)
        vacc = np.clip(np.asarray(vacc, dtype=float), 0.0, _VACC_MAX)
        keep = 1.0 - cr
        foi_matrix = (core["beta"] * core["contact"]
                      * keep[:, None] * keep[None, :]) / n_pop[None, :]
        p_ei = -math.expm1(-core["sigma"] * dt)
        p_ir = -math.expm1(-core["gamma"] * dt)
        p_wane = -math.expm1(-core["waning"] * dt)
        p_vacc = -np.expm1(-vacc * dt)
        ifr = core["ifr"]
        waning_on = core["waning"] > 0.0
        vacc_on = bool(np.any(p_vacc > 0.0))

        i_c = core["i0"].astype(float)
        s_c = n_pop - i_c
        e_c = np.zeros(k)
        r_c = np.zeros(k)
        d_c = np.zeros(k)
        doses_cap = core["doses_cap"].astype(float)
        doses = np.zeros(k)

        cum = i_c.copy()
        src = np.zeros((k, k))
        peak_tot = float(i_c.sum())
        peak_i = i_c.copy()
        peak_step = 0
        own_peak = i_c / n_pop
        own_step = np.zeros(k, dtype=np.int64)
        out_step = np.full(k, -1, dtype=np.int64)

        for t in range(n_steps):
            lam_parts = foi_matrix * i_c[None, :]
            p_inf = -np.expm1(-(foi_matrix @ i_c) * dt)
            new_e = s_c * p_inf
            new_v = (np.minimum((s_c - new_e) * p_vacc, doses_cap - doses)
                     if vacc_on else 0.0)
            out_e = e_c * p_ei
            out_i = i_c * p_ir
            new_d = out_i * ifr
            wane = r_c * p_wane if waning_on else 0.0
            lam_tot = lam_parts.sum(axis=1)
            safe = np.where(lam_tot > 0.0, lam_tot, 1.0)
            src += (new_e / safe)[:, None] * lam_parts
            s_c = s_c - new_e - new_v + wane
            e_c = e_c + new_e - out_e
            i_c = i_c + out_e - out_i
            r_c = r_c + out_i - new_d + new_v - wane
            d_c = d_c + new_d
            doses = doses + new_v
            cum = cum + new_e
            tot_i = float(i_c.sum())
            if tot_i > peak_tot:
                peak_tot = tot_i
                peak_i = i_c.copy()
                peak_step = t + 1
            f_i = i_c / n_pop
            up = f_i > own_peak
            own_peak = np.where(up, f_i, own_peak)
            own_step = np.where(up, t + 1, own_step)
            if vacc_on:
                done = ((out_step < 0) & (doses_cap > 0)
                        & (doses >= doses_cap * (1.0 - 1e-9)))
                out_step = np.where(done, t + 1, out_step)

        out_day = np.where(out_step >= 0, out_step * dt, np.nan)
        return {
            "own_peak": own_peak,
            "own_peak_day": own_step * dt,
            "peak_share": peak_i / max(peak_tot, 1e-12),
            "peak_day": peak_step * dt,
            "attack": cum / n_pop,
            "deaths": d_c,
            "doses": doses,
            "dose_out_day": out_day,
            "src": src,
            "vacc_on": vacc_on,
        }
