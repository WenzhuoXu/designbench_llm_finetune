"""Age-structured SEIR epidemic control with TIME-VARYING interventions (v2).

v1 (dataxp/domains/seir.py) let the agent scale one always-on contact
reduction and one vaccination rate per group. Here every group runs a timed
intervention programme, there are two population-wide measures with their own
timing (a categorical school policy and a mask-mandate toggle), detected cases
are isolated, and a fixed vaccine supply is allocated by share. v1 is untouched.

PHYSICS. K age groups (ages 0-80 split into K equal bins, sizes N_g),
compartments S, E, I (infectious, undetected), Q (infectious, detected and
isolated), R (recovered or vaccinated, immune), D (dead), plus an accounting
ICU census U_g. Contacts are the sum of four settings built from per-group
participation (each a reciprocal assortative + proportionate mixing matrix):
home H, school Sc (ages 5-19 plus 4% staff of working ages), work W (70% of
ages 20-64) and community Co (declining with age). beta is calibrated per
problem so that the next-generation matrix of H+Sc+W+Co OF THE 1-DAY CHAIN
has spectral radius R0: a case is counted in the force of infection for a
geometric number of days with mean 1/(1-exp(-gamma)), so
beta = R0 (1-exp(-gamma)) / rho(C N_g/N_h). (v1 used R0 gamma / rho, the
continuous-time value; in a 1-day chain that inflates the real R0 by
gamma/(1-exp(-gamma)), up to +18% for flu, and the stated R0, the R_t readout
and the final-size projection then disagreed with the simulated chain.)
Discrete-time chain, 1-day steps, exact exponential transition probabilities
(flows can never exceed their source compartment):

    lambda_g = sum_h B_t[g,h] I_h + beta*0.5*H[g,h] Q_h  (per capita of h)
    B_t      = beta*( H + m_t*f_t*Sc + m_t*(k_g k_h)*(W + Co) ) / N_h
    k_g      = 1 - contact_reduction_g while g's programme window is active
    f_t      = school factor of the school code while the school window runs
    m_t      = 1 - mask_efficacy while the mask window runs (else 1)
    I -> out at gamma + test_rate_g (inside g's window); the test share
             test/(gamma+test) goes to Q (isolated: transmits only at home, at
             half the household rate), the rest recovers
    deaths   = IFR_g x every exit from I or Q (Q dies at the same IFR)
    vaccine  = supply_frac*N doses delivered evenly from vax_day over
               rollout_days (clipped to day 180); group g receives the share
               vax_share_g / max(1, sum_h vax_share_h) of each day's doses;
               doses go to susceptibles only (S -> R); doses a group cannot use
               (no susceptibles left) are discarded and not paid for
    waning   = R -> S at omega (covid_waning family only)
    ICU      U_g' = icu_risk_g*(E->I flow) - U_g/icu_stay (census only)

Policy horizon: all windows and deliveries lie in days [0, 180). The chain then
runs a 90-day TAIL with every measure lifted (day 270 total), and a
final-size PROJECTION adds the deaths of the wave the remaining susceptibles
would suffer after day 270 if the disease is reintroduced (one imported case;
multi-group final-size equation x = s0 exp(-A (c + s0 - x)) with
A = beta C/(N_h (1-exp(-gamma))), which is exact for the deterministic 1-day
chain; minimal root by monotone Newton from below). A design that merely
postpones the epidemic past the horizon pays for it. (Checked: on 30 gold
final designs a 900-day tail breaches no peak/ICU cap, so the later wave's
uncounted peak is not exploited.)

ELEMENTS (fixed per problem; indices never change):
  0..K-1  group g (age band). Params:
    contact_reduction  continuous [0, 0.9]  share of g's work+community
                       contacts removed while g's window runs (scales g's row
                       AND column, so a g-h link keeps (1-cr_g)(1-cr_h))
    start_day          additive integer days [0, 179]  window start
    duration           additive integer days [0, 180]  window length; the
                       window is [start, min(start+duration, 180))
    test_rate          continuous [0, 0.5] per day  detection-and-isolation
                       rate of infectious members while g's window runs
    vax_share          continuous [0, 1]  share of the vaccine supply sent to
                       g; normalised in evaluate() as share/max(1, sum shares)
  K       schools (one global element). Params:
    school_policy      categorical {0 open, 1 small cohorts (school contacts
                       x0.6), 2 hybrid half attendance (x0.35), 3 closed /
                       remote (x0.05)}; cost per day 0 / 0.12 / 0.45 / 1.0 x
                       w_school x school-population share
    start_day, duration  the school measure's window (as above)
  K+1     mask mandate (one global element). Params:
    mask_mandate       toggle {0, 1}; multiplies school, work and community
                       transmission by (1 - mask_efficacy) during its window
    start_day, duration  the mandate's window
get() returns None for a param that does not apply to an element. set()
legalises deterministically: clamps to bounds, rounds continuous values to 4
decimals, days and the school code to integers, the mask toggle to {0, 1}.

MOVES (candidate_moves, ~40-125 per turn): continuous params scaled by
x0.5/0.8/1.3/2.0 (switch-on values when ~0, and 0 when small); start_day
shifted -21/-7/+7/+21, duration -21/-7/+14/+42; every other school code; the
mask flip. Global measures get all their moves every turn; group params are
stratified (all groups if K <= 5, else 5 groups drawn PER PARAM), so no
element type or move type is ever dropped. Window moves are offered only where
the window does something (cr or test > 0, school code > 0, mask on) and only
when the effective window changes.

INVALID DESIGNS: none are structurally possible (every legal setting is a
well-posed chain); any numerical failure returns finite worst-case metrics
outside the validity envelope (pop_conservation_err = 1), never NaN/inf.

OBJECTIVE (cost, minimise): 0.01 base
  + w_c  * sum_g (N_g/N) * cr_g^2   * window days_g      (closures)
  + w_t  * sum_g (N_g/N) * test_g   * window days_g      (testing)
  + w_s  * school_share * level_cost[code] * school days (schools)
  + w_m  * mask days * mask_on                           (mask mandate)
  + w_v  * doses administered / N                        (vaccine logistics)
CONSTRAINTS: maximum_peak_infected_frac (I+Q, days 0-270),
maximum_total_deaths_per_1k (simulated + projected), maximum_peak_icu_per_100k.
VALIDITY: peak_infected_frac <= 1, attack_rate <= 5 (reinfection allowed under
waning), total_deaths_per_1k <= 1000, pop_conservation_err <= 1e-6, cost > 0.

PROBLEMS. 4 families (flu_like, covid_like, measles_like, covid_waning) x the
v1 group ladder K in {2,3,4,6,8,10,12,16} (elements K+2). The initial design
is a weak, MISTIMED programme (small reductions in short windows that start
anywhere from early to after the free epidemic's peak; schools mostly open,
masks mostly off; a pro-rata / partial / youth-first vaccine plan). Goals are
calibrated by a seeded walk of 1-16 constraint-improving moves (tiers easy
1-3 / medium 4-8 / hard 9-16) from the initial design through the same move
set the agent has. For each constrained metric the walk improved, the cap lies
between the walk end and the start (witness + u x (start - witness), u in
[0, 0.25]); for one it did not improve, the cap is witness x [1, 1.12]. So
every problem has a known feasible design reachable in at most 16 moves (kept
in the spec as calib_actions) and the initial design violates every cap the
walk improved. STOCHASTIC variant: feasibility is a chance constraint on the
q=0.9 replicate quantile, so witness and start are measured as the q90 of 32
seeded stochastic replicates (witness x 1.10 margin) and u lies in
[0.05, 0.30]; calibrating on the mean-field witness instead left the witness
chance-infeasible in 26/30 problems (few-death counts are very noisy).

REFERENCE (heuristic): best of two seeded searches on the deterministic chain:
(1) the calibration witness followed by a cost-trimming descent (cheaper moves
that keep all three caps); (2) a penalised best-improvement local search from
the initial design over the agent's own move set (up to 32 steps, not
stopping at the first feasible design; score = the design-program value,
cost/max_cost plus a soft-hinge violation
term and a hard-violation offset, so slack-buying moves that enable later
cuts are taken), keeping the cheapest feasible design visited, then the same
trimming descent. reference = the cheapest feasible cost found. STOCHASTIC
variant: the caps are chance constraints, loose on the deterministic chain
(the search found near-free designs that fail them), so both searches run
against caps tightened by the witness's deterministic/q90 ratio per metric,
and their designs are accepted cheapest-first only if the q90 of 32 seeded
replicates meets the true caps (reference = that design's mean replicate
cost). It is still a heuristic, judged with more replicates than the
evaluator's 4-24, so obj/ref there carries estimator noise.

STOCHASTIC MODE: same chain with integer people and per-step binomial draws
(numpy default_rng(seed) only); vaccine deliveries are integer and
deterministic; the ICU census and the final-size projection are computed from
the realised state. Group sizes 1.5k-8k, so demographic noise is real.
Stochastic problems are seeded with 3x the day-0 cases (>= 24 per seeded
group): with 8 cases the flu_like peak varied ~10x across replicates and 4/8
flu problems were unrankable (G2-excluded) in a 30-problem run.

SIMPLIFICATIONS (explicit): 1-day steps; contact structure is synthetic
(four settings from participation profiles, not survey data); contact
reduction acts on work+community contacts only (home never, school only via
the school policy); masks act on all non-home settings with one efficacy;
isolation is perfect outside the home and leaks at half the household rate;
detection applies only to I (not E) and is a constant rate inside the window;
vaccine efficacy is folded into the dose count (one dose immunises one
susceptible), doses reach susceptibles only; the ICU census is an accounting
quantity (no capacity feedback on deaths); the projection ignores waning and
vaccination after day 270 and counts no ICU/peak for the projected wave; no
importation inside the simulated 270 days.

Verb surface (docs/dataxp_verb_surface.md; owner's decision of 2026-10-23)
------------
The move set (candidate_moves), set(), Phi and the stored {i, p, v, v_old}
dicts are the v2 ones above and are untouched; only the spelling in prompts,
action lines and reply parsing changes (class attributes verb_surface and
verb_table; format_move / parse_move / grammar are the table-driven defaults
of dataxp.domain). The bijection, per parameter (g = group number 0..K-1;
e = any element: a group g, K = schools, K+1 = mask mandate):
  contact_reduction v  <-> SET_CONTACT_REDUCTION(g, v)       share, 0..0.9
  test_rate v          <-> SET_TESTING_RATE(g, v)            per day, 0..0.5
  vax_share v          <-> SET_VACCINE_SHARE(g, v)           share, 0..1
  start_day v          <-> SET_INTERVENTION_START(e, v)      day, 0..179
  duration v           <-> SET_INTERVENTION_DURATION(e, v)   days, 0..180
  school_policy k      <-> SET_SCHOOL_POLICY(K, name), name = open /
                           small_cohorts / hybrid / closed (k = 0..3)
  mask_mandate 1 / 0   <-> MANDATE_MASKS(K+1) / LIFT_MASKS(K+1)
Judgment calls, recorded here as the surface rules ask:
  * Every verb carries the element number SET_PARAM used, including
    SET_SCHOOL_POLICY, MANDATE_MASKS and LIFT_MASKS, whose only legal element
    is the schools' (K) or the mandate's (K+1): the surface mechanism writes
    and reads an element argument on every verb, and the observation prints
    both numbers (PROBLEM line, DESIGN rows). A call on any other element
    parses, as the SET_PARAM did, and apply() changes nothing, exactly as
    set() did; legal_moves never offers it.
  * The window verbs SET_INTERVENTION_START / SET_INTERVENTION_DURATION are
    shared by the groups, the schools and the mask mandate (start_day and
    duration are one parameter each in `params`, so one table row each); the
    element argument says whose window.
  * Their values are the new absolute first day / length in days, as
    SET_PARAM's values were, not shifts; set() still rounds them to whole
    days and clamps.
  * School policies are spelled as identifiers built from the observation's
    names (open, small_cohorts, hybrid, closed). The observation's full names
    ("small cohorts", "hybrid half attendance", "closed (remote learning)")
    and "remote" are parse aliases, case ignored; format_move writes the
    canonical identifier only, so the surface is one-to-one on the legal
    moves. A bare code (SET_SCHOOL_POLICY(4, 3)) is not accepted.
  * mask_mandate is an on/off state, so it is the verb pair MANDATE_MASKS /
    LIFT_MASKS with no value argument.
  * Every continuous value is legalised by set() to 4 decimals or a whole
    day, so the 6-significant-digit action text (fmt_arg, as for SET_PARAM)
    is exact.
  * SET_PARAM(...) text is no longer parsed on this domain: the grammar the
    policy is served lists the verbs only.
  * describe()'s "Tunable" paragraph is written in the verbs (observation
    text only; no generated file contains it), conditional on verb_surface;
    the DESIGN rows keep their parameter columns (school_policy code and
    name, mask_mandate 0 / 1, start_day, duration), and the SET_SCHOOL_POLICY
    sentence ties each name to its code.
  * History and "change you are about to make" lines print the replaced value
    ("was 30", "policy open -> hybrid") for every verb but the mandate pair.

Set-parameter spelling (docs/dataxp_spellings.md; DATAXP_SPELLING=set_parameter)
-----------------------
The same moves through the one wrapper tool, generated from the verb table
by dataxp.domain (no override here): SET_CONTACT_REDUCTION(2, 0.4) <->
SET_PARAM(2, contact_reduction, 0.4), SET_INTERVENTION_START(K, 30) <->
SET_PARAM(K, start_day, 30), SET_SCHOOL_POLICY(K, closed) <-> SET_PARAM(K,
school_policy, 3) (the stored code; the grammar names the codes),
MANDATE_MASKS(K+1) / LIFT_MASKS(K+1) <-> SET_PARAM(K+1, mask_mandate, 1 / 0),
and so on. Judgment calls: the phrases are "Set a group's contact reduction /
testing rate / vaccine share", "Set a measure's start day / duration", "Set
the school policy", "Mandate masks" / "Lift the mask mandate"; the "Tunable"
paragraph reverts to its SET_PARAM wording (the run2 text) under this
spelling (spelled_in_verbs); the history "was ..." prints the stored
school_policy code.
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
from ..potential import DesignProgram, compute_potential_v2, program_from_goals

_FAMILIES = ("flu_like", "covid_like", "measles_like", "covid_waning")
_K_LADDER = (2, 3, 4, 6, 8, 10, 12, 16)
_T_POLICY = 180
_T_TAIL = 90
_T_SIM = _T_POLICY + _T_TAIL
_CR_MAX = 0.9
_TEST_MAX = 0.5
_BASE_COST = 0.01
_ISO_HOME = 0.5
_SCHOOL_FACTOR = (1.0, 0.6, 0.35, 0.05)
_SCHOOL_COST = (0.0, 0.12, 0.45, 1.0)
_SCHOOL_NAMES = ("open", "small cohorts", "hybrid half attendance",
                 "closed (remote learning)")
# verb-surface names of the school codes (docstring "Verb surface")
SCHOOL_POLICY_NAMES = ("open", "small_cohorts", "hybrid", "closed")
_GROUP_ARG = ArgSpec("group", "int", "age-group number (the number after g)")
_MEASURE_ARG = ArgSpec("measure", "int", "whose window: the group number g for that group's "
                                         "programme, the schools' element number for the "
                                         "school measure, the mask mandate's element number "
                                         "for the mandate (both numbers are printed in the "
                                         "observation)")
_SCHOOLS_ARG = ArgSpec("schools", "int", "the schools' element number (printed in the "
                                         "observation)")
_MANDATE_ARG = ArgSpec("mandate", "int", "the mask mandate's element number (printed in the "
                                         "observation)")
_GROUP_PARAMS = ("contact_reduction", "start_day", "duration", "test_rate",
                 "vax_share")
_SCHOOL_PARAMS = ("school_policy", "start_day", "duration")
_MASK_PARAMS = ("mask_mandate", "start_day", "duration")
_ALL_PARAMS = ("contact_reduction", "start_day", "duration", "test_rate",
               "vax_share", "school_policy", "mask_mandate")
_BOUNDS = {
    "contact_reduction": (0.0, _CR_MAX),
    "start_day": (0.0, float(_T_POLICY - 1)),
    "duration": (0.0, float(_T_POLICY)),
    "test_rate": (0.0, _TEST_MAX),
    "vax_share": (0.0, 1.0),
    "school_policy": (0.0, 3.0),
    "mask_mandate": (0.0, 1.0),
}
# move sets
_SCALE = (0.5, 0.8, 1.3, 2.0)
_START_SHIFTS = (-21, -7, 7, 21)
_DUR_SHIFTS = (-21, -7, 14, 42)
_GROUPS_PER_PARAM = 5
_CONSTRAINED = ("peak_infected_frac", "total_deaths_per_1k", "peak_icu_per_100k")
_GOAL_KEYS = {
    "peak_infected_frac": "maximum_peak_infected_frac",
    "total_deaths_per_1k": "maximum_total_deaths_per_1k",
    "peak_icu_per_100k": "maximum_peak_icu_per_100k",
}


class SEIR2Domain(Domain):
    """Age-structured SEIR with timed, typed interventions; see module doc."""
    # stochastic variants: phi_select compares candidates with a fresh
    # incumbent estimate (dataxp/ladder.py); no effect when deterministic
    rescore_incumbent = True

    name = "seir2"
    params = _ALL_PARAMS
    factors = _SCALE
    reference_class = "heuristic"
    max_elements = 18
    max_candidates = 160

    # ── verb surface (docstring "Verb surface"; docs/dataxp_verb_surface.md) ──
    # Spelling only: every SET_PARAM move candidate_moves enumerates has one
    # verb text and parses back to the same move. format_move / parse_move /
    # grammar are the table-driven defaults of dataxp.domain.
    verb_surface = True
    verb_table = (
        ParamVerb("contact_reduction", "SET_CONTACT_REDUCTION", _GROUP_ARG,
                  Continuous("share", "", "0..%g, the new share of the group's work and "
                                          "community contacts removed while its window runs "
                                          "(rounded to 4 decimals; a value outside is set to "
                                          "the bound)" % _CR_MAX),
                  doc="cut a group's work and community contacts inside its intervention "
                      "window; a g-h link keeps (1 - share_g)(1 - share_h). Costs w_closure "
                      "x (N_g/N) x share^2 per window day.",
                  phrase="Set a group's contact reduction"),
        ParamVerb("start_day", "SET_INTERVENTION_START", _MEASURE_ARG,
                  Continuous("day", "day", "0..%d, the new first day of the measure's window "
                                           "(a whole number, not a shift; the window is days "
                                           "start .. min(start + duration, %d) - 1)"
                             % (_T_POLICY - 1, _T_POLICY)),
                  doc="move the start of a group's programme, the school measure or the mask "
                      "mandate; the window matters only while that measure does something "
                      "(contact reduction or testing above 0, a school policy other than "
                      "open, a mandate in force).",
                  phrase="Set a measure's start day"),
        ParamVerb("duration", "SET_INTERVENTION_DURATION", _MEASURE_ARG,
                  Continuous("days", "days", "0..%d, the new length of the measure's window "
                                             "in whole days (not a shift; it is cut at day "
                                             "%d)" % (_T_POLICY, _T_POLICY)),
                  doc="lengthen or shorten the window of a group's programme, the school "
                      "measure or the mask mandate; every per-day cost scales with the "
                      "effective window days.",
                  phrase="Set a measure's duration"),
        ParamVerb("test_rate", "SET_TESTING_RATE", _GROUP_ARG,
                  Continuous("rate", "per day", "0..%g, the new detection-and-isolation rate "
                                                "of the group's infectious members inside "
                                                "its window (4 decimals)" % _TEST_MAX),
                  doc="test and isolate a group's infectious members inside its window: the "
                      "detected share rate/(rate + gamma) transmits only at home, at half "
                      "the household rate. Costs w_test x (N_g/N) x rate per window day.",
                  phrase="Set a group's testing rate"),
        ParamVerb("vax_share", "SET_VACCINE_SHARE", _GROUP_ARG,
                  Continuous("share", "", "0..1, the new share of the vaccine supply offered "
                                          "to the group (shares are divided by max(1, their "
                                          "sum); 4 decimals)"),
                  doc="reallocate the fixed vaccine supply to a group; doses reach its "
                      "susceptibles only and only the doses given are paid for.",
                  phrase="Set a group's vaccine share"),
        ParamVerb("school_policy", "SET_SCHOOL_POLICY", _SCHOOLS_ARG,
                  Categorical("policy", SCHOOL_POLICY_NAMES,
                              aliases=(("small cohorts", 1), ("hybrid half attendance", 2),
                                       ("closed (remote learning)", 3), ("remote", 3)),
                              doc="open = code 0 (school contacts x1, free), small_cohorts = "
                                  "1 (x%g, cost %g), hybrid = 2 half attendance (x%g, cost "
                                  "%g), closed = 3 remote learning (x%g, cost %g); the cost "
                                  "is per day of the school window x w_school x the school "
                                  "population share."
                                  % (_SCHOOL_FACTOR[1], _SCHOOL_COST[1], _SCHOOL_FACTOR[2],
                                     _SCHOOL_COST[2], _SCHOOL_FACTOR[3], _SCHOOL_COST[3])),
                  doc="set the school policy that applies during the school window.",
                  phrase="Set the school policy"),
        ParamVerb("mask_mandate", "MANDATE_MASKS", _MANDATE_ARG,
                  Toggle("LIFT_MASKS",
                         off_doc="take the mask mandate out of force (its window is kept, "
                                 "inactive).",
                         off_phrase="Lift the mask mandate"),
                  doc="put a population-wide mask mandate in force during its window: school, "
                      "work and community transmission x (1 - mask efficacy); costs w_mask "
                      "per day in force.",
                  phrase="Mandate masks"),
    )

    def __init__(self, stochastic: bool = False):
        super().__init__(stochastic)

    # ======================================================================
    # problems
    # ======================================================================
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[int(index) % len(_FAMILIES)]
        k = int(_K_LADDER[(int(index) // len(_FAMILIES)) % len(_K_LADDER)])
        n_pop = rng.integers(1500, 8001, size=k)

        if family == "flu_like":
            r0 = float(rng.uniform(1.4, 1.9))
            latent = float(rng.uniform(1.5, 2.5))
            infectious = float(rng.uniform(3.0, 5.0))
            waning = 0.0
            vax_day = int(rng.integers(25, 71))
            rollout = int(rng.integers(30, 61))
            supply = float(rng.uniform(0.25, 0.5))
        elif family in ("covid_like", "covid_waning"):
            r0 = float(rng.uniform(2.5, 3.8))
            latent = float(rng.uniform(3.0, 5.0))
            infectious = float(rng.uniform(5.0, 8.0))
            waning = (1.0 / float(rng.uniform(60.0, 120.0))
                      if family == "covid_waning" else 0.0)
            vax_day = int(rng.integers(50, 111))
            rollout = int(rng.integers(45, 91))
            supply = float(rng.uniform(0.45, 0.8))
        else:  # measles_like
            r0 = float(rng.uniform(6.0, 10.0))
            latent = float(rng.uniform(8.0, 12.0))
            infectious = float(rng.uniform(5.0, 8.0))
            waning = 0.0
            vax_day = int(rng.integers(0, 26))
            rollout = int(rng.integers(20, 46))
            supply = float(rng.uniform(0.7, 0.95))

        layers = {
            "home": {"rate": float(rng.uniform(2.0, 3.5)),
                     "assort": float(rng.uniform(0.15, 0.35))},
            "school": {"rate": float(rng.uniform(8.0, 14.0)),
                       "assort": float(rng.uniform(0.6, 0.85))},
            "work": {"rate": float(rng.uniform(4.0, 8.0)),
                     "assort": float(rng.uniform(0.3, 0.6))},
            "community": {"rate": float(rng.uniform(3.0, 7.0)),
                          "assort": float(rng.uniform(0.1, 0.4))},
        }

        edges = np.linspace(0.0, 80.0, k + 1)
        mid = 0.5 * (edges[:-1] + edges[1:]) / 80.0
        ifr0 = math.exp(rng.uniform(math.log(2e-5), math.log(2e-4)))
        ifr_ratio = math.exp(rng.uniform(math.log(50.0), math.log(400.0)))
        ifr = np.clip(ifr0 * np.power(ifr_ratio, (mid - mid[0]) / max(
            mid[-1] - mid[0], 1e-9)), 0.0, 0.15)
        icu_mult = float(rng.uniform(2.0, 3.5))

        n_seed_groups = 1 if k < 4 else int(rng.integers(1, 3))
        seed_groups = rng.choice(k, size=n_seed_groups, replace=False)
        seed_frac = float(rng.uniform(5e-4, 1.5e-3))
        i0 = np.zeros(k, dtype=np.int64)
        for g in seed_groups:
            i0[int(g)] = max(8, int(round(seed_frac * float(n_pop[int(g)]))))
        if self.stochastic:
            # 8 seed cases at low R0 made take-off timing so random that the
            # peak varied ~10x across replicates (flu_like) and Phi could not
            # rank moves; 3x the seed cases halves that spread. No rng draw,
            # so deterministic specs are unchanged.
            i0 = i0 * 3

        spec = {
            "problem_id": "seir2_%04d" % index,
            "family": family,
            "K": k,
            "N": [int(x) for x in n_pop],
            "age_edges": [float(x) for x in edges],
            "layers": layers,
            "r0": r0,
            "latent_days": latent,
            "infectious_days": infectious,
            "waning_per_day": float(waning),
            "ifr": [float(x) for x in ifr],
            "icu_risk": [float(min(0.3, x * icu_mult)) for x in ifr],
            "icu_stay_days": float(rng.uniform(7.0, 14.0)),
            "I0": [int(x) for x in i0],
            "mask_efficacy": float(rng.uniform(0.15, 0.30)),
            "vax_day": vax_day,
            "rollout_days": rollout,
            "supply_frac": supply,
            "w_closure": float(rng.uniform(0.7, 1.5)),
            "w_test": float(rng.uniform(0.25, 0.7)),
            "w_school": float(rng.uniform(0.5, 1.5)),
            "w_mask": float(rng.uniform(0.01, 0.04)),
            "w_vax": float(rng.uniform(4.0, 12.0)),
            "t_policy": _T_POLICY,
            "t_tail": _T_TAIL,
        }

        # free epidemic -> its peak day anchors the (mistimed) initial windows
        core = self._core(spec)
        k2 = k + 2
        free = {"cr": np.zeros(k), "test": np.zeros(k), "vax": np.zeros(k),
                "start": np.zeros(k2), "dur": np.zeros(k2),
                "school": 0.0, "mask": 0.0}
        mfree, _ = self._simulate(core, free, stochastic=False, seed=0)
        peak_free = max(5.0, float(mfree["peak_day"]))

        s0 = int(round(float(rng.uniform(0.15, 1.0)) * peak_free))
        s0 = int(min(max(s0, 0), 150))
        d0 = int(rng.integers(14, 43))
        start = np.zeros(k2)
        dur = np.zeros(k2)
        for e in range(k2):
            start[e] = float(min(max(s0 + int(rng.integers(-5, 6)), 0),
                                 _T_POLICY - 1))
            dur[e] = float(min(max(d0 + int(rng.integers(-5, 6)), 7),
                               _T_POLICY))
        cr = np.round(rng.uniform(0.03, 0.12, size=k), 3)
        test = np.round(rng.uniform(0.0, 0.03, size=k), 3)
        frac = n_pop / float(np.sum(n_pop))
        plan = ("pro_rata", "partial", "youth_first")[int(rng.integers(3))]
        if plan == "pro_rata":
            vax = frac.copy()
        elif plan == "partial":
            vax = frac * float(rng.uniform(0.4, 0.8))
        else:
            w = frac * (1.0 - 0.8 * mid)
            vax = w / float(np.sum(w))
        vax = np.round(vax, 3)
        school0 = (0, 0, 1)[int(rng.integers(3))]
        mask0 = 1 if float(rng.uniform()) < 0.25 else 0
        spec["design0"] = {
            "contact_reduction": [float(x) for x in cr],
            "test_rate": [float(x) for x in test],
            "vax_share": [float(x) for x in vax],
            "start_day": [float(x) for x in start],
            "duration": [float(x) for x in dur],
            "school_policy": float(school0),
            "mask_mandate": float(mask0),
            "vax_plan": plan,
        }

        # ---- goal calibration: seeded constraint-improving walk ------------
        st = self.load(spec)
        m0, _ = self._simulate(core, self._des(st), stochastic=False, seed=0)
        tier = float(rng.uniform())
        if tier < 0.15:
            n_walk = int(rng.integers(1, 4))
        elif tier < 0.50:
            n_walk = int(rng.integers(4, 9))
        else:
            n_walk = int(rng.integers(9, 17))
        walk_rng = np.random.default_rng(int(rng.integers(0, 2 ** 31 - 1)))

        def score(m):
            return sum(float(m[key]) / max(float(m0[key]), 1e-12)
                       for key in _CONSTRAINED)

        cur_score = score(m0)
        actions = []
        for _ in range(n_walk):
            cands = self.candidate_moves(st, walk_rng)
            if len(cands) > 16:
                keep = walk_rng.choice(len(cands), size=16, replace=False)
                cands = [cands[j] for j in sorted(keep.tolist())]
            best, best_s = None, cur_score - 1e-9
            for c in cands:
                child = self.clone(st)
                self.set(child, c.i, c.p, c.v)
                mc, _ = self._simulate(core, self._des(child),
                                       stochastic=False, seed=0)
                sc = score(mc)
                if sc < best_s:
                    best, best_s = c, sc
            if best is None:
                continue
            self.set(st, best.i, best.p, best.v)
            actions.append(best.as_dict())
            cur_score = best_s
        mt, _ = self._simulate(core, self._des(st), stochastic=False, seed=0)
        if self.stochastic:
            # Chance constraints are judged on the adverse q=0.9 quantile of
            # replicates, which sits well above the mean-field value when the
            # constrained counts are small (a handful of deaths). Calibrate on
            # 32-replicate q90 of witness (x1.10 margin) and start instead,
            # with replicate seeds that consume nothing from rng.
            base = stable_seed("seir2_stoch_cal", int(index), len(actions))
            st_start = self.load(spec)
            mt = self._q90(core, st, base)
            mt = {key: 1.10 * v for key, v in mt.items()}
            m0 = self._q90(core, st_start, base + 1)
        # caps: a metric the walk improved gets a cap between the witness and
        # the start (witness feasible, start violating); one it did not
        # improve gets a cap just above the witness (slack at the start).
        u_lo, u_hi = (0.0, 0.25) if not self.stochastic else (0.05, 0.30)
        goals = {}
        for key in _CONSTRAINED:
            t_j, s_j = float(mt[key]), float(m0[key])
            if s_j > t_j * (1.0 + 1e-6):
                cap = t_j + float(rng.uniform(u_lo, u_hi)) * (s_j - t_j)
            else:
                cap = t_j * float(rng.uniform(1.0, 1.12))
            goals[_GOAL_KEYS[key]] = float(max(cap, 1e-9))
        spec["goals"] = goals
        spec["calib_actions"] = actions
        spec["calib_tier"] = ("easy" if tier < 0.15 else
                              "medium" if tier < 0.50 else "hard")
        spec["calib_seed"] = int(rng.integers(0, 2 ** 31 - 1))
        return spec

    def _q90(self, core: dict, st, base: int, n_rep: int = 32) -> dict:
        """Adverse (upper) 0.9-quantile of each constrained metric over
        n_rep seeded stochastic replicates (linear interpolation, the same
        rule as potential.aggregate_metrics)."""
        reps = [self._simulate(core, self._des(st), stochastic=True,
                               seed=(int(base) + 7919 * r) & 0xFFFFFFFF)[0]
                for r in range(n_rep)]
        return {key: float(np.quantile([m[key] for m in reps], 0.9))
                for key in _CONSTRAINED}

    # ======================================================================
    # state
    # ======================================================================
    def load(self, spec: dict):
        core = self._core(spec)
        k = core["k"]
        d0 = spec.get("design0") or {}
        k2 = k + 2

        def arr(key, n, default):
            raw = d0.get(key)
            if raw is None:
                return np.full(n, float(default))
            a = np.asarray(raw, dtype=float).copy()
            return a if a.shape[0] == n else np.full(n, float(default))

        st = {
            "core": core,
            "cr": arr("contact_reduction", k, 0.05),
            "test": arr("test_rate", k, 0.0),
            "vax": arr("vax_share", k, 0.0),
            "start": arr("start_day", k2, 30.0),
            "dur": arr("duration", k2, 21.0),
            "school": float(d0.get("school_policy", 0.0)),
            "mask": float(d0.get("mask_mandate", 0.0)),
            "_cache": {},
            "_det": None,
            "_hint": None,
        }
        return st

    def clone(self, st):
        return {
            "core": st["core"],
            "cr": np.array(st["cr"], dtype=float, copy=True),
            "test": np.array(st["test"], dtype=float, copy=True),
            "vax": np.array(st["vax"], dtype=float, copy=True),
            "start": np.array(st["start"], dtype=float, copy=True),
            "dur": np.array(st["dur"], dtype=float, copy=True),
            "school": float(st["school"]),
            "mask": float(st["mask"]),
            "_cache": {},
            "_det": None,
            "_hint": None,
        }

    @staticmethod
    def _des(st) -> dict:
        return {"cr": st["cr"], "test": st["test"], "vax": st["vax"],
                "start": st["start"], "dur": st["dur"],
                "school": st["school"], "mask": st["mask"]}

    # ======================================================================
    # core / contacts
    # ======================================================================
    @staticmethod
    def _participation(edges: np.ndarray):
        k = edges.shape[0] - 1
        school = np.zeros(k)
        work = np.zeros(k)
        comm = np.zeros(k)
        for g in range(k):
            lo, hi = float(edges[g]), float(edges[g + 1])
            width = max(hi - lo, 1e-9)

            def overlap(a, b):
                return max(0.0, min(hi, b) - max(lo, a)) / width

            child = overlap(5.0, 19.0)
            wk = overlap(20.0, 65.0)
            school[g] = child + 0.04 * wk
            work[g] = 0.7 * wk
            comm[g] = 1.15 - 0.6 * (0.5 * (lo + hi) / 80.0)
        return np.ones(k), school, work, comm

    @staticmethod
    def _layer(rate: float, assort: float, part: np.ndarray,
               n_pop: np.ndarray) -> np.ndarray:
        """Reciprocal mixing matrix: contacts/day of a member of g with
        members of h; N_g M[g,h] == N_h M[h,g]."""
        k = part.shape[0]
        w = part * n_pop
        tot = float(np.sum(w))
        prop = (w / tot) if tot > 0 else np.zeros(k)
        return rate * part[:, None] * (assort * np.eye(k)
                                       + (1.0 - assort) * prop[None, :])

    def _core(self, spec: dict) -> dict:
        k = int(spec["K"])
        n_pop = np.asarray(spec["N"], dtype=float)
        edges = np.asarray(spec.get("age_edges",
                                    np.linspace(0.0, 80.0, k + 1)), dtype=float)
        p_home, p_school, p_work, p_comm = self._participation(edges)
        lay = spec["layers"]
        home = self._layer(lay["home"]["rate"], lay["home"]["assort"], p_home, n_pop)
        school = self._layer(lay["school"]["rate"], lay["school"]["assort"],
                             p_school, n_pop)
        work = self._layer(lay["work"]["rate"], lay["work"]["assort"], p_work, n_pop)
        comm = self._layer(lay["community"]["rate"], lay["community"]["assort"],
                           p_comm, n_pop)
        contact = home + school + work + comm
        gamma = 1.0 / float(spec["infectious_days"])
        # A case in the 1-day chain is counted in the force of infection for
        # a geometric number of days with mean 1/p_ir, p_ir = 1 - exp(-gamma),
        # not 1/gamma. Calibrating with gamma (the continuous-time formula, as
        # v1 did at 2 steps/day) inflates the chain's real R0 by
        # gamma/p_ir (+6% covid, up to +18% flu), and the stated R0, the R_t
        # readout and the final-size projection would all disagree with what
        # the chain simulates. With p_ir they agree exactly (linearised).
        p_ir = -math.expm1(-gamma)
        ngm_shape = contact * (n_pop[:, None] / n_pop[None, :])
        try:
            rho = float(np.max(np.abs(np.linalg.eigvals(ngm_shape))))
        except Exception:
            rho = float(np.max(ngm_shape.sum(axis=1)))
        beta = float(spec["r0"]) * p_ir / max(rho, 1e-9)
        n_total = float(np.sum(n_pop))
        school_share = float(np.sum(n_pop * p_school)) / n_total
        vax_day = int(spec.get("vax_day", 0))
        rollout = max(1, int(spec.get("rollout_days", 30)))
        supply = float(np.round(float(spec.get("supply_frac", 0.0)) * n_total))
        return {
            "k": k,
            "spec_goals": dict(spec.get("goals") or {}),
            "family": str(spec.get("family", "unknown")),
            "n_pop": n_pop,
            "n_int": np.asarray(spec["N"], dtype=np.int64),
            "n_total": n_total,
            "frac": n_pop / n_total,
            "edges": edges,
            "p_school": p_school,
            "p_work": p_work,
            "p_comm": p_comm,
            "home": home,
            "school": school,
            "work": work,
            "comm": comm,
            "wc": work + comm,
            "contact": contact,
            "beta": beta,
            "bN": beta / n_pop,
            "bq": beta * _ISO_HOME * home / n_pop[None, :],
            # final-size kernel of the chain: exact for the 1-day chain, whose
            # cumulative force of infection is beta*C/N x (infected)/p_ir
            "a_free": beta * contact / (p_ir * n_pop[None, :]),
            "p_ir": p_ir,
            "r0": float(spec["r0"]),
            "sigma": 1.0 / float(spec["latent_days"]),
            "gamma": gamma,
            "waning": float(spec.get("waning_per_day", 0.0)),
            "ifr": np.asarray(spec["ifr"], dtype=float),
            "icu_risk": np.asarray(spec["icu_risk"], dtype=float),
            "icu_stay": float(spec.get("icu_stay_days", 10.0)),
            "i0": np.asarray(spec["I0"], dtype=np.int64),
            "mask_eff": float(spec.get("mask_efficacy", 0.2)),
            "vax_day": vax_day,
            "rollout": rollout,
            "supply": supply,
            "supply_frac": float(spec.get("supply_frac", 0.0)),
            "school_share": school_share,
            "w_closure": float(spec["w_closure"]),
            "w_test": float(spec["w_test"]),
            "w_school": float(spec["w_school"]),
            "w_mask": float(spec["w_mask"]),
            "w_vax": float(spec["w_vax"]),
        }

    # ======================================================================
    # simulation
    # ======================================================================
    def evaluate(self, st, seed: int = 0) -> dict:
        self.calls += 1
        key = int(seed) if self.stochastic else 0
        cache = st.get("_cache")
        if cache is not None and key in cache:
            return dict(cache[key])
        try:
            metrics, _ = self._simulate(st["core"], self._des(st),
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
        """Finite, maximally bad: outside the validity envelope (conservation)
        and at the worst value of every constrained metric."""
        return {
            "cost": 1e6,
            "peak_infected_frac": 1.0,
            "total_deaths_per_1k": 1000.0,
            "peak_icu_per_100k": 1e5,
            "projected_deaths_per_1k": 1000.0,
            "attack_rate": 1.0,
            "doses_per_capita": 0.0,
            "susceptible_end_frac": 0.0,
            "peak_day": 0.0,
            "pop_conservation_err": 1.0,
        }

    @staticmethod
    def _windows(des: dict, k: int):
        start = np.asarray(des["start"], dtype=float)
        dur = np.asarray(des["dur"], dtype=float)
        s_idx = np.clip(np.round(start), 0, _T_POLICY - 1).astype(np.int64)
        d_idx = np.clip(np.round(dur), 0, _T_POLICY).astype(np.int64)
        e_idx = np.minimum(s_idx + d_idx, _T_POLICY)
        return s_idx, e_idx

    def _simulate(self, core: dict, des: dict, *, stochastic: bool, seed: int,
                  detail: bool = False):
        k = core["k"]
        n_pop = core["n_pop"]
        n_total = core["n_total"]
        cr = np.clip(np.asarray(des["cr"], dtype=float), 0.0, _CR_MAX)
        test = np.clip(np.asarray(des["test"], dtype=float), 0.0, _TEST_MAX)
        vax = np.clip(np.asarray(des["vax"], dtype=float), 0.0, 1.0)
        code = int(min(max(int(round(float(des["school"]))), 0), 3))
        mask_on = float(des["mask"]) >= 0.5
        s_idx, e_idx = self._windows(des, k)

        vd = int(core["vax_day"])
        ve = int(min(vd + core["rollout"], _T_POLICY))
        tot_share = float(np.sum(vax))
        alloc = vax / max(1.0, tot_share)
        thr = core["supply"] / float(core["rollout"])
        rate = alloc * thr
        vax_on = tot_share > 0.0 and ve > vd and thr > 0.0

        bps = {0, _T_SIM}
        if vax_on:
            bps.update((vd, ve))
        for e in range(k + 2):
            if e_idx[e] > s_idx[e]:
                bps.add(int(s_idx[e]))
                bps.add(int(e_idx[e]))
        bps = sorted(b for b in bps if 0 <= b <= _T_SIM)

        gamma = core["gamma"]
        p_ei = -math.expm1(-core["sigma"])
        p_ir = -math.expm1(-gamma)
        omega = core["waning"]
        p_wane = -math.expm1(-omega)
        waning_on = omega > 0.0
        u_dec = math.exp(-1.0 / max(core["icu_stay"], 1e-9))
        ifr = core["ifr"]
        icu_risk = core["icu_risk"]
        home = core["home"]
        sch = core["school"]
        wc = core["wc"]
        bN = core["bN"][None, :]
        bq = core["bq"]
        mask_mult = 1.0 - core["mask_eff"]

        if stochastic:
            rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
            # stacked integer state S, E, I, Q, R: the five independent
            # per-step flows are drawn in ONE binomial call
            xs = np.zeros((5, k), dtype=np.int64)
            xs[2] = core["i0"]
            xs[0] = core["n_int"] - xs[2]
            pm = np.zeros((5, k))
            pm[1] = p_ei
            pm[3] = p_ir
            pm[4] = p_wane if waning_on else 0.0
            d_c = np.zeros(k, dtype=np.int64)
            i_c = xs[2]
        else:
            rng = None
            i_c = core["i0"].astype(float)
            s_c = n_pop - i_c
            e_c = np.zeros(k)
            q_c = np.zeros(k)
            r_c = np.zeros(k)
            d_c = np.zeros(k)
        doses = np.zeros(k)
        u_c = np.zeros(k)
        cum_inf = float(np.sum(i_c))
        peak_tot = float(np.sum(i_c))
        peak_step = 0
        peak_icu = 0.0

        if detail:
            det_cum = i_c.astype(float).copy()
            det_src = np.zeros((k, k))
            det_peak_i = i_c.astype(float).copy()
            det_own_peak = i_c.astype(float) / n_pop
            det_own_day = np.zeros(k)
            det_isol = np.zeros(k)
            det_icu_at_peak = np.zeros(k)
            det_prev = []
            det_rt = {}
            sample_days = set(range(0, _T_SIM + 1, 30))

        for a, b in zip(bps[:-1], bps[1:]):
            if b <= a:
                continue
            act = (s_idx[:k] <= a) & (a < e_idx[:k])
            keep = 1.0 - cr * act
            school_f = (_SCHOOL_FACTOR[code]
                        if (code > 0 and s_idx[k] <= a < e_idx[k]) else 1.0)
            mm = mask_mult if (mask_on and s_idx[k + 1] <= a < e_idx[k + 1]) else 1.0
            bmat = (home + (mm * school_f) * sch
                    + mm * (keep[:, None] * keep[None, :]) * wc) * bN
            tau = test * act
            p_out = -np.expm1(-(gamma + tau))
            qf = tau / (gamma + tau)
            tau_on = bool(np.any(tau > 0.0))
            dosing = vax_on and (vd <= a < ve)
            if stochastic:
                pm[2] = p_out
                zero_i = np.zeros(k, dtype=np.int64)
                for t in range(a, b):
                    lam = bmat @ xs[2] + bq @ xs[3]
                    pm[0] = -np.expm1(-lam)
                    fl = rng.binomial(xs, pm)
                    new_e, out_e, out_i, out_q, wane = fl[0], fl[1], fl[2], fl[3], fl[4]
                    to_q = rng.binomial(out_i, qf) if tau_on else zero_i
                    rec = (out_i - to_q) + out_q
                    new_d = rng.binomial(rec, ifr)
                    if dosing:
                        target = (np.floor(rate * (t + 1 - vd))
                                  - np.floor(rate * (t - vd))).astype(np.int64)
                        new_v = np.minimum(target, xs[0] - new_e)
                        xs[0] += wane - new_e - new_v
                        xs[4] += rec - new_d + new_v - wane
                        doses = doses + new_v
                    else:
                        xs[0] += wane - new_e
                        xs[4] += rec - new_d - wane
                    xs[1] += new_e - out_e
                    xs[2] += out_e - out_i
                    xs[3] += to_q - out_q
                    d_c += new_d
                    u_c = u_c * u_dec + icu_risk * out_e
                    cum_inf += float(new_e.sum())
                    tot_i = float(xs[2].sum() + xs[3].sum())
                    icu_now = float(u_c.sum())
                    if tot_i > peak_tot:
                        peak_tot = tot_i
                        peak_step = t + 1
                    if icu_now > peak_icu:
                        peak_icu = icu_now
                continue
            for t in range(a, b):
                if detail and t in sample_days:
                    det_rt[t] = self._rt(core, bmat, tau, s_c)
                lam = bmat @ i_c + bq @ q_c
                p_inf = -np.expm1(-lam)
                new_e = s_c * p_inf
                if dosing:
                    new_v = np.minimum(rate, s_c - new_e)
                out_e = e_c * p_ei
                out_i = i_c * p_out
                to_q = out_i * qf
                out_q = q_c * p_ir
                rec = (out_i - to_q) + out_q
                new_d = rec * ifr
                wane = r_c * p_wane if waning_on else 0.0
                if detail:
                    parts = bmat * i_c[None, :] + bq * q_c[None, :]
                    ltot = parts.sum(axis=1)
                    safe = np.where(ltot > 0.0, ltot, 1.0)
                    det_src += (new_e / safe)[:, None] * parts
                    det_cum = det_cum + new_e
                    det_isol = det_isol + to_q
                if dosing:
                    s_c = s_c - new_e - new_v + wane
                    r_c = r_c + rec - new_d + new_v - wane
                    doses = doses + new_v
                else:
                    s_c = s_c - new_e + wane
                    r_c = r_c + rec - new_d - wane
                e_c = e_c + new_e - out_e
                i_c = i_c + out_e - out_i
                q_c = q_c + to_q - out_q
                d_c = d_c + new_d
                u_c = u_c * u_dec + icu_risk * out_e
                cum_inf += float(np.sum(new_e))
                infq = i_c + q_c
                tot_i = float(np.sum(infq))
                icu_now = float(np.sum(u_c))
                if tot_i > peak_tot:
                    peak_tot = tot_i
                    peak_step = t + 1
                    if detail:
                        det_peak_i = infq.astype(float)
                if icu_now > peak_icu:
                    peak_icu = icu_now
                    if detail:
                        det_icu_at_peak = u_c.copy()
                if detail:
                    f_i = infq / n_pop
                    up = f_i > det_own_peak
                    det_own_peak = np.where(up, f_i, det_own_peak)
                    det_own_day = np.where(up, t + 1, det_own_day)
                    if (t + 1) % 30 == 0 or t == 0:
                        det_prev.append((t + 1 if t > 0 else 0,
                                         tot_i / n_total))
        if stochastic:
            s_c, e_c, i_c, q_c, r_c = xs[0], xs[1], xs[2], xs[3], xs[4]
        if detail:
            det_rt[_T_SIM] = self._rt(core, core["contact"] * bN,
                                      np.zeros(k), s_c)

        # ---- projection: wave of the remaining susceptibles -----------------
        s_end = np.asarray(s_c, dtype=float)
        active = np.asarray(e_c + i_c + q_c, dtype=float)
        proj_new = self._final_size(core, s_end, active)
        proj_inf = proj_new + active
        proj_deaths = float(np.sum(ifr * proj_inf))

        # ---- cost ------------------------------------------------------------
        frac = core["frac"]
        days = (e_idx - s_idx).astype(float)
        closure = core["w_closure"] * float(np.sum(frac * cr * cr * days[:k]))
        testing = core["w_test"] * float(np.sum(frac * test * days[:k]))
        school_cost = (core["w_school"] * core["school_share"]
                       * _SCHOOL_COST[code] * float(days[k]))
        mask_cost = core["w_mask"] * float(days[k + 1]) * (1.0 if mask_on else 0.0)
        doses_tot = float(np.sum(doses))
        vax_cost = core["w_vax"] * doses_tot / n_total
        cost = _BASE_COST + closure + testing + school_cost + mask_cost + vax_cost

        deaths_sim = float(np.sum(d_c))
        conserv = (float(np.sum(s_c) + np.sum(e_c) + np.sum(i_c) + np.sum(q_c)
                         + np.sum(r_c) + np.sum(d_c)) - n_total) / n_total
        metrics = {
            "cost": float(cost),
            "peak_infected_frac": float(peak_tot / n_total),
            "total_deaths_per_1k": float(1000.0 * (deaths_sim + proj_deaths)
                                         / n_total),
            "peak_icu_per_100k": float(1e5 * peak_icu / n_total),
            "projected_deaths_per_1k": float(1000.0 * proj_deaths / n_total),
            "attack_rate": float(cum_inf / n_total),
            "doses_per_capita": float(doses_tot / n_total),
            "susceptible_end_frac": float(np.sum(s_end) / n_total),
            "peak_day": float(peak_step),
            "pop_conservation_err": float(conserv),
        }
        aux = None
        if detail:
            aux = {
                "s_idx": s_idx, "e_idx": e_idx, "code": code,
                "mask_on": mask_on, "alloc": alloc, "rate": rate,
                "vd": vd, "ve": ve, "vax_on": vax_on,
                "cum": det_cum, "src": det_src,
                "peak_share": det_peak_i / max(peak_tot, 1e-12),
                "own_peak": det_own_peak, "own_day": det_own_day,
                "isolated": det_isol, "deaths_sim": np.asarray(d_c, dtype=float),
                "deaths_proj": ifr * proj_inf,
                "doses": doses, "icu_at_peak": det_icu_at_peak,
                "prev": det_prev, "rt": det_rt,
                "costs": {"closure": closure, "testing": testing,
                          "schools": school_cost, "masks": mask_cost,
                          "vaccine": vax_cost},
                "closure_g": core["w_closure"] * frac * cr * cr * days[:k],
                "testing_g": core["w_test"] * frac * test * days[:k],
                "s_end": s_end,
            }
        return metrics, aux

    @staticmethod
    def _rt(core: dict, bmat: np.ndarray, tau: np.ndarray, s_c) -> float:
        """Effective reproduction number of the current regime and
        susceptible pool: next-generation matrix of the 1-day chain incl.
        isolation (a case is counted 1/(1-exp(-(gamma+test))) days in I, and
        a detected one a further 1/(1-exp(-gamma)) days in Q)."""
        gamma = core["gamma"]
        s = np.asarray(s_c, dtype=float)
        tau = np.asarray(tau, dtype=float)
        stay_i = 1.0 / (-np.expm1(-(gamma + tau)))
        to_q = tau / (gamma + tau)
        ngm = s[:, None] * (bmat * stay_i[None, :]
                            + core["bq"] * (to_q / core["p_ir"])[None, :])
        try:
            return float(np.max(np.abs(np.linalg.eigvals(ngm))))
        except Exception:
            return float("nan")

    @staticmethod
    def _final_size(core: dict, s0: np.ndarray, active: np.ndarray) -> np.ndarray:
        """People of each group (out of s0) infected by the free wave after
        the simulated period, with one imported case added pro-rata. Solves
        x = s0*exp(-A(c + s0 - x)) for the minimal root by monotone Newton
        from below (x = susceptibles left at the end of the wave)."""
        a_free = core["a_free"]
        c = active + core["frac"]           # + one imported person
        k = s0.shape[0]
        x = np.zeros(k)
        eye = np.eye(k)
        tol = 1e-9 * max(core["n_total"], 1.0)
        for _ in range(60):
            f = s0 * np.exp(-(a_free @ (c + s0 - x)))
            g = f - x
            if float(np.max(np.abs(g))) <= tol:
                x = f
                break
            jac = eye - f[:, None] * a_free
            try:
                dx = np.linalg.solve(jac, g)
                if not np.all(np.isfinite(dx)):
                    raise FloatingPointError
                x_new = np.minimum(np.maximum(x + dx, 0.0), s0)
            except Exception:
                x_new = f
            x = x_new
        x = np.minimum(np.maximum(x, 0.0), s0)
        return s0 - x

    def _detail(self, st) -> Tuple[dict, dict]:
        cached = st.get("_det")
        if cached is not None:
            return cached
        m, aux = self._simulate(st["core"], self._des(st), stochastic=False,
                                seed=0, detail=True)
        st["_det"] = (m, aux)
        return m, aux

    # ======================================================================
    # design variables
    # ======================================================================
    def n_elements(self, st) -> int:
        return int(st["core"]["k"]) + 2

    def _kind(self, st, i: int) -> Optional[str]:
        k = int(st["core"]["k"])
        if 0 <= i < k:
            return "group"
        if i == k:
            return "school"
        if i == k + 1:
            return "mask"
        return None

    def _applies(self, st, i: int, p: str) -> bool:
        kind = self._kind(st, i)
        if kind == "group":
            return p in _GROUP_PARAMS
        if kind == "school":
            return p in _SCHOOL_PARAMS
        if kind == "mask":
            return p in _MASK_PARAMS
        return False

    def get(self, st, i: int, p: str) -> Optional[float]:
        i = int(i)
        if not self._applies(st, i, p):
            return None
        if p == "contact_reduction":
            return float(st["cr"][i])
        if p == "test_rate":
            return float(st["test"][i])
        if p == "vax_share":
            return float(st["vax"][i])
        if p == "start_day":
            return float(st["start"][i])
        if p == "duration":
            return float(st["dur"][i])
        if p == "school_policy":
            return float(st["school"])
        if p == "mask_mandate":
            return float(st["mask"])
        return None

    @staticmethod
    def _legal(p: str, v: float) -> Optional[float]:
        """Legalise a value: clamp, round integer/categorical/toggle params,
        round continuous ones to 4 decimals. Deterministic."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        lo, hi = _BOUNDS[p]
        if p in ("contact_reduction", "test_rate", "vax_share"):
            return float(round(min(hi, max(lo, v)), 4))
        if p in ("start_day", "duration", "school_policy"):
            return float(int(round(min(hi, max(lo, v)))))
        if p == "mask_mandate":
            return 1.0 if v >= 0.5 else 0.0
        return None

    def set(self, st, i: int, p: str, v: float) -> None:
        i = int(i)
        if not self._applies(st, i, p):
            return
        val = self._legal(p, v)
        if val is None:
            return
        if p == "contact_reduction":
            st["cr"][i] = val
        elif p == "test_rate":
            st["test"][i] = val
        elif p == "vax_share":
            st["vax"][i] = val
        elif p == "start_day":
            st["start"][i] = val
        elif p == "duration":
            st["dur"][i] = val
        elif p == "school_policy":
            st["school"] = val
        elif p == "mask_mandate":
            st["mask"] = val
        st["_cache"] = {}
        st["_det"] = None
        st["_hint"] = None

    def bounds(self, st, p: str) -> Tuple[float, float]:
        if p not in _BOUNDS:
            raise ValueError("unknown param %r" % p)
        return _BOUNDS[p]

    # ======================================================================
    # moves
    # ======================================================================
    def _window(self, st, i: int) -> Tuple[int, int]:
        s = int(min(max(int(round(float(st["start"][i]))), 0), _T_POLICY - 1))
        d = int(min(max(int(round(float(st["dur"][i]))), 0), _T_POLICY))
        return s, min(s + d, _T_POLICY)

    def _scaled_values(self, cur: float, floor_vals, zero_ok: float):
        if cur < floor_vals[0] * 0.4:
            vals = list(floor_vals)
            if cur > 0.0:
                vals.append(0.0)
            return vals
        vals = [cur * f for f in _SCALE]
        if cur <= zero_ok:
            vals.append(0.0)
        return vals

    def candidate_values(self, st, i: int, p: str, cur: float) -> List[float]:
        """Raw move values for one (element, param); legalised and filtered
        for real (effective) change by candidate_moves."""
        if p == "contact_reduction":
            return self._scaled_values(cur, (0.05, 0.1), 0.03)
        if p == "test_rate":
            return self._scaled_values(cur, (0.02, 0.05), 0.01)
        if p == "vax_share":
            return self._scaled_values(cur, (0.05, 0.15), 0.03)
        if p == "start_day":
            return [cur + d for d in _START_SHIFTS]
        if p == "duration":
            return [cur + d for d in _DUR_SHIFTS]
        if p == "school_policy":
            return [float(c) for c in range(4) if c != int(round(cur))]
        if p == "mask_mandate":
            return [1.0 - float(cur >= 0.5)]
        return []

    def _effective_moves(self, st, i: int, p: str) -> List[Candidate]:
        cur = self.get(st, i, p)
        if cur is None:
            return []
        out = []
        seen = set()
        win0 = self._window(st, i) if p in ("start_day", "duration") else None
        for raw in self.candidate_values(st, i, p, cur):
            v = self._legal(p, raw)
            if v is None or v == cur or v in seen:
                continue
            if win0 is not None:
                s = float(st["start"][i]) if p == "duration" else v
                d = v if p == "duration" else float(st["dur"][i])
                s_i = int(min(max(int(round(s)), 0), _T_POLICY - 1))
                e_i = min(s_i + int(min(max(int(round(d)), 0), _T_POLICY)),
                          _T_POLICY)
                if (s_i, e_i) == win0 or (e_i <= s_i and win0[1] <= win0[0]):
                    continue        # no effective change of the window
            seen.add(v)
            out.append(Candidate(i=int(i), p=p, v=float(v), v_old=float(cur)))
        return out

    def _window_matters(self, st, i: int) -> bool:
        k = int(st["core"]["k"])
        if i < k:
            return float(st["cr"][i]) > 0.0 or float(st["test"][i]) > 0.0
        if i == k:
            return int(round(float(st["school"]))) > 0
        return float(st["mask"]) >= 0.5

    def candidate_moves(self, st, rng: np.random.Generator) -> List[Candidate]:
        """Stratified move set: every element TYPE and every move TYPE is
        offered every turn. Global measures: all their moves. Groups: per
        group-param, all groups when K <= 5, else 5 groups drawn per param
        (so a param can never drop out). Window moves are offered only where
        the window currently does something (an open school or an unset mask
        mandate has no window to move) and only when they change the
        effective window."""
        k = int(st["core"]["k"])
        cands: List[Candidate] = []
        # global measures
        cands += self._effective_moves(st, k, "school_policy")
        if self._window_matters(st, k):
            cands += self._effective_moves(st, k, "start_day")
            cands += self._effective_moves(st, k, "duration")
        cands += self._effective_moves(st, k + 1, "mask_mandate")
        if self._window_matters(st, k + 1):
            cands += self._effective_moves(st, k + 1, "start_day")
            cands += self._effective_moves(st, k + 1, "duration")
        # groups, stratified by param
        for p in _GROUP_PARAMS:
            if k <= _GROUPS_PER_PARAM:
                groups = list(range(k))
            else:
                groups = sorted(int(g) for g in rng.choice(
                    k, size=_GROUPS_PER_PARAM, replace=False))
            for g in groups:
                if p in ("start_day", "duration") and not self._window_matters(st, g):
                    continue
                cands += self._effective_moves(st, g, p)
        return cands

    # ======================================================================
    # program / reference
    # ======================================================================
    def program(self, spec: dict) -> DesignProgram:
        prog = program_from_goals(
            spec["goals"],
            objective_key="cost",
            objective_ref=self._max_cost(spec),
            objective_sense="min",
            domain=self.name,
        )
        return replace(prog, validity_bounds=(
            ("peak_infected_frac", 1.0),
            ("attack_rate", 5.0),
            ("total_deaths_per_1k", 1000.0),
            ("pop_conservation_err", 1e-6),
        ))

    def _max_cost(self, spec: dict) -> float:
        """Cost of everything at maximum over the whole policy horizon (full
        closure, max testing, schools closed, masks, the whole vaccine
        supply): the objective scale, so cost/scale lies in (0, 1]."""
        core = self._core(spec)
        t = float(_T_POLICY)
        return (_BASE_COST
                + t * (core["w_closure"] * _CR_MAX * _CR_MAX
                       + core["w_test"] * _TEST_MAX
                       + core["w_school"] * core["school_share"]
                       + core["w_mask"])
                + core["w_vax"] * core["supply_frac"])

    def _feasible_det(self, m: dict, goals: dict) -> bool:
        for key in _CONSTRAINED:
            cap = goals.get(_GOAL_KEYS[key])
            if cap is not None and float(m[key]) > float(cap):
                return False
        return True

    def reference(self, spec: dict) -> Optional[float]:
        """Heuristic, best of two seeded searches on the deterministic chain
        (cached in the spec):
          (1) the calibration witness (feasible by construction) followed by
              a cost-trimming descent (cheaper moves that keep every cap);
          (2) a penalised best-improvement local search from the INITIAL
              design over the agent's own typed move set (up to 32 steps,
              past the first feasible design; score = the design-program value with a soft hinge and a
              hard-violation offset, see _ref_search), keeping the cheapest
              feasible design visited, then the same trimming descent.
        reference = the cheapest feasible cost found. (1) alone was 2-4x
        worse than the depth-1 policy on a third of the problems: the
        calibration walk over-builds (e.g. 0.9 contact reduction for 128
        days) and pure trimming cannot leave that basin.

        STOCHASTIC variant: the caps are chance constraints calibrated on
        replicate q90, so on the deterministic chain they are loose and (2)
        found near-free designs (cost 0.19 vs 8.2 for the depth-1 policy)
        that fail the chance constraint. There, both searches run against
        caps tightened by the witness's deterministic/q90 ratio per metric,
        and the found designs are accepted cheapest-first only if their
        32-replicate q90 meets the true caps; reference = the mean
        replicate cost of the first accepted design (None if none passes)."""
        if "_reference_cost" in spec:
            cached = spec["_reference_cost"]
            return None if cached is None else float(cached)
        core = self._core(spec)
        goals = spec["goals"]
        seed = spec.get("calib_seed", 0)
        st_w = self.load(spec)
        for a in spec.get("calib_actions") or []:
            self.set(st_w, int(a["i"]), a["p"], float(a["v"]))
        m_w = self._sim_det(core, st_w)
        goals_s = goals
        if self.stochastic:
            q_w, _ = self._rep_stats(core, st_w, stable_seed("seir2_ref_q90", seed))
            goals_s = dict(goals)
            for key in _CONSTRAINED:
                gk = _GOAL_KEYS[key]
                if gk in goals_s:
                    ratio = float(m_w[key]) / max(float(q_w[key]), 1e-12)
                    goals_s[gk] = float(goals_s[gk]) * min(1.0, ratio)
        found = []          # (cost, state), feasible under goals_s
        # (1) witness + trim
        if self._feasible_det(m_w, goals_s):
            found.append((float(m_w["cost"]), st_w))
            rng = np.random.default_rng(stable_seed("seir2_ref", seed))
            found.append(self._trim(core, goals_s, st_w, float(m_w["cost"]),
                                    rng, budget=140))
        # (2) penalised local search from the initial design, then trim
        rng = np.random.default_rng(stable_seed("seir2_ref_ls", seed))
        st, m = self._ref_search(core, spec, goals_s, rng, steps=32)
        if m is not None and self._feasible_det(m, goals_s):
            found.append((float(m["cost"]), st))
            found.append(self._trim(core, goals_s, st, float(m["cost"]), rng,
                                    budget=140))
        best = None
        if not self.stochastic:
            best = min(c for c, _ in found) if found else None
        else:
            found.sort(key=lambda cs: cs[0])
            for j, (_, st_c) in enumerate(found):
                q, cost_mean = self._rep_stats(
                    core, st_c, stable_seed("seir2_ref_q90", seed, j + 1))
                if self._feasible_det(q, goals):
                    best = float(cost_mean)
                    break
        spec["_reference_cost"] = best
        return best

    def _rep_stats(self, core: dict, st, base: int, n_rep: int = 32):
        """(q90 of each constrained metric, mean cost) over n_rep seeded
        stochastic replicates; same quantile rule as _q90."""
        reps = [self._simulate(core, self._des(st), stochastic=True,
                               seed=(int(base) + 7919 * r) & 0xFFFFFFFF)[0]
                for r in range(n_rep)]
        q = {key: float(np.quantile([mm[key] for mm in reps], 0.9))
             for key in _CONSTRAINED}
        return q, float(np.mean([mm["cost"] for mm in reps]))

    def _sim_det(self, core: dict, st) -> dict:
        m, _ = self._simulate(core, self._des(st), stochastic=False, seed=0)
        return m

    @staticmethod
    def _rel_violation(m: dict, goals: dict) -> float:
        v = 0.0
        for key in _CONSTRAINED:
            cap = goals.get(_GOAL_KEYS[key])
            if cap is not None:
                v += max(0.0, float(m[key]) / max(float(cap), 1e-12) - 1.0)
        return v

    def _ref_search(self, core: dict, spec: dict, goals: dict, rng,
                    steps: int = 32):
        """Penalised best-improvement descent from the initial design over
        candidate_moves (seeded), on the deterministic chain. The score is
        the negated design-program value with a SOFT hinge (slack still pays
        a little) and a hard-violation offset: with a hard hinge the descent
        froze at the first feasible design, because every slack-buying move
        (e.g. more testing, which later lets an expensive school closure be
        shortened) raises cost. Returns the cheapest FEASIBLE (state,
        metrics) visited, or (None, None)."""
        st = self.load(spec)
        m = self._sim_det(core, st)
        prog = self.program(dict(spec, goals=goals)).with_violation_scale(
            m, tau=0.05)
        offset = 0.0 if prog.is_feasible(m) else 5.0

        def score(mm):
            return -compute_potential_v2(mm, prog, alpha=5.0, tau=0.05,
                                         feasibility_offset=offset,
                                         offset_on_hard=True)

        best_st, best_m = None, None
        if self._feasible_det(m, goals):
            best_st, best_m = st, m
        cur = score(m)
        for _ in range(int(steps)):
            best_c, best_s, best_cm = None, cur - 1e-12, None
            for c in self.candidate_moves(st, rng):
                child = self.clone(st)
                self.set(child, c.i, c.p, c.v)
                mc = self._sim_det(core, child)
                sc = score(mc)
                if sc < best_s:
                    best_c, best_s, best_cm = child, sc, mc
            if best_c is None:
                break
            st, cur, m = best_c, best_s, best_cm
            if self._feasible_det(m, goals) and (
                    best_m is None or float(m["cost"]) < float(best_m["cost"])):
                best_st, best_m = st, m
        return best_st, best_m

    def _trim(self, core: dict, goals: dict, st, best: float, rng,
              budget: int = 140):
        """Seeded first-improvement cost-trimming descent that keeps every
        cap; returns (cheapest feasible cost reached, its state)."""
        used = 0
        stalled = 0
        while used < budget and stalled < 2:
            moves = self._trim_moves(st, rng)
            improved = False
            for c in moves:
                if used >= budget:
                    break
                child = self.clone(st)
                self.set(child, c.i, c.p, c.v)
                mc = self._sim_det(core, child)
                used += 1
                if self._feasible_det(mc, goals) and float(mc["cost"]) < best - 1e-12:
                    st, best = child, float(mc["cost"])
                    improved = True
                    break
            stalled = 0 if improved else stalled + 1
        return float(best), st

    def _trim_moves(self, st, rng) -> List[Candidate]:
        """Cost-reducing moves only, in a seeded order, biggest saving
        first within a shuffled tie structure."""
        k = int(st["core"]["k"])
        moves = []
        for g in range(k):
            for p, fs in (("contact_reduction", (0.0, 0.5, 0.8)),
                          ("test_rate", (0.0, 0.5, 0.8)),
                          ("vax_share", (0.0, 0.5, 0.8))):
                cur = float(self.get(st, g, p))
                for f in fs:
                    v = self._legal(p, cur * f)
                    if v is not None and v < cur:
                        moves.append(Candidate(g, p, v, cur))
            if self._window_matters(st, g):
                cur = float(st["dur"][g])
                for d in (-21, -7):
                    v = self._legal("duration", cur + d)
                    if v is not None and v < cur:
                        moves.append(Candidate(g, "duration", v, cur))
        code = int(round(float(st["school"])))
        for c in range(code):
            moves.append(Candidate(k, "school_policy", float(c), float(code)))
        if code > 0:
            cur = float(st["dur"][k])
            for d in (-21, -7):
                v = self._legal("duration", cur + d)
                if v is not None and v < cur:
                    moves.append(Candidate(k, "duration", v, cur))
        if float(st["mask"]) >= 0.5:
            moves.append(Candidate(k + 1, "mask_mandate", 0.0, 1.0))
            cur = float(st["dur"][k + 1])
            for d in (-21, -7):
                v = self._legal("duration", cur + d)
                if v is not None and v < cur:
                    moves.append(Candidate(k + 1, "duration", v, cur))
        order = rng.permutation(len(moves))
        return [moves[j] for j in order.tolist()]

    # ======================================================================
    # feedback hints
    # ======================================================================
    def hint(self, st) -> Optional[Tuple[int, str, str]]:
        """One-knob diagnostic: blame the group carrying the largest share of
        infections at the overall peak; raise its contact reduction (or its
        testing once the reduction is maxed)."""
        cached = st.get("_hint")
        if cached is not None:
            return cached if cached != () else None
        try:
            _, aux = self._detail(st)
        except Exception:
            st["_hint"] = ()
            return None
        order = [int(g) for g in np.argsort(-aux["peak_share"], kind="stable")]
        for g in order:
            if float(st["cr"][g]) < _CR_MAX * (1.0 - 1e-9):
                st["_hint"] = (g, "contact_reduction", "up")
                return st["_hint"]
        for g in order:
            if float(st["test"][g]) < _TEST_MAX * (1.0 - 1e-9):
                st["_hint"] = (g, "test_rate", "up")
                return st["_hint"]
        st["_hint"] = ()
        return None

    def hint_move(self, st) -> Optional[Candidate]:
        """Greedy move from the cheap diagnostic, timing-aware but myopic:
        blame the group with the largest share of infections at the overall
        peak; if its programme is off, switch it on; if its window starts
        after (peak - 7) move it 14 days earlier; if it ends before
        (peak + 21) extend it by 14 days; otherwise double its contact
        reduction (then its testing). Never touches vaccines, schools or
        masks."""
        try:
            m, aux = self._detail(st)
        except Exception:
            return None
        peak = int(m["peak_day"])
        order = [int(g) for g in np.argsort(-aux["peak_share"], kind="stable")]
        for g in order:
            s, e = self._window(st, g)
            cr = float(st["cr"][g])
            props = []
            if cr < 0.02:
                props.append(("contact_reduction", 0.1))
            if s > max(peak - 7, 0):
                props.append(("start_day", float(st["start"][g]) - 14.0))
            if e < min(peak + 21, _T_POLICY):
                props.append(("duration", float(st["dur"][g]) + 14.0))
            props.append(("contact_reduction", cr * 2.0))
            props.append(("test_rate", max(float(st["test"][g]) * 2.0, 0.02)))
            for p, raw in props:
                cur = float(self.get(st, g, p))
                v = self._legal(p, raw)
                if v is None or v == cur:
                    continue
                if p in ("start_day", "duration"):
                    probe = self.clone(st)
                    self.set(probe, g, p, v)
                    if self._window(probe, g) == (s, e):
                        continue
                return Candidate(i=g, p=p, v=float(v), v_old=cur)
        return None

    # ======================================================================
    # observation: describe() / element_label() and their helpers ONLY.
    # Nothing above depends on them; they never modify the state. The
    # replay in _obs_trace mirrors the deterministic branch of _simulate step
    # for step (same regime, same update order) but only records what the
    # text reports; evaluate(), moves, problems and the reference never call it.
    # ======================================================================
    # start-of-day sample days of the timeline (weekly through the policy
    # period, then monthly through the tail)
    # (frozenset: in the class body the name `set` is the method above)
    _OBS_DAYS = tuple(sorted(frozenset(range(0, _T_POLICY + 3, 7))
                             | {210, 240, _T_SIM}))
    # lever table: R_t this many days before the infectious peak
    _OBS_LEAD = 14

    @staticmethod
    def _num(x, sig: int = 3) -> str:
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
    def _pct(cls, f) -> str:
        return cls._num(100.0 * float(f)) + "%"

    @staticmethod
    def _ipct(part, whole) -> str:
        """Integer percent (for splits that only need to be read roughly)."""
        whole = float(whole)
        return "%.0f" % (100.0 * float(part) / whole) if whole > 0 else "0"

    @staticmethod
    def _ages(core: dict, g: int) -> str:
        edges = core["edges"]
        k = core["k"]
        lo = int(round(float(edges[g])))
        hi = int(round(float(edges[g + 1])))
        if g == k - 1:
            return "ages %d+" % lo
        return "ages %d-%d" % (lo, hi - 1)

    def element_label(self, st, i) -> str:
        core = st["core"]
        k = int(core["k"])
        g = int(i)
        if 0 <= g < k:
            return "element %d = group g%d (%s, %s of pop, IFR %s)" % (
                g, g, self._ages(core, g), self._pct(core["frac"][g]),
                self._num(float(core["ifr"][g])))
        if g == k:
            return "element %d = schools (pupils and staff, %s of pop)" % (
                g, self._pct(core["school_share"]))
        if g == k + 1:
            return "element %d = mask mandate (population-wide)" % g
        return "element %d" % g

    # ------------------------------------------------------------ replay
    def _obs_trace(self, st) -> dict:
        """Deterministic replay of the current design that records, per day,
        what describe() reports: start-of-day prevalence and ICU census, the
        susceptibles at the start of every day (for R_t at any day), the ICU
        peak day, who infected whom in which setting, and doses offered/used.
        Same chain, regimes and update order as _simulate(stochastic=False)."""
        core = st["core"]
        des = self._des(st)
        k = int(core["k"])
        n_pop = core["n_pop"]
        n_total = core["n_total"]
        cr = np.clip(np.asarray(des["cr"], dtype=float), 0.0, _CR_MAX)
        test = np.clip(np.asarray(des["test"], dtype=float), 0.0, _TEST_MAX)
        vax = np.clip(np.asarray(des["vax"], dtype=float), 0.0, 1.0)
        code = int(min(max(int(round(float(des["school"]))), 0), 3))
        mask_on = float(des["mask"]) >= 0.5
        s_idx, e_idx = self._windows(des, k)
        vd = int(core["vax_day"])
        ve = int(min(vd + core["rollout"], _T_POLICY))
        tot_share = float(np.sum(vax))
        alloc = vax / max(1.0, tot_share)
        thr = core["supply"] / float(core["rollout"])
        rate = alloc * thr
        vax_on = tot_share > 0.0 and ve > vd and thr > 0.0
        bps = {0, _T_SIM}
        if vax_on:
            bps.update((vd, ve))
        for e in range(k + 2):
            if e_idx[e] > s_idx[e]:
                bps.add(int(s_idx[e]))
                bps.add(int(e_idx[e]))
        bps = sorted(b for b in bps if 0 <= b <= _T_SIM)

        gamma = core["gamma"]
        p_ei = -math.expm1(-core["sigma"])
        p_ir = -math.expm1(-gamma)
        omega = core["waning"]
        p_wane = -math.expm1(-omega)
        waning_on = omega > 0.0
        u_dec = math.exp(-1.0 / max(core["icu_stay"], 1e-9))
        ifr = core["ifr"]
        icu_risk = core["icu_risk"]
        home = core["home"]
        sch = core["school"]
        wc = core["wc"]
        bN = core["bN"][None, :]
        bq = core["bq"]
        mask_mult = 1.0 - core["mask_eff"]
        hm = home * bN

        i_c = core["i0"].astype(float)
        s_c = n_pop - i_c
        e_c = np.zeros(k)
        q_c = np.zeros(k)
        r_c = np.zeros(k)
        d_c = np.zeros(k)
        doses = np.zeros(k)
        u_c = np.zeros(k)
        peak_tot = float(np.sum(i_c))
        peak_icu = 0.0
        icu_day = 0
        s_hist = np.zeros((_T_SIM + 1, k))
        sample = set(self._OBS_DAYS)
        prev = {}
        # src[j][g, h]: infections of g caused by h's cases in setting j
        # (0 home, 1 school, 2 work+community, 3 home from isolated cases)
        src = np.zeros((4, k, k))
        for a, b in zip(bps[:-1], bps[1:]):
            if b <= a:
                continue
            act = (s_idx[:k] <= a) & (a < e_idx[:k])
            keep = 1.0 - cr * act
            school_f = (_SCHOOL_FACTOR[code]
                        if (code > 0 and s_idx[k] <= a < e_idx[k]) else 1.0)
            mm = mask_mult if (mask_on and s_idx[k + 1] <= a < e_idx[k + 1]) else 1.0
            bmat = (home + (mm * school_f) * sch
                    + mm * (keep[:, None] * keep[None, :]) * wc) * bN
            sm = ((mm * school_f) * sch) * bN
            wm = (mm * (keep[:, None] * keep[None, :]) * wc) * bN
            tau = test * act
            p_out = -np.expm1(-(gamma + tau))
            qf = tau / (gamma + tau)
            dosing = vax_on and (vd <= a < ve)
            for t in range(a, b):
                s_hist[t] = s_c
                if t in sample:
                    prev[t] = (float(np.sum(i_c + q_c)) / n_total,
                               1e5 * float(np.sum(u_c)) / n_total)
                lam = bmat @ i_c + bq @ q_c
                p_inf = -np.expm1(-lam)
                new_e = s_c * p_inf
                if dosing:
                    new_v = np.minimum(rate, s_c - new_e)
                out_e = e_c * p_ei
                out_i = i_c * p_out
                to_q = out_i * qf
                out_q = q_c * p_ir
                rec = (out_i - to_q) + out_q
                new_d = rec * ifr
                wane = r_c * p_wane if waning_on else 0.0
                parts = (hm * i_c[None, :], sm * i_c[None, :],
                         wm * i_c[None, :], bq * q_c[None, :])
                ltot = (parts[0].sum(axis=1) + parts[1].sum(axis=1)
                        + parts[2].sum(axis=1) + parts[3].sum(axis=1))
                w = new_e / np.where(ltot > 0.0, ltot, 1.0)
                for j in range(4):
                    src[j] += w[:, None] * parts[j]
                if dosing:
                    s_c = s_c - new_e - new_v + wane
                    r_c = r_c + rec - new_d + new_v - wane
                    doses = doses + new_v
                else:
                    s_c = s_c - new_e + wane
                    r_c = r_c + rec - new_d - wane
                e_c = e_c + new_e - out_e
                i_c = i_c + out_e - out_i
                q_c = q_c + to_q - out_q
                d_c = d_c + new_d
                u_c = u_c * u_dec + icu_risk * out_e
                tot_i = float(np.sum(i_c + q_c))
                icu_now = float(np.sum(u_c))
                if tot_i > peak_tot:
                    peak_tot = tot_i
                if icu_now > peak_icu:
                    peak_icu = icu_now
                    icu_day = t + 1
        s_hist[_T_SIM] = s_c
        prev[_T_SIM] = (float(np.sum(i_c + q_c)) / n_total,
                        1e5 * float(np.sum(u_c)) / n_total)
        return {
            "cr": cr, "test": test, "code": code, "mask_on": mask_on,
            "s_idx": s_idx, "e_idx": e_idx, "vd": vd, "ve": ve,
            "vax_on": vax_on, "tot_share": tot_share,
            "offered": rate * float(max(ve - vd, 0)) if vax_on else np.zeros(k),
            "doses": doses, "s_hist": s_hist, "src": src,
            "prev": sorted((d, p, u) for d, (p, u) in prev.items()),
            "icu_day": icu_day, "peak_frac": peak_tot / n_total,
            "peak_icu": 1e5 * peak_icu / n_total,
        }

    @staticmethod
    def _obs_inforce(core: dict, tr: dict, t: int):
        """What is in force on day t: (contact_reduction per group, test rate
        per group, school factor, mask multiplier)."""
        k = int(core["k"])
        s_idx, e_idx = tr["s_idx"], tr["e_idx"]
        act = (s_idx[:k] <= t) & (t < e_idx[:k])
        code = tr["code"]
        sf = (_SCHOOL_FACTOR[code]
              if (code > 0 and s_idx[k] <= t < e_idx[k]) else 1.0)
        mm = ((1.0 - core["mask_eff"])
              if (tr["mask_on"] and s_idx[k + 1] <= t < e_idx[k + 1]) else 1.0)
        return tr["cr"] * act, tr["test"] * act, sf, mm

    def _obs_rt(self, core: dict, crf, tau, sf: float, mm: float, s) -> float:
        """R_t of a regime given what is in force and the susceptibles."""
        keep = 1.0 - np.asarray(crf, dtype=float)
        bmat = (core["home"] + (mm * sf) * core["school"]
                + mm * (keep[:, None] * keep[None, :]) * core["wc"]) * core["bN"][None, :]
        return self._rt(core, bmat, np.asarray(tau, dtype=float), s)

    def _obs_rt_day(self, core: dict, tr: dict, t: int) -> float:
        crf, tau, sf, mm = self._obs_inforce(core, tr, t)
        return self._obs_rt(core, crf, tau, sf, mm, tr["s_hist"][t])

    # ---------------------------------------------------------- describe
    # character budget of describe(): above it the per-group rows are
    # condensed (window R_t only in the R_t line, top-2 infection sources,
    # no per-group later-wave deaths); every group stays listed.
    _OBS_CHARS = 10800

    def describe(self, st, spec) -> str:
        """Self-contained text observation: PROBLEM, DESIGN, RESULTS."""
        txt = self._obs_render(st, spec, compact=False)
        if len(txt) > self._OBS_CHARS:
            txt = self._obs_render(st, spec, compact=True)
        return txt

    def _obs_render(self, st, spec, compact: bool) -> str:
        core = st["core"]
        k = int(core["k"])
        num, pct, ipct = self._num, self._pct, self._ipct

        def n4(x):
            return num(x, 4)

        n_total = float(core["n_total"])
        n_pop = core["n_pop"]
        frac = core["frac"]
        family = str(spec.get("family", core.get("family", "unknown")))
        fam_r0 = {"flu_like": "1.4-1.9", "covid_like": "2.5-3.8",
                  "covid_waning": "2.5-3.8", "measles_like": "6-10"}
        waning = float(core["waning"])
        gamma = float(core["gamma"])
        p_ir = float(core["p_ir"])
        m, aux = self._detail(st)
        tr = self._obs_trace(st)
        s_idx, e_idx = tr["s_idx"], tr["e_idx"]
        s_hist = tr["s_hist"]
        code = int(tr["code"])
        mask_on = bool(tr["mask_on"])
        vd, ve = int(tr["vd"]), int(tr["ve"])
        vax_on = bool(tr["vax_on"])
        goals = dict(spec.get("goals") or core.get("spec_goals") or {})
        L = []
        # ------------------------------------------------------------ PROBLEM
        L.append("PROBLEM")
        L.append("Age-structured SEIR epidemic control, family %s (R0 range %s): "
                 "%d age groups (elements 0-%d), schools (element %d), mask "
                 "mandate (element %d); total population %d."
                 % (family, fam_r0.get(family, "?"), k, k - 1, k, k + 1,
                    int(round(n_total))))
        L.append("Horizon: measures run only in days 0-179; the simulation "
                 "continues to day %d with everything lifted. "
                 "peak_infected_frac and peak_icu_per_100k are taken over ALL "
                 "days 0-%d (a rebound after the measures end counts). "
                 "total_deaths_per_1k adds the deaths of one later wave after "
                 "day %d (disease reintroduced once, nothing in force, the "
                 "remaining susceptibles infected as in a free epidemic, "
                 "immunity assumed permanent): deaths only, no peak or ICU."
                 % (_T_SIM, _T_SIM, _T_SIM))
        imm = ("permanent" if waning <= 0.0 else
               "wanes at %s/day (mean %s days) for recovered and vaccinated"
               % (num(waning), num(1.0 / waning)))
        L.append("Disease: R0 %s, latent %s d, infectious %s d (gamma = %s/day; "
                 "in the 1-day chain an undetected case is infectious for a "
                 "mean of 1/(1-exp(-gamma)) = %s days), immunity %s. Day 0: %s "
                 "infectious, everyone else susceptible. One time step = 1 day."
                 % (num(core["r0"]), num(1.0 / core["sigma"]),
                    num(1.0 / gamma), num(gamma), num(1.0 / p_ir), imm,
                    ", ".join("%d in g%d" % (int(core["i0"][g]), g)
                              for g in range(k) if core["i0"][g] > 0)))
        lay = spec.get("layers") or {}

        def assort(name):
            try:
                return num(lay[name]["assort"])
            except (KeyError, TypeError):
                return "?"

        L.append("Contacts/day come from four settings: home (never reduced), "
                 "school (cut by the school code and masks), work and community "
                 "(cut by each group's contact_reduction and masks); masks cut "
                 "school, work and community transmission by %s. In each "
                 "setting a share a of contacts is within one's own age group, "
                 "the rest spread over all groups in proportion to "
                 "participation: a = home %s, "
                 "school %s, work %s, community %s. School: ages 5-19 plus "
                 "staff (4%% of ages 20-64); work: 70%% of ages 20-64; community "
                 "contacts decline with age."
                 % (pct(core["mask_eff"]), assort("home"), assort("school"),
                    assort("work"), assort("community")))
        if k <= 4:
            L.append("Work+community contacts/day of a member of g with members "
                     "of h (h = 0..%d): %s." % (k - 1, "; ".join(
                         "g%d: %s" % (g, "/".join(num(x) for x in core["wc"][g]))
                         for g in range(k))))
        per_day = core["supply"] / float(core["rollout"])
        clipped = vd + int(core["rollout"]) > _T_POLICY
        w_dose = float(core["w_vax"]) / n_total
        L.append("Vaccine (delivery schedule fixed; only vax_share is tunable): "
                 "%d doses (%s of population), %s per day on days %d-%d%s. "
                 "Group g is offered vax_share_g / max(1, sum of all vax_share) "
                 "of each day's doses, given to susceptible members only (one "
                 "dose = immune); doses g cannot use that day are discarded "
                 "unpaid; shares summing below 1 leave the rest unordered. "
                 "Cost = %s x doses given / N = %s per dose given (the whole "
                 "supply would cost %s)."
                 % (int(core["supply"]), pct(core["supply_frac"]),
                    num(per_day), vd, ve - 1,
                    (" (deliveries end with the policy period, so at most %d "
                     "doses arrive)" % int(per_day * (ve - vd)))
                    if clipped else "", num(core["w_vax"]), num(w_dose),
                    num(w_dose * min(core["supply"], per_day * max(ve - vd, 0)))))
        L.append("Outcomes: an infection in g ends in death with probability "
                 "IFR_g (when it ends, isolated or not); a share icu_risk_g of "
                 "g's infections enters ICU on the day it becomes infectious "
                 "(before any detection) and stays %s days on average (ICU "
                 "census = people in ICU now)." % num(core["icu_stay"]))
        if spelled_in_verbs(self):
            # the same facts as the SET_PARAM block below, in the verbs the
            # policy is served (docstring "Verb surface"); the SET_PARAM block
            # again under the set_parameter spelling
            L.append("Tunable, one change per turn, written as "
                     "SET_CONTACT_REDUCTION(g, share), SET_TESTING_RATE(g, rate), "
                     "SET_VACCINE_SHARE(g, share), SET_INTERVENTION_START(e, day), "
                     "SET_INTERVENTION_DURATION(e, days), SET_SCHOOL_POLICY(%d, "
                     "policy), MANDATE_MASKS(%d) or LIFT_MASKS(%d); g is the bare "
                     "group number (g2 -> 2), e a group number, %d (schools) or %d "
                     "(mask mandate). A value is the new ABSOLUTE setting, clamped "
                     "to the bounds; days are rounded to whole numbers; moving the "
                     "window of an open school, a lifted mandate, or a group whose "
                     "contact reduction and testing rate are both 0 changes "
                     "nothing. Per age group g (elements 0-%d):"
                     % (k, k + 1, k + 1, k, k + 1, k - 1))
            L.append("  SET_CONTACT_REDUCTION(g, share), share in [0, 0.9]: share of "
                     "g's work+community contacts removed while g's window runs; a "
                     "g-h link keeps (1-cr_g)(1-cr_h). Cost per window day = %s x "
                     "(N_g/N) x share^2." % num(core["w_closure"]))
            L.append("  SET_INTERVENTION_START(g, day), day in [0, 179], and "
                     "SET_INTERVENTION_DURATION(g, days), days in [0, 180] (whole "
                     "days): g's window is days start .. min(start+duration, "
                     "180)-1; contact reduction and testing act only inside it.")
            L.append("  SET_TESTING_RATE(g, rate), rate in [0, 0.5]/day: inside g's "
                     "window, g's infectious members are detected at this rate "
                     "(share detected = rate/(rate+gamma), gamma = %s); an "
                     "undetected case then stays infectious a mean "
                     "1/(1-exp(-(gamma+rate))) days. A detected case is isolated "
                     "for a further mean %s days: it infects only at home at half "
                     "the household rate, still counts in peak_infected_frac, and "
                     "keeps its IFR and ICU risk; so testing g cuts the infections "
                     "g's cases cause in others, not g's own risk. Cost per window "
                     "day = %s x (N_g/N) x rate."
                     % (num(gamma), num(1.0 / p_ir), num(core["w_test"])))
            L.append("  SET_VACCINE_SHARE(g, share), share in [0, 1]: g's share of "
                     "the vaccine supply (see Vaccine).")
            L.append("Schools (element %d): SET_SCHOOL_POLICY(%d, policy) with policy "
                     "open (code 0, school contacts x1), small_cohorts (1, x0.6), "
                     "hybrid (2, half attendance, x0.35) or closed (3, remote "
                     "learning, x0.05); cost per day %s x (0, 0.12, 0.45, "
                     "1.0)[code] (schools hold %s of the population); "
                     "SET_INTERVENTION_START(%d, day) / SET_INTERVENTION_DURATION(%d, "
                     "days) as for groups."
                     % (k, k, num(core["w_school"] * core["school_share"]),
                        pct(core["school_share"]), k, k))
            L.append("Mask mandate (element %d): MANDATE_MASKS(%d) puts it in force, "
                     "LIFT_MASKS(%d) lifts it; cost %s per day in force; "
                     "SET_INTERVENTION_START(%d, day) / SET_INTERVENTION_DURATION(%d, "
                     "days) as for groups."
                     % (k + 1, k + 1, k + 1, num(core["w_mask"]), k + 1, k + 1))
        else:
            L.append("Tunable (a change sets the new ABSOLUTE value, clamped to the "
                     "bounds; days and the school code are rounded to whole "
                     "numbers, mask_mandate to 0/1; moving the window of an open "
                     "school, an off mask, or a group whose contact_reduction and "
                     "test_rate are both 0 changes nothing). Per age group g "
                     "(elements 0-%d):" % (k - 1))
            L.append("  contact_reduction [0, 0.9]: share of g's work+community "
                     "contacts removed while g's window runs; a g-h link keeps "
                     "(1-cr_g)(1-cr_h). Cost per window day = %s x (N_g/N) x cr^2."
                     % num(core["w_closure"]))
            L.append("  start_day [0, 179], duration [0, 180] (whole days): g's "
                     "window is days start_day .. min(start_day+duration, 180)-1; "
                     "contact_reduction and test_rate act only inside it.")
            L.append("  test_rate [0, 0.5]/day: inside g's window, g's infectious "
                     "members are detected at this rate (share detected = "
                     "test/(test+gamma), gamma = %s); an undetected case then stays "
                     "infectious a mean 1/(1-exp(-(gamma+test))) days. A detected "
                     "case is isolated for a further mean %s days: it infects only "
                     "at home at half the household rate, still counts in "
                     "peak_infected_frac, and keeps its IFR and ICU risk; so "
                     "testing g cuts the infections g's cases cause in others, not "
                     "g's own risk. Cost per window day = %s x (N_g/N) x test_rate."
                     % (num(gamma), num(1.0 / p_ir), num(core["w_test"])))
            L.append("  vax_share [0, 1]: g's share of the vaccine supply (see "
                     "Vaccine).")
            L.append("Schools (element %d): school_policy code 0 open (school "
                     "contacts x1), 1 small cohorts (x0.6), 2 hybrid half "
                     "attendance (x0.35), 3 closed / remote (x0.05); cost per day "
                     "%s x (0, 0.12, 0.45, 1.0)[code] (schools hold %s of the "
                     "population); start_day/duration as for groups."
                     % (k, num(core["w_school"] * core["school_share"]),
                        pct(core["school_share"])))
            L.append("Mask mandate (element %d): mask_mandate 0 off / 1 on; cost %s "
                     "per day in force; start_day/duration as for groups."
                     % (k + 1, num(core["w_mask"])))
        L.append("Total cost (minimise) = %s + closures + testing + schools + "
                 "masks + vaccine." % num(_BASE_COST))
        # ICU risk is IFR x one multiplier unless capped, and everyone has the
        # same home contacts: say each once instead of per row when it holds
        ifr = np.asarray(core["ifr"], dtype=float)
        icu_r = np.asarray(core["icu_risk"], dtype=float)
        icu_mult = float(icu_r[0] / ifr[0]) if ifr[0] > 0 else 0.0
        same_icu = icu_mult > 0 and bool(np.allclose(icu_r, icu_mult * ifr,
                                                     rtol=1e-9, atol=0.0))
        home_c = core["home"].sum(axis=1)
        same_home = bool(np.allclose(home_c, home_c[0], rtol=1e-9, atol=0.0))
        L.append("GROUPS (g ages: size [share], IFR%s, contacts/day %s"
                 "school/work/community)%s%s:"
                 % ("" if same_icu else ", ICU risk per infection",
                    "" if same_home else "home/",
                    ("; ICU risk per infection = %s x IFR" % num(icu_mult))
                    if same_icu else "",
                    ("; home contacts %s/day for everyone" % num(home_c[0]))
                    if same_home else ""))
        for g in range(k):
            ages = self._ages(core, g)
            L.append("  g%d %s: N %d [%s], IFR %s%s, contacts %s%s/%s/%s"
                     % (g, ages[5:] if compact else ages, int(n_pop[g]),
                        pct(frac[g]), num(ifr[g]),
                        "" if same_icu else ", ICU %s" % num(icu_r[g]),
                        "" if same_home else num(home_c[g]) + "/",
                        num(core["school"][g].sum()),
                        num(core["work"][g].sum()), num(core["comm"][g].sum())))
        # ------------------------------------------------------------- DESIGN
        L.append("DESIGN")

        def win(e):
            if e_idx[e] <= s_idx[e]:
                return "no window (duration 0)"
            return "days %d-%d" % (int(s_idx[e]), int(e_idx[e]) - 1)

        def win_rt(e):
            if compact or e_idx[e] <= s_idx[e]:
                return ""
            return " [R_t %s -> %s]" % (
                num(self._obs_rt_day(core, tr, int(s_idx[e]))),
                num(self._obs_rt_day(core, tr, int(e_idx[e]))))

        tot_share = float(tr["tot_share"])
        L.append("Vaccine shares sum to %s%s." % (
            n4(tot_share),
            " (above 1: scaled down proportionally)" if tot_share > 1.0 + 1e-9
            else " (below 1: only that share of the supply is ordered)"
            if tot_share < 1.0 - 1e-9 else ""))
        # the share of supply offered equals vax_share unless shares sum > 1
        scaled = tot_share > 1.0 + 1e-9
        vax_txt = ("vax_share%s = doses offered (used) [share of g still "
                   "susceptible at the start of day %d / day %d]"
                   % (" -> share of supply offered" if scaled else "", vd, ve)
                   if vax_on else "vax_share (no doses delivered)")
        rt_txt = ("" if compact else " [R_t of the whole population on the "
                  "window's first day -> on the day after it ends]")
        L.append("Per group: contact_reduction, test_rate, start_day, duration "
                 "-> window%s | %s%s:"
                 % (rt_txt, vax_txt,
                    "" if compact else " | closure cost, testing cost"))
        for g in range(k):
            used = float(tr["cr"][g]) > 0.0 or float(tr["test"][g]) > 0.0
            w_txt = win(g)
            if used:
                w_txt += win_rt(g)
            elif e_idx[g] > s_idx[g]:
                w_txt += " (unused: contact_reduction and test_rate are 0)"
            if vax_on:
                sus = ((ipct(s_hist[vd][g], n_pop[g]) + "%",
                        ipct(s_hist[ve][g], n_pop[g]) + "%") if compact else
                       (pct(s_hist[vd][g] / n_pop[g]),
                        pct(s_hist[ve][g] / n_pop[g])))
                v_txt = "%s%s = %d (%d) [%s / %s]" % (
                    num(st["vax"][g]),
                    (" -> " + pct(aux["alloc"][g])) if scaled else "",
                    int(round(float(tr["offered"][g]))),
                    int(round(float(tr["doses"][g]))), sus[0], sus[1])
            else:
                v_txt = num(st["vax"][g])
            cost_txt = "" if compact else " | %s, %s" % (
                num(aux["closure_g"][g]), num(aux["testing_g"][g]))
            L.append("  g%d: %s, %s, %d, %d -> %s | %s%s"
                     % (g, num(st["cr"][g]), num(st["test"][g]),
                        int(st["start"][g]), int(st["dur"][g]), w_txt, v_txt,
                        cost_txt))
        L.append("  schools (element %d): school_policy %d (%s), start_day %d, "
                 "duration %d -> %s | cost %s"
                 % (k, code, _SCHOOL_NAMES[code], int(st["start"][k]),
                    int(st["dur"][k]),
                    (win(k) + win_rt(k)) if code > 0 else
                    "inactive while open (would apply on %s)" % win(k),
                    num(aux["costs"]["schools"])))
        L.append("  mask mandate (element %d): mask_mandate %d, start_day %d, "
                 "duration %d -> %s | cost %s"
                 % (k + 1, int(mask_on), int(st["start"][k + 1]),
                    int(st["dur"][k + 1]),
                    (win(k + 1) + win_rt(k + 1)) if mask_on else
                    "inactive while off (would apply on %s)" % win(k + 1),
                    num(aux["costs"]["masks"])))
        # ------------------------------------------------------------ RESULTS
        L.append("RESULTS")
        if self.stochastic:
            L.append("Figures below are a NOMINAL deterministic evaluation. "
                     "Feasibility is judged on the 90th percentile over random "
                     "replicates (the values under CONSTRAINT STATUS), which sit "
                     "above these nominal values, most of all for small death "
                     "and ICU counts; keep a margin below each cap.")
        c = aux["costs"]
        L.append("Cost %s = closures %s + testing %s + schools %s + masks %s + "
                 "vaccine %s (+%s)." % (n4(m["cost"]), num(c["closure"]),
                                        num(c["testing"]), num(c["schools"]),
                                        num(c["masks"]), num(c["vaccine"]),
                                        num(_BASE_COST)))

        def cap_txt(key):
            cap = goals.get(_GOAL_KEYS[key])
            if cap is None:
                return ""
            cap = float(cap)
            ratio = float(m[key]) / cap if cap > 0 else float("inf")
            return " (cap %s; value/cap %s)" % (n4(cap), num(ratio))

        L.append("Constrained metrics: peak_infected_frac %s (%s of population "
                 "infectious incl. isolated) on day %d%s; peak_icu_per_100k %s "
                 "on day %d%s; total_deaths_per_1k %s = %s by day %d + %s from "
                 "the later wave%s. Doses given %s per capita; susceptible at "
                 "day %d: %s."
                 % (n4(m["peak_infected_frac"]), pct(m["peak_infected_frac"]),
                    int(m["peak_day"]), cap_txt("peak_infected_frac"),
                    n4(m["peak_icu_per_100k"]), int(tr["icu_day"]),
                    cap_txt("peak_icu_per_100k"),
                    n4(m["total_deaths_per_1k"]),
                    n4(m["total_deaths_per_1k"] - m["projected_deaths_per_1k"]),
                    _T_SIM, num(m["projected_deaths_per_1k"]),
                    cap_txt("total_deaths_per_1k"),
                    num(m["doses_per_capita"]), _T_SIM,
                    pct(m["susceptible_end_frac"])))
        sig_t = 2 if compact else 3
        L.append("Timeline at the start of each day: infectious incl. isolated "
                 "(%% of pop) | ICU census per 100k: %s."
                 % ", ".join("d%d %s%%|%s" % (d, num(100.0 * p, sig_t),
                                               num(u, sig_t))
                             for d, p, u in tr["prev"]))
        # R_t at every change of what is in force
        ev = {}

        def add(day, txt):
            ev.setdefault(int(day), []).append(txt)

        for g in range(k):
            if e_idx[g] > s_idx[g] and (float(tr["cr"][g]) > 0.0
                                         or float(tr["test"][g]) > 0.0):
                add(s_idx[g], "+g%d" % g)
                add(e_idx[g], "-g%d" % g)
        if code > 0 and e_idx[k] > s_idx[k]:
            add(s_idx[k], "+school")
            add(e_idx[k], "-school")
        if mask_on and e_idx[k + 1] > s_idx[k + 1]:
            add(s_idx[k + 1], "+mask")
            add(e_idx[k + 1], "-mask")
        if vax_on:
            add(vd, "+vaccine")
            add(ve, "-vaccine")
        per = []
        for d in sorted(set(ev) | {0, _T_POLICY}):
            tag = ",".join(ev.get(d, [])) or ("none" if d == 0 else "")
            if d == _T_POLICY:
                tag = (tag + ": " if tag else "") + "all lifted"
            per.append("d%d %s [%s]" % (d, num(self._obs_rt_day(core, tr, d)), tag))
        per.append("d%d %s" % (_T_SIM, num(self._obs_rt_day(core, tr, _T_SIM))))
        L.append("R_t on each day what is in force changes (+ starts, - ends; "
                 "that regime with that day's susceptibles): %s."
                 % "; ".join(per))
        # levers: R_t on day T with one change to what is in force
        t_lev = int(min(max(0, int(m["peak_day"]) - self._OBS_LEAD), _T_POLICY - 1))
        crf, tau, sf, mm = self._obs_inforce(core, tr, t_lev)
        s_t = s_hist[t_lev]
        rt_now = self._obs_rt(core, crf, tau, sf, mm, s_t)
        rt_mask = [self._obs_rt(core, crf, tau, sf, mmx, s_t)
                   for mmx in (1.0, 1.0 - core["mask_eff"])]
        school_now = (code if (code > 0 and s_idx[k] <= t_lev < e_idx[k])
                      else 0)
        rt_school = [self._obs_rt(core, crf, tau, _SCHOOL_FACTOR[cc], mm, s_t)
                     for cc in range(4)]

        def star(vals, cur):
            return "/".join(num(v) + ("*" if j == cur else "")
                            for j, v in enumerate(vals))

        def dlt(x):
            x = float(x) - rt_now
            return "0" if abs(x) < 5e-5 * max(rt_now, 1e-9) else (
                ("+" if x > 0 else "") + num(x, 2))

        glev = []
        for g in range(k):
            if float(tr["cr"][g]) > 0.0 or float(tr["test"][g]) > 0.0:
                on_c, on_t = crf.copy(), tau.copy()
                on_c[g], on_t[g] = tr["cr"][g], tr["test"][g]
                off_c, off_t = crf.copy(), tau.copy()
                off_c[g], off_t[g] = 0.0, 0.0
                eff = (self._obs_rt(core, on_c, on_t, sf, mm, s_t)
                       - self._obs_rt(core, off_c, off_t, sf, mm, s_t))
                tog = dlt(rt_now + eff) + (
                    "*" if s_idx[g] <= t_lev < e_idx[g] else "")
            else:
                tog = "-"
            c2 = crf.copy()
            c2[g] = min(_CR_MAX, float(c2[g]) + 0.1)
            t2 = tau.copy()
            t2[g] = min(_TEST_MAX, float(t2[g]) + 0.05)
            glev.append("g%d %s/%s/%s" % (
                g, tog, dlt(self._obs_rt(core, c2, tau, sf, mm, s_t)),
                dlt(self._obs_rt(core, crf, t2, sf, mm, s_t))))
        L.append("Levers on day %d (the infectious peak day %d minus %d, at "
                 "most day 179): R_t then is %s. R_t that day with ONE change to "
                 "what is in force (* = current): mask off/on %s; school code "
                 "0/1/2/3 %s. Per group g, change in that day's R_t from (a) "
                 "g's current contact_reduction and test_rate in force vs not "
                 "(* = in force now, - = both 0) / (b) in-force "
                 "contact_reduction +0.1 / (c) in-force test_rate +0.05: %s."
                 % (t_lev, int(m["peak_day"]), self._OBS_LEAD, num(rt_now),
                    star(rt_mask, 1 if mm < 1.0 else 0),
                    star(rt_school, school_now), "; ".join(glev)))
        # where and by whom infections happened
        src = tr["src"]
        tot_set = [float(np.sum(src[j])) for j in range(4)]
        tot_inf = sum(tot_set)
        L.append("Where infections happened (days 0-%d): home %s%%, school %s%%, "
                 "work+community %s%%, at home from isolated cases %s%%."
                 % (_T_SIM, ipct(tot_set[0], tot_inf), ipct(tot_set[1], tot_inf),
                    ipct(tot_set[2], tot_inf), ipct(tot_set[3], tot_inf)))
        by_tgt = src.sum(axis=0)                     # [g, h]: g infected by h
        caused = by_tgt.sum(axis=0)
        caused_tot = float(np.sum(caused))
        deaths_g = aux["deaths_sim"] + aux["deaths_proj"]
        d_tot = float(np.sum(deaths_g))
        icu_pk = aux["icu_at_peak"]
        icu_tot = float(np.sum(icu_pk))
        # infected by: every source group when K <= 4; otherwise own / younger
        # / older groups plus the largest other sources
        # columns; the compact form (large K, see _OBS_CHARS) drops the own
        # peak, the ICU share (it tracks the deaths share), the later-wave
        # deaths, the largest other source and the setting split of the
        # infections g's cases caused
        cols = []
        if not compact:
            cols.append("own peak infectious (share of g, day)")
        cols += ["share of all infectious at the overall peak",
                 "infections by day %d per member of g (incl. reinfections; "
                 "can exceed 100%% when immunity wanes)" % _T_SIM]
        if not compact:
            cols.append("share of its infections detected and isolated")
        cols.append("deaths per 1k of total pop (share of all deaths%s)"
                    % ("" if compact else "; from the later wave"))
        if not compact:
            cols.append("share of the ICU census at the ICU peak")
        cols.append("where g was infected, home/school/work+comm %")
        if k <= 4:
            cols.append("infected by (source groups)")
        else:
            cols.append("infected by own/younger/older groups %%%s"
                        % ("" if compact else " (largest other sources)"))
        cols.append("share of all infections caused by g's cases%s"
                    % ("" if compact else
                       " (where: home/school/work+comm %)"))
        L.append("Per group: %s:" % " | ".join(cols))
        for g in range(k):
            cum = float(aux["cum"][g])
            row = by_tgt[g]
            row_tot = float(np.sum(row))
            parts = []
            if not compact:
                parts.append("%s d%d" % (pct(aux["own_peak"][g]),
                                         int(aux["own_day"][g])))
            d_txt = "%s (%s" % (num(1000.0 * deaths_g[g] / n_total),
                                pct(deaths_g[g] / d_tot) if d_tot > 0 else "0%")
            if not compact:
                d_txt += "; %s" % num(1000.0 * aux["deaths_proj"][g] / n_total)
            parts += [pct(aux["peak_share"][g]), pct(cum / n_pop[g])]
            if not compact:
                parts.append(pct(aux["isolated"][g] / cum) if cum > 0 else "0%")
            parts.append(d_txt + ")")
            if not compact:
                parts.append(pct(icu_pk[g] / icu_tot) if icu_tot > 0 else "0%")
            parts.append("/".join(ipct(x, row_tot) for x in (
                src[0][g].sum() + src[3][g].sum(), src[1][g].sum(),
                src[2][g].sum())))
            if k <= 4:
                parts.append(", ".join("g%d %s%%" % (h, ipct(row[h], row_tot))
                                       for h in range(k)))
            else:
                by_txt = "%s/%s/%s" % (
                    ipct(row[g], row_tot), ipct(np.sum(row[:g]), row_tot),
                    ipct(np.sum(row[g + 1:]), row_tot))
                if not compact:
                    others = [int(h) for h in np.argsort(-row, kind="stable")
                              if int(h) != g][:2]
                    by_txt += " (%s)" % ", ".join(
                        "g%d %s%%" % (h, ipct(row[h], row_tot)) for h in others)
                parts.append(by_txt)
            c_txt = pct(caused[g] / caused_tot) if caused_tot > 0 else "0%"
            if not compact:
                c_txt += " (%s)" % "/".join(ipct(x, caused[g]) for x in (
                    src[0][:, g].sum() + src[3][:, g].sum(),
                    src[1][:, g].sum(), src[2][:, g].sum()))
            parts.append(c_txt)
            L.append("  g%d: %s" % (g, " | ".join(parts)))
        return "\n".join(L)
