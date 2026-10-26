"""Grid2Op / L2RPN power-grid operation (dataxp native-verb domain "grid2op").

The simulator is Grid2Op (RTE, https://grid2op.readthedocs.io), the engine of
the "Learning to Run a Power Network" (L2RPN) competitions, with the
lightsim2grid AC power-flow backend. Nothing here is a re-implementation:
every evaluation is a real grid2op step on a real L2RPN environment, and
grid2op's own legality rules (cooldowns, one substation and one line status
per step, ramp limits, curtailment and storage limits) are the validity
envelope of the verbs.

What the system is (from the Grid2Op modelling documentation)
--------------------------------------------------------------
A transmission grid is a set of substations joined by power lines. Each
substation has TWO busbars; every element connected to it (a line end, a
generator, a load, a storage unit) sits on one of them. Elements on the same
busbar are electrically joined; elements on different busbars of the same
substation are not. The reference topology has everything on busbar 1.
Splitting a substation (moving some of its elements to busbar 2) changes the
meshing of the grid and reroutes power - the signature L2RPN remedial action,
which costs nothing. An element alone on a busbar is isolated: an isolated
load, generator or storage unit makes the power flow diverge (blackout); an
isolated line end carries no power.

Time passes in 5-minute steps. At every step the loads and the renewable
production take the values of the environment's time series (an exogenous
schedule the operator does not control), the dispatchable generators follow
their schedule plus the operator's redispatch, and an AC power flow gives the
current on every line. rho_l = current / thermal limit of line l; rho > 1 is
an overload. The protections disconnect a line that has been overloaded for
more than NB_TIMESTEP_OVERFLOW_ALLOWED consecutive steps, and immediately when
rho >= HARD_OVERFLOW_THRESHOLD (2.0); a line tripped by the protections cannot
be reconnected for NB_TIMESTEP_RECONNECTION steps. After the operator changes a
line's status, that line is frozen for NB_TIMESTEP_COOLDOWN_LINE steps; after
a substation's busbars are changed, that substation is frozen for
NB_TIMESTEP_COOLDOWN_SUB steps. One action changes at most one substation's
topology and at most one line's status (MAX_SUB_CHANGED = MAX_LINE_STATUS_CHANGED
= 1). The values are per environment and are printed in describe().

Redispatching moves output between dispatchable units, bounded per step by
each unit's ramp rate and by [pmin, pmax]; the deviation from the schedule
(target dispatch) persists until it is reversed. Curtailment caps a renewable
unit at a fraction of its pmax; the energy it then cannot deliver is produced
by the dispatchable units instead (grid2op redispatches them to balance, and
that compensation counts as redispatch). A storage unit absorbs (charges) or
injects (discharges) a set power for one step, bounded by its power rating,
its energy capacity and its efficiencies; the dispatchable units again cover
the balance.

grid2op's balancing rule (Environment/baseEnv.py, _make_redisp): the energy a
CURTAIL or SET_STORAGE order takes out of (or puts into) the grid, together
with the step's own load and renewable drift, must be covered by the
dispatchable units within their ramp rates and [pmin, pmax] at that step. An
order the units cannot balance ends the episode: grid2op raises
ImpossibleRedispatching and the step is a game over ("a generator would be
damaged in real life"). The parameter LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_
ACTION, which would clamp such an order to what the units can absorb, is
False in all four environments used here (checked on their parameters), so
the order is taken as written and the game ends. Evidence: CURTAIL(2, 0.3)
on a case14 contingency problem (wind unit at 48.6 MW cut to 30% of pmax,
24.8 MW off at once) asked the three dispatchable units for 30.7 MW while
their ramps allow 30 MW: done, failed_redispatching=True. The blackout
metric covers this case as well as a diverged power flow: blackout = 1 means
the episode ended, whether the AC power flow diverged (an isolated load,
generator or storage unit, a non-connected grid) or the dispatch could not be
balanced after a curtailment / storage order; the observation after such a
move names the cause (and grid2op's own account of the MW asked for versus
the MW available). A blackout is outside the model's validity.

Verbs (one grid2op action = one 5-minute step; DO_NOTHING also advances time)
------------------------------------------------------------------------------
  DO_NOTHING()                      let the step pass
  RECONNECT_LINE(l)                 put line l back in service (both ends on busbar 1)
  DISCONNECT_LINE(l)                open line l
  SET_BUS(sub, element, bus)        move one element end of substation sub to busbar 1 or 2
  SPLIT_SUB(sub, bus2)              set substation sub's topology in one action: the listed
                                    elements on busbar 2, every other element on busbar 1
  MERGE_SUB(sub)                    every element of substation sub back on busbar 1
  REDISPATCH(g_from, g_to, mw)      move mw MW of output from unit g_from to unit g_to
  CURTAIL(g, frac)                  cap renewable unit g at frac x pmax (1.0 lifts the cap)
  SET_STORAGE(s, mw)                storage s absorbs mw MW this step (negative = injects)
Element names: line<l>_or (origin end), line<l>_ex (extremity end), gen<g>,
load<c>, stor<s>. SET_BUS carries the substation as well as the element: the
move is refused when the element is not at that substation.

A line's status is changed ONLY by RECONNECT_LINE and DISCONNECT_LINE (and by
the protections / maintenance). The two ends of a line out of service have no
busbar: SET_BUS refuses them, a SPLIT_SUB bus2 list naming one is refused, and
SPLIT_SUB / MERGE_SUB leave them untouched (grid2op value 0 = unchanged) while
they set every other element of the substation. This departs from raw grid2op,
where a busbar assignment on an end of a disconnected line reconnects that line
(Action/baseAction.py, _aux_get_topo_impact_update_with_powerline_status):
that path is illegal under a line cooldown / maintenance / protection trip, and
it would bypass the problem's forced outage (which grid2op knows nothing of) on
the grids without a line cooldown. With the ends left at 0 a topology verb can
never change a line's status; apply() additionally checks it after the step.

Problems (make_problem; 4 families x 4 environments)
----------------------------------------------------
A problem is an environment, a time series (chronic), a start step t0 and a
short list of pre-actions applied at the steps before t0 that put the grid
in the state the operator inherits. Environments (grid2op's bundled test
data, no download): l2rpn_case14_sandbox (14 substations, 20 lines, 3 chronics
of 575 steps, no cooldowns), educ_case14_storage (14 substations + 2 storage
units, 7 chronics of 288 steps, cooldowns 3), l2rpn_neurips_2020_track1 (36
substations, 59 lines, 2 chronics of 864 steps, cooldowns 3) and l2rpn_idf_2023
(118 substations, 186 lines, 7 storage units, 2 chronics of 575 steps,
cooldowns 3, scheduled maintenance). The opponent (random line attacks) of the
L2RPN environments is disabled so that every replay is exact; contingencies
are injected explicitly instead. Families:
  overload      a step of the time series at which the reference topology is
                overloaded on its own (1 < rho_max <= 1.6; case36 and case118
                only - the 14-bus chronics never overload by themselves)
  contingency   one line is tripped at t0-1 (the outage that overloads the grid
                most, among those that do not black it out); the tripped line is
                a forced outage for K = 2..6 steps (as grid2op does after a
                protection trip) and must be reconnected afterwards
  topology      a substation was badly split at t0-1 (a random 2-busbar partition
                that overloads a line); with cooldowns the substation is frozen
                for 3 steps so the overload must first be managed another way
  dispatch      a standing redispatch of 3..8 steps at the ramp rate (the pair
                of units that loads the grid most) was applied before t0, and
                at t0-1 the line whose outage overloads the so-loaded grid was
                tripped (forced outage K = 2..6 steps, as in contingency). The
                deviation costs its premium every hour until it is unwound.
                Redispatch alone never overloads these grids (10 ramp-steps
                reach max rho 0.90 / 0.80 / 0.82 / 1.03 on the four grids), so
                a pure-redispatch family would always start feasible.
A problem set has no duplicate spec: the module keeps the (family, env,
chronic, t0, fault) of every problem built in the process and make_problem
re-draws its start on a repeat (problems are built in index order, so this is
reproducible from a fresh process).
Goals (minimum_/maximum_ convention): maximum_max_rho = 1.0, maximum_overload_sum
= 0 (sum over lines of max(0, rho - 1): the same feasible set, but it credits
relieving any overloaded line, not just the worst), maximum_lines_out = 0 (every
line the operator can put back is in service; lines in maintenance or under a
forced outage do not count until they become available). Objective
operating_cost [$/h] = energy cost (sum over units of output x price) +
redispatch cost (sum of |redispatch| x the unit's price: the balancing premium
paid for ordering a unit away from its schedule, in either direction) + storage
cost (|storage power| x the storage marginal cost), minimised. Validity
envelope: blackout = 0 and max_rho <= 8 (an ended episode - diverged or
islanded power flow, or an infeasible redispatch after a CURTAIL /
SET_STORAGE order - returns FINITE metrics outside the envelope: max_rho 10,
blackout 1, cost 10 x ref).
reference(spec) = the energy cost of the time series' own schedule at t0
(zero redispatch), reference_class "heuristic": it is the floor of the
operating cost only up to the drift of the load during the window (the
objective is read at a later step than the reference), so a policy's obj/ref
can fall below 1 (the ladder's greedy median was 0.86) - it is a reference
level, not a bound.

Time within an episode: the agent's first observation is step t0; every move
advances one step; the ladder's turn budget (8 quick / 20 full) is the window.
The domain is time-stepping (advances_time = True, null_move = DO_NOTHING();
docs/corpus_native_domains.md, "Time-stepping domains"): when no enumerated
move improves on the incumbent, the ladder applies DO_NOTHING so the chronic
advances instead of the same step being scored and harvested again, and
harvest drops exact duplicate pairs by (problem, step, chosen, rejected).
evaluate(st, seed) ignores seed: the environment is deterministic once the
opponent is off, and a state is (chronic, t0, pre-actions, move history),
replayed from the chronic start when it is first needed (reset, fast-forward,
pre-actions, moves). States that were reached by apply() keep their own
env.copy(), so scoring a child is one env copy (12-25 ms) and one step (1 ms);
the copy was checked to reproduce the original step for step (rho, flows,
statuses bit-equal).

Candidate moves (legal_moves): the whole vocabulary is enumerated at every
state in which it applies, restricted to a neighbourhood ON EVERY GRID (a
sizing rule, the same on the 14-bus grids as on case118) -
SET_BUS / SPLIT_SUB / MERGE_SUB at the substations at the ends of the 3 most
loaded lines, every split substation and 2 random substations (random
balanced partitions of the substation's LIVE elements, 3 per substation);
DISCONNECT_LINE for the 8 most loaded lines; REDISPATCH between up to 24
(from, to) pairs drawn among the 6 largest producers and the 6 units with most
headroom plus random pairs, at the full and half ramp, the MW quantised down
to 0.1 MW and dropped below 0.5 MW; CURTAIL for the 8 largest renewable
producers (half their output, and lifting an active cap); SET_STORAGE at
+-rating and 0; every reconnectable line; DO_NOTHING. apply() accepts ANY
move grid2op accepts, not only enumerated ones (a REDISPATCH of any positive
MW, a SET_BUS of any element of a line in service). SET_BUS moves that would
leave a load, generator or storage unit on a busbar without a line end
(always a blackout) are not enumerated, but apply() takes them like grid2op
does. RECONNECT_LINE names both busbars (1) in the grid2op action: with the
status alone grid2op puts each end back on the busbar it last occupied.
Every enumerated move and every hint_move is accepted by apply() (checked by
walking the quick-generation problems: 0 refusals in 8-turn walks).

Simplifications and exclusions (explicit)
-----------------------------------------
* The L2RPN opponent is disabled; contingencies are injected by the problem.
* Forecasts are not shown (the observation is the current step only).
* Only grid2op's bundled test chronics are used (no download); the 118-bus
  environment is l2rpn_idf_2023's test data.
* rte_case5_example is not used: it has one dispatchable unit, so REDISPATCH
  (a pair) can never apply; l2rpn_wcci_2022_dev (one chronic) is covered by
  idf_2023.
* The "dispatch" family's violation comes from a line outage, not from the
  standing redispatch itself (see Families): the redispatch is the cost side.
* The objective is a per-step rate ($/h) at the current step, not a window
  integral, so the chronic's load drift moves it independently of the agent.
* A dangling line end (alone on a busbar) is counted in lines_out: it is a
  line out of service by another name.
* No alarm / attention budget, no load shedding (grid2op has neither here).
* Topology verbs never touch the ends of a line out of service (see Verbs):
  raw grid2op would reconnect the line through a busbar assignment on such an
  end; here the line comes back only through RECONNECT_LINE, so the forced
  outage and the cooldowns cannot be bypassed.
* DISCONNECT_LINE on a line already out of service is refused by apply()
  (the mirror of RECONNECT_LINE's "already in service" check): grid2op takes
  such an order and it equals DO_NOTHING, which the grammar ("in service and
  not frozen") does not promise. legal_moves never enumerated it.
* The infeasible-redispatch game over is kept exactly as grid2op has it
  (LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION is False): apply() does not
  pre-check a CURTAIL / SET_STORAGE order against the units' headroom (the
  balance also depends on the next step's exogenous drift, which the
  observation does not show) and does not clamp it; the resulting state is a
  blackout (metrics outside the envelope) and the observation says why. The
  enumerated CURTAIL candidates (half the unit's output) can hit the rule on
  a grid whose dispatchable units have little ramp room; such a child is
  priced as a blackout and never chosen.
* The reference is a level, not a bound (reference_class "heuristic").

Set-parameter spelling (docs/dataxp_spellings.md; DATAXP_SPELLING=set_parameter)
--------------------------------------------------------------------------------
The same moves written through the one wrapper tool, by the generic encoding
of dataxp.domain (native_set_param_format / parse / grammar from the `verbs`
table; nothing overridden here): element = the verb's first int argument,
operation = the verb in lower case, value = the remaining argument, a JSON
list of the remaining arguments in table order, or 0.

  DO_NOTHING()                        SET_PARAM(0, do_nothing, 0)
  RECONNECT_LINE(3)                   SET_PARAM(3, reconnect_line, 0)
  DISCONNECT_LINE(3)                  SET_PARAM(3, disconnect_line, 0)
  SET_BUS(4, line3_ex, 2)             SET_PARAM(4, set_bus, ["line3_ex", 2])
  SPLIT_SUB(4, ["line3_ex","gen1"])   SET_PARAM(4, split_sub, ["line3_ex","gen1"])
  MERGE_SUB(4)                        SET_PARAM(4, merge_sub, 0)
  REDISPATCH(3, 5, 20)                SET_PARAM(3, redispatch, [5, 20])
  CURTAIL(2, 0.5)                     SET_PARAM(2, curtail, 0.5)
  SET_STORAGE(0, -3.5)                SET_PARAM(0, set_storage, -3.5)

Judgment calls:
* DO_NOTHING has neither element nor argument: element 0 and value 0 are the
  placeholders the parser requires exactly (SET_PARAM(1, do_nothing, 0) and
  SET_PARAM(0, do_nothing, 1) do not parse), so the null move has one
  spelling, as every other move.
* REDISPATCH's element is g_from, the unit whose output goes down (the table's
  first int argument); g_to rides first in the value list. SET_BUS's element
  is the substation; its grid2op element NAME (line3_ex) is the first item of
  the value list, quoted as JSON, the busbar the second. The grammar block
  says so ("value: JSON list [element, bus] in this order, element in double
  quotes") and the parser also takes the list as the block head literally
  shows it, with the bare name (SET_PARAM(4, set_bus, [line3_ex, 2]); the
  verifier of 2026-10-23 found that form rejected while SET_BUS(4, line3_ex,
  2) parsed, SET_BUS being 31% of the legal moves); format_move still writes
  the JSON form only. SPLIT_SUB's single remaining argument is the JSON list
  itself, compact as fmt_arg writes it (["line3_ex","gen1"]), exactly as the
  multitool call writes it, and only as JSON (the block shows the example).
* The operation name parses ignoring case (SET_PARAM(3, RECONNECT_LINE, 0)
  parses); the verb call text (RECONNECT_LINE(3)) is rejected, as the grammar
  lists SET_PARAM only.
* describe() names the operations through Domain.op_name and shows the three
  call signatures (REDISPATCH, CURTAIL, SET_STORAGE) through Domain.call_form,
  so under set_parameter the observation reads "set_bus refuses them", "a
  curtail or set_storage order", "Redispatch: SET_PARAM(g_from, redispatch,
  [g_to, mw]) lowers unit g_from ..." and "no operation puts it back earlier
  ... the topology operations do not touch its ends"; nothing else in the
  observation changes, and the multitool text is byte for byte the previous
  one. VerbSpec.phrase heads each operation block of the generated grammar
  ("Reconnect a line: SET_PARAM(l, reconnect_line, 0)").
"""

from __future__ import annotations

import gc
import logging
import math
import re
import warnings
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..domain import ArgSpec, Domain, Move, VerbSpec, spelled_in_verbs, stable_seed
from ..potential import DesignProgram, program_from_goals

_FAMILIES = ("overload", "contingency", "topology", "dispatch")

_ENVS = {
    "case14": {"name": "l2rpn_case14_sandbox",
               "label": "IEEE 14-bus L2RPN sandbox grid"},
    "case14s": {"name": "educ_case14_storage",
                "label": "IEEE 14-bus grid with two storage units"},
    "case36": {"name": "l2rpn_neurips_2020_track1",
               "label": "36-substation grid (L2RPN NeurIPS 2020, track 1)"},
    "case118": {"name": "l2rpn_idf_2023",
                "label": "118-substation grid (L2RPN Paris region 2023)"},
}
_ENV_WEIGHTS = {"case14": 0.3, "case14s": 0.3, "case36": 0.25, "case118": 0.15}
_OVERLOAD_ENVS = ("case36", "case118")

RHO_LO, RHO_HI = 1.0, 1.6        # target initial overload band
RHO_CAP = 8.0                    # validity envelope on rho
FAIL_RHO = 10.0
MAX_CANDIDATES = 96
N_TOP_LINES = 3                  # substations of the 3 most loaded lines get topology moves
N_DISC = 8                       # DISCONNECT_LINE offered for the 8 most loaded lines
N_SPLITS_PER_SUB = 3
N_REDISP_PAIRS = 24
N_CURTAIL = 8
MIN_WINDOW = 30                  # steps left in the chronic after t0

_log = logging.getLogger(__name__)


# ── environment handles (one grid2op env per environment, per process) ──────
class _EnvHandle:
    """A grid2op environment plus its static data. The handle's own env is
    only ever used as a template: every state holds a copy."""

    def __init__(self, key: str):
        try:
            import grid2op
            from grid2op.Action import DontAct, PlayableAction
            from grid2op.Chronics import MultifolderWithCache
            from grid2op.Opponent import BaseOpponent, NeverAttackBudget
        except ImportError as e:  # pragma: no cover - documented in the report
            raise ImportError("dataxp.domains.grid2op needs the grid2op package "
                              "(pip install -r dataxp/requirements_grid2op.txt): %s" % e)
        try:
            from lightsim2grid import LightSimBackend
            backend = LightSimBackend()
            self.backend_name = "lightsim2grid"
        except ImportError:        # pandapower fallback (10-30x slower)
            backend = None
            self.backend_name = "pandapower"
        self.key = key
        self.env_name = _ENVS[key]["name"]
        self.label = _ENVS[key]["label"]
        logging.getLogger("grid2op").setLevel(logging.ERROR)
        kw = dict(test=True, action_class=PlayableAction, chronics_class=MultifolderWithCache,
                  opponent_class=BaseOpponent, opponent_budget_class=NeverAttackBudget,
                  opponent_action_class=DontAct, opponent_init_budget=0.0,
                  opponent_budget_per_ts=0.0, opponent_attack_cooldown=999999,
                  opponent_attack_duration=0)
        if backend is not None:
            kw["backend"] = backend
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = grid2op.make(self.env_name, **kw)
            env.chronics_handler.real_data.set_filter(lambda x: True)
            env.chronics_handler.real_data.reset()
            env.deactivate_forecast()
        self.env = env
        self.n_sub, self.n_line = int(env.n_sub), int(env.n_line)
        self.n_gen, self.n_load, self.n_stor = int(env.n_gen), int(env.n_load), int(env.n_storage)
        self.sub_info = np.asarray(env.sub_info, dtype=int)
        self.sub_start = np.concatenate([[0], np.cumsum(self.sub_info)[:-1]]).astype(int)
        self.l_or_sub = np.asarray(env.line_or_to_subid, dtype=int)
        self.l_ex_sub = np.asarray(env.line_ex_to_subid, dtype=int)
        self.g_sub = np.asarray(env.gen_to_subid, dtype=int)
        self.c_sub = np.asarray(env.load_to_subid, dtype=int)
        self.s_sub = (np.asarray(env.storage_to_subid, dtype=int) if self.n_stor
                      else np.zeros(0, dtype=int))
        self.l_or_pos = np.asarray(env.line_or_pos_topo_vect, dtype=int)
        self.l_ex_pos = np.asarray(env.line_ex_pos_topo_vect, dtype=int)
        self.g_pos = np.asarray(env.gen_pos_topo_vect, dtype=int)
        self.c_pos = np.asarray(env.load_pos_topo_vect, dtype=int)
        self.s_pos = (np.asarray(env.storage_pos_topo_vect, dtype=int) if self.n_stor
                      else np.zeros(0, dtype=int))
        self.gen_type = [str(x) for x in env.gen_type]
        self.gen_pmax = np.asarray(env.gen_pmax, dtype=float)
        self.gen_pmin = np.asarray(env.gen_pmin, dtype=float)
        self.ramp_up = np.asarray(env.gen_max_ramp_up, dtype=float)
        self.ramp_down = np.asarray(env.gen_max_ramp_down, dtype=float)
        self.gen_redisp = np.asarray(env.gen_redispatchable, dtype=bool)
        self.gen_renew = np.asarray(env.gen_renewable, dtype=bool)
        self.gen_cost = np.asarray(env.gen_cost_per_MW, dtype=float)
        if self.n_stor:
            self.stor_emax = np.asarray(env.storage_Emax, dtype=float)
            self.stor_emin = np.asarray(env.storage_Emin, dtype=float)
            self.stor_pprod = np.asarray(env.storage_max_p_prod, dtype=float)
            self.stor_pabs = np.asarray(env.storage_max_p_absorb, dtype=float)
            self.stor_cost = np.asarray(env.storage_marginal_cost, dtype=float)
            self.stor_eff_c = np.asarray(env.storage_charging_efficiency, dtype=float)
            self.stor_eff_d = np.asarray(env.storage_discharging_efficiency, dtype=float)
            self.stor_loss = np.asarray(env.storage_loss, dtype=float)
        self.thermal = np.asarray(env.get_thermal_limit(), dtype=float)
        p = env.parameters
        self.params = {k: getattr(p, k) for k in (
            "NB_TIMESTEP_OVERFLOW_ALLOWED", "NB_TIMESTEP_RECONNECTION",
            "NB_TIMESTEP_COOLDOWN_LINE", "NB_TIMESTEP_COOLDOWN_SUB",
            "HARD_OVERFLOW_THRESHOLD", "MAX_SUB_CHANGED", "MAX_LINE_STATUS_CHANGED")}
        self.delta_min = int(round(float(env.delta_time_seconds) / 60.0))
        subpaths = list(env.chronics_handler.real_data.subpaths)
        names = [str(s).replace("\\", "/").rstrip("/").split("/")[-1] for s in subpaths]
        self.chronics = sorted(names)
        self._chronic_index = {n: i for i, n in enumerate(names)}
        self.max_step: Dict[str, int] = {}
        self.scan: Dict[str, np.ndarray] = {}
        # element names per substation, in topo_vect order
        self.sub_elems: List[List[str]] = [[] for _ in range(self.n_sub)]
        self.elem_pos: Dict[str, int] = {}
        self.elem_sub: Dict[str, int] = {}
        order = []
        for l in range(self.n_line):
            order.append(("line%d_or" % l, int(self.l_or_pos[l]), int(self.l_or_sub[l])))
            order.append(("line%d_ex" % l, int(self.l_ex_pos[l]), int(self.l_ex_sub[l])))
        for g in range(self.n_gen):
            order.append(("gen%d" % g, int(self.g_pos[g]), int(self.g_sub[g])))
        for c in range(self.n_load):
            order.append(("load%d" % c, int(self.c_pos[c]), int(self.c_sub[c])))
        for s in range(self.n_stor):
            order.append(("stor%d" % s, int(self.s_pos[s]), int(self.s_sub[s])))
        order.sort(key=lambda x: x[1])
        for name, pos, sub in order:
            self.sub_elems[sub].append(name)
            self.elem_pos[name] = pos
            self.elem_sub[name] = sub
        self.n_disp = int(self.gen_redisp.sum())

    # -- positioning ----------------------------------------------------------
    def start(self, chronic: str, t_first: int):
        """A fresh env copy positioned at chronic step t_first (reference
        topology, no redispatch, counters at zero) and its observation."""
        env = self.env
        if chronic not in self._chronic_index:
            raise ValueError("unknown chronic %r for %s" % (chronic, self.env_name))
        env.set_id(self._chronic_index[chronic])
        env.seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obs = env.reset()
            if t_first > 0:
                env.fast_forward_chronics(int(t_first))
                obs = env.get_obs()
        self.max_step[chronic] = int(obs.max_step)
        return env.copy(), obs

    def scan_rho(self, chronic: str) -> np.ndarray:
        """max rho of the reference topology at every step of the chronic,
        cached per chronic. The scan steps through the chronic with the
        protections off and puts every line back as soon as its scheduled
        maintenance is over (one reconnection per step, grid2op's limit):
        grid2op never reconnects a line by itself, so a plain do-nothing scan
        accumulates every past maintenance outage and no longer describes the
        state fast_forward_chronics(t) produces (that state carries only the
        maintenance active at t). With the reconnections the scan reproduces
        start(chronic, t) step for step (checked: max |diff| = 0 on case36)."""
        if chronic in self.scan:
            return self.scan[chronic]
        env, obs = self.start(chronic, 0)
        p = env.parameters
        p.NO_OVERFLOW_DISCONNECTION = True
        env.change_parameters(p)
        env.set_id(self._chronic_index[chronic])
        env.seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obs = env.reset()
        out = [float(np.max(obs.rho))]
        done = False
        while not done:
            status = np.asarray(obs.line_status, bool)
            cd = np.asarray(obs.time_before_cooldown_line, dtype=int)
            maint = np.asarray(obs.time_next_maintenance) == 0
            rec = [int(l) for l in range(self.n_line)
                   if (not status[l]) and cd[l] == 0 and not maint[l]]
            act = env.action_space(
                {"set_line_status": [(rec[0], +1)],
                 "set_bus": {"lines_or_id": [(rec[0], 1)], "lines_ex_id": [(rec[0], 1)]}}
                if rec else {})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obs, _, done, _ = env.step(act)
            out.append(float(np.max(obs.rho)) if not done else float("nan"))
        env.close()
        arr = np.asarray(out, dtype=float)
        self.scan[chronic] = arr
        return arr


_HANDLES: Dict[str, _EnvHandle] = {}

# (family, env, chronic, t0, fault) of every problem built in this process.
# make_problem re-draws its start on a repeat, so the problem set of one
# generation run has no duplicate spec (gate G1). Problem i is therefore
# reproducible from a fresh process when problems are built in index order,
# which is what run_generation and smoke do.
_BUILT: set = set()


def _handle(key: str) -> _EnvHandle:
    h = _HANDLES.get(key)
    if h is None:
        h = _EnvHandle(key)
        _HANDLES[key] = h
    return h


def _step(env, act_dict: dict):
    """One grid2op step; (obs, done, info). A malformed action dict raises
    ValueError before the step."""
    try:
        act = env.action_space(act_dict)
    except Exception as e:
        raise ValueError("grid2op rejected the action %r: %s" % (act_dict, e))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs, _, done, info = env.step(act)
    return obs, bool(done), info


def _flagged(info) -> Optional[str]:
    if info.get("is_illegal"):
        return "illegal (cooldown, or more than one substation / line status in one step)"
    if info.get("is_ambiguous"):
        exc = info.get("exception") or []
        return "ambiguous: %s" % ("; ".join(str(e)[:120] for e in exc) if exc else "see grid2op")
    return None


_RE_ASKED = re.compile(r"should (increase|decrease) their total production of\s*([\d.]+)\s*MW")
_RE_AVAIL = re.compile(r"at most\s*([\d.]+)\s*MW")


def _done_reason(info, obs=None) -> str:
    """Why grid2op ended the episode at this step, for the observation
    (describe) and the refusal of any further move. The three cases grid2op
    has here: an infeasible redispatch after a curtailment / storage order
    (failed_redispatching; LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION is
    False so the order is not clamped), a diverged / non-connected power
    flow, and the end of the time series."""
    exc = info.get("exception") or []
    texts = [str(e) for e in exc]
    if info.get("failed_redispatching"):
        why = ("infeasible redispatch (grid2op ImpossibleRedispatching game over): the energy "
               "the order took out of or put into the grid, plus the step's own load / renewable "
               "drift, could not be covered by the dispatchable units within their ramp rates "
               "and pmin / pmax")
        for t in texts:
            m1, m2 = _RE_ASKED.search(t), _RE_AVAIL.search(t)
            if m1 and m2:
                why += (" - the units had to %s their output by %s MW this step but can %s it "
                        "by at most %s MW" % (m1.group(1), m1.group(2), m1.group(1), m2.group(1)))
                break
        return why
    if any("ivergence" in t or "non connected" in t for t in texts):
        return ("power flow diverged (no AC solution: an isolated load, generator or storage "
                "unit, or a non-connected grid)")
    if texts:
        return "grid2op ended the episode: " + texts[0][:200]
    if obs is not None and int(getattr(obs, "max_step", 0) or 0) > 0 \
            and int(obs.current_step) >= int(obs.max_step):
        return "the time series ended at step %d" % int(obs.current_step)
    return "grid2op ended the episode (no exception reported)"


# ── the domain ───────────────────────────────────────────────────────────────
class Grid2OpDomain(Domain):
    """Grid operation on L2RPN environments with grid2op's own action
    vocabulary (topology, line status, redispatch, curtailment, storage)."""

    name = "grid2op"
    native_moves = True
    params = ()
    reference_class = "heuristic"    # the schedule's cost at t0: a level, not a bound (drift)
    max_candidates = MAX_CANDIDATES

    verbs = (
        VerbSpec("DO_NOTHING", (), "take no action; the grid advances one 5-minute step",
                 phrase="Do nothing"),
        VerbSpec("RECONNECT_LINE", (ArgSpec("l", "int", "line id 0..n_line-1, currently out of "
                                            "service, not frozen, not in maintenance / forced outage"),),
                 "put line l back in service, both ends on busbar 1", phrase="Reconnect a line"),
        VerbSpec("DISCONNECT_LINE", (ArgSpec("l", "int", "line id, in service and not frozen"),),
                 "open line l (it is frozen for NB_TIMESTEP_COOLDOWN_LINE steps afterwards)",
                 phrase="Disconnect a line"),
        VerbSpec("SET_BUS", (ArgSpec("sub", "int", "substation id 0..n_sub-1, not frozen"),
                             ArgSpec("element", "str", "an element of that substation: line<l>_or or "
                                     "line<l>_ex of a line IN SERVICE, gen<g>, load<c> or stor<s>"),
                             ArgSpec("bus", "int", "1 or 2")),
                 "move that element end to busbar bus of its substation (one substation per step; "
                 "the substation is then frozen for NB_TIMESTEP_COOLDOWN_SUB steps). An end of a "
                 "line out of service has no busbar and is refused: the line comes back only "
                 "through RECONNECT_LINE", phrase="Move an element end to a busbar"),
        VerbSpec("SPLIT_SUB", (ArgSpec("sub", "int", "substation id, not frozen"),
                               ArgSpec("bus2", "json", "list of element names of that substation, "
                                       'e.g. ["line3_ex","gen1"]; an end of a line out of service '
                                       "may not be listed")),
                 "set the whole substation in one action: the listed elements on busbar 2, all "
                 "its other elements on busbar 1 (an element alone on a busbar is isolated); the "
                 "ends of lines out of service are left as they are (out)",
                 phrase="Split a substation"),
        VerbSpec("MERGE_SUB", (ArgSpec("sub", "int", "a split substation, not frozen"),),
                 "put every element of the substation back on busbar 1 (the ends of lines out of "
                 "service stay out; the line itself comes back only through RECONNECT_LINE)",
                 phrase="Merge a substation"),
        VerbSpec("REDISPATCH", (ArgSpec("g_from", "int", "dispatchable unit whose output goes down"),
                                ArgSpec("g_to", "int", "dispatchable unit whose output goes up"),
                                ArgSpec("mw", "float", "MW moved this step, 0 < mw <= the smaller "
                                        "of the two ramp rates and of the two units' room")),
                 "lower unit g_from by mw and raise unit g_to by mw; the deviation from the "
                 "schedule persists and costs |redispatch| x price every hour until reversed",
                 phrase="Redispatch between two units"),
        VerbSpec("CURTAIL", (ArgSpec("g", "int", "a renewable unit"),
                             ArgSpec("frac", "float", "cap as a fraction of pmax, 0 <= frac <= 1; "
                                     "1 lifts the cap")),
                 "cap renewable unit g at frac x pmax; the shortfall is produced by the "
                 "dispatchable units (counted as redispatch). A cap whose shortfall, with the "
                 "step's load / renewable drift, exceeds what the dispatchable units can raise "
                 "within their ramp rates ends the episode (blackout: infeasible redispatch)",
                 phrase="Curtail a renewable unit"),
        VerbSpec("SET_STORAGE", (ArgSpec("s", "int", "storage unit id"),
                                 ArgSpec("mw", "float", "-max_p_prod <= mw <= max_p_absorb; "
                                         "positive absorbs (charges), negative injects")),
                 "storage s takes/gives mw MW during this one step (0 afterwards unless set "
                 "again); the dispatchable units cover the balance. An order the dispatchable "
                 "units cannot balance within their ramp rates ends the episode (blackout: "
                 "infeasible redispatch)", phrase="Set a storage unit's power"),
    )

    # time-stepping domain: every move advances the chronic one step. When no
    # enumerated move improves on the incumbent the ladder applies null_move
    # (DO_NOTHING) so the world advances (docs/corpus_native_domains.md,
    # "Time-stepping domains"); harvest keys duplicate pairs on time_key.
    advances_time = True
    time_key = "step"

    def null_move(self, st) -> Move:
        return Move("DO_NOTHING", {})

    def __init__(self, stochastic: bool = False):
        super().__init__(False)      # grid2op with the opponent off is deterministic

    # ── problems ─────────────────────────────────────────────────────────────
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[index % len(_FAMILIES)]
        keys = list(_OVERLOAD_ENVS) if family == "overload" else list(_ENVS)
        w = np.asarray([_ENV_WEIGHTS[k] for k in keys], dtype=float)
        key = keys[int(rng.choice(len(keys), p=w / w.sum()))]
        h = _handle(key)
        chronic = h.chronics[int(rng.integers(len(h.chronics)))]
        scan = h.scan_rho(chronic)
        n_steps = len(scan)
        hi = n_steps - MIN_WINDOW
        built = None
        n_attempts = 12
        for attempt in range(n_attempts):
            if family == "overload":
                ok = [t for t in range(4, hi) if math.isfinite(scan[t]) and RHO_LO < scan[t] <= RHO_HI]
                if not ok:
                    key = _OVERLOAD_ENVS[(list(_OVERLOAD_ENVS).index(key) + 1) % len(_OVERLOAD_ENVS)]
                    h = _handle(key)
                    chronic = h.chronics[int(rng.integers(len(h.chronics)))]
                    scan = h.scan_rho(chronic)
                    n_steps, hi = len(scan), len(scan) - MIN_WINDOW
                    continue
                # a third of the time the first step of an overload spell (there
                # are few of them: drawn more often they repeat), else any step
                starts = [t for t in ok if scan[t - 1] <= RHO_LO] or ok
                t0 = int(starts[int(rng.integers(len(starts)))]) if rng.random() < 0.35 \
                    else int(ok[int(rng.integers(len(ok)))])
                built = self._build_overload(h, chronic, t0)
            else:
                # loaded moments: the upper half of the reference rho profile;
                # the dispatch family needs up to 9 steps of history before t0
                t_min = 12 if family == "dispatch" else 6
                fin = [t for t in range(t_min, hi) if math.isfinite(scan[t]) and scan[t] <= RHO_LO]
                if not fin:
                    fin = [t for t in range(t_min, hi) if math.isfinite(scan[t])]
                if not fin:
                    chronic = h.chronics[int(rng.integers(len(h.chronics)))]
                    scan = h.scan_rho(chronic)
                    n_steps, hi = len(scan), len(scan) - MIN_WINDOW
                    continue
                med = float(np.median([scan[t] for t in fin]))
                loaded = [t for t in fin if scan[t] >= med] or fin
                t0 = int(loaded[int(rng.integers(len(loaded)))])
                if family == "contingency":
                    built = self._build_contingency(h, chronic, t0, rng)
                elif family == "topology":
                    built = self._build_topology(h, chronic, t0, rng)
                else:
                    built = self._build_dispatch(h, chronic, t0, rng)
            if built is not None:
                ident = (family, key, chronic, int(built[2].current_step), str(built[3].get("fault")))
                if ident in _BUILT:
                    built = None          # already in this run's problem set: re-draw
                    continue
                _BUILT.add(ident)
                break
        if built is None:
            raise RuntimeError("grid2op make_problem: no %s start found on %s after %d attempts"
                               % (family, key, n_attempts))
        pre, forced, obs0, extra = built
        rho0 = float(np.max(obs0.rho))
        goals = {"maximum_max_rho": 1.0, "maximum_overload_sum": 0.0, "maximum_lines_out": 0.0}
        sched_cost = float(np.sum(h.gen_cost * np.maximum(
            np.asarray(obs0.gen_p, float) - np.asarray(obs0.actual_dispatch, float), 0.0)))
        klass = ("feasible" if rho0 <= 1.0 else "near" if rho0 <= 1.15
                 else "mid" if rho0 <= 1.4 else "deep")
        spec = {
            "family": family,
            "env": key,
            "env_name": h.env_name,
            "chronic": chronic,
            "t0": int(obs0.current_step),
            "pre_actions": [m.as_dict() for m in pre],
            "forced_outage": {str(l): int(u) for l, u in forced.items()},
            "goals": goals,
            "ref_cost": max(sched_cost, 1.0),
            "n_sub": h.n_sub, "n_line": h.n_line, "n_gen": h.n_gen,
            "n_load": h.n_load, "n_storage": h.n_stor,
            "initial_max_rho": rho0,
            "initial_lines_out": int(np.sum(~np.asarray(obs0.line_status, bool))),
            "difficulty_class": klass,
            "backend": h.backend_name,
        }
        spec.update(extra)
        return spec

    # -- family builders: (pre_moves, forced_outage, obs0, extra) or None -----
    def _build_overload(self, h, chronic, t0):
        env, obs = h.start(chronic, t0)
        env.close()
        if not (RHO_LO < float(np.max(obs.rho)) <= RHO_HI):
            return None
        return [], {}, obs, {"fault": "natural overload of the reference topology"}

    def _pick_outage(self, h, base, obs, rng):
        """The line whose outage overloads the grid at this state into the
        band (random among those; failing that the heaviest outage that
        stays under RHO_HI, if it loads the grid beyond 0.85), as (line,
        rho), or None. `base` is stepped on copies only; on the big grids
        the lines around the 12 most loaded ones plus 16 random lines are
        tried."""
        lines = [l for l in range(h.n_line) if obs.line_status[l]]
        if h.n_line > 48:
            top = np.argsort(-np.asarray(obs.rho), kind="stable")[:12].tolist()
            subs = set(int(h.l_or_sub[l]) for l in top) | set(int(h.l_ex_sub[l]) for l in top)
            near = [l for l in lines if int(h.l_or_sub[l]) in subs or int(h.l_ex_sub[l]) in subs]
            rest = [l for l in lines if l not in set(near)]
            pick = rng.choice(len(rest), size=min(16, len(rest)), replace=False).tolist()
            lines = sorted(set(near) | set(rest[k] for k in pick))
        cands, best = [], (None, -1.0)
        for l in lines:
            e = base.copy()
            o, done, info = _step(e, {"set_line_status": [(int(l), -1)]})
            e.close()
            if done or _flagged(info):
                continue
            r = float(np.max(o.rho))
            if RHO_LO < r <= RHO_HI:
                cands.append((l, r))
            if r > best[1] and r <= RHO_HI:
                best = (l, r)
        if cands:
            return cands[int(rng.integers(len(cands)))]
        if best[0] is not None and best[1] > 0.85:
            return best
        return None

    def _build_contingency(self, h, chronic, t0, rng):
        base, obs = h.start(chronic, t0 - 1)
        pick = self._pick_outage(h, base, obs, rng)
        base.close()
        if pick is None:
            return None
        l, r = pick
        K = int(rng.integers(2, 7))
        env, obs = h.start(chronic, t0 - 1)
        obs, done, info = _step(env, {"set_line_status": [(int(l), -1)]})
        env.close()
        if done or _flagged(info):
            return None
        return ([Move("DISCONNECT_LINE", {"l": int(l)})], {int(l): t0 + K}, obs,
                {"fault": "line %d tripped at step %d, forced outage for %d steps" % (l, t0 - 1, K)})

    @staticmethod
    def _dead_ends(h, obs) -> set:
        """Element names of both ends of every line out of service at this
        observation: they have no busbar (topo_vect -1) and no topology verb
        may address them (a busbar assignment on one would reconnect the line
        in grid2op; here only RECONNECT_LINE does that)."""
        status = np.asarray(obs.line_status, bool)
        out = set()
        for l in np.nonzero(~status)[0].tolist():
            out.add("line%d_or" % l)
            out.add("line%d_ex" % l)
        return out

    def _partitions(self, h, obs, sub, rng, n):
        """n random 2-busbar partitions of the LIVE elements of substation sub
        (the ends of lines out of service are never listed: they keep no
        busbar) with at least two elements on each busbar and a line end on
        each."""
        dead = self._dead_ends(h, obs)
        elems = [e for e in h.sub_elems[sub] if e not in dead]
        k = len(elems)
        out, seen = [], set()
        for _ in range(12 * n):
            if len(out) >= n:
                break
            mask = rng.random(k) < 0.5
            if mask.sum() < 2 or (~mask).sum() < 2:
                continue
            b2 = [e for e, m in zip(elems, mask) if m]
            b1 = [e for e, m in zip(elems, mask) if not m]
            if not any(e.startswith("line") for e in b2) or not any(e.startswith("line") for e in b1):
                continue
            key = tuple(sorted(b2))
            if key in seen:
                continue
            seen.add(key)
            out.append(b2)
        return out

    def _build_topology(self, h, chronic, t0, rng):
        base, obs = h.start(chronic, t0 - 1)
        subs = [s for s in range(h.n_sub) if h.sub_info[s] >= 4]
        if h.n_sub > 40:
            top = np.argsort(-np.asarray(obs.rho), kind="stable")[:8].tolist()
            near = set(int(h.l_or_sub[l]) for l in top) | set(int(h.l_ex_sub[l]) for l in top)
            subs = [s for s in subs if s in near] + [
                subs[k] for k in rng.choice(len(subs), size=min(6, len(subs)), replace=False).tolist()]
            subs = sorted(set(subs))
        trials = []
        for s in subs:
            for b2 in self._partitions(h, obs, s, rng, 3):
                trials.append((s, b2))
        order = rng.permutation(len(trials)).tolist()
        cands = []
        for k in order[:min(len(order), 36)]:
            s, b2 = trials[k]
            e = base.copy()
            o, done, info = _step(e, self._split_action(h, obs, s, b2))
            e.close()
            if done or _flagged(info):
                continue
            r = float(np.max(o.rho))
            if RHO_LO < r <= RHO_HI:
                cands.append((s, b2, r))
        base.close()
        if cands:
            s, b2, r = cands[int(rng.integers(len(cands)))]
        else:
            return None
        env, obs = h.start(chronic, t0 - 1)
        obs, done, info = _step(env, self._split_action(h, obs, s, b2))
        env.close()
        if done or _flagged(info):
            return None
        return ([Move("SPLIT_SUB", {"sub": int(s), "bus2": list(b2)})], {}, obs,
                {"fault": "substation %d split at step %d (busbar 2: %s)" % (s, t0 - 1, ", ".join(b2))})

    def _build_dispatch(self, h, chronic, t0, rng):
        """A standing redispatch inherited from the previous operator (the
        pair that loads the grid most, n_pre = 3..8 steps at the ramp rate,
        so 3..8 x ramp MW stands on each unit) and, at t0-1, the line outage
        that overloads the so-loaded grid into the band. Redispatch alone
        never overloads these grids (10 ramp-steps reach max rho 0.90 /
        0.80 / 0.82 / 1.03 on case14 / case14s / case36 / case118), so the
        outage supplies the violation and the deviation supplies the cost:
        its premium is paid every hour until it is unwound."""
        disp = np.nonzero(h.gen_redisp)[0].tolist()
        if len(disp) < 2:
            return None
        pairs = [(a, b) for a in disp for b in disp if a != b]
        if len(pairs) > N_REDISP_PAIRS:
            pairs = [pairs[k] for k in rng.choice(len(pairs), size=N_REDISP_PAIRS, replace=False).tolist()]
        n_pre = int(rng.integers(3, 9))
        base, _ = h.start(chronic, t0 - 1 - n_pre)
        best = (None, -1.0)
        for a, b in pairs:
            mw = float(min(h.ramp_down[a], h.ramp_up[b]))
            if mw <= 0:
                continue
            e = base.copy()
            ok, o = True, None
            for _ in range(n_pre):
                o, done, info = _step(e, {"redispatch": [(int(a), -mw), (int(b), +mw)]})
                if done or _flagged(info):
                    ok = False
                    break
            e.close()
            if not ok or o is None:
                continue
            r = float(np.max(o.rho))
            if r > best[1] and r <= RHO_HI:
                best = ((a, b, mw), r)
        base.close()
        if best[0] is None:
            return None
        a, b, mw = best[0]
        env, obs = h.start(chronic, t0 - 1 - n_pre)
        pre = []
        for _ in range(n_pre):
            m = Move("REDISPATCH", {"g_from": int(a), "g_to": int(b), "mw": float(mw)})
            obs, done, info = _step(env, self._action(h, obs, m))
            if done or _flagged(info):
                env.close()
                return None
            pre.append(m)
        pick = self._pick_outage(h, env, obs, rng)
        if pick is None:
            env.close()
            return None
        l, r = pick
        K = int(rng.integers(2, 7))
        m = Move("DISCONNECT_LINE", {"l": int(l)})
        obs, done, info = _step(env, self._action(h, obs, m))
        env.close()
        if done or _flagged(info):
            return None
        pre.append(m)
        return (pre, {int(l): t0 + K}, obs,
                {"fault": "%d MW redispatched from unit %d to unit %d over the %d steps before %d, "
                          "then line %d tripped at step %d (forced outage for %d steps)"
                          % (int(round(n_pre * mw)), a, b, n_pre, t0 - 1, l, t0 - 1, K)})

    # ── state ────────────────────────────────────────────────────────────────
    def load(self, spec: dict):
        h = _handle(spec["env"])
        fx = {
            "h": h,
            "chronic": str(spec["chronic"]),
            "t0": int(spec["t0"]),
            "pre": [Move.from_dict(a) for a in spec.get("pre_actions", [])],
            "forced": {int(k): int(v) for k, v in (spec.get("forced_outage") or {}).items()},
            "goals": dict(spec.get("goals") or {}),
            "ref": float(spec.get("ref_cost") or 1.0),
            "family": str(spec.get("family", "")),
            "fault": str(spec.get("fault", "")),
        }
        return {"fx": fx, "hist": [], "cell": {"env": None, "obs": None, "done": False, "reason": None}}

    def clone(self, st):
        return {"fx": st["fx"], "hist": list(st["hist"]), "cell": st["cell"]}

    def _materialise(self, st) -> dict:
        cell = st["cell"]
        if cell["obs"] is not None:
            return cell
        fx = st["fx"]
        h = fx["h"]
        env, obs = h.start(fx["chronic"], fx["t0"] - len(fx["pre"]))
        done, reason = False, None
        for m in fx["pre"] + [Move.from_dict(a) for a in st["hist"]]:
            if done:
                raise ValueError("replay beyond the end of the episode (%s)" % reason)
            obs, done, info = _step(env, self._action(h, obs, m))
            flag = _flagged(info)
            if flag:
                raise ValueError("replayed move %s is %s" % (self.format_move(m), flag))
            if done:
                reason = _done_reason(info, obs)
        cell = {"env": env, "obs": obs, "done": done, "reason": reason}
        st["cell"] = cell
        return cell

    # ── simulation ───────────────────────────────────────────────────────────
    def _fail(self, fx) -> dict:
        h = fx["h"]
        return {"max_rho": FAIL_RHO, "overload_sum": FAIL_RHO, "lines_out": float(h.n_line),
                "blackout": 1.0, "operating_cost": 10.0 * fx["ref"], "energy_cost": 10.0 * fx["ref"],
                "redispatch_cost": 0.0, "storage_cost": 0.0, "redispatch_mw": 0.0,
                "curtailed_mw": 0.0, "served_load_mw": 0.0, "n_overloaded": float(h.n_line),
                "lines_unavailable": 0.0, "step": -1.0}

    def _dangling(self, h, obs) -> List[int]:
        """Lines in service with an end alone on a busbar (carrying nothing)."""
        tv = np.asarray(obs.topo_vect, dtype=int)
        out = []
        for l in range(h.n_line):
            if not obs.line_status[l]:
                continue
            for pos, sub in ((int(h.l_or_pos[l]), int(h.l_or_sub[l])),
                             (int(h.l_ex_pos[l]), int(h.l_ex_sub[l]))):
                b = int(tv[pos])
                if b <= 0:
                    continue
                s0 = int(h.sub_start[sub])
                seg = tv[s0:s0 + int(h.sub_info[sub])]
                if int(np.sum(seg == b)) <= 1:
                    out.append(l)
                    break
        return out

    def _lines_out(self, fx, obs) -> Tuple[List[int], List[int]]:
        """(lines out the operator is responsible for, lines unavailable)."""
        h = fx["h"]
        step = int(obs.current_step)
        status = np.asarray(obs.line_status, bool)
        maint = np.asarray(obs.time_next_maintenance) == 0
        resp, unavail = [], []
        for l in range(h.n_line):
            if status[l]:
                continue
            if maint[l] or step < fx["forced"].get(l, -1):
                unavail.append(l)
            else:
                resp.append(l)
        return resp, unavail

    def evaluate(self, st, seed: int = 0) -> dict:
        self.calls += 1
        fx = st["fx"]
        h = fx["h"]
        try:
            cell = self._materialise(st)
        except Exception:
            return self._fail(fx)
        obs = cell["obs"]
        if cell["done"] or obs is None:
            return self._fail(fx)
        rho = np.asarray(obs.rho, dtype=float)
        rho = np.where(np.isfinite(rho), rho, FAIL_RHO)
        resp, unavail = self._lines_out(fx, obs)
        dang = self._dangling(h, obs)
        p = np.maximum(np.asarray(obs.gen_p, float), 0.0)
        d = np.asarray(obs.actual_dispatch, float)
        energy = float(np.sum(h.gen_cost * p))
        redisp = float(np.sum(h.gen_cost * np.abs(d)))
        stor = float(np.sum(h.stor_cost * np.abs(np.asarray(obs.storage_power, float)))) if h.n_stor else 0.0
        curt = float(np.sum(np.asarray(obs.curtailment_mw, float))) if hasattr(obs, "curtailment_mw") else 0.0
        cost = energy + redisp + stor
        if not math.isfinite(cost) or cost <= 0.0:
            cost = 1e-3
        return {
            "max_rho": float(np.max(rho)) if rho.size else 0.0,
            "overload_sum": float(np.sum(np.maximum(rho - 1.0, 0.0))),
            "lines_out": float(len(resp) + len(dang)),
            "blackout": 0.0,
            "operating_cost": float(cost),
            "energy_cost": energy,
            "redispatch_cost": redisp,
            "storage_cost": stor,
            "redispatch_mw": float(np.sum(np.abs(d))),
            "curtailed_mw": curt,
            "served_load_mw": float(np.sum(np.asarray(obs.load_p, float))),
            "n_overloaded": float(int(np.sum(rho > 1.0))),
            "lines_unavailable": float(len(unavail)),
            "step": float(int(obs.current_step)),
        }

    # ── design variables (none: native verbs only) ───────────────────────────
    def n_elements(self, st) -> int:
        raise NotImplementedError("grid2op has no SET_PARAM addressing")

    def get(self, st, i: int, p: str) -> Optional[float]:
        return None

    def set(self, st, i: int, p: str, v: float) -> None:
        raise ValueError("grid2op has no SET_PARAM verb")

    def bounds(self, st, p: str) -> Tuple[float, float]:
        return (0.0, 0.0)

    # ── program / reference ──────────────────────────────────────────────────
    def program(self, spec: dict) -> DesignProgram:
        prog = program_from_goals(
            spec["goals"],
            objective_key="operating_cost",
            objective_ref=self.reference(spec),
            objective_sense="min",
            scales={"max_rho": 0.1, "overload_sum": 0.1, "lines_out": 1.0},
            domain=self.name,
        )
        return replace(prog, validity_bounds=(("blackout", 0.5), ("max_rho", RHO_CAP)))

    def reference(self, spec: dict) -> Optional[float]:
        ref = spec.get("ref_cost")
        if ref is None:
            st = self.load(spec)
            cell = self._materialise(st)
            obs = cell["obs"]
            h = st["fx"]["h"]
            ref = float(np.sum(h.gen_cost * np.maximum(
                np.asarray(obs.gen_p, float) - np.asarray(obs.actual_dispatch, float), 0.0)))
            ref = max(ref, 1.0)
            spec["ref_cost"] = ref
            cell["env"].close()
        return float(ref)

    # ── moves ────────────────────────────────────────────────────────────────
    def _action(self, h, obs, move: Move) -> dict:
        """The grid2op action dict of a move, after the cheap structural
        checks (ValueError when the move cannot be expressed)."""
        v, a = move.verb, move.args
        if v == "DO_NOTHING":
            return {}
        if v == "RECONNECT_LINE":
            l = self._line_id(h, a.get("l"))
            # both ends named explicitly: with the status alone grid2op puts
            # each end back on the busbar it last occupied (checked on case14:
            # an end split to busbar 2, disconnected, came back on busbar 2).
            # Naming the buses of a reconnected line is legal, counts as no
            # substation change and starts no substation cooldown (checked on
            # case14 and case14s, also alongside a split of a third substation).
            return {"set_line_status": [(l, +1)],
                    "set_bus": {"lines_or_id": [(l, 1)], "lines_ex_id": [(l, 1)]}}
        if v == "DISCONNECT_LINE":
            l = self._line_id(h, a.get("l"))
            return {"set_line_status": [(l, -1)]}
        if v == "SET_BUS":
            sub = self._sub_id(h, a.get("sub"))
            el = str(a.get("element", ""))
            if el not in h.elem_sub:
                raise ValueError("unknown element %r" % el)
            if h.elem_sub[el] != sub:
                raise ValueError("element %s is at substation %d, not %d" % (el, h.elem_sub[el], sub))
            bus = int(a.get("bus", 0))
            if bus not in (1, 2):
                raise ValueError("bus must be 1 or 2")
            if el.startswith("line"):
                l = self._elem_index(el)
                if not bool(obs.line_status[l]):
                    raise ValueError("%s is an end of line %d, which is out of service: its ends "
                                     "have no busbar and SET_BUS cannot address them (the line "
                                     "comes back only through RECONNECT_LINE(%d))" % (el, l, l))
            return {"set_bus": {self._bus_key(el): [(self._elem_index(el), bus)]}}
        if v == "SPLIT_SUB":
            sub = self._sub_id(h, a.get("sub"))
            b2 = a.get("bus2")
            if not isinstance(b2, (list, tuple)):
                raise ValueError("bus2 must be a list of element names")
            return self._split_action(h, obs, sub, [str(x) for x in b2])
        if v == "MERGE_SUB":
            sub = self._sub_id(h, a.get("sub"))
            return self._split_action(h, obs, sub, [])
        if v == "REDISPATCH":
            g1, g2 = self._gen_id(h, a.get("g_from")), self._gen_id(h, a.get("g_to"))
            mw = float(a.get("mw", 0.0))
            if g1 == g2:
                raise ValueError("g_from and g_to must differ")
            if not (h.gen_redisp[g1] and h.gen_redisp[g2]):
                raise ValueError("units %d and %d must both be dispatchable" % (g1, g2))
            if not (math.isfinite(mw) and mw > 0.0):
                raise ValueError("mw must be positive")
            return {"redispatch": [(g1, -mw), (g2, +mw)]}
        if v == "CURTAIL":
            g = self._gen_id(h, a.get("g"))
            frac = float(a.get("frac", 1.0))
            if not h.gen_renew[g]:
                raise ValueError("unit %d is not renewable" % g)
            if not (math.isfinite(frac) and 0.0 <= frac <= 1.0):
                raise ValueError("frac must be within [0, 1]")
            return {"curtail": [(g, frac)]}
        if v == "SET_STORAGE":
            if not h.n_stor:
                raise ValueError("this grid has no storage unit")
            s = int(a.get("s", -1))
            if not 0 <= s < h.n_stor:
                raise ValueError("storage id out of range")
            mw = float(a.get("mw", 0.0))
            if not math.isfinite(mw):
                raise ValueError("mw must be finite")
            return {"set_storage": [(s, mw)]}
        raise ValueError("domain grid2op has no verb %r" % v)

    @staticmethod
    def _line_id(h, l) -> int:
        try:
            l = int(l)
        except (TypeError, ValueError):
            raise ValueError("line id must be an integer")
        if not 0 <= l < h.n_line:
            raise ValueError("line id %d out of range 0..%d" % (l, h.n_line - 1))
        return l

    @staticmethod
    def _sub_id(h, s) -> int:
        try:
            s = int(s)
        except (TypeError, ValueError):
            raise ValueError("substation id must be an integer")
        if not 0 <= s < h.n_sub:
            raise ValueError("substation id %d out of range" % s)
        return s

    @staticmethod
    def _gen_id(h, g) -> int:
        try:
            g = int(g)
        except (TypeError, ValueError):
            raise ValueError("unit id must be an integer")
        if not 0 <= g < h.n_gen:
            raise ValueError("unit id %d out of range" % g)
        return g

    @staticmethod
    def _bus_key(el: str) -> str:
        if el.startswith("line") and el.endswith("_or"):
            return "lines_or_id"
        if el.startswith("line") and el.endswith("_ex"):
            return "lines_ex_id"
        if el.startswith("gen"):
            return "generators_id"
        if el.startswith("load"):
            return "loads_id"
        return "storages_id"

    @staticmethod
    def _elem_index(el: str) -> int:
        digits = "".join(ch for ch in el.split("_")[0] if ch.isdigit())
        return int(digits)

    def _split_action(self, h, obs, sub: int, bus2: List[str]) -> dict:
        """The grid2op substation action of SPLIT_SUB(sub, bus2) / MERGE_SUB
        (bus2 = []): listed elements to busbar 2, the other live elements to
        busbar 1, and the ends of lines out of service left at 0 (unchanged),
        so the action never reconnects a line. A bus2 list naming such an end
        is refused."""
        elems = h.sub_elems[sub]
        dead = self._dead_ends(h, obs)
        b2 = set(bus2)
        for e in bus2:
            if e not in elems:
                raise ValueError("element %r is not at substation %d (its elements: %s)"
                                 % (e, sub, ", ".join(elems)))
            if e in dead:
                l = self._elem_index(e)
                raise ValueError("%s is an end of line %d, which is out of service: it cannot be "
                                 "given a busbar (the line comes back only through "
                                 "RECONNECT_LINE(%d))" % (e, l, l))
        topo = [0 if e in dead else (2 if e in b2 else 1) for e in elems]
        return {"set_bus": {"substations_id": [(int(sub), topo)]}}

    def apply(self, st, move: Move) -> None:
        fx = st["fx"]
        h = fx["h"]
        cell = self._materialise(st)
        if cell["done"]:
            raise ValueError("the episode has ended (%s); no further action is possible"
                             % (cell.get("reason") or "blackout"))
        obs = cell["obs"]
        act = self._action(h, obs, move)
        if move.verb == "RECONNECT_LINE":
            l = int(move.args["l"])
            if obs.line_status[l]:
                raise ValueError("line %d is already in service" % l)
            if int(obs.current_step) < fx["forced"].get(l, -1):
                raise ValueError("line %d is under forced outage until step %d"
                                 % (l, fx["forced"][l]))
        if move.verb == "DISCONNECT_LINE":
            # the mirror of the RECONNECT_LINE check: grid2op takes a status
            # order on a line already out (it then equals DO_NOTHING), but the
            # grammar promises a line in service
            l = int(move.args["l"])
            if not obs.line_status[l]:
                raise ValueError("line %d is already out of service" % l)
        env = cell["env"].copy()
        obs2, done, info = _step(env, act)
        flag = _flagged(info)
        if flag:
            env.close()
            raise ValueError("%s is %s at this state" % (self.format_move(move), flag))
        if move.verb in ("SET_BUS", "SPLIT_SUB", "MERGE_SUB") and not done:
            # the stated semantics: a topology verb never changes a line's status
            # (the ends of lines out of service are left at 0 in the action)
            back = np.nonzero(~np.asarray(obs.line_status, bool) & np.asarray(obs2.line_status, bool))[0]
            if back.size:
                env.close()
                raise ValueError("%s would put line %d back in service; a line comes back only "
                                 "through RECONNECT_LINE" % (self.format_move(move), int(back[0])))
        st["hist"].append(move.as_dict())
        # a move grid2op accepts but that ends the episode (infeasible redispatch
        # after a curtailment / storage order, diverged power flow) is kept as
        # the blackout state it produces, with the cause for describe()
        st["cell"] = {"env": env, "obs": obs2, "done": done,
                      "reason": _done_reason(info, obs2) if done else None}

    # -- enumeration ------------------------------------------------------------
    def legal_moves(self, st, rng: np.random.Generator,
                    max_candidates: Optional[int] = None) -> List[Move]:
        cell = self._materialise(st)
        if cell["done"]:
            return []
        gc.collect()          # env copies of scored children are cyclic; free them per turn
        fx = st["fx"]
        h = fx["h"]
        obs = cell["obs"]
        step = int(obs.current_step)
        rho = np.asarray(obs.rho, dtype=float)
        status = np.asarray(obs.line_status, bool)
        cd_line = np.asarray(obs.time_before_cooldown_line, dtype=int)
        cd_sub = np.asarray(obs.time_before_cooldown_sub, dtype=int)
        maint = np.asarray(obs.time_next_maintenance) == 0
        tv = np.asarray(obs.topo_vect, dtype=int)
        out: List[Move] = []

        # reconnections
        for l in range(h.n_line):
            if (not status[l]) and cd_line[l] == 0 and not maint[l] and step >= fx["forced"].get(l, -1):
                out.append(Move("RECONNECT_LINE", {"l": int(l)}))

        # topology: substations around the most loaded lines, split ones, random
        order = np.argsort(-rho, kind="stable").tolist()
        live = [l for l in order if status[l]]
        subs = []
        for l in live[:N_TOP_LINES]:
            for s in (int(h.l_or_sub[l]), int(h.l_ex_sub[l])):
                if s not in subs:
                    subs.append(s)
        split = [s for s in range(h.n_sub) if self._is_split(h, tv, s)]
        for s in split:
            if s not in subs:
                subs.append(s)
        pool = [s for s in range(h.n_sub) if s not in subs and h.sub_info[s] >= 4]
        if pool:
            for k in rng.choice(len(pool), size=min(2, len(pool)), replace=False).tolist():
                subs.append(pool[k])
        for s in subs:
            if cd_sub[s] > 0:
                continue
            elems_s = h.sub_elems[s]
            bus_of = {e: int(tv[h.elem_pos[e]]) for e in elems_s}
            for e in elems_s:
                b = bus_of[e]
                if b <= 0:
                    continue
                other = 2 if b == 1 else 1
                stay = [x for x in elems_s if x != e and bus_of[x] == b]
                dest = [x for x in elems_s if bus_of[x] == other]
                if not e.startswith("line") and not any(x.startswith("line") for x in dest):
                    continue       # a load/gen/storage on a busbar without a line end: blackout
                if stay and not any(x.startswith("line") for x in stay):
                    continue       # the busbar it leaves would keep units/loads but no line end
                # a line end moved to an empty busbar dangles (that line carries nothing);
                # legal in grid2op, so it stays enumerated and is priced by lines_out
                out.append(Move("SET_BUS", {"sub": int(s), "element": e, "bus": int(other)}))
            if h.sub_info[s] >= 4:
                for b2 in self._partitions(h, obs, s, rng, N_SPLITS_PER_SUB):
                    out.append(Move("SPLIT_SUB", {"sub": int(s), "bus2": list(b2)}))
            if s in split:
                out.append(Move("MERGE_SUB", {"sub": int(s)}))

        # redispatch
        disp = [g for g in range(h.n_gen) if h.gen_redisp[g]]
        if len(disp) >= 2:
            p = np.asarray(obs.gen_p, float)
            room_up = np.asarray([min(h.ramp_up[g], h.gen_pmax[g] - p[g]) for g in disp])
            room_dn = np.asarray([min(h.ramp_down[g], p[g] - h.gen_pmin[g]) for g in disp])
            if len(disp) <= 6:
                pairs = [(a, b) for a in disp for b in disp if a != b]
            else:
                big = [disp[k] for k in np.argsort(-p[disp], kind="stable")[:6].tolist()]
                roomy = [disp[k] for k in np.argsort(-room_up, kind="stable")[:6].tolist()]
                pairs = [(a, b) for a in big for b in roomy if a != b]
                extra = [(a, b) for a in disp for b in disp if a != b and (a, b) not in set(pairs)]
                if extra:
                    pick = rng.choice(len(extra), size=min(8, len(extra)), replace=False).tolist()
                    pairs += [extra[k] for k in pick]
                if len(pairs) > N_REDISP_PAIRS:
                    pick = rng.choice(len(pairs), size=N_REDISP_PAIRS, replace=False).tolist()
                    pairs = [pairs[k] for k in sorted(pick)]
            idx = {g: k for k, g in enumerate(disp)}
            for a, b in pairs:
                mw = float(min(room_dn[idx[a]], room_up[idx[b]]))
                mw = math.floor(mw * 10.0) / 10.0
                if mw < 0.5:
                    continue
                out.append(Move("REDISPATCH", {"g_from": int(a), "g_to": int(b), "mw": mw}))
                half = math.floor(mw * 5.0) / 10.0
                if half >= 0.5 and half < mw:
                    out.append(Move("REDISPATCH", {"g_from": int(a), "g_to": int(b), "mw": half}))

        # curtailment
        ren = [g for g in range(h.n_gen) if h.gen_renew[g]]
        if ren:
            p = np.asarray(obs.gen_p, float)
            lim = np.asarray(obs.curtailment_limit, float)
            prod = sorted([g for g in ren if p[g] > 1.0], key=lambda g: -p[g])[:N_CURTAIL]
            for g in prod:
                frac = math.floor(0.5 * p[g] / h.gen_pmax[g] * 100.0) / 100.0
                if frac >= 0.0 and frac < lim[g] - 1e-9:
                    out.append(Move("CURTAIL", {"g": int(g), "frac": float(frac)}))
            for g in ren:
                if lim[g] < 1.0 - 1e-9:
                    out.append(Move("CURTAIL", {"g": int(g), "frac": 1.0}))

        # storage
        for s in range(h.n_stor):
            ch = float(obs.storage_charge[s])
            pw = float(obs.storage_power[s])
            if ch < h.stor_emax[s] - 1e-6:
                out.append(Move("SET_STORAGE", {"s": int(s), "mw": float(h.stor_pabs[s])}))
            if ch > h.stor_emin[s] + 1e-6:
                out.append(Move("SET_STORAGE", {"s": int(s), "mw": -float(h.stor_pprod[s])}))
            if abs(pw) > 1e-6:
                out.append(Move("SET_STORAGE", {"s": int(s), "mw": 0.0}))

        # disconnections of the most loaded lines
        for l in live[:N_DISC]:
            if cd_line[l] == 0:
                out.append(Move("DISCONNECT_LINE", {"l": int(l)}))

        out.append(Move("DO_NOTHING", {}))
        cap = max_candidates if max_candidates is not None else self.max_candidates
        return self.subsample_moves(out, rng, cap)

    @staticmethod
    def _is_split(h, tv, s) -> bool:
        s0 = int(h.sub_start[s])
        seg = tv[s0:s0 + int(h.sub_info[s])]
        return bool(np.any(seg == 2))

    # -- greedy (depth-0 diagnostic) ----------------------------------------------
    def hint_move(self, st) -> Optional[Move]:
        cell = self._materialise(st)
        if cell["done"]:
            return None
        fx = st["fx"]
        h = fx["h"]
        obs = cell["obs"]
        step = int(obs.current_step)
        status = np.asarray(obs.line_status, bool)
        cd_line = np.asarray(obs.time_before_cooldown_line, dtype=int)
        cd_sub = np.asarray(obs.time_before_cooldown_sub, dtype=int)
        maint = np.asarray(obs.time_next_maintenance) == 0
        rho = np.asarray(obs.rho, float)
        p = np.asarray(obs.gen_p, float)
        d = np.asarray(obs.actual_dispatch, float)
        for l in range(h.n_line):
            if (not status[l]) and cd_line[l] == 0 and not maint[l] and step >= fx["forced"].get(l, -1):
                return Move("RECONNECT_LINE", {"l": int(l)})
        worst = int(np.argmax(rho))
        if rho[worst] > 1.0:
            ends = (int(h.l_or_sub[worst]), int(h.l_ex_sub[worst]))
            # a producing renewable at either end: curtail it
            ren = [g for g in range(h.n_gen) if h.gen_renew[g] and int(h.g_sub[g]) in ends and p[g] > 2.0]
            if ren:
                g = max(ren, key=lambda g: p[g])
                frac = math.floor(0.5 * p[g] / h.gen_pmax[g] * 100.0) / 100.0
                return Move("CURTAIL", {"g": int(g), "frac": float(frac)})
            # a dispatchable unit at either end: move its output to the unit with most room
            disp = [g for g in range(h.n_gen) if h.gen_redisp[g]]
            at = [g for g in disp if int(h.g_sub[g]) in ends and p[g] - h.gen_pmin[g] > 0.5]
            if at and len(disp) >= 2:
                a = max(at, key=lambda g: p[g])
                others = [g for g in disp if g != a]
                b = max(others, key=lambda g: min(h.ramp_up[g], h.gen_pmax[g] - p[g]))
                mw = min(h.ramp_down[a], p[a] - h.gen_pmin[a], h.ramp_up[b], h.gen_pmax[b] - p[b])
                mw = math.floor(mw * 10.0) / 10.0
                if mw >= 0.5:
                    return Move("REDISPATCH", {"g_from": int(a), "g_to": int(b), "mw": float(mw)})
            # otherwise split the busier end: the overloaded line's end with the least
            # loaded line's end among the lines IN SERVICE there that are not dangling
            # (an end of a line out of service has no busbar and may not be listed; a
            # dangling end carries nothing and would be picked for its rho 0)
            dang = set(self._dangling(h, obs))
            for s in sorted(ends, key=lambda s: -int(h.sub_info[s])):
                if cd_sub[s] > 0 or h.sub_info[s] < 4:
                    continue
                me = "line%d_%s" % (worst, "or" if int(h.l_or_sub[worst]) == s else "ex")
                lines_here = [e for e in h.sub_elems[s] if e.startswith("line") and e != me
                              and status[self._elem_index(e)] and self._elem_index(e) not in dang]
                if not lines_here:
                    continue
                other = min(lines_here, key=lambda e: rho[self._elem_index(e)])
                units_here = [e for e in h.sub_elems[s] if not e.startswith("line")]
                if units_here and len(lines_here) < 2:
                    continue       # busbar 1 would keep units/loads but no live line end: blackout
                return Move("SPLIT_SUB", {"sub": int(s), "bus2": [me, other]})
            return None
        # feasible: unwind the most expensive deviation one ramp at a time
        disp = [g for g in range(h.n_gen) if h.gen_redisp[g]]
        up = [g for g in disp if d[g] > 0.5]
        dn = [g for g in disp if d[g] < -0.5]
        if up and dn:
            a = max(up, key=lambda g: d[g] * h.gen_cost[g])
            b = max(dn, key=lambda g: -d[g] * h.gen_cost[g])
            mw = min(d[a], -d[b], h.ramp_down[a], h.ramp_up[b], h.gen_pmax[b] - p[b], p[a] - h.gen_pmin[a])
            mw = math.floor(mw * 10.0) / 10.0
            if mw >= 0.5:
                return Move("REDISPATCH", {"g_from": int(a), "g_to": int(b), "mw": float(mw)})
        return Move("DO_NOTHING", {})

    # -- labels ---------------------------------------------------------------------
    def move_label(self, st, move: Move) -> str:
        fx = st["fx"]
        h = fx["h"]
        try:
            cell = self._materialise(st)
            obs = cell["obs"]
        except Exception:
            obs = None
        v, a = move.verb, move.args

        def line_txt(l):
            l = int(l)
            t = "line %d (sub %d - sub %d)" % (l, int(h.l_or_sub[l]), int(h.l_ex_sub[l]))
            if obs is not None:
                t += ", rho %s" % _num(float(obs.rho[l]), 3) if obs.line_status[l] else ", out of service"
            return t
        if v == "DO_NOTHING":
            return "let the step pass"
        if v in ("RECONNECT_LINE", "DISCONNECT_LINE"):
            return line_txt(a.get("l", -1)) if 0 <= int(a.get("l", -1)) < h.n_line else ""
        if v == "SET_BUS":
            el = str(a.get("element", ""))
            if obs is not None and el in h.elem_pos:
                return "sub %s: %s from busbar %d" % (a.get("sub"), el, int(obs.topo_vect[h.elem_pos[el]]))
            return "sub %s: %s" % (a.get("sub"), el)
        if v == "SPLIT_SUB":
            return "sub %s: %d of its %d elements to busbar 2" % (
                a.get("sub"), len(a.get("bus2") or []), int(h.sub_info[int(a.get("sub", 0))]))
        if v == "MERGE_SUB":
            return "sub %s back to one busbar" % a.get("sub")
        if v == "REDISPATCH":
            g1, g2 = int(a.get("g_from", 0)), int(a.get("g_to", 0))
            return "unit %d (%s, %s $/MWh) to unit %d (%s, %s $/MWh)" % (
                g1, h.gen_type[g1], _num(h.gen_cost[g1]), g2, h.gen_type[g2], _num(h.gen_cost[g2]))
        if v == "CURTAIL":
            g = int(a.get("g", 0))
            t = "unit %d (%s, pmax %s MW" % (g, h.gen_type[g], _num(h.gen_pmax[g]))
            if obs is not None:
                t += ", now %s MW" % _num(float(obs.gen_p[g]))
            return t + ")"
        if v == "SET_STORAGE":
            s = int(a.get("s", 0))
            t = "storage %d (sub %d" % (s, int(h.s_sub[s]))
            if obs is not None:
                t += ", charge %s/%s MWh" % (_num(float(obs.storage_charge[s])), _num(h.stor_emax[s]))
            return t + ")"
        return ""

    # ── observation ──────────────────────────────────────────────────────────
    def describe(self, st, spec) -> str:
        fx = st["fx"]
        h = fx["h"]
        P = h.params
        goals = (spec or {}).get("goals") or fx["goals"]
        try:
            cell = self._materialise(st)
            obs = cell["obs"] if not cell["done"] else None
            reason = (cell.get("reason") or "blackout") if cell["done"] else None
        except Exception as e:
            obs, reason = None, "the move history cannot be replayed (%s)" % str(e)[:160]
        n_ren = int(h.gen_renew.sum())
        disp_up = float(np.sum(h.ramp_up[h.gen_redisp]))
        disp_dn = float(np.sum(h.ramp_down[h.gen_redisp]))
        # the operations as the served grammar names / writes them in the
        # process's spelling (docstring, "Set-parameter spelling")
        op, call = self.op_name, self.call_form
        vb, vbs = ("verb", "verbs") if spelled_in_verbs(self) else ("operation", "operations")
        out = ["PROBLEM"]
        A = out.append
        A("Real-time operation of the %s: %d substations, %d power lines, %d generators (%d "
          "dispatchable: %s; %d renewable: %s), %d loads%s. Grid2Op environment %s (L2RPN), time "
          "series %s, %d-minute steps; the window began at step %d (%s)."
          % (h.label, h.n_sub, h.n_line, h.n_gen, h.n_disp,
             ", ".join("gen%d %s" % (g, h.gen_type[g]) for g in range(h.n_gen) if h.gen_redisp[g]),
             n_ren, ", ".join("gen%d %s" % (g, h.gen_type[g]) for g in range(h.n_gen) if h.gen_renew[g]),
             h.n_load, (", %d storage units" % h.n_stor) if h.n_stor else "",
             h.env_name, fx["chronic"], h.delta_min, fx["t0"], fx["fault"] or fx["family"]))
        A("Every substation has two busbars (1 and 2). Each element connected to it (a line end, a "
          "generator, a load, a storage unit) sits on one busbar; elements on the same busbar are "
          "electrically joined, elements on different busbars of the same substation are not. The "
          "reference topology has everything on busbar 1; splitting a substation (some elements on "
          "busbar 2) changes the meshing and reroutes power, at no cost. An element alone on a "
          "busbar is isolated: an isolated load, generator or storage unit makes the power flow "
          "diverge (blackout); an isolated line end carries nothing (that line is effectively out). "
          "The two ends of a line out of service have no busbar (shown as out): %s refuses "
          "them, a %s list may not name them, and %s / %s leave them as they "
          "are while setting the substation's other elements. A line changes status only through "
          "%s and %s (and the protections / maintenance); %s "
          "is refused while the line is frozen, in maintenance or under forced outage."
          % (op("SET_BUS"), op("SPLIT_SUB"), op("SPLIT_SUB"), op("MERGE_SUB"),
             op("RECONNECT_LINE"), op("DISCONNECT_LINE"), op("RECONNECT_LINE")))
        A("Each step the loads and the renewable output take the next values of the time series "
          "(an exogenous schedule), the dispatchable units follow their schedule plus the "
          "redispatch in force, and an AC power flow gives the current on every line. rho = "
          "current / thermal limit; rho > 1 is an overload. The protections disconnect a line "
          "overloaded for more than %d consecutive steps, and at once when rho >= %s; a line "
          "tripped by the protections cannot be reconnected for %d steps. After the operator "
          "changes a line's status that line is frozen for %d steps; after a substation's "
          "busbars are changed that substation is frozen for %d steps. One action changes at most "
          "one substation and one line status; a line reconnected by the operator comes back with "
          "both ends on busbar 1."
          % (int(P["NB_TIMESTEP_OVERFLOW_ALLOWED"]), _num(P["HARD_OVERFLOW_THRESHOLD"]),
             int(P["NB_TIMESTEP_RECONNECTION"]), int(P["NB_TIMESTEP_COOLDOWN_LINE"]),
             int(P["NB_TIMESTEP_COOLDOWN_SUB"])))
        A("Redispatch: %s lowers unit g_from by mw MW and raises g_to by "
          "mw MW for this step and every later one (the deviation persists until reversed); mw "
          "is limited by each unit's ramp rate per step and by [pmin, pmax]. The units' schedule "
          "is the time series' own; the redispatch cost is |redispatch| x the unit's price, paid "
          "every hour the deviation stands, in either direction. Curtailment: %s "
          "caps renewable unit g at frac x pmax from this step on (1.0 lifts the cap); the energy "
          "it then cannot deliver is produced by the dispatchable units instead and counts as "
          "redispatch (their price is paid). " % (call("REDISPATCH"), call("CURTAIL")) + (
              "Storage: %s makes unit s absorb mw MW (negative: inject) during "
              "this one step within its power rating and energy limits (charging efficiency %s, "
              "discharging %s; a standing loss of %s MW drains each unit every step whatever it "
              "does); the dispatchable units cover the balance and the storage marginal cost "
              "(%s $/MWh) is paid on |mw|. "
              % (call("SET_STORAGE"), _num(float(h.stor_eff_c[0])), _num(float(h.stor_eff_d[0])),
                 _num(float(h.stor_loss[0])), _num(float(h.stor_cost[0])))
              if h.n_stor else "This grid has no storage unit. ")
          + "Balancing rule (grid2op): the energy a %s or %s order takes out of or "
            "puts into the grid, together with the step's own load and renewable drift, must be "
            "covered by the dispatchable units within their ramp rates and pmin / pmax at that "
            "step. On this grid they can together raise their output by at most %s MW per step "
            "and lower it by at most %s MW per step (less when a unit sits at pmax / pmin or is "
            "already ramping for a redispatch). An order they cannot balance ENDS THE EPISODE: "
            "grid2op does not clamp it (LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION is off), it "
            "declares an infeasible redispatch (ImpossibleRedispatching) and the game is over - "
            "a blackout for the metrics below. Cut a renewable unit in steps the units can "
            "follow, or lift a cap gradually, rather than at once."
            % (op("CURTAIL"), op("SET_STORAGE"), _num(disp_up), _num(disp_dn)))
        forced = fx["forced"]
        if forced:
            A("Forced outages: " + "; ".join(
                "line %d cannot be reconnected before step %d" % (l, u) for l, u in sorted(forced.items()))
              + " (a protection trip; no %s puts it back earlier - %s is refused and "
                "the topology %s do not touch its ends - and it must be put back once available)."
                % (vb, op("RECONNECT_LINE"), vbs))
        A("Time: every move is applied at the start of the next %d-minute step and the result you "
          "see is the power flow of that step; %s also advances time. The chronic ends at "
          "step %d." % (h.delta_min, op("DO_NOTHING"), h.max_step.get(fx["chronic"], 0) or 0))
        A("Metrics: max_rho = max over lines of rho; overload_sum = sum over lines of max(0, rho - 1); "
          "lines_out = lines out of service that the operator could put back (not in maintenance "
          "or forced outage) plus lines with an end alone on a busbar; blackout = 1 if the episode "
          "has ended - the power flow diverged (an isolated load, generator or storage unit, a "
          "non-connected grid) or a %s / %s order could not be balanced by the "
          "dispatchable units (infeasible redispatch, see the balancing rule above); "
          "operating_cost [$/h] = energy_cost (sum over units of output x price) + "
          "redispatch_cost (sum of |redispatch| x price) + storage_cost; redispatch_mw = sum of "
          "|redispatch|; curtailed_mw = renewable output held back by caps; served_load_mw; "
          "n_overloaded = lines with rho > 1; lines_unavailable = lines out in maintenance or "
          "forced outage; step = time series step." % (op("CURTAIL"), op("SET_STORAGE")))
        lim = goals.get("maximum_max_rho", 1.0)
        A("Goals: max_rho <= %s and overload_sum <= %s (no line over its thermal limit), lines_out "
          "<= %s; minimise operating_cost. A blackout, or any rho above %s, is outside the model's "
          "validity and counts as failed." % (_num(lim), _num(goals.get("maximum_overload_sum", 0.0)),
                                              _num(goals.get("maximum_lines_out", 0.0)), _num(RHO_CAP)))
        A("Element names: line<l>_or / line<l>_ex (origin / extremity end of line l), gen<g>, "
          "load<c>, stor<s>. Lines are listed as line l: sub a - sub b (origin - extremity); flow "
          "is the active power entering the line at its origin end (negative = flowing from the "
          "extremity to the origin).")

        out.append("")
        if obs is None:
            last = st["hist"][-1] if st.get("hist") else None
            after = ""
            if last is not None:
                try:
                    after = " after %s" % self.format_move(Move.from_dict(last))
                except Exception:
                    after = ""
            A("DESIGN / RESULTS: the episode has ended%s: %s. No further action is possible; the "
              "metrics are the failure values (blackout 1, max_rho %s, lines_out %d, operating_cost "
              "10 x the schedule's cost), outside the model's validity."
              % (after, reason, _num(FAIL_RHO), h.n_line))
            return "\n".join(out)

        # ── DESIGN ──
        step = int(obs.current_step)
        tv = np.asarray(obs.topo_vect, dtype=int)
        rho = np.asarray(obs.rho, float)
        status = np.asarray(obs.line_status, bool)
        cd_line = np.asarray(obs.time_before_cooldown_line, dtype=int)
        cd_sub = np.asarray(obs.time_before_cooldown_sub, dtype=int)
        maint = np.asarray(obs.time_next_maintenance) == 0
        p = np.asarray(obs.gen_p, float)
        d = np.asarray(obs.actual_dispatch, float)
        tgt = np.asarray(obs.target_dispatch, float)
        clim = np.asarray(obs.curtailment_limit, float)
        A("DESIGN (step %d, %04d-%02d-%02d %02d:%02d)" % (
            step, int(obs.year), int(obs.month), int(obs.day), int(obs.hour_of_day), int(obs.minute_of_hour)))
        split = [s for s in range(h.n_sub) if self._is_split(h, tv, s)]
        if split:
            A("Split substations: " + "; ".join(
                "sub %d (busbar 2: %s)" % (s, ", ".join(e for e in h.sub_elems[s] if tv[h.elem_pos[e]] == 2))
                for s in split) + ". Every other substation is on a single busbar.")
        else:
            A("Topology: every substation on a single busbar (reference topology).")
        dang = self._dangling(h, obs)
        if dang:
            A("Dangling line ends (alone on a busbar, carrying nothing): " + ", ".join(
                "line %d" % l for l in dang) + ".")
        resp, unavail = self._lines_out(fx, obs)
        outs = [l for l in range(h.n_line) if not status[l]]
        if outs:
            parts = []
            for l in outs:
                why = []
                if maint[l]:
                    why.append("in maintenance")
                if step < forced.get(l, -1):
                    why.append("forced outage until step %d" % forced[l])
                if cd_line[l] > 0:
                    why.append("frozen %d more step%s" % (cd_line[l], "" if cd_line[l] == 1 else "s"))
                if not why:
                    why.append("reconnectable now")
                parts.append("line %d (sub %d - sub %d; %s)" % (l, int(h.l_or_sub[l]), int(h.l_ex_sub[l]),
                                                                ", ".join(why)))
            A("Lines out of service: " + "; ".join(parts) + ".")
        else:
            A("Every line is in service.")
        dev = [g for g in range(h.n_gen) if abs(d[g]) > 1e-3]
        if dev:
            A("Redispatch in force: " + ", ".join("gen%d %s MW" % (g, _signed(d[g])) for g in dev)
              + " (target " + ", ".join("gen%d %s" % (g, _signed(tgt[g])) for g in dev) + ").")
        else:
            A("No redispatch in force (every unit on its schedule).")
        caps = [g for g in range(h.n_gen) if h.gen_renew[g] and clim[g] < 1.0 - 1e-9]
        if caps:
            A("Curtailment caps: " + ", ".join("gen%d at %s%% of pmax" % (g, _num(100 * clim[g])) for g in caps) + ".")
        if h.n_stor:
            A("Storage: " + "; ".join(
                "stor%d charge %s/%s MWh, power %s MW" % (
                    s, _num(float(obs.storage_charge[s])), _num(h.stor_emax[s]), _signed(float(obs.storage_power[s])))
                for s in range(h.n_stor)) + ".")
        frozen_s = [s for s in range(h.n_sub) if cd_sub[s] > 0]
        frozen_l = [l for l in range(h.n_line) if cd_line[l] > 0 and status[l]]
        if frozen_s or frozen_l:
            A("Frozen (cooldown): " + "; ".join(
                (["substations " + ", ".join("sub %d (%d)" % (s, cd_sub[s]) for s in frozen_s)] if frozen_s else [])
                + (["lines " + ", ".join("line %d (%d)" % (l, cd_line[l]) for l in frozen_l)] if frozen_l else []))
              + " - steps until they can be changed.")
        energy = float(np.sum(h.gen_cost * np.maximum(p, 0.0)))
        redisp = float(np.sum(h.gen_cost * np.abs(d)))
        stor = float(np.sum(h.stor_cost * np.abs(np.asarray(obs.storage_power, float)))) if h.n_stor else 0.0
        tl = float(np.sum(np.asarray(obs.load_p, float)))
        tg = float(np.sum(p))
        A("Total load %s MW, generation %s MW (losses %s MW). Operating cost %s $/h = energy %s + "
          "redispatch %s + storage %s; the schedule's own energy cost at the window start was %s $/h."
          % (_num(tl, 4), _num(tg, 4), _num(tg - tl), _num(energy + redisp + stor, 5), _num(energy, 5),
             _num(redisp, 4), _num(stor, 4), _num(fx["ref"], 5)))

        # ── RESULTS ──
        out.append("")
        A("RESULTS (AC power flow at step %d)" % step)
        over = [l for l in np.argsort(-rho, kind="stable").tolist() if rho[l] > 1.0]
        if over:
            A("Overloaded (%d line%s): %s." % (len(over), "" if len(over) == 1 else "s", "; ".join(
                "line %d (sub %d - sub %d) rho %s, over by %s, overloaded for %d step%s" % (
                    l, int(h.l_or_sub[l]), int(h.l_ex_sub[l]), _num(rho[l], 3), _num(rho[l] - 1.0),
                    int(obs.timestep_overflow[l]), "" if int(obs.timestep_overflow[l]) == 1 else "s")
                for l in over)))
        else:
            top = np.argsort(-rho, kind="stable").tolist()[:3]
            A("No line is overloaded; closest: " + ", ".join(
                "line %d rho %s" % (l, _num(rho[l], 3)) for l in top) + ".")
        A("Lines (id: origin sub - extremity sub, rho, flow at origin MW, thermal limit A, status), "
          "most loaded first:")
        p_or = np.asarray(obs.p_or, float)
        for l in np.argsort(-rho, kind="stable").tolist():
            if status[l]:
                st_txt = "in service" + (", frozen %d" % cd_line[l] if cd_line[l] > 0 else "")
                A("  line %d: sub %d - sub %d, rho %s, flow %s MW, limit %s A, %s" % (
                    l, int(h.l_or_sub[l]), int(h.l_ex_sub[l]), _num(rho[l], 3), _signed(p_or[l]),
                    _num(h.thermal[l]), st_txt))
            else:
                A("  line %d: sub %d - sub %d, OUT, limit %s A" % (
                    l, int(h.l_or_sub[l]), int(h.l_ex_sub[l]), _num(h.thermal[l])))
        A("Substations (elements with their busbar; a frozen substation shows the steps left):")
        for s in range(h.n_sub):
            cells = []
            for e in h.sub_elems[s]:
                b = int(tv[h.elem_pos[e]])
                cells.append("%s:%s" % (e, b if b > 0 else "out"))
            A("  sub %d%s: %s" % (s, (" (frozen %d)" % cd_sub[s]) if cd_sub[s] > 0 else "", " ".join(cells)))
        A("Generators (type, substation, output MW; dispatchable: schedule, redispatch, pmin-pmax, "
          "ramp MW/step, price $/MWh; renewable: pmax, cap):")
        for g in range(h.n_gen):
            if h.gen_redisp[g]:
                A("  gen%d %s sub %d: p %s MW (schedule %s, redispatch %s), %s-%s MW, ramp %s, price %s"
                  % (g, h.gen_type[g], int(h.g_sub[g]), _num(p[g]), _num(p[g] - d[g]), _signed(d[g]),
                     _num(h.gen_pmin[g]), _num(h.gen_pmax[g]), _num(h.ramp_up[g]), _num(h.gen_cost[g])))
            else:
                cap = "" if clim[g] >= 1.0 - 1e-9 else ", cap %s%% of pmax" % _num(100 * clim[g])
                A("  gen%d %s sub %d: p %s MW, pmax %s%s%s" % (
                    g, h.gen_type[g], int(h.g_sub[g]), _num(p[g]), _num(h.gen_pmax[g]), cap,
                    " (renewable, curtailable)" if h.gen_renew[g] else " (not dispatchable)"))
        lp = np.asarray(obs.load_p, float)
        A("Loads (substation, MW): " + "; ".join(
            "load%d sub %d %s" % (c, int(h.c_sub[c]), _num(lp[c])) for c in range(h.n_load)) + ".")
        if h.n_stor:
            A("Storage units: " + "; ".join(
                "stor%d sub %d: charge %s of %s MWh, power %s MW, rating %s MW" % (
                    s, int(h.s_sub[s]), _num(float(obs.storage_charge[s])), _num(h.stor_emax[s]),
                    _signed(float(obs.storage_power[s])), _num(h.stor_pabs[s]))
                for s in range(h.n_stor)) + ".")
        return "\n".join(out)


# ── small formatting helpers ─────────────────────────────────────────────────
def _num(x, sig: int = 3) -> str:
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


def _signed(x, sig: int = 3) -> str:
    x = float(x)
    s = _num(abs(x), sig)
    if s == "0":
        return "0"
    return ("+" if x > 0 else "-") + s
