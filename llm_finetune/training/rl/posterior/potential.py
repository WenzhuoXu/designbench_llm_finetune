"""Potential-based reward shaping for posterior evaluation."""

from __future__ import annotations

import math


def _finite_or_default(value: object, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def softplus(x: float) -> float:
    if x > 20:
        return x
    if x < -20:
        return math.exp(x)
    return math.log1p(math.exp(x))


def violation_score(
    state: dict,
    *,
    target_fos: float = 1.5,
    max_deflection: float = 0.01,
) -> float:
    # Clamp FOS to [0, inf): negative values from failed FEA have no physical
    # meaning and would cause softplus(1.5 - (-1e25)) = 1e25, blowing up rewards.
    fos_b = max(0.0, _finite_or_default(state.get("fos_buckling", 0.0) or 0.0, 0.0))
    fos_y = max(0.0, _finite_or_default(state.get("fos_yielding", 0.0) or 0.0, 0.0))
    _max_d = max(max_deflection, 1e-6)
    # Cap deflection at 100× the limit so softplus stays finite (<= ~100).
    deflection = min(
        _finite_or_default(state.get("deflection", 0.0) or 0.0, _max_d * 10.0),
        _max_d * 100.0,
    )
    return (
        softplus(target_fos - fos_b)
        + softplus(target_fos - fos_y)
        + softplus((deflection / _max_d) - 1.0)
    )


def compute_potential(
    state: dict,
    *,
    initial_mass: float,
    alpha: float = 5.0,
    target_fos: float = 1.5,
    max_deflection: float = 0.01,
) -> float:
    mass = _finite_or_default(state.get("mass", 0.0) or 0.0, 0.0)
    initial_mass = _finite_or_default(initial_mass, 0.0)
    if mass <= 0.0 or initial_mass <= 0.0:
        return -alpha * violation_score(
            state,
            target_fos=target_fos,
            max_deflection=max_deflection,
        )
    mass_utility = math.log(max(initial_mass, 1e-6) / max(mass, 1e-6))
    return mass_utility - alpha * violation_score(
        state,
        target_fos=target_fos,
        max_deflection=max_deflection,
    )


def compute_step_reward(
    prev_state: dict,
    next_state: dict,
    *,
    initial_mass: float,
    gamma: float = 0.99,
    alpha: float = 5.0,
    target_fos: float = 1.5,
    max_deflection: float = 0.01,
) -> float:
    prev_potential = compute_potential(
        prev_state,
        initial_mass=initial_mass,
        alpha=alpha,
        target_fos=target_fos,
        max_deflection=max_deflection,
    )
    next_potential = compute_potential(
        next_state,
        initial_mass=initial_mass,
        alpha=alpha,
        target_fos=target_fos,
        max_deflection=max_deflection,
    )
    return gamma * next_potential - prev_potential


# ═════════════════════════════════════════════════════════════════════════════
# v2: the design program and its Lagrangian potential  (DOMAIN-AGNOSTIC)
#
# The v1 potential above hardcodes one domain's constraint set: FOS >= 1.5 for
# both failure modes and deflection <= 0.01 m for every problem, with mass
# priced as an unbounded log-utility against the *initial* design. That is wrong
# even inside the truss domain -- auditing DesignBench's 130 truss problems
# shows two disjoint families, 100 constraining {fos_b, fos_y, mass <= m*} with
# NO deflection goal and 30 constraining {fos_b, fos_y, deflection <= d*} with
# NO mass goal -- so v1 prices a phantom constraint on 100/130 problems at
# weight alpha and never prices the real mass cap at all. On that family
# max_mass = 1.1 x optimal_mass and the initial design is a degraded, under-built
# version of the optimum, so mass MUST rise to reach feasibility while v1's
# log(m0/m) term penalises exactly that move; and because softplus never
# saturates, raising FOS far past 1.5 keeps paying. The v1 argmax is therefore an
# over-stiffened, over-mass design.
#
# It is also, more fundamentally, un-portable: nothing about batteries, linkages
# or any other design domain is expressible in it.
#
# v2 replaces the hardcoded set with a DesignProgram -- (objective, constraints)
# read from the problem statement itself -- and defines the potential as the
# negated Lagrangian of that program:
#
#     Phi(s) = -w * f_hat(s)  -  alpha * sum_c hinge(g_hat_c(s))
#
# f_hat is the objective normalised by a problem-supplied reference and g_hat_c
# is the RELATIVE violation of constraint c, hinged so that over-satisfying a
# constraint stops paying. Both are dimensionless, so alpha means the same thing
# in every domain, and a problem contributes no term for a constraint it does
# not have.
#
# Constraints are recovered from the goals dict by naming convention
# ("minimum_x"/"min_x" => x >= limit, "maximum_x"/"max_x" => x <= limit), which
# is the convention DesignBench already uses in both its truss goals
# (minimum_fos_buckling, maximum_mass, maximum_deflection) and its battery goals
# (max_temperature, max_plating, max_charge_time, min_neg_potential). A new
# domain is onboarded by naming its goals, not by writing a potential.
#
# Offline evidence that the specification is what mattered, not the search
# (scripts/search_ladder.py, all 130 truss problems, no LLM, deterministic):
# identical one-step lookahead reaches feasibility on 67% of the mass family
# under Phi_v1 and 88% under Phi_v2, solving a strict superset (+21 problems,
# -0). Searching harder against a misspecified potential is worse than not
# searching at all (greedy heuristic: 86%).
# ═════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Optional, Sequence

# goal-key prefix -> constraint sense. Longest prefix wins, so "minimum_" is
# tried before "min_".
_GOAL_PREFIXES: tuple[tuple[str, str], ...] = (
    ("minimum_", "lower"),
    ("maximum_", "upper"),
    ("min_", "lower"),
    ("max_", "upper"),
)

# Goal keys that describe the problem rather than constrain the design.
_NON_CONSTRAINT_GOALS = frozenset({"objective", "objective_sense", "objective_ref"})


@dataclass(frozen=True)
class Constraint:
    """One inequality of a design program, in the units of its own metric."""

    key: str          # metric name to read out of the state dict
    limit: float
    sense: str        # "upper" => value <= limit ; "lower" => value >= limit
    scale: float = 1.0   # normaliser for the relative violation
    label: str = ""

    def violation(self, state: Mapping, *, tau: float = 0.05, missing: float = 0.0) -> float:
        """Relative violation, hinged to 0.0 when the constraint is satisfied."""
        raw = state.get(self.key, None)
        if raw is None:
            return missing
        value = _finite_or_default(raw, float("nan"))
        if not math.isfinite(value):
            # A diverged simulation is treated as a maximally violated constraint
            # rather than as a missing one: it must not look attractive.
            return 1.0
        scale = self.scale if self.scale > 0 else max(abs(self.limit), 1e-9)
        if self.sense == "lower":
            gap = (self.limit - value) / scale
        else:
            gap = (value - self.limit) / scale
        # Cap so one diverged metric cannot dominate the whole potential.
        return min(hinge(gap, tau), 100.0)


@dataclass(frozen=True)
class DesignProgram:
    """(objective, constraints) descriptor of a design problem — no domain code.

    ``objective_key`` names the metric to optimise and ``objective_ref`` the
    scale that makes it dimensionless; ``objective_sense`` is "min" or "max".
    A program with no objective is a pure feasibility problem.
    """

    constraints: tuple[Constraint, ...] = ()
    objective_key: Optional[str] = None
    objective_sense: str = "min"
    objective_ref: float = 1.0
    domain: str = "generic"
    # Most design objectives are physically positive quantities (mass, charge
    # time, cost). A simulator that returns zero or a negative one has left its
    # domain of validity, and a potential that passes such a value through
    # rewards the exploit: driving a truss pipe to t > 2r makes its cross-section
    # area pi*t*(2r - t) NEGATIVE, so mass goes negative and -mass/ref becomes an
    # unbounded bonus. A min-mass lookahead found exactly that and reported a
    # mass ratio of -508. Treat it as an invalid state instead.
    objective_must_be_positive: bool = True
    # Divisor that makes the constraint price alpha DIMENSIONLESS across domains.
    # Relative violations are not commensurable between problems: a truss starting
    # at FOS 0.44 against a 1.5 target has V = 0.71, while a battery cell 2% over
    # its temperature limit has V = 0.02 -- a factor of 34. With one alpha the
    # battery's objective term (a charge-time gap of 0.33) simply outbids its
    # constraint term (5 x 0.02 = 0.10), and the potential ranks an infeasible
    # fast-charging cell above a feasible one. Setting violation_ref to the
    # INITIAL state's violation expresses every violation in units of "how far
    # this problem started from feasible", after which alpha means the same thing
    # in every domain. 1.0 leaves the raw relative scale.
    violation_ref: float = 1.0
    # Sanity bounds marking states where the simulator's outputs stop meaning
    # anything, INDEPENDENT of the problem's stated constraints. Search will
    # optimise into any such region the potential does not exclude: on the
    # mass-constrained truss family, which states no deflection limit, a one-step
    # lookahead learned to thin a member until the structure became a mechanism --
    # forces vanish, FOS explodes, mass drops, and the design is scored FEASIBLE
    # with a deflection of 1.5e14 m. That was 17% of its "successes".
    # Each entry is (metric_key, upper_bound_on_absolute_value).
    validity_bounds: tuple[tuple[str, float], ...] = ()

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)

    def limit_for(self, key: str) -> Optional[float]:
        for c in self.constraints:
            if c.key == key:
                return c.limit
        return None

    def violations(self, state: Mapping, *, tau: float = 0.05) -> dict[str, float]:
        return {c.label or c.key: c.violation(state, tau=tau) for c in self.constraints}

    def total_violation(self, state: Mapping, *, tau: float = 0.05,
                        normalised: bool = True) -> float:
        raw = sum(c.violation(state, tau=tau) for c in self.constraints)
        if normalised and self.violation_ref > 0.0:
            return raw / self.violation_ref
        return raw

    def with_violation_scale(self, initial_state: Mapping, *, tau: float = 0.05,
                             floor: float = 1e-3) -> "DesignProgram":
        """Copy of this program with alpha made dimensionless for this problem."""
        raw = self.total_violation(initial_state, tau=tau, normalised=False)
        return replace(self, violation_ref=max(raw, floor))

    def is_valid(self, state: Mapping) -> bool:
        """False when the state is outside the simulator's domain of validity."""
        if self.objective_key is not None and self.objective_must_be_positive:
            value = _finite_or_default(state.get(self.objective_key), float("nan"))
            if not math.isfinite(value) or value <= 0.0:
                return False
        for key, bound in self.validity_bounds:
            raw = state.get(key)
            if raw is None:
                continue
            value = _finite_or_default(raw, float("nan"))
            if not math.isfinite(value) or abs(value) > bound:
                return False
        return True

    def objective(self, state: Mapping) -> float:
        """Normalised objective, oriented so that LOWER is always better."""
        if self.objective_key is None:
            return 0.0
        raw = state.get(self.objective_key)
        value = _finite_or_default(raw, float("nan"))
        ref = self.objective_ref if abs(self.objective_ref) > 1e-12 else 1.0
        if not math.isfinite(value):
            return 10.0  # diverged: treat as very bad, but bounded
        if self.objective_must_be_positive and value <= 0.0:
            return 10.0  # outside the simulator's domain of validity
        for key, bound in self.validity_bounds:
            raw = state.get(key)
            if raw is None:
                continue
            metric = _finite_or_default(raw, float("nan"))
            if not math.isfinite(metric) or abs(metric) > bound:
                return 10.0
        normalised = value / ref
        return normalised if self.objective_sense == "min" else -normalised

    def is_feasible(self, state: Mapping, *, tol: float = 1e-9) -> bool:
        if not self.is_valid(state):
            return False
        return self.total_violation(state, tau=0.0) <= tol


def hinge(x: float, tau: float = 0.05) -> float:
    """Smooth hinge: tau*softplus(x/tau) -> max(0, x) as tau -> 0.

    Unlike a bare softplus this saturates at 0 for satisfied constraints, so
    over-satisfying a constraint earns no further potential.
    """
    if tau <= 0.0:
        return max(0.0, x)
    return tau * softplus(x / tau)


def _resolve_metric_key(goal_key: str, sense_prefix: str, state_keys: Optional[Sequence[str]]) -> str:
    """Map a goal name onto the metric it constrains.

    ``maximum_mass`` -> ``mass``; ``max_temperature`` -> ``max_temperature`` when
    the state exposes that name verbatim (PyBaMM-style metrics keep the max_ in
    the metric itself); ``max_charge_time`` -> ``charge_time``.
    """
    stripped = goal_key[len(sense_prefix):]
    if state_keys is None:
        return stripped
    keys = set(state_keys)
    if goal_key in keys:      # metric named exactly like the goal
        return goal_key
    if stripped in keys:
        return stripped
    return stripped


def program_from_goals(
    goals: Mapping,
    *,
    state_keys: Optional[Sequence[str]] = None,
    objective_key: Optional[str] = None,
    objective_ref: Optional[float] = None,
    objective_sense: str = "min",
    scales: Optional[Mapping[str, float]] = None,
    domain: str = "generic",
) -> DesignProgram:
    """Recover a DesignProgram from a goals dict by naming convention.

    Any ``minimum_x``/``min_x`` key becomes ``x >= limit``; any
    ``maximum_x``/``max_x`` key becomes ``x <= limit``. Non-finite limits
    (``inf`` bounds) are dropped: an unbounded constraint is not a constraint.
    """
    constraints: list[Constraint] = []
    if isinstance(goals, Mapping):
        for goal_key, raw_limit in goals.items():
            if goal_key in _NON_CONSTRAINT_GOALS:
                continue
            sense = None
            prefix = ""
            for pref, sns in _GOAL_PREFIXES:
                if goal_key.startswith(pref):
                    sense, prefix = sns, pref
                    break
            if sense is None:
                continue
            try:
                limit = float(raw_limit)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(limit):
                continue  # e.g. maximum_deflection: inf -> unconstrained
            metric = _resolve_metric_key(goal_key, prefix, state_keys)
            scale = None
            if scales:
                scale = scales.get(metric) or scales.get(goal_key)
            constraints.append(
                Constraint(
                    key=metric,
                    limit=limit,
                    sense=sense,
                    scale=float(scale) if scale else max(abs(limit), 1e-9),
                    label=metric,
                )
            )
    return DesignProgram(
        constraints=tuple(constraints),
        objective_key=objective_key,
        objective_sense=objective_sense,
        objective_ref=float(objective_ref) if objective_ref else 1.0,
        domain=domain,
    )


def compute_potential_v2(
    state: Mapping,
    program: DesignProgram,
    *,
    alpha: float = 5.0,
    tau: float = 0.05,
    objective_weight: float = 1.0,
    feasibility_offset: float = 0.0,
) -> float:
    """Negated Lagrangian of the design program: -w*objective - alpha*violations.

    ``feasibility_offset`` (M) adds a smooth step so that ANY violating design
    ranks below ANY satisfying one:

        Phi = -w*f(s) - alpha*V(s) - M*tanh(V(s)/tau)

    At M = 0 this is the plain Lagrangian, where the two terms are commensurable
    and a small violation can be bought with a large objective gain. That is a
    real failure, not a hypothetical: on the battery `thermal_hard` family the
    plain potential scored 1/3 where UNDIRECTED enumeration over the same
    candidates scored 3/3. A cell 2% over its temperature limit but charging in
    half the time scores -0.60 against a feasible cell's -0.83, because the
    violation is 0.02 in relative units while the objective gap is 0.33 -- alpha
    would have to exceed ~16 to overcome it, and no single alpha is right for
    every family.

    Note the fix is NOT to shrink the objective while infeasible: that makes
    infeasibility *cheaper* (the violating design stops paying for the objective
    it is failing to earn) and inverts the ranking further. The offset is the
    lexicographic construction -- satisfy the constraints, then optimise -- made
    smooth so the landscape stays differentiable across the boundary.
    """
    violation = program.total_violation(state, tau=tau)
    value = -objective_weight * program.objective(state) - alpha * violation
    if feasibility_offset > 0.0 and violation > 0.0:
        value -= feasibility_offset * math.tanh(violation / max(tau, 1e-9))
    return value


def compute_step_reward_v2(
    prev_state: Mapping,
    next_state: Mapping,
    program: DesignProgram,
    *,
    gamma: float = 0.99,
    alpha: float = 5.0,
    tau: float = 0.05,
    objective_weight: float = 1.0,
    feasibility_offset: float = 0.0,
) -> float:
    """Potential-based shaping term gamma*Phi(s') - Phi(s) under the v2 potential."""
    kwargs = dict(alpha=alpha, tau=tau, objective_weight=objective_weight,
                  feasibility_offset=feasibility_offset)
    return (
        gamma * compute_potential_v2(next_state, program, **kwargs)
        - compute_potential_v2(prev_state, program, **kwargs)
    )


# ── domain adapters ──────────────────────────────────────────────────────────
# Each adapter only has to say which metric is the objective and what scale
# makes it dimensionless. The constraints come from the goals dict for free.

_TRUSS_STATE_KEYS = (
    "mass", "fos_buckling", "fos_yielding", "deflection", "is_feasible",
)


def program_from_truss_spec(problem_spec: Mapping, initial_mass: float | None = None) -> DesignProgram:
    """DesignBench truss problem -> DesignProgram (objective: minimise mass).

    ``objective_ref`` prefers the LP optimum recorded in ``_metadata.optimal_mass``
    so that a reported mass ratio is interpretable as "x times the known optimum";
    it falls back to the mass cap, then the initial mass.
    """
    spec = problem_spec or {}
    goals = spec.get("goals") or {}
    if not isinstance(goals, Mapping):
        goals = {}
    meta = spec.get("_metadata") or {}
    if not isinstance(meta, Mapping):
        meta = {}

    def _pos(value) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) and numeric > 0.0 else None

    ref = (
        _pos(meta.get("optimal_mass"))
        or _pos(goals.get("maximum_mass"))
        or _pos(initial_mass)
        or 1.0
    )
    # A truss that deflects by a tenth of its own span is a mechanism, not a
    # design, whatever the goals say. Derive the bound from the geometry so it
    # scales with the problem; fall back to 1 m if the topology is unreadable.
    span = 0.0
    topology = spec.get("topology") or {}
    joints = topology.get("joints") if isinstance(topology, Mapping) else None
    if joints:
        xs, ys = [], []
        for joint in joints:
            position = (joint or {}).get("position") or []
            if len(position) >= 2:
                try:
                    xs.append(float(position[0]))
                    ys.append(float(position[1]))
                except (TypeError, ValueError):
                    continue
        if xs and ys:
            span = max(max(xs) - min(xs), max(ys) - min(ys))
    deflection_bound = 0.1 * span if span > 0 else 1.0

    program = program_from_goals(
        goals,
        state_keys=_TRUSS_STATE_KEYS,
        objective_key="mass",
        objective_ref=ref,
        objective_sense="min",
        domain="truss",
    )
    return replace(program, validity_bounds=(("deflection", deflection_bound),))


_BATTERY_STATE_KEYS = (
    "charge_time", "max_temperature", "max_plating", "min_neg_potential",
    "energy_density", "capacity", "success",
)


def program_from_battery_goals(
    goals: Mapping,
    *,
    objective_key: str = "charge_time",
    objective_ref: float | None = None,
    objective_sense: str = "min",
) -> DesignProgram:
    """DesignBench battery problem -> DesignProgram.

    Same convention parser as the truss adapter; only the objective differs.
    ``objective_ref`` defaults to the charge-time goal so the objective is
    expressed in units of "fraction of the allowed charge time".
    """
    if objective_ref is None and isinstance(goals, Mapping):
        try:
            objective_ref = float(goals.get("max_charge_time", 900.0))
        except (TypeError, ValueError):
            objective_ref = 900.0
    return program_from_goals(
        goals,
        state_keys=_BATTERY_STATE_KEYS,
        objective_key=objective_key,
        objective_ref=objective_ref or 900.0,
        objective_sense=objective_sense,
        domain="battery",
    )


# Back-compat alias: the truss adapter under the name the RL code first used.
constraints_from_spec = program_from_truss_spec
ProblemConstraints = DesignProgram
