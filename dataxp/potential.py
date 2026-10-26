"""Domain-agnostic design-program potential (Phi_v2), vendored from
llm_finetune/training/rl/posterior/potential.py and extended for stochastic
simulators.

Extensions over the vendored version:
  * aggregate_metrics(): chance-constraint aggregation of per-seed metric
    samples (adverse-direction quantile per constraint, mean objective).
  * auto_feasibility_offset(): the turn5 SS8 enabling condition, computed from
    the initial state alone.
Phi under noise is scored as E[Phi] over replicates (each replicate pays its
own hinge), while FEASIBILITY is judged on the quantile-aggregated state.

Target runtime: Python >= 3.8, numpy/scipy only (trace: 3.10.9 / 1.23.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence


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


def hinge(x: float, tau: float = 0.05) -> float:
    """Smooth hinge: tau*softplus(x/tau) -> max(0, x) as tau -> 0."""
    if tau <= 0.0:
        return max(0.0, x)
    return tau * softplus(x / tau)


_GOAL_PREFIXES = (
    ("minimum_", "lower"),
    ("maximum_", "upper"),
    ("min_", "lower"),
    ("max_", "upper"),
)

_NON_CONSTRAINT_GOALS = frozenset({"objective", "objective_sense", "objective_ref"})


@dataclass(frozen=True)
class Constraint:
    """One inequality of a design program, in the units of its own metric."""

    key: str          # metric name to read out of the state dict
    limit: float
    sense: str        # "upper" => value <= limit ; "lower" => value >= limit
    scale: float = 1.0
    label: str = ""

    def violation(self, state: Mapping, *, tau: float = 0.05, missing: float = 0.0) -> float:
        raw = state.get(self.key, None)
        if raw is None:
            return missing
        value = _finite_or_default(raw, float("nan"))
        if not math.isfinite(value):
            # A diverged simulation is a maximally violated constraint, never
            # a missing one: it must not look attractive.
            return 1.0
        scale = self.scale if self.scale > 0 else max(abs(self.limit), 1e-9)
        if self.sense == "lower":
            gap = (self.limit - value) / scale
        else:
            gap = (value - self.limit) / scale
        return min(hinge(gap, tau), 100.0)


@dataclass(frozen=True)
class DesignProgram:
    """(objective, constraints) descriptor of a design problem - no domain code."""

    constraints: tuple = ()
    objective_key: Optional[str] = None
    objective_sense: str = "min"
    objective_ref: float = 1.0
    domain: str = "generic"
    objective_must_be_positive: bool = True
    violation_ref: float = 1.0
    # (metric_key, upper_bound_on_absolute_value): the validity envelope.
    # States beyond it are outside the simulator's domain of validity whatever
    # the goals say; search WILL optimise into any region left unbounded (D9).
    validity_bounds: tuple = ()

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)

    def violations(self, state: Mapping, *, tau: float = 0.05) -> dict:
        return {c.label or c.key: c.violation(state, tau=tau) for c in self.constraints}

    def total_violation(self, state: Mapping, *, tau: float = 0.05,
                        normalised: bool = True) -> float:
        raw = sum(c.violation(state, tau=tau) for c in self.constraints)
        if normalised and self.violation_ref > 0.0:
            return raw / self.violation_ref
        return raw

    def with_violation_scale(self, initial_state: Mapping, *, tau: float = 0.05,
                             floor: float = 1e-3) -> "DesignProgram":
        """Copy with alpha made dimensionless for this problem (turn5 SS8)."""
        raw = self.total_violation(initial_state, tau=tau, normalised=False)
        return replace(self, violation_ref=max(raw, floor))

    def is_valid(self, state: Mapping) -> bool:
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
        value = _finite_or_default(state.get(self.objective_key), float("nan"))
        ref = self.objective_ref if abs(self.objective_ref) > 1e-12 else 1.0
        if not math.isfinite(value):
            return 10.0
        if self.objective_must_be_positive and value <= 0.0:
            return 10.0
        if not self.is_valid(state):
            return 10.0
        normalised = value / ref
        return normalised if self.objective_sense == "min" else -normalised

    def is_feasible(self, state: Mapping, *, tol: float = 1e-9) -> bool:
        if not self.is_valid(state):
            return False
        return self.total_violation(state, tau=0.0) <= tol


def _resolve_metric_key(goal_key: str, sense_prefix: str,
                        state_keys: Optional[Sequence[str]]) -> str:
    stripped = goal_key[len(sense_prefix):]
    if state_keys is None:
        return stripped
    keys = set(state_keys)
    if goal_key in keys:      # metric named exactly like the goal (PyBaMM style)
        return goal_key
    return stripped


def program_from_goals(
    goals: Mapping,
    *,
    state_keys: Optional[Sequence[str]] = None,
    objective_key: Optional[str] = None,
    objective_ref: Optional[float] = None,
    objective_sense: str = "min",
    scales: Optional[Mapping] = None,
    domain: str = "generic",
) -> DesignProgram:
    """Recover a DesignProgram from a goals dict by naming convention.

    ``minimum_x``/``min_x`` => x >= limit; ``maximum_x``/``max_x`` => x <= limit.
    Non-finite limits are dropped: an unbounded constraint is not a constraint.
    """
    constraints = []
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
                continue
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
    offset_on_hard: bool = False,
) -> float:
    """Negated Lagrangian: -w*objective - alpha*violations [- M*tanh(V/tau)].

    offset_on_hard=True applies the offset to the HARD violation (tau=0), so a
    satisfying design pays nothing and any violating one pays up to M - the
    "rank any violating design below any satisfying one" the offset exists
    for (turn5 SS8). With the soft hinge (the run1 form) a feasible design
    near a limit also paid most of M: the soft hinge is 0.035 at gap 0, which
    is already several tau after violation scaling."""
    violation = program.total_violation(state, tau=tau)
    value = -objective_weight * program.objective(state) - alpha * violation
    if feasibility_offset > 0.0:
        v_off = program.total_violation(state, tau=0.0) if offset_on_hard else violation
        if v_off > 0.0:
            value -= feasibility_offset * math.tanh(v_off / max(tau, 1e-9))
    return value


OFFSET_RULES = ("raw", "infeasible", "legacy_scaled")


def auto_feasibility_offset(program: DesignProgram, initial_state: Mapping,
                            *, alpha: float = 5.0, tau: float = 0.05,
                            objective_range: float = 1.0,
                            M: float = 5.0, rule: str = "infeasible") -> float:
    """turn5 SS8 rule: a feasible start needs no offset; an infeasible one gets
    it when the objective's dynamic range can outbid alpha x the typical
    violation, measured by the same initial-state statistic
    with_violation_scale uses - the RAW (unscaled) initial violation.
    Deterministic, per problem, no human input.

    rule="infeasible"     (default) on whenever the start is infeasible. On
                          720 problems x 6 domains this removed the moves that
                          step out of the feasible set and back (circuit 146 -> 7,
                          seir 25 -> 0) and raised final feasibility (circuit
                          0.82 -> 0.91) at +7-9% objective in those two domains.
    rule="raw"            the turn5 SS8 condition above, as written. With
                          violation scaling it almost never fires (raw initial
                          violations are 0.2-14, so alpha*V0 >= 1): same
                          trajectories as legacy on 5/6 domains.
    rule="legacy_scaled"  the run1 behaviour (2026-09), kept only to reproduce
                          run1. It tested the violation AFTER
                          with_violation_scale, which is 1 by construction, so
                          the offset was on only when V_raw0 < ~2e-4; and its
                          "starts feasible" test (v0 <= 0) could never fire
                          because the soft hinge is never exactly 0.
    """
    if rule not in OFFSET_RULES:
        raise ValueError("unknown offset rule %r" % (rule,))
    if rule == "legacy_scaled":
        v0 = program.total_violation(initial_state, tau=tau)
        if v0 <= 0.0:
            return 0.0
        return M if alpha * v0 < objective_range else 0.0
    if program.is_feasible(initial_state):
        return 0.0          # starts feasible: plain Lagrangian is fine
    if rule == "infeasible":
        return M
    raw0 = program.total_violation(initial_state, tau=tau, normalised=False)
    return M if alpha * raw0 < objective_range else 0.0


# ── stochastic aggregation ───────────────────────────────────────────────────

def aggregate_metrics(samples: Mapping, program: DesignProgram,
                      *, q: float = 0.9) -> dict:
    """Collapse per-seed metric samples into ONE conservative metrics dict.

    ``samples`` maps metric key -> sequence of replicate values. For each
    CONSTRAINED metric the adverse-direction q-quantile is taken (upper
    constraint -> high quantile, lower constraint -> low quantile), so
    ``is_feasible`` on the result is a chance constraint at level q. The
    objective and every unconstrained metric aggregate by mean. Non-finite
    replicates propagate as non-finite (a diverged draw must stay visible).
    """
    sense_by_key = {c.key: c.sense for c in program.constraints}
    out = {}
    for key, values in samples.items():
        vals = [float(v) for v in values]
        if not vals:
            continue
        if any(not math.isfinite(v) for v in vals):
            out[key] = float("inf") if sense_by_key.get(key) != "lower" else float("-inf")
            continue
        sense = sense_by_key.get(key)
        if sense == "upper":
            out[key] = _quantile(vals, q)
        elif sense == "lower":
            out[key] = _quantile(vals, 1.0 - q)
        else:
            out[key] = sum(vals) / len(vals)
    return out


def _quantile(vals, q: float) -> float:
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac
