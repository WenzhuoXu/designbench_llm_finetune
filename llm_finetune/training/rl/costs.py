"""
RL Cost Functions — Research Hook.

Cost functions penalize undesirable behaviors in rollouts. Unlike rewards,
costs are typically used in constrained RL (RLHF with safety constraints),
Lagrangian methods, or as negative reward components.

Use cases:
  - TokenBudgetCost: discourage verbose outputs (important for long-context models)
  - ConstraintViolationCost: penalize severity of constraint violations beyond a threshold
  - ComputationalCost: penalize expensive rollouts (many FEA calls)
  - CustomCost: any research-specific penalty you want to impose

To add a new cost function:
  1. Subclass CostFunction
  2. Implement compute() and name()
  3. Register in COST_REGISTRY

Usage:
    cost_fn = COST_REGISTRY["token_budget"](max_tokens_per_step=512)
    cost = cost_fn.compute(rollout, problem_spec)
    # Subtract from reward or use in Lagrangian objective
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional

from llm_finetune.training.rl.rewards import RolloutResult

log = logging.getLogger(__name__)


MAX_COST_VALUE = 100.0


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return numeric if math.isfinite(numeric) else default


def _bounded_cost(value: object, *, max_value: float = MAX_COST_VALUE) -> float:
    numeric = _finite_float(value, max_value)
    return max(0.0, min(float(max_value), numeric))


class CostFunction(ABC):
    """Abstract base class for RL cost functions.

    Research hook: implement compute() to penalize undesirable rollout properties.
    Costs are non-negative scalars. They can be used as:
      - Direct subtraction from reward: final_reward = reward - λ * cost
      - Lagrangian constraints: maximize reward s.t. E[cost] ≤ threshold
      - Separate logging signal for monitoring rollout behavior
    """

    @abstractmethod
    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        """Compute scalar cost for a completed rollout.

        Args:
            rollout: Complete rollout result.
            problem_spec: Problem specification dict.

        Returns:
            Non-negative scalar cost. Higher = worse.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Unique name for logging and registry."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TokenBudgetCost(CostFunction):
    """Cost for generating more tokens than the budget allows.

    Encourages concise, focused responses. Particularly important for
    thinking models (Qwen3, DeepSeek-R1) which may generate very long
    <think> blocks.

    cost = max(0, mean_tokens_per_step - max_tokens_per_step) / max_tokens_per_step

    Research note: for reasoning models, the thinking length may correlate
    with answer quality. Consider a softer penalty that allows longer thinking
    for harder problems.
    """

    def __init__(
        self,
        max_tokens_per_step: int = 512,
        thinking_tokens_per_step: int = 2048,
        penalize_thinking: bool = False,
    ):
        """
        Args:
            max_tokens_per_step: Soft limit on tokens per step (answer only).
            thinking_tokens_per_step: Soft limit on thinking tokens per step.
            penalize_thinking: If False, thinking tokens are not penalized.
        """
        self.max_tokens_per_step = max_tokens_per_step
        self.thinking_tokens_per_step = thinking_tokens_per_step
        self.penalize_thinking = penalize_thinking

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        if not rollout.token_counts:
            return 0.0

        token_counts = [_finite_float(t, 0.0) for t in rollout.token_counts]
        mean_tokens = sum(token_counts) / len(token_counts)
        budget = self.max_tokens_per_step
        excess = max(0.0, mean_tokens - budget)
        return _bounded_cost(excess / max(budget, 1))  # normalized cost in [0, max]

    def name(self) -> str:
        return "token_budget"


class ConstraintViolationCost(CostFunction):
    """Cost for severity of constraint violations in the final design.

    Measures how far the final design is from satisfying constraints:
        - FOS violation: max(0, TARGET_FOS - final_FOS)^2 / TARGET_FOS
        - Mass violation: max(0, final_mass - max_mass) / max_mass

    This provides a smooth, continuous signal even for infeasible designs,
    complementing binary feasibility reward.

    Research note: this cost can guide constraint satisfaction even when
    the model hasn't learned to achieve full feasibility yet.
    """

    TARGET_FOS = 1.5  # DesignBench requirement

    def __init__(
        self,
        fos_weight: float = 1.0,
        mass_weight: float = 0.5,
        quadratic: bool = True,
        max_cost: float = 10.0,
    ):
        self.fos_weight = fos_weight
        self.mass_weight = mass_weight
        self.quadratic = quadratic
        self.max_cost = max_cost

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        if not rollout.final_state:
            return 1.0  # Maximum cost for empty rollout

        final = rollout.final_state
        total_cost = 0.0

        # FOS constraints
        fos_b = max(0.0, _finite_float(final.get("fos_buckling"), 0.0))
        fos_y = max(0.0, _finite_float(final.get("fos_yielding"), 0.0))

        for fos in [fos_b, fos_y]:
            if fos < self.TARGET_FOS:
                violation = (self.TARGET_FOS - fos) / self.TARGET_FOS
                if self.quadratic:
                    violation = violation ** 2
                total_cost += self.fos_weight * violation / 2  # divide by 2 (two FOS constraints)

        # Mass constraint
        goal_mass = _finite_float(
            problem_spec.get("goals", {}).get("maximum_mass"), float("inf")
        )
        final_mass_raw = final.get("mass", 0.0)
        final_mass = _finite_float(final_mass_raw, float("inf"))
        if not math.isfinite(final_mass):
            total_cost += self.mass_weight * self.max_cost
        elif math.isfinite(goal_mass) and goal_mass > 0 and final_mass > goal_mass:
            mass_violation = (final_mass - goal_mass) / goal_mass
            if self.quadratic:
                mass_violation = mass_violation ** 2
            total_cost += self.mass_weight * min(self.max_cost, mass_violation)

        return _bounded_cost(total_cost, max_value=self.max_cost)

    def name(self) -> str:
        return "constraint_violation"


class ComputationalCost(CostFunction):
    """Cost for number of FEA evaluations used (computational efficiency).

    Each FEA call takes 5-15ms; over thousands of rollouts this becomes
    significant. This cost discourages redundant or exploratory actions
    that don't improve the design.

    cost = n_fea_calls / max_fea_calls

    Research note: on the H100 cluster with 104 CPUs, FEA can be parallelized
    heavily. Use this cost primarily to discourage unnecessarily long rollouts.
    """

    def __init__(self, max_fea_calls: int = 20):
        self.max_fea_calls = max_fea_calls

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        n_calls = rollout.n_fea_calls or len(rollout.action_sequence)
        return min(1.0, _finite_float(n_calls, 0.0) / max(self.max_fea_calls, 1))

    def name(self) -> str:
        return "computational"


class RepetitionCost(CostFunction):
    """Cost for repeating the same action (exploration penalty).

    Measures fraction of actions that are duplicates of previous actions
    in the same rollout. High repetition indicates the model is stuck
    in a local optimum.

    Research note: combine with StepEfficiencyReward for rollouts that
    make diverse, directional improvements.
    """

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        actions = rollout.action_sequence
        if len(actions) <= 1:
            return 0.0
        unique_actions = len(set(actions))
        repetition_rate = 1.0 - (unique_actions / len(actions))
        return repetition_rate

    def name(self) -> str:
        return "repetition"


class CompositeCost(CostFunction):
    """Weighted sum of multiple cost functions.

    Usage in config:
        cost_fn: composite
        cost_weights:
          token_budget: 0.1
          constraint_violation: 1.0
          repetition: 0.2
    """

    def __init__(self, weights: Optional[dict[str, float]] = None):
        if weights is None:
            weights = {"constraint_violation": 1.0}
        self.weights = weights
        self.components: dict[str, CostFunction] = {}
        for name, weight in weights.items():
            if name in COST_REGISTRY and weight != 0.0:
                self.components[name] = COST_REGISTRY[name]()

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        total = 0.0
        for name, component in self.components.items():
            weight = _finite_float(self.weights.get(name, 0.0), 0.0)
            total += weight * _bounded_cost(component.compute(rollout, problem_spec))
        return _bounded_cost(total)

    def get_breakdown(self, rollout: RolloutResult, problem_spec: dict) -> dict[str, float]:
        return {
            name: _bounded_cost(component.compute(rollout, problem_spec))
            for name, component in self.components.items()
        }

    def name(self) -> str:
        return "composite_cost"


class DeadEndAvoidanceCost(CostFunction):
    """Penalty for entering a structural dead-end configuration (§2.4).

    Detects dead-ends via two simultaneous signals:
      1. Monotone-worsening min-FOS over the last window_size steps.
      2. All recent actions are parameter-only (SCALE_PARAM / MODIFY_PARAM),
         indicating the policy is stuck in a local-search loop.

    Self-gating: returns 0.0 when neither condition holds, so it is silent on
    states with no matching dead-end signature.

    Config: cost_fn: dead_end_avoidance
    Params: dead_end_cost (default 0.20 per §6.2 ξ), window_size (3).
    """

    def __init__(self, dead_end_cost: float = 0.20, window_size: int = 3) -> None:
        self.dead_end_cost = dead_end_cost
        self.window_size = window_size

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        history = rollout.state_history
        actions = rollout.action_sequence
        if len(history) < self.window_size + 1 or len(actions) < self.window_size:
            return 0.0

        def _min_fos(s: dict) -> float:
            return min(
                max(0.0, _finite_float(s.get("fos_buckling"), 0.0)),
                max(0.0, _finite_float(s.get("fos_yielding"), 0.0)),
            )

        window_states = history[-(self.window_size + 1):]
        fos_vals = [_min_fos(s) for s in window_states]
        monotone_worse = all(fos_vals[i] >= fos_vals[i + 1] for i in range(len(fos_vals) - 1))
        if not monotone_worse:
            return 0.0

        recent_actions = actions[-self.window_size:]
        _local = {"SCALE_PARAM", "SCALE_MULTI_PARAM", "MODIFY_PARAM"}
        all_local = all(
            any(a.upper().startswith(p) for p in _local)
            for a in recent_actions
        )
        return self.dead_end_cost if all_local else 0.0

    def name(self) -> str:
        return "dead_end_avoidance"


# ── Registry ──────────────────────────────────────────────────────────────────
COST_REGISTRY: dict[str, type[CostFunction]] = {
    "token_budget": TokenBudgetCost,
    "constraint_violation": ConstraintViolationCost,
    "computational": ComputationalCost,
    "repetition": RepetitionCost,
    "composite": CompositeCost,
    "dead_end_avoidance": DeadEndAvoidanceCost,
}


def build_cost_from_config(cfg) -> Optional[CostFunction]:
    """Build a CostFunction from an OmegaConf config.

    Config format:
        cost_fn: composite_cost
        cost_weights:
          constraint_violation: 1.0
          token_budget: 0.1

    Returns None if no cost function is configured (cost_fn: null).
    """
    from omegaconf import DictConfig, OmegaConf
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    cost_fn_name = cfg.get("cost_fn", None)
    if cost_fn_name is None:
        return None

    if cost_fn_name == "composite":
        weights = cfg.get("cost_weights", {})
        return CompositeCost(weights=weights)

    if cost_fn_name not in COST_REGISTRY:
        raise ValueError(
            f"Unknown cost function: {cost_fn_name!r}. "
            f"Available: {list(COST_REGISTRY.keys())}"
        )
    return COST_REGISTRY[cost_fn_name]()
