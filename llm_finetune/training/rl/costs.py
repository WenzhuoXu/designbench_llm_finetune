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
from abc import ABC, abstractmethod
from typing import Optional

from llm_finetune.training.rl.rewards import RolloutResult

log = logging.getLogger(__name__)


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

        mean_tokens = sum(rollout.token_counts) / len(rollout.token_counts)
        budget = self.max_tokens_per_step
        excess = max(0.0, mean_tokens - budget)
        return excess / budget  # normalized cost in [0, ∞)

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
    ):
        self.fos_weight = fos_weight
        self.mass_weight = mass_weight
        self.quadratic = quadratic

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        if not rollout.final_state:
            return 1.0  # Maximum cost for empty rollout

        final = rollout.final_state
        total_cost = 0.0

        # FOS constraints
        fos_b = float(final.get("fos_buckling", 0.0) or 0.0)
        fos_y = float(final.get("fos_yielding", 0.0) or 0.0)

        for fos in [fos_b, fos_y]:
            if fos < self.TARGET_FOS:
                violation = (self.TARGET_FOS - fos) / self.TARGET_FOS
                if self.quadratic:
                    violation = violation ** 2
                total_cost += self.fos_weight * violation / 2  # divide by 2 (two FOS constraints)

        # Mass constraint
        goal_mass = float(
            problem_spec.get("goals", {}).get("maximum_mass", float("inf"))
        )
        final_mass = float(final.get("mass", 0.0) or 0.0)
        if goal_mass < float("inf") and final_mass > goal_mass:
            mass_violation = (final_mass - goal_mass) / goal_mass
            if self.quadratic:
                mass_violation = mass_violation ** 2
            total_cost += self.mass_weight * mass_violation

        return total_cost

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
        return min(1.0, n_calls / self.max_fea_calls)

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
            total += self.weights.get(name, 0.0) * component.compute(rollout, problem_spec)
        return total

    def get_breakdown(self, rollout: RolloutResult, problem_spec: dict) -> dict[str, float]:
        return {
            name: component.compute(rollout, problem_spec)
            for name, component in self.components.items()
        }

    def name(self) -> str:
        return "composite_cost"


# ── Registry ──────────────────────────────────────────────────────────────────
COST_REGISTRY: dict[str, type[CostFunction]] = {
    "token_budget": TokenBudgetCost,
    "constraint_violation": ConstraintViolationCost,
    "computational": ComputationalCost,
    "repetition": RepetitionCost,
    "composite": CompositeCost,
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
