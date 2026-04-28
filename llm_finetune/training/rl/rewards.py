"""
RL Reward Functions — Research Hook.

Defines how rollout trajectories are scored during GRPO training.
This is a primary research variable: different reward functions produce
different optimization objectives and learned behaviors.

GRPO uses a group of K rollouts per prompt. Each rollout gets a scalar reward.
The group mean is subtracted (group normalization) to compute advantages.

To add a new reward function:
  1. Subclass RewardFunction
  2. Implement compute() and name()
  3. Register in REWARD_REGISTRY
  4. Reference in configs/rl/grpo_truss.yaml as: reward_fn: "composite"
     with reward_weights: {feasibility: 1.0, fos_improvement: 0.5, ...}

Usage:
    reward_fn = REWARD_REGISTRY["composite"](
        weights={"feasibility": 1.0, "fos_improvement": 0.5, "grammar": 0.2}
    )
    reward = reward_fn.compute(rollout, problem_spec)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Complete result of one GRPO rollout episode.

    A rollout represents one LLM trajectory through a truss design problem:
    the model is shown the problem, generates actions, each action is executed
    via FEA, and the model sees the updated state.

    Research hook: extend this dataclass to add fields like:
      - reasoning_quality: LLM-judged quality of reasoning steps
      - node_visits: MCTS visit count for explored states
      - policy_entropy: entropy of action distribution at each step
      - thought_depth: length of thinking blocks per step
    """
    problem_id: str
    action_sequence: list[str]            # Parsed grammar actions per step
    raw_outputs: list[str]                # Raw LLM outputs per step (including thinking)
    state_history: list[dict]             # FEA state at each step: {mass, fos_b, fos_y, feasible}
    final_state: dict                     # Final design state after all actions
    initial_state: dict                   # Problem initial state
    token_counts: list[int]               # Tokens generated per step
    parse_success: list[bool]             # Grammar parse success per step
    reaches_solution: bool                # Did final state satisfy all constraints?
    n_fea_calls: int = 0                  # Total FEA evaluations (cost signal)
    n_steps: int = 0                      # Number of completed steps
    error_message: str = ""              # Error if rollout failed
    metadata: dict = field(default_factory=dict)  # Extra info for logging
    step_records: list[dict] = field(default_factory=list)
    tree_metrics: dict = field(default_factory=dict)
    action_class_sequence: list[str] = field(default_factory=list)
    prediction_texts: list[str] = field(default_factory=list)
    candidate_sets: list[dict] = field(default_factory=list)


class RewardFunction(ABC):
    """Abstract base class for GRPO reward functions.

    Research hook: implement compute() to define your reward signal.
    The reward must be a scalar float — it will be normalized within the
    GRPO group before computing advantages.
    """

    @abstractmethod
    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        """Compute scalar reward for a completed rollout.

        Args:
            rollout: Complete rollout result with state history and actions.
            problem_spec: Problem specification dict (goals, topology, loading).

        Returns:
            Scalar reward. Higher is better.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Unique name for logging and registry."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class FeasibilityReward(RewardFunction):
    """Binary reward: +1.0 if final design is feasible, 0.0 otherwise.

    Simplest possible reward: did the design satisfy all constraints?
    Sparse but clear signal. Best combined with a dense shaping reward.

    Research note: pure feasibility reward may cause the model to find
    any feasible solution regardless of mass efficiency.
    """

    def __init__(self, reward_value: float = 1.0):
        self.reward_value = reward_value

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        return self.reward_value if rollout.reaches_solution else 0.0

    def name(self) -> str:
        return "feasibility"


class FOSImprovementReward(RewardFunction):
    """Continuous reward based on Factor of Safety improvement.

    Measures how much the design improved toward satisfying FOS constraints:
        reward = min(FOS_buckling, FOS_yielding) - min(initial_FOS_b, initial_FOS_y)

    Normalized by target FOS (1.5 for DesignBench) so reward ∈ [-1, 1].

    Research note: this reward provides dense feedback even for infeasible designs,
    helping the model learn directional improvements.
    """

    TARGET_FOS = 1.5  # DesignBench requirement

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        if not rollout.state_history:
            return 0.0

        initial_fos = self._min_fos(rollout.initial_state)
        final_fos = self._min_fos(rollout.final_state)
        delta = final_fos - initial_fos

        if self.normalize:
            # Normalize by target FOS
            delta = delta / self.TARGET_FOS
            return max(-1.0, min(1.0, delta))
        return delta

    def _min_fos(self, state: dict) -> float:
        fos_b = float(state.get("fos_buckling", 0.0) or 0.0)
        fos_y = float(state.get("fos_yielding", 0.0) or 0.0)
        if fos_b <= 0 and fos_y <= 0:
            return 0.0
        if fos_b <= 0:
            return fos_y
        if fos_y <= 0:
            return fos_b
        return min(fos_b, fos_y)

    def name(self) -> str:
        return "fos_improvement"


class MassReductionReward(RewardFunction):
    """Reward for reducing structural mass while maintaining feasibility.

    Computes normalized mass reduction:
        reward = (initial_mass - final_mass) / goal_mass

    Only positive when mass decreases. Can be combined with FeasibilityReward
    to create a reward that encourages lightweight feasible designs.

    Research note: be careful about unconstrained mass minimization — the model
    may learn to make structures infeasibly light. Use with FeasibilityReward.
    """

    def __init__(self, only_if_feasible: bool = False):
        self.only_if_feasible = only_if_feasible

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        if self.only_if_feasible and not rollout.reaches_solution:
            return 0.0

        initial_mass = float(rollout.initial_state.get("mass", 0.0) or 0.0)
        final_mass = float(rollout.final_state.get("mass", 0.0) or 0.0)

        if initial_mass <= 0:
            return 0.0

        # Get goal mass from problem spec
        goal_mass = float(
            problem_spec.get("goals", {}).get("maximum_mass", initial_mass) or initial_mass
        )

        mass_reduction = (initial_mass - final_mass) / goal_mass
        return float(mass_reduction)

    def name(self) -> str:
        return "mass_reduction"


class GrammarComplianceReward(RewardFunction):
    """Reward for generating valid grammar actions (format compliance).

    Fraction of steps where the model's output was parseable as a valid
    grammar action. Encourages the model to maintain the structured output format.

    Range: [0.0, 1.0]

    Research note: high grammar compliance is necessary but not sufficient.
    Models quickly learn format compliance; this reward matters most in early training.
    """

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        if not rollout.parse_success:
            return 0.0
        return sum(rollout.parse_success) / len(rollout.parse_success)

    def name(self) -> str:
        return "grammar_compliance"


class StepEfficiencyReward(RewardFunction):
    """Penalty for long trajectories — encourages efficient optimization paths.

    A model that reaches the solution in fewer steps is rewarded more.
    This discourages exhaustive brute-force strategies.

    reward = -α * (n_steps / max_steps) if not solution
           = β * (1 - n_steps / max_steps) if solution

    Research note: too strong an efficiency penalty may prevent exploration.
    Start with a small coefficient (0.1-0.2) and increase gradually.
    """

    def __init__(
        self,
        max_steps: int = 20,
        solution_bonus: float = 0.2,
        step_penalty: float = 0.05,
    ):
        self.max_steps = max_steps
        self.solution_bonus = solution_bonus
        self.step_penalty = step_penalty

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        n_steps = rollout.n_steps or len(rollout.action_sequence)
        step_fraction = n_steps / max(self.max_steps, 1)

        if rollout.reaches_solution:
            # Reward for solving efficiently
            return self.solution_bonus * (1.0 - step_fraction)
        else:
            # Penalty for using steps without solving
            return -self.step_penalty * step_fraction

    def name(self) -> str:
        return "step_efficiency"


class ProgressReward(RewardFunction):
    """Step-level reward: positive for each step that improves FOS.

    Computes average per-step FOS improvement:
        reward = mean(max(0, FOS_{t+1} - FOS_t) for all steps)

    This provides a dense, step-level signal rather than just outcome-based reward.

    Research note: corresponds to a process reward model (PRM) that uses
    FEA improvement as the per-step signal. Can be used as a proxy for PRM labels.
    """

    TARGET_FOS = 1.5

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        if len(rollout.state_history) < 2:
            return 0.0

        improvements = []
        for i in range(1, len(rollout.state_history)):
            prev = rollout.state_history[i - 1]
            curr = rollout.state_history[i]
            prev_fos = self._min_fos(prev)
            curr_fos = self._min_fos(curr)
            improvements.append(max(0.0, curr_fos - prev_fos))

        if not improvements:
            return 0.0
        return sum(improvements) / len(improvements) / self.TARGET_FOS

    def _min_fos(self, state: dict) -> float:
        fos_b = float(state.get("fos_buckling", 0.0) or 0.0)
        fos_y = float(state.get("fos_yielding", 0.0) or 0.0)
        return min(fos_b, fos_y) if fos_b > 0 and fos_y > 0 else max(fos_b, fos_y)

    def name(self) -> str:
        return "progress"


class CompositeReward(RewardFunction):
    """Weighted sum of multiple reward functions.

    This is the recommended reward for most experiments — combine dense
    (FOS improvement, grammar compliance) and sparse (feasibility) signals.

    Example config:
        reward_fn: composite
        reward_weights:
          feasibility: 1.0
          fos_improvement: 0.5
          mass_reduction: 0.2
          grammar_compliance: 0.1
          step_efficiency: 0.1

    Research note: weight tuning matters significantly. Start with feasibility
    as the dominant signal and add dense rewards to guide exploration.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None, **component_kwargs):
        if weights is None:
            weights = {
                "feasibility": 1.0,
                "fos_improvement": 0.5,
                "grammar_compliance": 0.1,
            }
        self.weights = weights
        self.components: dict[str, RewardFunction] = {}

        # Instantiate component reward functions
        for name, weight in weights.items():
            if name in REWARD_REGISTRY and weight != 0.0:
                self.components[name] = REWARD_REGISTRY[name](**component_kwargs.get(name, {}))

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        total = 0.0
        for name, component in self.components.items():
            weight = self.weights.get(name, 0.0)
            component_reward = component.compute(rollout, problem_spec)
            total += weight * component_reward
        return total

    def get_breakdown(self, rollout: RolloutResult, problem_spec: dict) -> dict[str, float]:
        """Return per-component rewards (for logging)."""
        return {
            name: component.compute(rollout, problem_spec)
            for name, component in self.components.items()
        }

    def name(self) -> str:
        return "composite"


# ── Registry ──────────────────────────────────────────────────────────────────
REWARD_REGISTRY: dict[str, type[RewardFunction]] = {
    "feasibility": FeasibilityReward,
    "fos_improvement": FOSImprovementReward,
    "mass_reduction": MassReductionReward,
    "grammar_compliance": GrammarComplianceReward,
    "step_efficiency": StepEfficiencyReward,
    "progress": ProgressReward,
    "composite": CompositeReward,
}


def build_reward_from_config(cfg) -> RewardFunction:
    """Build a RewardFunction from an OmegaConf config.

    Config format (in configs/rl/grpo_truss.yaml):
        reward_fn: composite
        reward_weights:
          feasibility: 1.0
          fos_improvement: 0.5
          grammar_compliance: 0.1
    """
    from omegaconf import DictConfig, OmegaConf
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    reward_fn_name = cfg.get("reward_fn", "composite")

    if reward_fn_name == "composite":
        weights = cfg.get("reward_weights", {})
        return CompositeReward(weights=weights)

    if reward_fn_name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward function: {reward_fn_name!r}. "
            f"Available: {list(REWARD_REGISTRY.keys())}"
        )
    return REWARD_REGISTRY[reward_fn_name]()
