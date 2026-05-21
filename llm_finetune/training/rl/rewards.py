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
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


MAX_REWARD_ABS = 10_000.0


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return numeric if math.isfinite(numeric) else default


def _bounded_reward(value: object, *, limit: float = MAX_REWARD_ABS) -> float:
    numeric = _finite_float(value, 0.0)
    return max(-limit, min(limit, numeric))


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
        return _bounded_reward(delta)

    def _min_fos(self, state: dict) -> float:
        fos_b = max(0.0, _finite_float(state.get("fos_buckling"), 0.0))
        fos_y = max(0.0, _finite_float(state.get("fos_yielding"), 0.0))
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

        initial_mass = _finite_float(rollout.initial_state.get("mass"), 0.0)
        final_mass = _finite_float(rollout.final_state.get("mass"), initial_mass)

        if initial_mass <= 0:
            return 0.0

        # Get goal mass from problem spec
        goal_mass = _finite_float(
            problem_spec.get("goals", {}).get("maximum_mass"), initial_mass
        )
        if goal_mass <= 0:
            return 0.0

        mass_reduction = (initial_mass - final_mass) / goal_mass
        return _bounded_reward(mass_reduction)

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
        fos_b = max(0.0, _finite_float(state.get("fos_buckling"), 0.0))
        fos_y = max(0.0, _finite_float(state.get("fos_yielding"), 0.0))
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
            weight = _finite_float(self.weights.get(name, 0.0), 0.0)
            component_reward = _bounded_reward(component.compute(rollout, problem_spec))
            total += weight * component_reward
        return _bounded_reward(total)

    def get_breakdown(self, rollout: RolloutResult, problem_spec: dict) -> dict[str, float]:
        """Return per-component rewards (for logging)."""
        return {
            name: _bounded_reward(component.compute(rollout, problem_spec))
            for name, component in self.components.items()
        }

    def name(self) -> str:
        return "composite"


# ── Posterior reward hooks (§1-§4 of plan_posterior_reward_walkthrough.md) ────

def _action_class(action_str: str) -> str:
    """Extract action type prefix from a grammar action string."""
    if not action_str:
        return "UNKNOWN"
    for prefix in (
        "SCALE_MULTI_PARAM", "SCALE_PARAM", "ADD_MEMBER",
        "MODIFY_PARAM", "REMOVE_MEMBER", "MOVE_JOINT",
    ):
        if action_str.upper().startswith(prefix):
            return prefix
    return "UNKNOWN"


class LagrangianPotentialReward(RewardFunction):
    """Potential-based per-step reward r_env = γΦ(s') - Φ(s) (§1).

    Sums discounted potential differences along the trajectory.
    Provides dense, policy-invariant shaping aligned with the mass/FOS objective.

    Config: reward_fn: lagrangian_potential
    Params: alpha (Lagrangian constraint price, default 5.0), gamma (0.99).
    """

    def __init__(self, alpha: float = 5.0, gamma: float = 0.99) -> None:
        self.alpha = alpha
        self.gamma = gamma

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        from llm_finetune.training.rl.posterior.potential import compute_step_reward
        history = rollout.state_history
        if len(history) < 2:
            return 0.0
        initial_mass = _finite_float((rollout.initial_state or {}).get("mass"), 1.0)
        total = 0.0
        for t in range(len(history) - 1):
            total += compute_step_reward(
                history[t],
                history[t + 1],
                initial_mass=initial_mass,
                gamma=self.gamma,
                alpha=self.alpha,
            )
        return _bounded_reward(total)

    def name(self) -> str:
        return "lagrangian_potential"


class MacroCompletionReward(RewardFunction):
    """Bonus for completing a known macro-action pattern (§2.3).

    Detects two-step action-class sequences that correspond to multi-step
    design moves (e.g. ADD_MEMBER → SCALE_PARAM for reinforce-and-tune).
    Self-gating: returns 0.0 when no pattern is matched.

    Config: reward_fn: macro_completion
    Params: macro_bonus (default 0.05 per §6.2 β₂).
    """

    _PATTERNS: tuple[tuple[str, str], ...] = (
        ("ADD_MEMBER", "SCALE_PARAM"),
        ("ADD_MEMBER", "MODIFY_PARAM"),
        ("REMOVE_MEMBER", "MOVE_JOINT"),
        ("REMOVE_MEMBER", "SCALE_PARAM"),
        ("SCALE_PARAM", "ADD_MEMBER"),
        ("MOVE_JOINT", "SCALE_PARAM"),
    )

    def __init__(self, macro_bonus: float = 0.05) -> None:
        self.macro_bonus = macro_bonus

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        actions = rollout.action_sequence
        if len(actions) < 2:
            return 0.0
        types = [_action_class(a) for a in actions]
        for i in range(len(types) - 1):
            if (types[i], types[i + 1]) in self._PATTERNS:
                return self.macro_bonus
        return 0.0

    def name(self) -> str:
        return "macro_completion"


class ForwardPredictionReward(RewardFunction):
    """Reward grounding CoT to FEA outcomes via forward prediction (§4.1).

    Parses <predict>mass: …, fos: …</predict> tags from raw LLM outputs and
    compares to the actual FEA result for that step. Returns normalised
    agreement score × prediction_weight. Returns 0.0 when no tags are found.

    Config: reward_fn: forward_prediction
    Params: prediction_weight (default 0.10 per §6.2 μ).
    """

    def __init__(self, prediction_weight: float = 0.10) -> None:
        self.prediction_weight = prediction_weight

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        import re
        scores: list[float] = []
        for i, raw in enumerate(rollout.raw_outputs or []):
            m = re.search(r"<predict>(.*?)</predict>", raw, re.DOTALL | re.IGNORECASE)
            if m is None:
                continue
            if i + 1 >= len(rollout.state_history):
                continue
            score = self._compare(m.group(1), rollout.state_history[i + 1])
            scores.append(score)
        if not scores:
            return 0.0
        return self.prediction_weight * (sum(scores) / len(scores))

    @staticmethod
    def _parse_float(text: str, pattern: str) -> Optional[float]:
        import re
        m = re.search(pattern, text, re.IGNORECASE)
        return _finite_float(m.group(1), 0.0) if m else None

    def _compare(self, prediction_text: str, actual: dict) -> float:
        mass_pred = self._parse_float(prediction_text, r"mass[:\s]+([0-9.]+)")
        fos_pred = self._parse_float(prediction_text, r"fos[:\s]+([0-9.]+)")
        if mass_pred is None and fos_pred is None:
            return 0.0
        n_terms = 0
        score = 0.0
        actual_mass = _finite_float(actual.get("mass"), 0.0)
        if mass_pred is not None and actual_mass > 0:
            score += max(0.0, 1.0 - abs(mass_pred - actual_mass) / actual_mass)
            n_terms += 1
        actual_fos = min(
            max(0.0, _finite_float(actual.get("fos_buckling"), 0.0)),
            max(0.0, _finite_float(actual.get("fos_yielding"), 0.0)),
        )
        if fos_pred is not None and actual_fos > 0:
            score += max(0.0, 1.0 - abs(fos_pred - actual_fos) / actual_fos)
            n_terms += 1
        return score / n_terms if n_terms > 0 else 0.0

    def name(self) -> str:
        return "forward_prediction"


class StagnationEscapeReward(RewardFunction):
    """Bonus for breaking a stagnation plateau by switching action class (§4.3).

    Detects stagnation: last window_size steps improved min-FOS by less than
    plateau_tolerance. Rewards the step that breaks the plateau with a
    different action class than the plateau's dominant class.
    Self-gating: returns 0.0 outside stagnation.

    Config: reward_fn: stagnation_escape
    Params: escape_bonus (default 0.30 per §6.2 κ), window_size (4),
            plateau_tolerance (0.02).
    """

    def __init__(
        self,
        escape_bonus: float = 0.30,
        window_size: int = 4,
        plateau_tolerance: float = 0.02,
    ) -> None:
        self.escape_bonus = escape_bonus
        self.window_size = window_size
        self.plateau_tolerance = plateau_tolerance

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        history = rollout.state_history
        actions = rollout.action_sequence
        if len(history) < self.window_size + 1 or not actions:
            return 0.0

        def _min_fos(s: dict) -> float:
            return min(
                max(0.0, _finite_float(s.get("fos_buckling"), 0.0)),
                max(0.0, _finite_float(s.get("fos_yielding"), 0.0)),
            )

        window_states = history[-(self.window_size + 1):-1]
        fos_vals = [_min_fos(s) for s in window_states]
        if max(fos_vals) - min(fos_vals) >= self.plateau_tolerance:
            return 0.0  # not stagnant

        window_actions = actions[-(self.window_size + 1):-1] if len(actions) > 1 else []
        recent_action = actions[-1]
        recent_type = _action_class(recent_action)
        if not window_actions:
            return 0.0
        window_types = [_action_class(a) for a in window_actions]
        dominant = max(set(window_types), key=window_types.count)
        return self.escape_bonus if recent_type != dominant else 0.0

    def name(self) -> str:
        return "stagnation_escape"


# ── Tree-signal rewards (§3 of plan_posterior_reward_walkthrough.md) ─────────


class OnlineTreeAdvantageReward(RewardFunction):
    """Tree-expanded advantage as training reward (§3.1).

    When the trainer runs online MCTS (use_tree_expansion=True in config),
    it stores the pre-computed tree advantage in rollout.tree_metrics under
    the key "tree_advantage".  This reward simply reads that value.

    Without tree expansion (the common case during initial training), it falls
    back to the full-trajectory potential gain Φ(s_H)−Φ(s₀), which is the
    D=∞ rollout estimate and a reasonable proxy.  Enable use_tree_expansion
    to replace the proxy with the proper D=1, b=b lookahead advantage.

    Domain-agnostic: uses only Φ via compute_potential.  The advantage is
    expressed relative to the group mean by GRPO's group-normalisation, so
    no extra baseline subtraction is needed here.

    Config: reward_fn: tree_advantage
    Params: alpha (5.0), gamma (0.99)
    """

    def __init__(self, alpha: float = 5.0, gamma: float = 0.99) -> None:
        self.alpha = alpha
        self.gamma = gamma

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        # Primary path: trainer populated tree_metrics["tree_advantage"]
        if rollout.tree_metrics and "tree_advantage" in rollout.tree_metrics:
            return _bounded_reward(rollout.tree_metrics["tree_advantage"])

        # Fallback: full-trajectory Φ gain as D=∞ rollout estimate
        from llm_finetune.training.rl.posterior.potential import compute_potential
        if not rollout.initial_state or not rollout.final_state:
            return 0.0
        initial_mass = _finite_float(rollout.initial_state.get("mass"), 1.0)
        phi_0 = compute_potential(
            rollout.initial_state, initial_mass=initial_mass, alpha=self.alpha
        )
        phi_H = compute_potential(
            rollout.final_state, initial_mass=initial_mass, alpha=self.alpha
        )
        return _bounded_reward(self.gamma * phi_H - phi_0)

    def name(self) -> str:
        return "tree_advantage"


class StepNormalizedPhiReturn(RewardFunction):
    """Phi gain per step — rewards efficient, large-scope actions (§3.5).

    reward = (Φ(s_H) − Φ(s₀)) / n_steps

    When GRPO normalises across the group, rollouts that achieve the same
    total Φ gain in fewer steps rank higher.  This incentivises compound
    actions that move the state far in a single step rather than many
    small incremental moves.

    Domain-agnostic: only Φ and step count are used.

    Config: reward_fn: step_normalized_phi
    Params: alpha (5.0)
    """

    def __init__(self, alpha: float = 5.0) -> None:
        self.alpha = alpha

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        if rollout.n_steps == 0:
            return 0.0
        from llm_finetune.training.rl.posterior.potential import compute_potential
        initial_mass = _finite_float((rollout.initial_state or {}).get("mass"), 1.0)
        phi_0 = compute_potential(
            rollout.initial_state or {}, initial_mass=initial_mass, alpha=self.alpha
        )
        phi_H = compute_potential(
            rollout.final_state or {}, initial_mass=initial_mass, alpha=self.alpha
        )
        return _bounded_reward((phi_H - phi_0) / rollout.n_steps)

    def name(self) -> str:
        return "step_normalized_phi"


class InitialDifficultyWeightedReturn(RewardFunction):
    """Scales base reward by initial-state difficulty (§4.2, generalised).

    R′(τ) = R_base(τ) × (1 + λ × d(s₀))
    d(s₀) = max(0, −Φ(s₀)) / difficulty_scale

    States with deeply negative initial Φ (far from feasible, high violation)
    get proportionally stronger gradient signal so that hard problems are not
    under-weighted relative to easy ones during GRPO training.

    Domain-agnostic: difficulty is derived solely from Φ(s₀), not from any
    domain-specific quantity such as FOS or deflection.

    Config: reward_fn: difficulty_weighted
    Params: base_reward_fn ("lagrangian_potential"), lambda_difficulty (0.5),
            difficulty_scale (10.0), alpha (5.0)
    """

    def __init__(
        self,
        base_reward_fn: str = "lagrangian_potential",
        lambda_difficulty: float = 0.5,
        difficulty_scale: float = 10.0,
        alpha: float = 5.0,
    ) -> None:
        self.lambda_difficulty = lambda_difficulty
        self.difficulty_scale = max(difficulty_scale, 1e-6)
        self.alpha = alpha
        # REWARD_REGISTRY is defined below; safe to access at instantiation time
        # (this __init__ is only ever called after the full module is imported).
        self._base_reward_fn_name = base_reward_fn
        self._base: Optional[RewardFunction] = None

    def _get_base(self) -> RewardFunction:
        if self._base is None:
            if self._base_reward_fn_name not in REWARD_REGISTRY:
                raise ValueError(
                    f"InitialDifficultyWeightedReturn: unknown base_reward_fn "
                    f"{self._base_reward_fn_name!r}"
                )
            self._base = REWARD_REGISTRY[self._base_reward_fn_name]()
        return self._base

    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        from llm_finetune.training.rl.posterior.potential import compute_potential
        base_r = _bounded_reward(self._get_base().compute(rollout, problem_spec))
        initial_mass = _finite_float((rollout.initial_state or {}).get("mass"), 1.0)
        phi_0 = compute_potential(
            rollout.initial_state or {}, initial_mass=initial_mass, alpha=self.alpha
        )
        d = max(0.0, -phi_0) / self.difficulty_scale
        return _bounded_reward(base_r * (1.0 + self.lambda_difficulty * d))

    def name(self) -> str:
        return "difficulty_weighted"


# ── Registry ──────────────────────────────────────────────────────────────────
REWARD_REGISTRY: dict[str, type[RewardFunction]] = {
    "feasibility": FeasibilityReward,
    "fos_improvement": FOSImprovementReward,
    "mass_reduction": MassReductionReward,
    "grammar_compliance": GrammarComplianceReward,
    "step_efficiency": StepEfficiencyReward,
    "progress": ProgressReward,
    "composite": CompositeReward,
    "lagrangian_potential": LagrangianPotentialReward,
    "macro_completion": MacroCompletionReward,
    "forward_prediction": ForwardPredictionReward,
    "stagnation_escape": StagnationEscapeReward,
    # Tree-signal rewards (§3): domain-agnostic, use Φ and tree_metrics
    "tree_advantage": OnlineTreeAdvantageReward,
    "step_normalized_phi": StepNormalizedPhiReturn,
    "difficulty_weighted": InitialDifficultyWeightedReturn,
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
