"""
Tests for RL reward and cost functions.

Tests all built-in reward/cost functions with synthetic RolloutResult objects.
Run with: pytest tests/test_rewards.py -v
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from llm_finetune.envs.truss_env import _sanitize_fea_state
from llm_finetune.training.rl.costs import (
    CompositeCost,
    ConstraintViolationCost,
    RepetitionCost,
    TokenBudgetCost,
)
from llm_finetune.training.rl.rewards import (
    REWARD_REGISTRY,
    CompositeReward,
    FeasibilityReward,
    FOSImprovementReward,
    GrammarComplianceReward,
    InitialDifficultyWeightedReturn,
    MassReductionReward,
    OnlineTreeAdvantageReward,
    RolloutResult,
    StepEfficiencyReward,
    StepNormalizedPhiReturn,
    build_reward_from_config,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_rollout(
    reaches_solution: bool = False,
    fos_b_initial: float = 0.8,
    fos_b_final: float = 1.6,
    fos_y_initial: float = 1.2,
    fos_y_final: float = 1.8,
    mass_initial: float = 200.0,
    mass_final: float = 180.0,
    n_steps: int = 5,
    parse_success: list = None,
    token_counts: list = None,
) -> RolloutResult:
    if parse_success is None:
        parse_success = [True] * n_steps
    if token_counts is None:
        token_counts = [256] * n_steps
    return RolloutResult(
        problem_id="auto_problem_090",
        action_sequence=[f"SCALE_PARAM({i}, radius, 1.1)" for i in range(n_steps)],
        raw_outputs=[f"<think>reasoning</think>\nSCALE_PARAM({i}, radius, 1.1)" for i in range(n_steps)],
        state_history=[
            {"mass": mass_initial, "fos_buckling": fos_b_initial, "fos_yielding": fos_y_initial, "is_feasible": False},
            *[{"mass": mass_initial - i * 4, "fos_buckling": fos_b_initial + i * 0.16,
               "fos_yielding": fos_y_initial + i * 0.12, "is_feasible": False}
              for i in range(1, n_steps)],
            {"mass": mass_final, "fos_buckling": fos_b_final, "fos_yielding": fos_y_final,
             "is_feasible": reaches_solution},
        ],
        final_state={"mass": mass_final, "fos_buckling": fos_b_final,
                     "fos_yielding": fos_y_final, "is_feasible": reaches_solution},
        initial_state={"mass": mass_initial, "fos_buckling": fos_b_initial,
                       "fos_yielding": fos_y_initial, "is_feasible": False},
        token_counts=token_counts,
        parse_success=parse_success,
        reaches_solution=reaches_solution,
        n_fea_calls=n_steps,
        n_steps=n_steps,
    )


PROBLEM_SPEC = {
    "problem_id": "auto_problem_090",
    "goals": {"minimum_fos_buckling": 1.5, "minimum_fos_yielding": 1.5, "maximum_mass": 225.0},
}


def test_sanitize_fea_state_clamps_nonfinite_outputs():
    state = {
        "mass": float("inf"),
        "fos_buckling": float("nan"),
        "fos_yielding": -float("inf"),
        "deflection": float("inf"),
        "is_feasible": True,
    }
    clean = _sanitize_fea_state(state, {"maximum_mass": 225.0, "maximum_deflection": 0.01})
    assert clean["mass"] == pytest.approx(2250.0)
    assert clean["fos_buckling"] == 0.0
    assert clean["fos_yielding"] == 0.0
    assert clean["deflection"] == pytest.approx(1.0)
    assert clean["is_feasible"] is False
    assert clean["fea_sanitized"] is True


# ── FeasibilityReward ─────────────────────────────────────────────────────────

def test_feasibility_reward_solved():
    reward = FeasibilityReward()
    rollout = make_rollout(reaches_solution=True)
    assert reward.compute(rollout, PROBLEM_SPEC) == 1.0

def test_feasibility_reward_unsolved():
    reward = FeasibilityReward()
    rollout = make_rollout(reaches_solution=False)
    assert reward.compute(rollout, PROBLEM_SPEC) == 0.0

def test_feasibility_reward_custom_value():
    reward = FeasibilityReward(reward_value=2.0)
    rollout = make_rollout(reaches_solution=True)
    assert reward.compute(rollout, PROBLEM_SPEC) == 2.0


# ── FOSImprovementReward ──────────────────────────────────────────────────────

def test_fos_improvement_positive():
    reward = FOSImprovementReward()
    # FOS improves from 0.8 to 1.6 — positive reward
    rollout = make_rollout(fos_b_initial=0.8, fos_b_final=1.6, fos_y_initial=1.2, fos_y_final=1.8)
    r = reward.compute(rollout, PROBLEM_SPEC)
    assert r > 0

def test_fos_improvement_negative():
    # FOS gets worse
    reward = FOSImprovementReward()
    rollout = make_rollout(fos_b_initial=1.2, fos_b_final=0.5, fos_y_initial=1.8, fos_y_final=0.8)
    r = reward.compute(rollout, PROBLEM_SPEC)
    assert r < 0

def test_fos_improvement_normalized_range():
    reward = FOSImprovementReward(normalize=True)
    rollout = make_rollout(fos_b_initial=0.0, fos_b_final=2.0, fos_y_initial=0.0, fos_y_final=2.0)
    r = reward.compute(rollout, PROBLEM_SPEC)
    assert -1.0 <= r <= 1.0


# ── MassReductionReward ───────────────────────────────────────────────────────

def test_mass_reduction_positive():
    reward = MassReductionReward()
    rollout = make_rollout(mass_initial=200.0, mass_final=180.0)
    r = reward.compute(rollout, PROBLEM_SPEC)
    assert r > 0

def test_mass_reduction_only_if_feasible():
    reward = MassReductionReward(only_if_feasible=True)
    rollout = make_rollout(mass_initial=200.0, mass_final=180.0, reaches_solution=False)
    assert reward.compute(rollout, PROBLEM_SPEC) == 0.0

def test_mass_reduction_feasible():
    reward = MassReductionReward(only_if_feasible=True)
    rollout = make_rollout(mass_initial=200.0, mass_final=180.0, reaches_solution=True)
    assert reward.compute(rollout, PROBLEM_SPEC) > 0


# ── GrammarComplianceReward ───────────────────────────────────────────────────

def test_grammar_perfect():
    reward = GrammarComplianceReward()
    rollout = make_rollout(parse_success=[True, True, True])
    assert reward.compute(rollout, PROBLEM_SPEC) == 1.0

def test_grammar_partial():
    reward = GrammarComplianceReward()
    rollout = make_rollout(parse_success=[True, False, True, False])
    assert reward.compute(rollout, PROBLEM_SPEC) == pytest.approx(0.5)

def test_grammar_all_fail():
    reward = GrammarComplianceReward()
    rollout = make_rollout(parse_success=[False, False])
    assert reward.compute(rollout, PROBLEM_SPEC) == 0.0

def test_grammar_empty():
    reward = GrammarComplianceReward()
    rollout = make_rollout(parse_success=[])
    assert reward.compute(rollout, PROBLEM_SPEC) == 0.0


# ── StepEfficiencyReward ──────────────────────────────────────────────────────

def test_step_efficiency_solved_early():
    reward = StepEfficiencyReward(max_steps=20, solution_bonus=0.2)
    rollout = make_rollout(reaches_solution=True, n_steps=5)
    r = reward.compute(rollout, PROBLEM_SPEC)
    assert r > 0  # solved early → bonus

def test_step_efficiency_unsolved_late():
    reward = StepEfficiencyReward(max_steps=20, step_penalty=0.05)
    rollout = make_rollout(reaches_solution=False, n_steps=20)
    r = reward.compute(rollout, PROBLEM_SPEC)
    assert r < 0  # all steps used, no solution → penalty


# ── CompositeReward ───────────────────────────────────────────────────────────

def test_composite_weights():
    reward = CompositeReward(weights={"feasibility": 1.0, "grammar_compliance": 0.5})
    rollout = make_rollout(reaches_solution=True, parse_success=[True, True])
    r = reward.compute(rollout, PROBLEM_SPEC)
    # Should be ≥ 1.0 (feasibility) + 0.5 (grammar) = 1.5
    assert r >= 1.0

def test_composite_breakdown():
    reward = CompositeReward(weights={"feasibility": 1.0, "grammar_compliance": 0.5})
    rollout = make_rollout(reaches_solution=True, parse_success=[True])
    breakdown = reward.get_breakdown(rollout, PROBLEM_SPEC)
    assert "feasibility" in breakdown
    assert "grammar_compliance" in breakdown
    assert breakdown["feasibility"] == 1.0


# ── Cost functions ────────────────────────────────────────────────────────────

def test_token_budget_within_budget():
    cost = TokenBudgetCost(max_tokens_per_step=512)
    rollout = make_rollout(token_counts=[256, 300])
    assert cost.compute(rollout, PROBLEM_SPEC) == 0.0

def test_token_budget_exceeded():
    cost = TokenBudgetCost(max_tokens_per_step=100)
    rollout = make_rollout(token_counts=[500, 600])
    assert cost.compute(rollout, PROBLEM_SPEC) > 0

def test_constraint_violation_feasible():
    cost = ConstraintViolationCost()
    rollout = make_rollout(fos_b_final=1.8, fos_y_final=2.0)
    spec = {"goals": {"maximum_mass": 300.0}}
    c = cost.compute(rollout, spec)
    assert c == 0.0  # no violations

def test_constraint_violation_infeasible():
    cost = ConstraintViolationCost()
    rollout = make_rollout(fos_b_final=0.5, fos_y_final=0.8)
    c = cost.compute(rollout, PROBLEM_SPEC)
    assert c > 0

def test_constraint_violation_nonfinite_state_is_bounded():
    cost = ConstraintViolationCost()
    rollout = make_rollout()
    rollout.final_state = {
        "mass": float("inf"),
        "fos_buckling": float("nan"),
        "fos_yielding": -float("inf"),
        "is_feasible": False,
    }
    c = cost.compute(rollout, PROBLEM_SPEC)
    assert math.isfinite(c)
    assert 0.0 <= c <= cost.max_cost

def test_composite_cost_nonfinite_components_are_bounded():
    cost = CompositeCost(weights={"constraint_violation": 1.0, "token_budget": 0.1})
    rollout = make_rollout(token_counts=[float("inf")])
    rollout.final_state = {"mass": float("inf"), "fos_buckling": 0.0, "fos_yielding": 0.0}
    c = cost.compute(rollout, PROBLEM_SPEC)
    assert math.isfinite(c)
    assert 0.0 <= c <= 100.0

def test_repetition_no_repeat():
    cost = RepetitionCost()
    rollout = make_rollout()
    rollout.action_sequence = ["SCALE_PARAM(1, r, 1.1)", "ADD_MEMBER(0,5,Al,Pipe,0.03,0.004)"]
    assert cost.compute(rollout, {}) == 0.0

def test_repetition_all_same():
    cost = RepetitionCost()
    rollout = make_rollout()
    rollout.action_sequence = ["SCALE_PARAM(1, r, 1.1)"] * 4
    assert cost.compute(rollout, {}) == pytest.approx(0.75)


# ── Registry tests ────────────────────────────────────────────────────────────

def test_all_rewards_in_registry():
    rollout = make_rollout()
    for name, cls in REWARD_REGISTRY.items():
        if name == "composite":
            continue
        reward = cls()
        r = reward.compute(rollout, PROBLEM_SPEC)
        assert isinstance(r, (int, float)), f"{name} returned non-numeric: {r}"

def test_build_reward_from_config():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "reward_fn": "composite",
        "reward_weights": {"feasibility": 1.0, "grammar_compliance": 0.1},
    })
    reward = build_reward_from_config(cfg)
    assert reward.name() == "composite"
    rollout = make_rollout(reaches_solution=True, parse_success=[True])
    r = reward.compute(rollout, PROBLEM_SPEC)
    assert r >= 1.0


# ── OnlineTreeAdvantageReward ─────────────────────────────────────────────────

def test_tree_advantage_from_tree_metrics():
    reward = OnlineTreeAdvantageReward()
    rollout = make_rollout()
    rollout.tree_metrics = {"tree_advantage": 3.14}
    assert reward.compute(rollout, PROBLEM_SPEC) == pytest.approx(3.14)

def test_tree_advantage_fallback_positive():
    # FOS/mass improve → positive phi gain → positive fallback advantage
    reward = OnlineTreeAdvantageReward()
    rollout = make_rollout(fos_b_initial=0.5, fos_b_final=1.8,
                           fos_y_initial=0.5, fos_y_final=1.8,
                           mass_initial=200.0, mass_final=160.0)
    assert rollout.tree_metrics == {}
    r = reward.compute(rollout, PROBLEM_SPEC)
    assert r > 0

def test_tree_advantage_fallback_negative():
    # FOS degrades → potential drops → negative advantage
    reward = OnlineTreeAdvantageReward()
    rollout = make_rollout(fos_b_initial=1.8, fos_b_final=0.3,
                           fos_y_initial=1.8, fos_y_final=0.3,
                           mass_initial=180.0, mass_final=220.0)
    r = reward.compute(rollout, PROBLEM_SPEC)
    assert r < 0

def test_tree_advantage_tree_metrics_overrides_fallback():
    reward = OnlineTreeAdvantageReward()
    rollout = make_rollout()
    rollout.tree_metrics = {"tree_advantage": -99.0}
    # Even though fallback would be positive, tree_metrics takes priority
    assert reward.compute(rollout, PROBLEM_SPEC) == pytest.approx(-99.0)

def test_tree_advantage_nonfinite_tree_metric_is_bounded():
    reward = OnlineTreeAdvantageReward()
    rollout = make_rollout()
    rollout.tree_metrics = {"tree_advantage": float("inf")}
    r = reward.compute(rollout, PROBLEM_SPEC)
    assert math.isfinite(r)


# ── StepNormalizedPhiReturn ───────────────────────────────────────────────────

def test_step_normalized_phi_fewer_steps_ranks_higher():
    reward = StepNormalizedPhiReturn()
    # Same problem, same start/end states, different n_steps
    few = make_rollout(fos_b_initial=0.5, fos_b_final=1.8,
                       fos_y_initial=0.5, fos_y_final=1.8,
                       mass_initial=200.0, mass_final=160.0, n_steps=3)
    many = make_rollout(fos_b_initial=0.5, fos_b_final=1.8,
                        fos_y_initial=0.5, fos_y_final=1.8,
                        mass_initial=200.0, mass_final=160.0, n_steps=15)
    assert reward.compute(few, PROBLEM_SPEC) > reward.compute(many, PROBLEM_SPEC)

def test_step_normalized_phi_zero_steps():
    reward = StepNormalizedPhiReturn()
    rollout = make_rollout(n_steps=0)
    rollout.n_steps = 0
    assert reward.compute(rollout, PROBLEM_SPEC) == 0.0

def test_step_normalized_phi_sign():
    reward = StepNormalizedPhiReturn()
    # Improving trajectory → positive
    good = make_rollout(fos_b_initial=0.5, fos_b_final=1.8,
                        fos_y_initial=0.5, fos_y_final=1.8,
                        mass_initial=200.0, mass_final=150.0)
    assert reward.compute(good, PROBLEM_SPEC) > 0
    # Degrading trajectory → negative
    bad = make_rollout(fos_b_initial=1.8, fos_b_final=0.3,
                       fos_y_initial=1.8, fos_y_final=0.3,
                       mass_initial=160.0, mass_final=210.0)
    assert reward.compute(bad, PROBLEM_SPEC) < 0


# ── InitialDifficultyWeightedReturn ──────────────────────────────────────────

def test_difficulty_weighted_hard_gt_easy():
    reward = InitialDifficultyWeightedReturn(base_reward_fn="feasibility")
    # Hard: deeply infeasible initial state (FOS = 0.1 → large violation, very negative Φ)
    hard = make_rollout(fos_b_initial=0.1, fos_y_initial=0.1,
                        mass_initial=200.0, reaches_solution=True)
    # Easy: nearly feasible initial state
    easy = make_rollout(fos_b_initial=1.4, fos_y_initial=1.4,
                        mass_initial=200.0, reaches_solution=True)
    r_hard = reward.compute(hard, PROBLEM_SPEC)
    r_easy = reward.compute(easy, PROBLEM_SPEC)
    assert r_hard > r_easy

def test_difficulty_weighted_no_amplification_on_zero_base():
    # If base reward is 0 (unsolved), difficulty weighting should still return 0
    reward = InitialDifficultyWeightedReturn(base_reward_fn="feasibility")
    rollout = make_rollout(fos_b_initial=0.1, fos_y_initial=0.1,
                           mass_initial=200.0, reaches_solution=False)
    assert reward.compute(rollout, PROBLEM_SPEC) == pytest.approx(0.0)

def test_difficulty_weighted_multiplier_positive():
    reward = InitialDifficultyWeightedReturn(
        base_reward_fn="feasibility", lambda_difficulty=1.0, difficulty_scale=10.0
    )
    rollout = make_rollout(fos_b_initial=0.1, fos_y_initial=0.1,
                           mass_initial=200.0, reaches_solution=True)
    r = reward.compute(rollout, PROBLEM_SPEC)
    assert r > 1.0  # base=1.0, multiplier > 1 for hard state
