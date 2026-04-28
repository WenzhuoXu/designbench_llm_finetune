"""
Tests for RL reward and cost functions.

Tests all built-in reward/cost functions with synthetic RolloutResult objects.
Run with: pytest tests/test_rewards.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from llm_finetune.training.rl.rewards import (
    RolloutResult,
    FeasibilityReward,
    FOSImprovementReward,
    MassReductionReward,
    GrammarComplianceReward,
    StepEfficiencyReward,
    ProgressReward,
    CompositeReward,
    REWARD_REGISTRY,
    build_reward_from_config,
)
from llm_finetune.training.rl.costs import (
    TokenBudgetCost,
    ConstraintViolationCost,
    ComputationalCost,
    RepetitionCost,
    COST_REGISTRY,
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
