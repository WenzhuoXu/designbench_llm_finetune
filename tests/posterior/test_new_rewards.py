"""
Tests for the posterior reward hooks added in §1-§4 integration:
  LagrangianPotentialReward, MacroCompletionReward, ForwardPredictionReward,
  StagnationEscapeReward, DeadEndAvoidanceCost.

All tests are CPU-only with synthetic RolloutResult objects — no FEA, no model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from llm_finetune.training.rl.rewards import (
    RolloutResult,
    LagrangianPotentialReward,
    MacroCompletionReward,
    ForwardPredictionReward,
    StagnationEscapeReward,
    REWARD_REGISTRY,
)
from llm_finetune.training.rl.costs import (
    DeadEndAvoidanceCost,
    COST_REGISTRY,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _state(mass=150.0, fos_b=1.2, fos_y=1.8, deflection=0.005, feasible=False):
    return {
        "mass": mass,
        "fos_buckling": fos_b,
        "fos_yielding": fos_y,
        "deflection": deflection,
        "is_feasible": feasible,
    }


def _rollout(
    states=None,
    actions=None,
    initial_mass=200.0,
    raw_outputs=None,
    reaches_solution=False,
) -> RolloutResult:
    if states is None:
        states = [_state(mass=200.0), _state(mass=190.0), _state(mass=180.0)]
    if actions is None:
        actions = [f"SCALE_PARAM({i}, radius, 1.1)" for i in range(len(states) - 1)]
    if raw_outputs is None:
        raw_outputs = ["<think>reasoning</think>"] * len(actions)
    initial = states[0]
    return RolloutResult(
        problem_id="test_p",
        action_sequence=actions,
        raw_outputs=raw_outputs,
        state_history=states,
        final_state=states[-1],
        initial_state=initial,
        token_counts=[128] * len(actions),
        parse_success=[True] * len(actions),
        reaches_solution=reaches_solution,
        n_fea_calls=len(actions),
        n_steps=len(actions),
    )


PROBLEM_SPEC = {"problem_id": "test_p", "goals": {}}


# ── LagrangianPotentialReward ─────────────────────────────────────────────────

class TestLagrangianPotentialReward:
    def test_returns_float(self):
        r = LagrangianPotentialReward()
        rollout = _rollout()
        assert isinstance(r.compute(rollout, PROBLEM_SPEC), float)

    def test_positive_for_improving_trajectory(self):
        states = [
            _state(mass=200.0, fos_b=0.8, fos_y=1.0),
            _state(mass=190.0, fos_b=1.2, fos_y=1.4),
            _state(mass=175.0, fos_b=1.6, fos_y=1.8, feasible=True),
        ]
        r = LagrangianPotentialReward()
        reward = r.compute(_rollout(states=states), PROBLEM_SPEC)
        assert reward > 0.0

    def test_negative_for_worsening_trajectory(self):
        states = [
            _state(mass=150.0, fos_b=1.8, fos_y=2.0, feasible=True),
            _state(mass=180.0, fos_b=1.2, fos_y=1.4),
            _state(mass=210.0, fos_b=0.7, fos_y=0.8),
        ]
        r = LagrangianPotentialReward()
        reward = r.compute(_rollout(states=states), PROBLEM_SPEC)
        assert reward < 0.0

    def test_zero_for_single_step(self):
        r = LagrangianPotentialReward()
        rollout = _rollout(states=[_state()])
        assert r.compute(rollout, PROBLEM_SPEC) == 0.0

    def test_alpha_scaling(self):
        states = [_state(mass=200.0, fos_b=0.5), _state(mass=190.0, fos_b=0.6)]
        r_low = LagrangianPotentialReward(alpha=1.0)
        r_high = LagrangianPotentialReward(alpha=10.0)
        rollout = _rollout(states=states)
        # High alpha penalises violations more → bigger negative impact on r_env
        # for infeasible states; reward difference should exist
        assert r_low.compute(rollout, PROBLEM_SPEC) != r_high.compute(rollout, PROBLEM_SPEC)

    def test_name(self):
        assert LagrangianPotentialReward().name() == "lagrangian_potential"

    def test_registered(self):
        assert "lagrangian_potential" in REWARD_REGISTRY


# ── MacroCompletionReward ─────────────────────────────────────────────────────

class TestMacroCompletionReward:
    def test_detects_add_then_scale(self):
        r = MacroCompletionReward()
        rollout = _rollout(actions=[
            "ADD_MEMBER(0, 5, 6061_T6_Aluminum, Pipe, 0.03, 0.003)",
            "SCALE_PARAM(5, radius, 1.15)",
        ])
        assert r.compute(rollout, PROBLEM_SPEC) == pytest.approx(0.05)

    def test_detects_remove_then_move(self):
        r = MacroCompletionReward()
        rollout = _rollout(actions=[
            "REMOVE_MEMBER(4)",
            "MOVE_JOINT(3, [0.0, 2.0], [0.5, 2.0])",
        ])
        assert r.compute(rollout, PROBLEM_SPEC) == pytest.approx(0.05)

    def test_no_bonus_for_same_class(self):
        r = MacroCompletionReward()
        rollout = _rollout(actions=[
            "SCALE_PARAM(0, radius, 1.1)",
            "SCALE_PARAM(1, radius, 0.9)",
        ])
        assert r.compute(rollout, PROBLEM_SPEC) == 0.0

    def test_no_bonus_for_single_action(self):
        r = MacroCompletionReward()
        rollout = _rollout(actions=["ADD_MEMBER(0, 1, M, P, 0.1, 0.01)"])
        assert r.compute(rollout, PROBLEM_SPEC) == 0.0

    def test_custom_bonus_value(self):
        r = MacroCompletionReward(macro_bonus=0.2)
        rollout = _rollout(actions=[
            "SCALE_PARAM(0, r, 1.1)",
            "ADD_MEMBER(0, 1, M, P, 0.1, 0.01)",
        ])
        assert r.compute(rollout, PROBLEM_SPEC) == pytest.approx(0.2)

    def test_name(self):
        assert MacroCompletionReward().name() == "macro_completion"

    def test_registered(self):
        assert "macro_completion" in REWARD_REGISTRY


# ── ForwardPredictionReward ───────────────────────────────────────────────────

class TestForwardPredictionReward:
    def test_zero_without_predict_tags(self):
        r = ForwardPredictionReward()
        rollout = _rollout()
        assert r.compute(rollout, PROBLEM_SPEC) == 0.0

    def test_positive_for_accurate_mass_prediction(self):
        actual_mass = 175.0
        states = [_state(mass=200.0), _state(mass=actual_mass)]
        raw_outputs = [f"<predict>mass: {actual_mass:.1f}</predict>"]
        r = ForwardPredictionReward(prediction_weight=0.10)
        rollout = _rollout(states=states, actions=["SCALE_PARAM(0, r, 0.9)"],
                           raw_outputs=raw_outputs)
        reward = r.compute(rollout, PROBLEM_SPEC)
        assert reward > 0.0
        assert reward <= 0.10 + 1e-6

    def test_zero_for_wrong_prediction(self):
        actual_mass = 175.0
        predicted_mass = 350.0  # 100% off
        states = [_state(mass=200.0), _state(mass=actual_mass)]
        raw_outputs = [f"<predict>mass: {predicted_mass:.1f}</predict>"]
        r = ForwardPredictionReward()
        rollout = _rollout(states=states, actions=["SCALE_PARAM(0, r, 0.9)"],
                           raw_outputs=raw_outputs)
        reward = r.compute(rollout, PROBLEM_SPEC)
        assert reward == pytest.approx(0.0, abs=0.01)

    def test_name(self):
        assert ForwardPredictionReward().name() == "forward_prediction"

    def test_registered(self):
        assert "forward_prediction" in REWARD_REGISTRY


# ── StagnationEscapeReward ────────────────────────────────────────────────────

class TestStagnationEscapeReward:
    def _flat_fos_rollout(self, escape_action: str, escape_with_same_class: bool = False):
        # window_size=3: need 4 states in window + 1 more for the escape
        plateau_states = [_state(fos_b=1.0)] * 4  # flat FOS window
        escape_state = _state(fos_b=1.0)  # final state (FOS doesn't matter for detection)
        states = plateau_states + [escape_state]
        window_actions = ["SCALE_PARAM(0, r, 1.0)"] * 3  # dominant class
        actions = window_actions + [escape_action]
        return _rollout(states=states, actions=actions)

    def test_fires_on_non_local_escape(self):
        r = StagnationEscapeReward(window_size=3, plateau_tolerance=0.05)
        rollout = self._flat_fos_rollout("ADD_MEMBER(0, 1, M, P, 0.1, 0.01)")
        reward = r.compute(rollout, PROBLEM_SPEC)
        assert reward > 0.0

    def test_silent_for_same_class_escape(self):
        r = StagnationEscapeReward(window_size=3, plateau_tolerance=0.05)
        rollout = self._flat_fos_rollout("SCALE_PARAM(1, thickness, 0.95)")
        assert r.compute(rollout, PROBLEM_SPEC) == 0.0

    def test_silent_without_stagnation(self):
        r = StagnationEscapeReward(window_size=3, plateau_tolerance=0.02)
        # FOS is changing significantly — not stagnant
        states = [
            _state(fos_b=0.5), _state(fos_b=0.8), _state(fos_b=1.1),
            _state(fos_b=1.4), _state(fos_b=1.7),
        ]
        rollout = _rollout(
            states=states,
            actions=["SCALE_PARAM(0,r,1.1)"] * 3 + ["ADD_MEMBER(0,1,M,P,0.1,0.01)"],
        )
        assert r.compute(rollout, PROBLEM_SPEC) == 0.0

    def test_silent_for_short_history(self):
        r = StagnationEscapeReward(window_size=4)
        rollout = _rollout(states=[_state()] * 2, actions=["ADD_MEMBER(0,1,M,P,0.1,0.01)"])
        assert r.compute(rollout, PROBLEM_SPEC) == 0.0

    def test_name(self):
        assert StagnationEscapeReward().name() == "stagnation_escape"

    def test_registered(self):
        assert "stagnation_escape" in REWARD_REGISTRY


# ── DeadEndAvoidanceCost ──────────────────────────────────────────────────────

class TestDeadEndAvoidanceCost:
    def _dead_end_rollout(self, use_non_local_at_end: bool = False):
        # Monotone-worsening FOS + all SCALE_PARAM
        states = [
            _state(fos_b=1.4), _state(fos_b=1.3),
            _state(fos_b=1.2), _state(fos_b=1.1),
        ]
        if use_non_local_at_end:
            actions = ["SCALE_PARAM(0,r,0.9)"] * 2 + ["ADD_MEMBER(0,1,M,P,0.1,0.01)"]
        else:
            actions = ["SCALE_PARAM(0,r,0.9)"] * 3
        return _rollout(states=states, actions=actions)

    def test_fires_for_monotone_worsening_and_all_local(self):
        cost = DeadEndAvoidanceCost(window_size=3)
        rollout = self._dead_end_rollout(use_non_local_at_end=False)
        assert cost.compute(rollout, PROBLEM_SPEC) > 0.0

    def test_silent_when_non_local_action_present(self):
        cost = DeadEndAvoidanceCost(window_size=3)
        rollout = self._dead_end_rollout(use_non_local_at_end=True)
        assert cost.compute(rollout, PROBLEM_SPEC) == 0.0

    def test_silent_when_fos_not_monotone(self):
        cost = DeadEndAvoidanceCost(window_size=3)
        states = [_state(fos_b=1.0), _state(fos_b=0.9), _state(fos_b=1.1), _state(fos_b=0.8)]
        rollout = _rollout(states=states, actions=["SCALE_PARAM(0,r,0.9)"] * 3)
        assert cost.compute(rollout, PROBLEM_SPEC) == 0.0

    def test_silent_for_short_history(self):
        cost = DeadEndAvoidanceCost(window_size=3)
        rollout = _rollout(states=[_state()] * 2, actions=["SCALE_PARAM(0,r,0.9)"])
        assert cost.compute(rollout, PROBLEM_SPEC) == 0.0

    def test_custom_cost_value(self):
        cost = DeadEndAvoidanceCost(dead_end_cost=0.5, window_size=3)
        rollout = self._dead_end_rollout()
        assert cost.compute(rollout, PROBLEM_SPEC) == pytest.approx(0.5)

    def test_name(self):
        assert DeadEndAvoidanceCost().name() == "dead_end_avoidance"

    def test_registered(self):
        assert "dead_end_avoidance" in COST_REGISTRY
