"""Regression tests for the design-program potential and the lookahead estimator.

Each test pins a defect found on 2026-08-23 (see docs/turn5_potential_specification.md)
so it cannot silently come back.
"""

from __future__ import annotations

import math

import pytest

from llm_finetune.training.rl.posterior.potential import (
    DesignProgram,
    compute_potential,
    compute_potential_v2,
    program_from_battery_goals,
    program_from_goals,
    program_from_truss_spec,
)

TRUSS_MASS_SPEC = {
    "problem_id": "mass_family",
    "goals": {"minimum_fos_buckling": 1.5, "minimum_fos_yielding": 1.5, "maximum_mass": 264.8},
    "_metadata": {"optimal_mass": 240.7},
}
TRUSS_DEFL_SPEC = {
    "problem_id": "deflection_family",
    "goals": {"minimum_fos_buckling": 1.5, "minimum_fos_yielding": 1.5, "maximum_deflection": 0.015},
}
TRUSS_UNBOUNDED_SPEC = {
    "problem_id": "fos_only",
    "goals": {"minimum_fos_buckling": 1.5, "minimum_fos_yielding": 1.5,
              "maximum_deflection": float("inf")},
}


def _keys(program: DesignProgram) -> set[str]:
    return {c.key for c in program.constraints}


class TestConstraintRecovery:
    """D3: the potential must encode the constraint set the problem actually states."""

    def test_mass_family_has_no_deflection_constraint(self):
        program = program_from_truss_spec(TRUSS_MASS_SPEC)
        assert _keys(program) == {"fos_buckling", "fos_yielding", "mass"}

    def test_deflection_family_has_no_mass_constraint(self):
        program = program_from_truss_spec(TRUSS_DEFL_SPEC)
        assert _keys(program) == {"fos_buckling", "fos_yielding", "deflection"}
        assert program.limit_for("deflection") == pytest.approx(0.015)

    def test_infinite_limit_is_not_a_constraint(self):
        program = program_from_truss_spec(TRUSS_UNBOUNDED_SPEC)
        assert "deflection" not in _keys(program)

    def test_objective_reference_prefers_the_lp_optimum(self):
        assert program_from_truss_spec(TRUSS_MASS_SPEC).objective_ref == pytest.approx(240.7)
        # no optimum recorded -> falls back to the initial mass
        assert program_from_truss_spec(TRUSS_DEFL_SPEC, initial_mass=88.0).objective_ref == pytest.approx(88.0)

    def test_convention_parser_is_domain_agnostic(self):
        program = program_from_battery_goals(
            {"max_temperature": 333.15, "max_plating": 1e-5,
             "max_charge_time": 900.0, "min_neg_potential": 0.0}
        )
        assert _keys(program) == {"max_temperature", "max_plating", "charge_time", "min_neg_potential"}
        senses = {c.key: c.sense for c in program.constraints}
        assert senses["min_neg_potential"] == "lower"
        assert senses["max_temperature"] == "upper"

    def test_unknown_domain_needs_no_new_code(self):
        program = program_from_goals(
            {"maximum_cost": 100.0, "minimum_stiffness": 4.0},
            objective_key="cost", objective_ref=100.0,
        )
        assert _keys(program) == {"cost", "stiffness"}


class TestOrdering:
    """The claim the whole study turns on: which design does the potential prefer?"""

    LEAN = {"mass": 250.0, "fos_buckling": 1.55, "fos_yielding": 1.60, "deflection": 0.03}
    HEAVY = {"mass": 900.0, "fos_buckling": 6.00, "fos_yielding": 8.00, "deflection": 0.004}

    def test_v1_prefers_an_infeasible_over_stiffened_design(self):
        # The defect, pinned. Phi_v1 scores HEAVY (mass 900 > cap 265) above LEAN.
        v1_lean = compute_potential(self.LEAN, initial_mass=102.55, alpha=5.0)
        v1_heavy = compute_potential(self.HEAVY, initial_mass=102.55, alpha=5.0)
        assert v1_heavy > v1_lean

    def test_v2_prefers_the_feasible_design(self):
        program = program_from_truss_spec(TRUSS_MASS_SPEC)
        assert program.is_feasible(self.LEAN)
        assert not program.is_feasible(self.HEAVY)
        assert compute_potential_v2(self.LEAN, program, alpha=5.0) > \
               compute_potential_v2(self.HEAVY, program, alpha=5.0)

    def test_over_satisfying_a_constraint_almost_stops_paying(self):
        """The hinge saturates; v1's softplus never does.

        Same mass, FOS raised from 1.55 to 6.0. v1 keeps paying for that (it is
        why its argmax is an over-stiffened design); v2's remaining credit is the
        soft band around the constraint, an order of magnitude smaller.
        """
        program = program_from_truss_spec(TRUSS_MASS_SPEC)
        at_limit = {"mass": 250.0, "fos_buckling": 1.55, "fos_yielding": 1.55, "deflection": 0.03}
        far_over = {"mass": 250.0, "fos_buckling": 6.00, "fos_yielding": 6.00, "deflection": 0.03}

        gap_v2 = compute_potential_v2(far_over, program, alpha=5.0) - \
                 compute_potential_v2(at_limit, program, alpha=5.0)
        gap_v1 = compute_potential(far_over, initial_mass=102.55, alpha=5.0) - \
                 compute_potential(at_limit, initial_mass=102.55, alpha=5.0)

        assert gap_v1 > 6.0          # v1 still pays richly for pure over-design
        assert 0.0 < gap_v2 < 0.3    # v2 pays only inside the soft band
        assert gap_v1 / gap_v2 > 20.0

    def test_mass_cap_is_actually_priced(self):
        program = program_from_truss_spec(TRUSS_MASS_SPEC)
        under = {"mass": 260.0, "fos_buckling": 1.6, "fos_yielding": 1.6, "deflection": 0.03}
        over = {"mass": 400.0, "fos_buckling": 1.6, "fos_yielding": 1.6, "deflection": 0.03}
        assert compute_potential_v2(under, program, alpha=5.0) - \
               compute_potential_v2(over, program, alpha=5.0) > 1.0


class TestValidityGuard:
    """A simulator exploit must not read as a good design."""

    def test_negative_mass_is_invalid_not_excellent(self):
        program = program_from_truss_spec(TRUSS_MASS_SPEC)
        # pipe area = pi*t*(2r - t) goes negative for t > 2r, reachable inside the
        # problem's own bounds; a min-mass lookahead drove straight into it.
        exploit = {"mass": -40.0, "fos_buckling": 2.0, "fos_yielding": 2.0, "deflection": 0.02}
        valid = {"mass": 250.0, "fos_buckling": 1.55, "fos_yielding": 1.6, "deflection": 0.03}
        assert not program.is_feasible(exploit)
        assert compute_potential_v2(exploit, program, alpha=5.0) < \
               compute_potential_v2(valid, program, alpha=5.0)

    def test_diverged_simulation_is_bounded_and_far_below_any_feasible_design(self):
        """9 of 130 truss problems diverge at their initial state (mechanisms).

        The potential must stay finite there -- an inf would poison the whole
        GRPO group -- and must rank far below anything that actually works.
        """
        program = program_from_truss_spec(TRUSS_MASS_SPEC)
        diverged = {"mass": float("inf"), "fos_buckling": 0.0, "fos_yielding": 0.0,
                    "deflection": float("inf")}
        feasible = {"mass": 250.0, "fos_buckling": 1.55, "fos_yielding": 1.6, "deflection": 0.03}
        value = compute_potential_v2(diverged, program, alpha=5.0)
        assert math.isfinite(value)
        assert value < compute_potential_v2(feasible, program, alpha=5.0) - 10.0
        assert not program.is_feasible(diverged)


class TestRewardPlumbing:
    """D4/D5: config values must reach the reward, and nothing may arrive by accident."""

    def test_alpha_reaches_the_reward_component(self):
        from llm_finetune.training.rl.rewards import build_reward_from_config
        for alpha in (2.0, 5.0, 10.0):
            reward = build_reward_from_config({
                "reward_fn": "composite",
                "reward_weights": {"lagrangian_potential": 1.0},
                "posterior": {"alpha": alpha, "gamma": 0.99},
            })
            assert reward.components["lagrangian_potential"].alpha == alpha

    def test_reward_kwargs_override_the_posterior_block(self):
        from llm_finetune.training.rl.rewards import build_reward_from_config
        reward = build_reward_from_config({
            "reward_fn": "composite",
            "reward_weights": {"lagrangian_potential_v2": 1.0},
            "posterior": {"alpha": 3.0},
            "reward_kwargs": {"lagrangian_potential_v2": {"tau": 0.02}},
        })
        component = reward.components["lagrangian_potential_v2"]
        assert (component.alpha, component.tau) == (3.0, 0.02)

    def test_unknown_kwargs_do_not_crash_a_run(self):
        from llm_finetune.training.rl.rewards import build_reward_from_config
        build_reward_from_config({
            "reward_fn": "composite",
            "reward_weights": {"feasibility": 1.0},
            "posterior": {"alpha": 5.0, "nonsense_key": 1},
        })


class TestNoStallBonus:
    """D8: a step that changes nothing must be worth nothing.

    The shaping sum was accumulated without the gamma**t factor, so the return
    was a path integral rather than a boundary term and a no-op paid
    (1-gamma)*|Phi| wherever Phi < 0 -- i.e. everywhere infeasible. On a typical
    infeasible truss state that was +0.25 a turn, +1.27 over five: a direct
    reward for burning the turn budget, proportional to how bad the violation is.
    """

    BAD = {"mass": 150.0, "fos_buckling": 0.44, "fos_yielding": 2.5, "deflection": 0.043}

    def _stall(self, n):
        from llm_finetune.training.rl.rewards import RolloutResult
        history = [dict(self.BAD) for _ in range(n + 1)]
        return RolloutResult(
            problem_id="p", action_sequence=["x"] * n, raw_outputs=[],
            state_history=history, final_state=history[-1], initial_state=history[0],
            token_counts=[], parse_success=[True] * n, reaches_solution=False, n_steps=n,
        )

    def test_stalling_pays_nothing_at_gamma_one(self):
        from llm_finetune.training.rl.rewards import (
            LagrangianPotentialReward, LagrangianPotentialV2Reward,
        )
        for reward in (LagrangianPotentialReward(alpha=5.0, gamma=1.0),
                       LagrangianPotentialV2Reward(alpha=5.0, gamma=1.0)):
            assert reward.compute(self._stall(5), TRUSS_MASS_SPEC) == pytest.approx(0.0, abs=1e-9)

    def test_the_defect_is_reproduced_at_gamma_below_one(self):
        from llm_finetune.training.rl.rewards import LagrangianPotentialReward
        assert LagrangianPotentialReward(alpha=5.0, gamma=0.99).compute(
            self._stall(5), TRUSS_MASS_SPEC) > 1.0

    def test_shipped_configs_use_gamma_one(self):
        import yaml
        from pathlib import Path
        for name in ("grpo_mt_t5a_phi1", "grpo_mt_t5b_phi2", "grpo_mt_t5c_lookahead"):
            path = Path("configs/rl") / f"{name}.yaml"
            if not path.exists():
                pytest.skip(f"{name} not present")
            cfg = yaml.safe_load(path.read_text())
            assert cfg["posterior"]["gamma"] == 1.0, name


class TestLookaheadEstimator:
    """D1: the estimator must not be an affine image of a per-trajectory quantity."""

    def test_advantage_sums_per_step_counterfactuals(self):
        from llm_finetune.training.rl.rewards import LookaheadAdvantageReward, RolloutResult
        rollout = RolloutResult(
            problem_id="p", action_sequence=["a", "b"], raw_outputs=[],
            state_history=[{}, {}, {}], final_state={}, initial_state={},
            token_counts=[], parse_success=[True, True], reaches_solution=False, n_steps=2,
        )
        rollout.tree_metrics["lookahead_probes"] = [{"advantage": 0.31}, {"advantage": -0.12}]
        assert LookaheadAdvantageReward().compute(rollout, {}) == pytest.approx(0.19)

    def test_missing_probe_data_is_visibly_flat(self):
        from llm_finetune.training.rl.rewards import LookaheadAdvantageReward, RolloutResult
        rollout = RolloutResult(
            problem_id="p", action_sequence=[], raw_outputs=[], state_history=[],
            final_state={}, initial_state={}, token_counts=[], parse_success=[],
            reaches_solution=False, n_steps=0,
        )
        assert LookaheadAdvantageReward().compute(rollout, {}) == 0.0

    def test_probe_ranks_the_policy_against_counterfactuals(self):
        from llm_finetune.training.rl.posterior.lookahead_probe import (
            counterfactual_advantage, probe_state,
        )
        program = program_from_truss_spec(TRUSS_MASS_SPEC)
        successors = {
            "good": {"mass": 250.0, "fos_buckling": 1.6, "fos_yielding": 1.6, "deflection": 0.03},
            "ok": {"mass": 300.0, "fos_buckling": 1.2, "fos_yielding": 1.6, "deflection": 0.03},
            "bad": {"mass": 800.0, "fos_buckling": 0.4, "fos_yielding": 1.6, "deflection": 0.03},
        }
        probe = probe_state(
            program=program, candidates=list(successors),
            transition=successors.get, policy_next_state=successors["good"], alpha=5.0,
        )
        assert probe.rho == 1.0 and probe.regret == pytest.approx(0.0)
        assert counterfactual_advantage(probe) > 0.0

        worse = probe_state(
            program=program, candidates=list(successors),
            transition=successors.get, policy_next_state=successors["bad"], alpha=5.0,
        )
        assert worse.rho == 0.0
        assert worse.regret > probe.regret
        assert counterfactual_advantage(worse) < 0.0

    def test_reliability_gate_fires_when_the_top_two_are_indistinguishable(self):
        from llm_finetune.training.rl.posterior.lookahead_probe import probe_state
        program = program_from_truss_spec(TRUSS_MASS_SPEC)
        # two near-identical leaders inside a wide spread -> the lookahead cannot rank them
        spread = {
            "a": {"mass": 250.0, "fos_buckling": 1.60, "fos_yielding": 1.6, "deflection": 0.03},
            "b": {"mass": 250.5, "fos_buckling": 1.60, "fos_yielding": 1.6, "deflection": 0.03},
            "c": {"mass": 900.0, "fos_buckling": 0.2, "fos_yielding": 1.6, "deflection": 0.03},
        }
        probe = probe_state(program=program, candidates=list(spread),
                            transition=spread.get, policy_next_state=spread["a"], alpha=5.0)
        assert probe.margin < 2.0 * (0.99 ** 2) * probe.sigma
        assert probe.reliable is False


class TestConfigsSayWhatTheyMean:
    """D5: Hydra merges dict-valued keys, so a config's declared reward is not
    necessarily its effective reward. Every experiment config is pinned here."""

    EXPECTED = {
        "grpo_mt_t5a_phi1": {"feasibility", "fos_improvement", "grammar_compliance",
                             "step_efficiency", "lagrangian_potential"},
        "grpo_mt_t5b_phi2": {"feasibility", "fos_improvement", "grammar_compliance",
                             "step_efficiency", "lagrangian_potential_v2"},
        "grpo_mt_t5c_lookahead": {"feasibility", "fos_improvement", "grammar_compliance",
                                  "step_efficiency", "lagrangian_potential_v2",
                                  "lookahead_advantage"},
        "grpo_mt_t5ao_phionly_v1": {"lagrangian_potential"},
        "grpo_mt_t5bo_phionly_v2": {"lagrangian_potential_v2"},
    }

    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_effective_components_match_intent(self, name):
        from pathlib import Path
        if not (Path("configs/rl") / f"{name}.yaml").exists():
            pytest.skip(f"{name} not present")
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from llm_finetune.training.rl.rewards import build_reward_from_config
        with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
            cfg = OmegaConf.to_container(
                compose(config_name="grpo_config", overrides=[f"rl={name}"]).rl, resolve=True)
        assert set(build_reward_from_config(cfg).components) == self.EXPECTED[name]

    def test_ab_arms_differ_in_exactly_one_term(self):
        a, b = self.EXPECTED["grpo_mt_t5a_phi1"], self.EXPECTED["grpo_mt_t5b_phi2"]
        assert a ^ b == {"lagrangian_potential", "lagrangian_potential_v2"}
        b2, c = self.EXPECTED["grpo_mt_t5b_phi2"], self.EXPECTED["grpo_mt_t5c_lookahead"]
        assert c ^ b2 == {"lookahead_advantage"}
        # the potential-only pair must share NOTHING but the potential slot
        ao, bo = self.EXPECTED["grpo_mt_t5ao_phionly_v1"], self.EXPECTED["grpo_mt_t5bo_phionly_v2"]
        assert len(ao) == len(bo) == 1 and ao != bo


class TestCrossDomainScaling:
    """Relative violations are not commensurable across domains, and one alpha
    cannot serve both. Measured on the battery `thermal_hard` family, where the
    plain Lagrangian scored 1/3 against undirected enumeration's 3/3."""

    BATTERY_GOALS = {"max_temperature": 333.15, "max_plating": 1e-5,
                     "max_charge_time": 1800.0, "min_neg_potential": 0.0}
    HOT_FAST = {"charge_time": 900.0, "max_temperature": 340.0,
                "max_plating": 0.0, "min_neg_potential": 0.01}   # 2% over temperature
    COOL_SLOW = {"charge_time": 1500.0, "max_temperature": 330.0,
                 "max_plating": 0.0, "min_neg_potential": 0.01}  # feasible
    START = {"charge_time": 2400.0, "max_temperature": 338.0,
             "max_plating": 0.0, "min_neg_potential": 0.01}

    def test_the_defect_is_reproduced(self):
        """Without an offset, a small violation is bought with a large objective gain."""
        program = program_from_battery_goals(self.BATTERY_GOALS)
        assert not program.is_feasible(self.HOT_FAST)
        assert program.is_feasible(self.COOL_SLOW)
        hot = compute_potential_v2(self.HOT_FAST, program, alpha=5.0, tau=0.005)
        cool = compute_potential_v2(self.COOL_SLOW, program, alpha=5.0, tau=0.005)
        assert hot > cool, "expected the plain Lagrangian to mis-rank this pair"

    def test_feasibility_offset_restores_the_ordering(self):
        program = program_from_battery_goals(self.BATTERY_GOALS)
        hot = compute_potential_v2(self.HOT_FAST, program, alpha=5.0, tau=0.005,
                                   feasibility_offset=5.0)
        cool = compute_potential_v2(self.COOL_SLOW, program, alpha=5.0, tau=0.005,
                                    feasibility_offset=5.0)
        assert cool > hot

    def test_offset_does_not_disturb_a_domain_that_already_worked(self):
        program = program_from_truss_spec(TRUSS_MASS_SPEC)
        lean = {"mass": 250.0, "fos_buckling": 1.55, "fos_yielding": 1.60, "deflection": 0.03}
        heavy = {"mass": 900.0, "fos_buckling": 6.00, "fos_yielding": 8.00, "deflection": 0.004}
        for offset in (0.0, 5.0):
            assert compute_potential_v2(lean, program, alpha=5.0, tau=0.005,
                                        feasibility_offset=offset) > \
                   compute_potential_v2(heavy, program, alpha=5.0, tau=0.005,
                                        feasibility_offset=offset)

    def test_violation_normalisation_makes_alpha_portable(self):
        """34x scale gap between domains, closed to ~2x."""
        battery = program_from_battery_goals(self.BATTERY_GOALS).with_violation_scale(
            self.START, tau=0.005)
        truss_start = {"mass": 102.5, "fos_buckling": 0.44, "fos_yielding": 2.5, "deflection": 0.04}
        truss = program_from_truss_spec(TRUSS_MASS_SPEC).with_violation_scale(
            truss_start, tau=0.005)
        ratio = max(battery.violation_ref, truss.violation_ref) / \
            min(battery.violation_ref, truss.violation_ref)
        assert ratio < 4.0, f"effective alpha still differs by {ratio:.1f}x across domains"
        # and normalisation must not change any single-domain ordering
        assert truss.total_violation(truss_start, tau=0.005) == pytest.approx(1.0, abs=1e-6)

    def test_normalisation_alone_is_not_enough(self):
        """Honest bound on the claim: the offset is what guarantees the ordering."""
        program = program_from_battery_goals(self.BATTERY_GOALS).with_violation_scale(
            self.START, tau=0.005)
        hot = compute_potential_v2(self.HOT_FAST, program, alpha=5.0, tau=0.005)
        cool = compute_potential_v2(self.COOL_SLOW, program, alpha=5.0, tau=0.005)
        assert hot > cool, "normalisation alone was expected to be insufficient here"


class TestPerTokenAdvantageDelivery:
    """GRPO broadcasts one advantage per sequence; per-step credit needs its own channel.

    Decompose a per-token advantage vector as A = mean(A)*1 + A~ with A~ orthogonal
    to 1. A scalar reward term can only ever produce the first component, so the
    summed form of the lookahead advantage (arms T5c, T5co) was structurally unable
    to deliver per-step credit -- and measurably did not (t = +0.67 against its
    control over 14 paired steps, wrong sign).
    """

    def _fixture(self):
        import torch
        out = {
            "advantages": torch.tensor([1.0, -1.0]),
            "completion_ids": torch.zeros(2, 6, dtype=torch.long),
            "completion_mask": torch.tensor([[1, 1, 1, 0, 1, 1], [1, 1, 1, 0, 1, 1]]),
        }
        inputs = [
            {"step_token_advantages": [0.5, 0.5, 0.5, 0.0, -0.5, -0.5]},
            {"step_token_advantages": [-0.2, -0.2, -0.2, 0.0, 0.9, 0.9]},
        ]
        return out, inputs

    def test_delivers_a_per_token_tensor(self):
        from llm_finetune.training.rl.lookahead_grpo_trainer import inject_per_token_advantages
        out, inputs = self._fixture()
        res = inject_per_token_advantages(dict(out), inputs, kappa=1.0)
        assert res["advantages"].shape == (2, 6)

    def test_carries_the_component_grpo_cannot(self):
        """Within-row variation is exactly what a per-sequence scalar cannot express."""
        from llm_finetune.training.rl.lookahead_grpo_trainer import inject_per_token_advantages
        out, inputs = self._fixture()
        res = inject_per_token_advantages(dict(out), inputs, kappa=1.0)
        row = res["advantages"][0]
        assert abs(row[0].item() - row[4].item()) > 0.1

    def test_environment_tokens_get_no_step_credit(self):
        from llm_finetune.training.rl.lookahead_grpo_trainer import inject_per_token_advantages
        out, inputs = self._fixture()
        res = inject_per_token_advantages(dict(out), inputs, kappa=1.0)
        assert res["advantages"][0, 3].item() == pytest.approx(1.0, abs=1e-5)
        assert res["advantages"][1, 3].item() == pytest.approx(-1.0, abs=1e-5)

    def test_kappa_zero_is_exactly_stock_grpo(self):
        """The control arm must be bit-identical to unmodified GRPO."""
        from llm_finetune.training.rl.lookahead_grpo_trainer import inject_per_token_advantages
        out, inputs = self._fixture()
        res = inject_per_token_advantages(dict(out), inputs, kappa=0.0)
        assert res["advantages"].dim() == 1

    def test_misalignment_falls_back_instead_of_corrupting(self):
        """A length mismatch would attribute one rollout's credit to another's tokens."""
        from llm_finetune.training.rl.lookahead_grpo_trainer import inject_per_token_advantages
        out, inputs = self._fixture()
        res = inject_per_token_advantages(dict(out), inputs[:1], kappa=1.0)
        assert res["advantages"].dim() == 1

    def test_config_requires_an_all_steps_probe(self):
        """A subsampled probe would silently zero the credit on most turns."""
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load((Path("configs/rl") / "grpo_mt_t8_pertoken.yaml").read_text())
        assert cfg["lookahead_kappa"] != 0.0
        assert cfg["lookahead_probe"]["enabled"] and cfg["lookahead_probe"]["all_steps"]
