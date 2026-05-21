"""
Tests for TrussRolloutEnv.

Tests the environment step/reset cycle, action parsing, and parallel rollouts.
Note: FEA-dependent tests require DesignBench to be importable (they are skipped
if DesignBench is not available in the current environment).

Run with: pytest tests/test_env.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/DesignBench")


import pytest

from llm_finetune.envs.truss_env import TrussRolloutEnv, parse_grammar_action

# ── Grammar action parsing ────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("SCALE_PARAM(3, radius, 1.15)", "SCALE_PARAM(3, radius, 1.15)"),
    ("<think>Let me scale member 3</think>\nSCALE_PARAM(3, radius, 1.15)", "SCALE_PARAM(3, radius, 1.15)"),
    ("ADD_MEMBER(0, 5, 6061_T6_Aluminum, Pipe, 0.03, 0.004)", "ADD_MEMBER(0, 5, 6061_T6_Aluminum, Pipe, 0.03, 0.004)"),
    ("MODIFY_PARAM(2, thickness, 0.004, 0.006)", "MODIFY_PARAM(2, thickness, 0.004, 0.006)"),
    ("REMOVE_MEMBER(4)", "REMOVE_MEMBER(4)"),
    ("MOVE_JOINT(3, [0.0, 2.0], [0.5, 2.0])", "MOVE_JOINT(3, [0.0, 2.0], [0.5, 2.0])"),
    ("SCALE_MULTI_PARAM([1,2], [radius:1.1, thickness:0.9])", "SCALE_MULTI_PARAM([1,2], [radius:1.1, thickness:0.9])"),
    ("No action here", None),
    ("", None),
])
def test_parse_grammar_action(text, expected):
    result = parse_grammar_action(text)
    assert result == expected


def test_parse_grammar_action_with_preamble():
    text = (
        "Looking at the current FOS values, I should increase the radius of member 3.\n"
        "SCALE_PARAM(3, radius, 1.20)\n"
        "This will improve buckling resistance."
    )
    result = parse_grammar_action(text)
    assert result is not None
    assert result.startswith("SCALE_PARAM")


def test_parse_grammar_action_thinking_block():
    text = """<think>
The buckling FOS is 0.785, which is below 1.5. Member 3 is most critical.
I should scale its radius.
</think>
SCALE_PARAM(3, radius, 1.541)"""
    result = parse_grammar_action(text)
    assert result == "SCALE_PARAM(3, radius, 1.541)"


def test_parse_grammar_action_rejects_semantic_placeholders():
    assert parse_grammar_action("SCALE_MULTI_PARAM([ids], [param:factor, ...])") is None
    assert parse_grammar_action("SCALE_MULTI_PARAM([0-11], [r:1.1, t:1.1])") is None


# ── TrussRolloutEnv initialization ────────────────────────────────────────────

def test_env_initialization():
    env = TrussRolloutEnv(max_steps=20, n_workers=2)
    assert env.max_steps == 20
    assert env.n_workers == 2


def test_env_split_completion_into_steps():
    env = TrussRolloutEnv()
    completion = (
        "SCALE_PARAM(3, radius, 1.15)\n\n"
        "[FEA Result]\n  Mass: 140.0 kg\n\n"
        "ADD_MEMBER(0, 5, 6061_T6_Aluminum, Pipe, 0.03, 0.004)"
    )
    steps = env._split_completion_into_steps(completion)
    assert len(steps) >= 2


def test_env_split_completion_single():
    env = TrussRolloutEnv()
    completion = "SCALE_PARAM(3, radius, 1.15)"
    steps = env._split_completion_into_steps(completion)
    assert len(steps) >= 1
    assert "SCALE_PARAM" in steps[0]


# ── Tests requiring DesignBench FEA (skip if not available) ──────────────────

try:
    sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/DesignBench")
    DESIGNBENCH_AVAILABLE = True
except Exception:
    DESIGNBENCH_AVAILABLE = False

SKIP_FEA = pytest.mark.skipif(
    not DESIGNBENCH_AVAILABLE,
    reason="DesignBench not importable in current environment",
)

# Minimal problem spec for testing
MINI_PROBLEM_SPEC = {
    "problem_id": "test_problem",
    "topology": {
        "joints": [
            {"id": 0, "position": [0.0, 0.0, 0.0], "support": "pinned"},
            {"id": 1, "position": [1.0, 0.0, 0.0], "support": "roller"},
            {"id": 2, "position": [0.5, 0.5, 0.0], "support": None},
        ],
        "members": [
            {"id": 0, "joint_1": 0, "joint_2": 2, "material": "6061_T6_Aluminum",
             "shape": {"type": "Pipe", "r": 0.03, "t": 0.004}},
            {"id": 1, "joint_1": 1, "joint_2": 2, "material": "6061_T6_Aluminum",
             "shape": {"type": "Pipe", "r": 0.03, "t": 0.004}},
        ],
    },
    "loading": [{"joint": 2, "force": [0, -10000, 0]}],
    "goals": {
        "minimum_fos_buckling": 1.5,
        "minimum_fos_yielding": 1.5,
        "maximum_mass": 50.0,
    },
}


@SKIP_FEA
def test_env_reset():
    env = TrussRolloutEnv(max_steps=5)
    state = env.reset(MINI_PROBLEM_SPEC)
    assert isinstance(state, dict)
    assert "mass" in state or "error" in state


@SKIP_FEA
def test_env_run_completion_parse_failure():
    env = TrussRolloutEnv(max_steps=5)
    rollout = env.run_completion(MINI_PROBLEM_SPEC, "No valid action here")
    assert rollout.problem_id == "test_problem"
    assert len(rollout.action_sequence) == 0


@SKIP_FEA
def test_env_run_completion_with_action():
    env = TrussRolloutEnv(max_steps=5)
    rollout = env.run_completion(
        MINI_PROBLEM_SPEC,
        "SCALE_PARAM(0, radius, 1.2)"
    )
    assert rollout.problem_id == "test_problem"
    # Grammar parse should succeed
    assert len(rollout.parse_success) > 0


@SKIP_FEA
def test_env_run_batch():
    env = TrussRolloutEnv(max_steps=5, n_workers=2)
    completions = ["SCALE_PARAM(0, radius, 1.2)", "ADD_MEMBER(0, 1, 6061_T6_Aluminum, Pipe, 0.03, 0.004)"]
    specs = [MINI_PROBLEM_SPEC, MINI_PROBLEM_SPEC]
    rollouts = env.run_batch(specs, completions)
    assert len(rollouts) == 2
