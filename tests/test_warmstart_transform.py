"""
Tests for warmstart message transform.

Tests the WarmstartTransform class which reformats DesignBench SFT messages
for warmstart pre-training. These tests run on the login node (no GPU needed).

Run:
    pytest tests/test_warmstart_transform.py -v
"""

import json
import re
from pathlib import Path

import pytest

from llm_finetune.data.processors.warmstart_transform import (
    WarmstartTransform,
    ACTION_PATTERN,
)


# ── Sample data ──────────────────────────────────────────────────────────────

SAMPLE_MESSAGES = [
    {
        "role": "system",
        "content": (
            "PROBLEM: Truss optimization problem (ground structure variant 79)\n\n"
            "INITIAL STRUCTURE:\nJoints (8 total):\n  J0: pos=(0.00, 0.00) [pinned]\n"
            "Members (13 total):\n  M0: J0-J1 [A36_Steel, Pipe(r=0.0275, t=0.0037)]\n"
            "LOADING:\n  Joint J5: Force=(0, -34175, 0) N\n\n"
            "DESIGN GOALS:\n  - Minimum Factor of Safety (buckling): 1.5\n"
            "  - Maximum allowable mass: 272.70 kg"
        ),
    },
    {
        "role": "system",
        "content": (
            "INITIAL STATE ANALYSIS:\nMass: 213.21 kg\n"
            "Factor of Safety (buckling): 0.73\n"
            "Factor of Safety (yielding): 2.68\n"
            "Maximum deflection: 4.1762e-03 m\n"
            "Status: INFEASIBLE \u2717\n\n"
            "Constraint violations:\n  - Buckling FOS 0.73 < 1.5 (minimum required)"
        ),
    },
    {
        "role": "assistant",
        "content": "<think>Modification: Add member connecting joints 1 and 6\nAction: ADD_MEMBER(1, 6, A36_Steel, Pipe, 0.023171, 0.003862)</think>",
    },
    {
        "role": "system",
        "content": (
            "STRUCTURAL ANALYSIS RESULT:\nMass: 229.89 kg\n"
            "Factor of Safety (buckling): 0.73\n"
            "Factor of Safety (yielding): 4.52\n"
            "Status: INFEASIBLE \u2717"
        ),
    },
    {
        "role": "assistant",
        "content": "<think>Modification: Increase 6 thickness by 63%\nAction: SCALE_PARAM(6, thickness, 1.631)</think>",
    },
    {
        "role": "system",
        "content": (
            "STRUCTURAL ANALYSIS RESULT:\nMass: 247.91 kg\n"
            "Factor of Safety (buckling): 1.52\n"
            "Status: FEASIBLE \u2713"
        ),
    },
    {
        "role": "assistant",
        "content": "<answer>Optimal feasible solution found via LP</answer>",
    },
]


@pytest.fixture
def transform():
    return WarmstartTransform()


@pytest.fixture
def transformed(transform):
    return transform.transform_messages(SAMPLE_MESSAGES)


# ── Role remapping tests ─────────────────────────────────────────────────────

class TestRoleRemapping:
    """Test that roles are correctly remapped."""

    def test_first_message_is_system(self, transformed):
        assert transformed[0]["role"] == "system"

    def test_system_messages_merged(self, transformed):
        """messages[0] + messages[1] should be merged into one system message."""
        system_content = transformed[0]["content"]
        assert "PROBLEM:" in system_content
        assert "INITIAL STATE ANALYSIS:" in system_content

    def test_fea_feedback_becomes_user(self, transformed):
        """FEA/simulation feedback messages should become user role."""
        user_msgs = [m for m in transformed if m["role"] == "user"]
        assert len(user_msgs) >= 1
        for m in user_msgs:
            assert m["content"].startswith("[Simulation Result]")

    def test_assistant_stays_assistant(self, transformed):
        """Assistant messages should remain assistant role."""
        asst_msgs = [m for m in transformed if m["role"] == "assistant"]
        assert len(asst_msgs) >= 2  # at least action + answer

    def test_no_consecutive_same_role(self, transformed):
        """After transform, no two consecutive messages should have system role."""
        for i in range(1, len(transformed)):
            if transformed[i]["role"] == "system":
                assert transformed[i - 1]["role"] != "system", (
                    f"Consecutive system messages at index {i-1} and {i}"
                )


# ── Action extraction tests ──────────────────────────────────────────────────

class TestActionExtraction:
    """Test that actions are correctly extracted and wrapped."""

    def test_action_in_action_tags(self, transformed):
        """Grammar actions should be in <action>...</action> tags."""
        asst_msgs = [m for m in transformed if m["role"] == "assistant"]
        action_msgs = [m for m in asst_msgs if "<action>" in m["content"]]
        assert len(action_msgs) >= 1

        for m in action_msgs:
            match = re.search(r"<action>(.*?)</action>", m["content"], re.DOTALL)
            assert match is not None, f"No <action> tag found in: {m['content']}"
            action = match.group(1).strip()
            assert ACTION_PATTERN.match(action), f"Unparseable action: {action}"

    def test_action_outside_think(self, transformed):
        """Actions should be AFTER </think>, not inside <think>."""
        asst_msgs = [m for m in transformed if m["role"] == "assistant"]
        for m in asst_msgs:
            if "<think>" in m["content"] and "<action>" in m["content"]:
                think_end = m["content"].index("</think>")
                action_start = m["content"].index("<action>")
                assert action_start > think_end, (
                    f"<action> should come after </think>: {m['content'][:200]}"
                )

    def test_add_member_parsed(self, transformed):
        """ADD_MEMBER action should be correctly extracted."""
        asst_msgs = [m for m in transformed if m["role"] == "assistant"]
        add_member_found = any(
            "ADD_MEMBER" in m["content"] for m in asst_msgs
        )
        assert add_member_found

    def test_scale_param_parsed(self, transformed):
        """SCALE_PARAM action should be correctly extracted."""
        asst_msgs = [m for m in transformed if m["role"] == "assistant"]
        scale_found = any(
            "SCALE_PARAM" in m["content"] for m in asst_msgs
        )
        assert scale_found


# ── Think content tests ──────────────────────────────────────────────────────

class TestThinkContent:
    """Test that <think> blocks contain cleaned descriptions."""

    def test_think_has_description(self, transformed):
        """<think> blocks should contain a human-readable description."""
        asst_msgs = [m for m in transformed if m["role"] == "assistant"]
        for m in asst_msgs:
            if "<think>" in m["content"] and "<action>" in m["content"]:
                think_match = re.search(r"<think>(.*?)</think>", m["content"], re.DOTALL)
                assert think_match is not None
                think_text = think_match.group(1).strip()
                assert len(think_text) > 5, f"Think block too short: {think_text}"

    def test_think_no_action_prefix(self, transformed):
        """<think> blocks should not contain 'Action:' prefix."""
        asst_msgs = [m for m in transformed if m["role"] == "assistant"]
        for m in asst_msgs:
            if "<think>" in m["content"]:
                think_match = re.search(r"<think>(.*?)</think>", m["content"], re.DOTALL)
                if think_match:
                    assert "Action:" not in think_match.group(1), (
                        f"'Action:' should not be in <think>: {think_match.group(1)}"
                    )


# ── Answer message tests ─────────────────────────────────────────────────────

class TestAnswerMessage:
    """Test that <answer> messages are preserved."""

    def test_answer_preserved(self, transformed):
        """<answer>...</answer> messages should pass through unchanged."""
        answer_msgs = [m for m in transformed if "<answer>" in m.get("content", "")]
        assert len(answer_msgs) == 1
        assert "Optimal feasible solution found" in answer_msgs[0]["content"]

    def test_answer_is_assistant(self, transformed):
        """<answer> messages should remain assistant role."""
        for m in transformed:
            if "<answer>" in m.get("content", ""):
                assert m["role"] == "assistant"


# ── Message count and structure tests ────────────────────────────────────────

class TestStructure:
    """Test overall message structure after transform."""

    def test_fewer_messages_after_merge(self, transformed):
        """Should have fewer messages since system msgs are merged."""
        assert len(transformed) < len(SAMPLE_MESSAGES)

    def test_alternating_roles(self, transformed):
        """After system msg, roles should roughly alternate user/assistant."""
        roles = [m["role"] for m in transformed]
        # First should be system
        assert roles[0] == "system"
        # Rest should alternate (assistant, user, assistant, ..., assistant)
        for i in range(2, len(roles)):
            if roles[i] == roles[i - 1]:
                # Two consecutive same roles is unexpected
                assert False, (
                    f"Unexpected consecutive roles at {i-1},{i}: "
                    f"{roles[i-1]}, {roles[i]} in {roles}"
                )


# ── Integration test with real data ──────────────────────────────────────────

TRAIN_JSONL = Path("/ocean/projects/mch250030p/wxu7/DesignBench/data/sft/train.jsonl")


@pytest.mark.skipif(
    not TRAIN_JSONL.exists(),
    reason="DesignBench SFT data not available",
)
class TestRealData:
    """Integration tests against actual DesignBench SFT data."""

    def test_transform_first_10_examples(self, transform):
        """Transform first 10 real examples without errors."""
        with open(TRAIN_JSONL) as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                ex = json.loads(line)
                result = transform.transform_messages(ex["messages"])
                assert len(result) >= 3, f"Example {i}: too few messages after transform"

                # Check at least one <action> tag exists
                action_count = sum(
                    1 for m in result
                    if m["role"] == "assistant" and "<action>" in m["content"]
                )
                assert action_count >= 1, f"Example {i}: no <action> tags found"

    def test_all_actions_parseable(self, transform):
        """All extracted actions should match grammar patterns."""
        with open(TRAIN_JSONL) as f:
            for i, line in enumerate(f):
                if i >= 50:
                    break
                ex = json.loads(line)
                result = transform.transform_messages(ex["messages"])

                for m in result:
                    if m["role"] == "assistant" and "<action>" in m["content"]:
                        match = re.search(
                            r"<action>(.*?)</action>", m["content"], re.DOTALL
                        )
                        assert match, f"Example {i}: malformed <action> tag"
                        action = match.group(1).strip()
                        assert ACTION_PATTERN.match(action), (
                            f"Example {i}: unparseable action: {action}"
                        )
