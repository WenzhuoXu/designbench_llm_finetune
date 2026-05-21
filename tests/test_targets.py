"""
Tests for SFT target functions.

Tests get_loss_mask() for each target with synthetic token sequences.
Run with: pytest tests/test_targets.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock

import pytest
import torch

from llm_finetune.training.sft.targets import (
    TARGET_REGISTRY,
    FullSequenceTarget,
    GoldCurriculumWarmstartTarget,
    WarmstartReasoningTarget,
    build_target_from_config,
)


def make_mock_tokenizer(decode_map: dict = None):
    """Make a mock tokenizer that returns text based on token IDs."""
    tokenizer = MagicMock()
    if decode_map:
        def decode(ids, **kwargs):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(ids, list):
                return "".join(decode_map.get(i, f"<tok_{i}>") for i in ids)
            return decode_map.get(ids, f"<tok_{ids}>")

        tokenizer.decode.side_effect = decode
    else:
        tokenizer.decode.side_effect = lambda ids, **kwargs: f"<tok_{ids}>"
    return tokenizer


def make_char_tokenized(text: str):
    """Return token ids plus a tokenizer that decodes one character per token."""
    decode_map = {idx: ch for idx, ch in enumerate(text)}
    return torch.tensor(list(range(len(text)))), make_mock_tokenizer(decode_map)


# ── FullSequenceTarget ────────────────────────────────────────────────────────

def test_full_sequence_supervises_non_ignored():
    target = FullSequenceTarget()
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    labels = torch.tensor([-100, 2, 3, 4, -100])  # first and last ignored
    tokenizer = make_mock_tokenizer()

    mask = target.get_loss_mask(input_ids, labels, tokenizer)
    assert mask.shape == (5,)
    assert mask[0].item() == 0   # -100 → no loss
    assert mask[1].item() == 1   # valid label → loss
    assert mask[4].item() == 0   # -100 → no loss

def test_full_sequence_all_valid():
    target = FullSequenceTarget()
    input_ids = torch.tensor([1, 2, 3])
    labels = torch.tensor([1, 2, 3])
    mask = target.get_loss_mask(input_ids, labels, make_mock_tokenizer())
    assert mask.sum().item() == 3

def test_full_sequence_passthrough():
    """FullSequenceTarget should pass examples through unchanged."""
    target = FullSequenceTarget()
    example = {"input_ids": [1, 2, 3], "labels": [1, 2, 3]}
    result = target.transform_example(example, make_mock_tokenizer())
    assert result == example


# ── WarmstartReasoningTarget ─────────────────────────────────────────────────

def test_warmstart_reasoning_transforms():
    """WarmstartReasoningTarget should reformat messages."""
    target = WarmstartReasoningTarget()
    example = {
        "messages": [
            {"role": "system", "content": "PROBLEM: test"},
            {"role": "system", "content": "INITIAL STATE: test"},
            {"role": "assistant", "content": "<think>Modification: Scale param\nAction: SCALE_PARAM(1, thickness, 1.2)</think>"},
            {"role": "system", "content": "RESULT: Mass: 100"},
            {"role": "assistant", "content": "<answer>Done</answer>"},
        ]
    }
    tokenizer = make_mock_tokenizer()
    result = target.transform_example(example, tokenizer)
    # Messages should be reformatted (not passthrough)
    assert result is not example
    msgs = result["messages"]
    # Raw problem/state should become the first user turn under a stable
    # instruction system prompt.
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "INITIAL STATE" in msgs[1]["content"]
    # Action should be in <action> tags
    asst = [m for m in msgs if m["role"] == "assistant" and "<action>" in m["content"]]
    assert len(asst) >= 1
    assert "SCALE_PARAM" in asst[0]["content"]

def test_warmstart_reasoning_does_not_mutate_original():
    """Transform should not modify the original example dict."""
    target = WarmstartReasoningTarget()
    original_messages = [
        {"role": "system", "content": "PROBLEM"},
        {"role": "system", "content": "STATE"},
        {"role": "assistant", "content": "<think>Modification: X\nAction: SCALE_PARAM(1, t, 1.1)</think>"},
    ]
    example = {"messages": original_messages, "problem_id": "test"}
    target.transform_example(example, make_mock_tokenizer())
    # Original should be untouched
    assert example["messages"] is original_messages
    assert example["messages"][0]["role"] == "system"


# ── GoldCurriculumWarmstartTarget ────────────────────────────────────────────

def test_gold_curriculum_turn_slices_actions():
    target = GoldCurriculumWarmstartTarget(curriculum_stage=2)
    example = {
        "problem_id": "p0",
        "trace_id": "t0",
        "trace_quality": 1.0,
        "strategy_type": "Mixed",
        "messages": [
            {"role": "system", "content": "PROBLEM: test"},
            {"role": "system", "content": "INITIAL STATE: infeasible"},
            {
                "role": "assistant",
                "content": "<think>Modification: Scale member\nAction: SCALE_PARAM(1, thickness, 1.2)</think>",
            },
            {"role": "system", "content": "STRUCTURAL ANALYSIS RESULT:\nStatus: INFEASIBLE"},
            {
                "role": "assistant",
                "content": "<think>Modification: Add member\nAction: ADD_MEMBER(0, 1, A36_Steel, Pipe, 0.03, 0.004)</think>",
            },
        ],
    }

    result = target.transform_example(example, make_mock_tokenizer())
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["step_index"] == 0
    assert result[0]["action_type"] == "SCALE_PARAM"
    assert result[1]["step_index"] == 1
    assert result[1]["messages"][-1]["role"] == "assistant"
    assert "<action>ADD_MEMBER" in result[1]["messages"][-1]["content"]


def test_gold_curriculum_stage_one_action_only():
    target = GoldCurriculumWarmstartTarget(curriculum_stage=1)
    example = {
        "messages": [
            {"role": "system", "content": "PROBLEM"},
            {"role": "system", "content": "STATE"},
            {
                "role": "assistant",
                "content": "<think>Modification: X\nAction: SCALE_PARAM(1, t, 1.1)</think>",
            },
        ]
    }
    result = target.transform_example(example, make_mock_tokenizer())
    assert result[0]["messages"][-1]["content"] == "<action>SCALE_PARAM(1, t, 1.1)</action>"


def test_gold_curriculum_masks_only_final_assistant_span_and_stop():
    target = GoldCurriculumWarmstartTarget(curriculum_stage=2)
    text = (
        "<|im_start|>system\ninstructions<|im_end|>\n"
        "<|im_start|>user\nproblem<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<action>OLD_ACTION()</action><|im_end|>\n"
        "<|im_start|>user\nfeedback<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\nreason\n</think>\n\n<action>NEW_ACTION()</action><|im_end|>\n"
    )
    input_ids, tokenizer = make_char_tokenized(text)
    labels = input_ids.clone()

    mask = target.get_loss_mask(input_ids, labels, tokenizer)

    old_pos = text.index("OLD_ACTION")
    new_pos = text.index("NEW_ACTION")
    final_end_pos = text.rfind("<|im_end|>") + len("<|im_end|>") - 1
    user_pos = text.index("feedback")

    assert mask[old_pos].item() == 0
    assert mask[user_pos].item() == 0
    assert mask[new_pos].item() == 1
    assert mask[final_end_pos].item() == 1


# ── Registry tests ────────────────────────────────────────────────────────────

def test_registry_has_expected_targets():
    expected = ["gold_curriculum_warmstart", "warmstart_reasoning", "full_sequence"]
    for name in expected:
        assert name in TARGET_REGISTRY, f"Missing: {name}"

def test_registry_has_no_extras():
    """Registry should only contain targets that work with the real data."""
    assert set(TARGET_REGISTRY.keys()) == {
        "gold_curriculum_warmstart",
        "warmstart_reasoning",
        "full_sequence",
    }

def test_build_target_from_config():
    for name in TARGET_REGISTRY:
        target = build_target_from_config(name)
        assert target.name() is not None

def test_build_target_invalid():
    with pytest.raises(ValueError, match="Unknown SFT target"):
        build_target_from_config("nonexistent_target")
