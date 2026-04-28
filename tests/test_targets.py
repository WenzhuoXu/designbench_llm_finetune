"""
Tests for SFT target functions.

Tests get_loss_mask() for each target with synthetic token sequences.
Run with: pytest tests/test_targets.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from unittest.mock import MagicMock

from llm_finetune.training.sft.targets import (
    FullSequenceTarget,
    WarmstartReasoningTarget,
    TARGET_REGISTRY,
    build_target_from_config,
)


def make_mock_tokenizer(decode_map: dict = None):
    """Make a mock tokenizer that returns text based on token IDs."""
    tokenizer = MagicMock()
    if decode_map:
        tokenizer.decode.side_effect = lambda ids, **kwargs: decode_map.get(
            ids[0] if isinstance(ids, list) else ids.item(), f"<tok_{ids}>"
        )
    else:
        tokenizer.decode.side_effect = lambda ids, **kwargs: f"<tok_{ids}>"
    return tokenizer


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
    # System messages merged
    assert msgs[0]["role"] == "system"
    assert "INITIAL STATE" in msgs[0]["content"]
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


# ── Registry tests ────────────────────────────────────────────────────────────

def test_registry_has_expected_targets():
    expected = ["warmstart_reasoning", "full_sequence"]
    for name in expected:
        assert name in TARGET_REGISTRY, f"Missing: {name}"

def test_registry_has_no_extras():
    """Registry should only contain targets that work with the real data."""
    assert set(TARGET_REGISTRY.keys()) == {"warmstart_reasoning", "full_sequence"}

def test_build_target_from_config():
    for name in TARGET_REGISTRY:
        target = build_target_from_config(name)
        assert target.name() is not None

def test_build_target_invalid():
    with pytest.raises(ValueError, match="Unknown SFT target"):
        build_target_from_config("nonexistent_target")
