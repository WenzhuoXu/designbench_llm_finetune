"""Tests for strict DesignBench grammar validation."""

from llm_finetune.data.grammar import extract_action, find_actions, validate_action
from scripts.check_format_compliance import evaluate_generation_text


def test_extract_action_accepts_valid_action():
    text = "<think>Scale the critical member.</think>\n<action>SCALE_PARAM(3, radius, 1.15)</action>"
    assert extract_action(text) == "SCALE_PARAM(3, radius, 1.15)"


def test_validate_rejects_placeholder_ids():
    result = validate_action("SCALE_MULTI_PARAM([ids], [param:factor, ...])")
    assert not result.is_valid
    assert result.reason == "placeholder_token"


def test_validate_rejects_range_ids():
    result = validate_action("SCALE_MULTI_PARAM([0-11], [r:1.1, t:1.1])")
    assert not result.is_valid
    assert result.reason == "range_member_ids"


def test_validate_accepts_explicit_multi_member_list():
    result = validate_action("SCALE_MULTI_PARAM([0, 11], [r:1.1, t:1.1])")
    assert result.is_valid
    assert result.action_type == "SCALE_MULTI_PARAM"


def test_validate_checks_known_member_ids():
    state = {"member_dimensions": {"0": {}, "1": {}}}
    result = validate_action("SCALE_PARAM(3, radius, 1.15)", state)
    assert not result.is_valid
    assert result.reason == "unknown_member_id"


def test_find_actions_prefers_action_tags():
    text = "<action>SCALE_PARAM(1, t, 1.1)</action> SCALE_PARAM(2, t, 1.1)"
    assert find_actions(text) == ["SCALE_PARAM(1, t, 1.1)"]


def test_compliance_rejects_multiple_actions():
    result = evaluate_generation_text(
        "<action>SCALE_PARAM(1, t, 1.1)</action><action>SCALE_PARAM(2, t, 1.1)</action>"
    )
    assert not result["semantic_ok"]
    assert not result["single_action"]


def test_compliance_rejects_answer_spam():
    result = evaluate_generation_text(
        "<action>SCALE_PARAM(1, t, 1.1)</action><answer>Done</answer>"
    )
    assert result["semantic_ok"]
    assert result["answer_spam"]
