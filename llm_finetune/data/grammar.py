"""Shared DesignBench grammar extraction and validation utilities.

The validator is intentionally stricter than the old regex parser: it rejects
placeholder examples and shorthand ranges that look parseable to a regex but
cannot execute in the truss environment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

ACTION_NAMES = (
    "SCALE_MULTI_PARAM",
    "SCALE_PARAM",
    "ADD_MEMBER",
    "MODIFY_PARAM",
    "REMOVE_MEMBER",
    "MOVE_JOINT",
    "OPTIMAL_STATE",
)

ACTION_NAME_PATTERN = "|".join(ACTION_NAMES)
ACTION_PATTERN = re.compile(
    rf"\b({ACTION_NAME_PATTERN})\s*(?:\((?:[^()]|\[[^\]]*\])*\))?",
    re.IGNORECASE,
)
ACTION_TAG_PATTERN = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)
ANSWER_TAG_PATTERN = re.compile(r"<answer\b[^>]*>.*?</answer>", re.DOTALL | re.IGNORECASE)
THINK_TAG_PATTERN = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)

PLACEHOLDER_PATTERN = re.compile(
    r"\b(ids?|member_ids?|all_members|param|factor|param:factor)\b",
    re.IGNORECASE,
)
RANGE_LIST_PATTERN = re.compile(r"\[[^\]]*\b\d+\s*-\s*\d+\b[^\]]*\]")

PARAM_ALIASES = {"r", "radius", "t", "thickness"}


@dataclass(frozen=True)
class ValidationResult:
    """Result of strict grammar validation."""

    is_valid: bool
    action: str = ""
    action_type: str = ""
    reason: str = ""
    canonical_action: str = ""


def find_actions(text: str, *, prefer_tags: bool = True) -> list[str]:
    """Return action-looking strings from text.

    If ``prefer_tags`` is true and any ``<action>`` tags are present, only the
    tag contents are considered. This lets compliance checks reject generations
    with multiple action tags instead of silently accepting the first one.
    """
    if not text:
        return []

    tag_contents = ACTION_TAG_PATTERN.findall(text)
    if prefer_tags and tag_contents:
        return [_strip_action_candidate(item) for item in tag_contents if item.strip()]

    return [_strip_action_candidate(m.group(0)) for m in ACTION_PATTERN.finditer(text)]


def extract_action(text: str) -> Optional[str]:
    """Extract the first semantically valid grammar action from model output."""
    for candidate in find_actions(text):
        result = validate_action(candidate)
        if result.is_valid:
            return result.action
    return None


def canonicalize_action(action: str) -> str:
    """Return a stable, whitespace-normalized action string."""
    action = _strip_action_candidate(action)
    match = re.match(rf"^\s*({ACTION_NAME_PATTERN})\s*(?:\((.*)\))?\s*$", action, re.I | re.S)
    if not match:
        return action.strip()

    name = match.group(1).upper()
    args = match.group(2)
    if args is None:
        return name
    return f"{name}({', '.join(_split_args(args))})"


def validate_action(action: str, problem_or_state: Optional[dict] = None) -> ValidationResult:
    """Validate one DesignBench grammar action.

    Args:
        action: Raw action string, optionally wrapped in ``<action>`` tags.
        problem_or_state: Optional state/spec dict. When member ids are present,
            the validator checks that referenced ids exist.
    """
    action = _strip_action_candidate(action)
    if not action:
        return ValidationResult(False, reason="empty_action")

    if PLACEHOLDER_PATTERN.search(action):
        return ValidationResult(False, action=action, reason="placeholder_token")
    if RANGE_LIST_PATTERN.search(action):
        return ValidationResult(False, action=action, reason="range_member_ids")

    match = re.match(rf"^\s*({ACTION_NAME_PATTERN})\s*(?:\((.*)\))?\s*$", action, re.I | re.S)
    if not match:
        return ValidationResult(False, action=action, reason="unknown_action")

    action_type = match.group(1).upper()
    args_text = match.group(2)
    args = _split_args(args_text or "")
    canonical = canonicalize_action(action)

    validator = {
        "SCALE_PARAM": _validate_scale_param,
        "SCALE_MULTI_PARAM": _validate_scale_multi_param,
        "ADD_MEMBER": _validate_add_member,
        "MODIFY_PARAM": _validate_modify_param,
        "REMOVE_MEMBER": _validate_remove_member,
        "MOVE_JOINT": _validate_move_joint,
        "OPTIMAL_STATE": _validate_optimal_state,
    }[action_type]
    reason = validator(args, problem_or_state)
    if reason:
        return ValidationResult(
            False,
            action=action,
            action_type=action_type,
            reason=reason,
            canonical_action=canonical,
        )
    return ValidationResult(
        True,
        action=action,
        action_type=action_type,
        canonical_action=canonical,
    )


def count_answer_tags(text: str) -> int:
    return len(ANSWER_TAG_PATTERN.findall(text or ""))


def count_think_tags(text: str) -> int:
    return len(THINK_TAG_PATTERN.findall(text or ""))


def _strip_action_candidate(text: str) -> str:
    text = re.sub(r"^\s*<action>\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*</action>\s*$", "", text.strip(), flags=re.I)
    match = ACTION_PATTERN.search(text)
    return match.group(0).strip() if match else text.strip()


def _split_args(args_text: str) -> list[str]:
    if not args_text:
        return []
    args: list[str] = []
    current: list[str] = []
    bracket_depth = 0
    for char in args_text:
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        if char == "," and bracket_depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current or args_text.endswith(","):
        args.append("".join(current).strip())
    return args


def _validate_scale_param(args: list[str], problem_or_state: Optional[dict]) -> str:
    if len(args) != 3:
        return "scale_param_arity"
    if not _is_int(args[0]):
        return "member_id_not_int"
    if not _valid_param(args[1]):
        return "invalid_parameter"
    if not _is_float(args[2]):
        return "factor_not_float"
    return _check_member_ids([int(args[0])], problem_or_state)


def _validate_scale_multi_param(args: list[str], problem_or_state: Optional[dict]) -> str:
    if len(args) != 2:
        return "scale_multi_param_arity"
    ids = _parse_int_list(args[0])
    if not ids:
        return "member_list_not_explicit_ints"
    params = _parse_param_factor_list(args[1])
    if not params:
        return "param_factor_list_invalid"
    return _check_member_ids(ids, problem_or_state)


def _validate_add_member(args: list[str], problem_or_state: Optional[dict]) -> str:
    if len(args) != 6:
        return "add_member_arity"
    if not _is_int(args[0]) or not _is_int(args[1]):
        return "joint_id_not_int"
    if not args[2] or not args[3]:
        return "missing_material_or_shape"
    if not _is_float(args[4]) or not _is_float(args[5]):
        return "dimension_not_float"
    return ""


def _validate_modify_param(args: list[str], problem_or_state: Optional[dict]) -> str:
    if len(args) != 4:
        return "modify_param_arity"
    if not _is_int(args[0]):
        return "member_id_not_int"
    if not _valid_param(args[1]):
        return "invalid_parameter"
    if not _is_float(args[2]) or not _is_float(args[3]):
        return "value_not_float"
    return _check_member_ids([int(args[0])], problem_or_state)


def _validate_remove_member(args: list[str], problem_or_state: Optional[dict]) -> str:
    if len(args) != 1:
        return "remove_member_arity"
    if not _is_int(args[0]):
        return "member_id_not_int"
    return _check_member_ids([int(args[0])], problem_or_state)


def _validate_move_joint(args: list[str], problem_or_state: Optional[dict]) -> str:
    if len(args) != 3:
        return "move_joint_arity"
    if not _is_int(args[0]):
        return "joint_id_not_int"
    # trussme's executor takes 3-vectors; this validator demanded 2 and so rejected every
    # MOVE_JOINT the executor accepts. Accept either, since the executor is the authority.
    if (_parse_float_list(args[1], expected_len=2) is None
            and _parse_float_list(args[1], expected_len=3) is None):
        return "old_position_invalid"
    if (_parse_float_list(args[2], expected_len=2) is None
            and _parse_float_list(args[2], expected_len=3) is None):
        return "new_position_invalid"
    return ""


def _validate_optimal_state(args: list[str], problem_or_state: Optional[dict]) -> str:
    return "" if len(args) == 0 else "optimal_state_takes_no_args"


def _parse_int_list(text: str) -> list[int]:
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return []
    inner = text[1:-1].strip()
    if not inner:
        return []
    values = [item.strip() for item in inner.split(",")]
    if any(not _is_int(item) for item in values):
        return []
    return [int(item) for item in values]


def _parse_param_factor_list(text: str) -> dict[str, float]:
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return {}
    inner = text[1:-1].strip()
    if not inner:
        return {}
    result: dict[str, float] = {}
    for item in inner.split(","):
        if ":" not in item:
            return {}
        key, value = [part.strip() for part in item.split(":", 1)]
        if not _valid_param(key) or not _is_float(value):
            return {}
        result[key] = float(value)
    return result


def _parse_float_list(text: str, expected_len: int) -> Optional[list[float]]:
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    values = [item.strip() for item in text[1:-1].split(",")]
    if len(values) != expected_len or any(not _is_float(item) for item in values):
        return None
    return [float(item) for item in values]


def _valid_param(param: str) -> bool:
    return param.strip().lower() in PARAM_ALIASES


def _is_int(value: str) -> bool:
    return re.fullmatch(r"[+-]?\d+", value.strip()) is not None


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _check_member_ids(ids: list[int], problem_or_state: Optional[dict]) -> str:
    known_ids = _known_member_ids(problem_or_state)
    if known_ids is None:
        return ""
    missing = [member_id for member_id in ids if member_id not in known_ids]
    return "unknown_member_id" if missing else ""


def _known_member_ids(problem_or_state: Optional[dict]) -> Optional[set[int]]:
    if not isinstance(problem_or_state, dict):
        return None

    member_dimensions = problem_or_state.get("member_dimensions")
    if isinstance(member_dimensions, dict) and member_dimensions:
        known: set[int] = set()
        for key in member_dimensions:
            try:
                known.add(int(key))
            except (TypeError, ValueError):
                pass
        return known or None

    topology = problem_or_state.get("topology")
    if isinstance(topology, dict):
        members = topology.get("members")
        if isinstance(members, list):
            known = {int(m["id"]) for m in members if isinstance(m, dict) and "id" in m}
            return known or None

    return None
