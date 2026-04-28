"""State and action featurization for posterior evaluation."""

from __future__ import annotations

import math
from typing import Iterable

STATE_FEATURE_NAMES = [
    "fos_b_ratio",
    "fos_y_ratio",
    "defl_ratio",
    "mass_ratio",
    "is_feasible",
    "buckling_slack",
    "yielding_slack",
    "defl_slack",
    "dv_buckling",
    "dv_yielding",
    "dv_deflection",
    "dv_none",
    "n_members",
    "depth_norm",
]

ACTION_CLASS_ORDER = [
    "SCALE_PARAM",
    "MODIFY_PARAM",
    "ADD_MEMBER",
    "REMOVE_MEMBER",
    "MOVE_JOINT",
]

LOCAL_ACTIONS = {"SCALE_PARAM", "MODIFY_PARAM"}
MID_ACTIONS = {"MOVE_JOINT"}
NON_LOCAL_ACTIONS = {"ADD_MEMBER", "REMOVE_MEMBER"}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def action_class(action: str) -> str:
    if not action:
        return "UNKNOWN"
    prefix = action.split("(", 1)[0].strip().upper()
    return prefix


def action_locality(action: str) -> str:
    cls = action_class(action)
    if cls in LOCAL_ACTIONS:
        return "local"
    if cls in MID_ACTIONS:
        return "mid"
    if cls in NON_LOCAL_ACTIONS:
        return "non_local"
    return "unknown"


def dominant_violation(state: dict, target_fos: float = 1.5, max_deflection: float = 0.01) -> str:
    fos_b = _safe_float(state.get("fos_buckling"))
    fos_y = _safe_float(state.get("fos_yielding"))
    deflection = _safe_float(state.get("deflection"))
    gaps = {
        "buckling": max(0.0, target_fos - fos_b),
        "yielding": max(0.0, target_fos - fos_y),
        "deflection": max(0.0, deflection - max_deflection),
    }
    key = max(gaps, key=gaps.get)
    return "none" if gaps[key] <= 0.0 else key


def extract_state_features(
    state: dict,
    *,
    initial_mass: float | None = None,
    mean_horizon: float = 9.0,
    target_fos: float = 1.5,
    max_deflection: float = 0.01,
) -> list[float]:
    """Map an env state into the 14-dim feature vector from the master doc."""
    fos_b = _safe_float(state.get("fos_buckling"))
    fos_y = _safe_float(state.get("fos_yielding"))
    deflection = _safe_float(state.get("deflection"))
    mass = _safe_float(state.get("mass"))
    is_feasible = 1.0 if state.get("is_feasible", False) else 0.0
    n_members = float(len(state.get("member_dimensions", {}) or {}))
    step = _safe_float(state.get("step"))

    if initial_mass is None:
        initial_mass = mass if mass > 0.0 else 1.0
    initial_mass = max(initial_mass, 1e-6)
    depth_norm = step / max(mean_horizon, 1.0)

    dv = dominant_violation(
        state,
        target_fos=target_fos,
        max_deflection=max_deflection,
    )

    dv_flags = {
        "buckling": 1.0 if dv == "buckling" else 0.0,
        "yielding": 1.0 if dv == "yielding" else 0.0,
        "deflection": 1.0 if dv == "deflection" else 0.0,
        "none": 1.0 if dv == "none" else 0.0,
    }

    return [
        fos_b / target_fos if target_fos > 0.0 else 0.0,
        fos_y / target_fos if target_fos > 0.0 else 0.0,
        deflection / max_deflection if max_deflection > 0.0 else 0.0,
        mass / initial_mass,
        is_feasible,
        max(0.0, target_fos - fos_b),
        max(0.0, target_fos - fos_y),
        max(0.0, deflection - max_deflection),
        dv_flags["buckling"],
        dv_flags["yielding"],
        dv_flags["deflection"],
        dv_flags["none"],
        n_members,
        depth_norm,
    ]


def difficulty_score(
    state: dict,
    *,
    target_fos: float = 1.5,
    max_deflection: float = 0.01,
) -> float:
    """Difficulty proxy from the master doc's section 4.2."""
    fos_b = _safe_float(state.get("fos_buckling"))
    fos_y = _safe_float(state.get("fos_yielding"))
    deflection = _safe_float(state.get("deflection"))
    fos_min = min(
        fos_b if fos_b > 0.0 else 0.0,
        fos_y if fos_y > 0.0 else 0.0,
    )
    fos_term = max(1.0 - (fos_min / max(target_fos, 1e-6)), 0.0)
    deflection_term = max((deflection / max(max_deflection, 1e-6)) - 1.0, 0.0)
    return fos_term + deflection_term


def shannon_entropy(probabilities: Iterable[float]) -> float:
    probs = [p for p in probabilities if p > 0.0]
    return -sum(p * math.log(p) for p in probs)
