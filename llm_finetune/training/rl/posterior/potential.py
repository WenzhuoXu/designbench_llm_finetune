"""Potential-based reward shaping for posterior evaluation."""

from __future__ import annotations

import math


def _finite_or_default(value: object, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def softplus(x: float) -> float:
    if x > 20:
        return x
    if x < -20:
        return math.exp(x)
    return math.log1p(math.exp(x))


def violation_score(
    state: dict,
    *,
    target_fos: float = 1.5,
    max_deflection: float = 0.01,
) -> float:
    # Clamp FOS to [0, inf): negative values from failed FEA have no physical
    # meaning and would cause softplus(1.5 - (-1e25)) = 1e25, blowing up rewards.
    fos_b = max(0.0, _finite_or_default(state.get("fos_buckling", 0.0) or 0.0, 0.0))
    fos_y = max(0.0, _finite_or_default(state.get("fos_yielding", 0.0) or 0.0, 0.0))
    _max_d = max(max_deflection, 1e-6)
    # Cap deflection at 100× the limit so softplus stays finite (<= ~100).
    deflection = min(
        _finite_or_default(state.get("deflection", 0.0) or 0.0, _max_d * 10.0),
        _max_d * 100.0,
    )
    return (
        softplus(target_fos - fos_b)
        + softplus(target_fos - fos_y)
        + softplus((deflection / _max_d) - 1.0)
    )


def compute_potential(
    state: dict,
    *,
    initial_mass: float,
    alpha: float = 5.0,
    target_fos: float = 1.5,
    max_deflection: float = 0.01,
) -> float:
    mass = _finite_or_default(state.get("mass", 0.0) or 0.0, 0.0)
    initial_mass = _finite_or_default(initial_mass, 0.0)
    if mass <= 0.0 or initial_mass <= 0.0:
        return -alpha * violation_score(
            state,
            target_fos=target_fos,
            max_deflection=max_deflection,
        )
    mass_utility = math.log(max(initial_mass, 1e-6) / max(mass, 1e-6))
    return mass_utility - alpha * violation_score(
        state,
        target_fos=target_fos,
        max_deflection=max_deflection,
    )


def compute_step_reward(
    prev_state: dict,
    next_state: dict,
    *,
    initial_mass: float,
    gamma: float = 0.99,
    alpha: float = 5.0,
    target_fos: float = 1.5,
    max_deflection: float = 0.01,
) -> float:
    prev_potential = compute_potential(
        prev_state,
        initial_mass=initial_mass,
        alpha=alpha,
        target_fos=target_fos,
        max_deflection=max_deflection,
    )
    next_potential = compute_potential(
        next_state,
        initial_mass=initial_mass,
        alpha=alpha,
        target_fos=target_fos,
        max_deflection=max_deflection,
    )
    return gamma * next_potential - prev_potential
