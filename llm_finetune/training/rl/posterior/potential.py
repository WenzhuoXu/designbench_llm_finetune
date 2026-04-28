"""Potential-based reward shaping for posterior evaluation."""

from __future__ import annotations

import math


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
    fos_b = float(state.get("fos_buckling", 0.0) or 0.0)
    fos_y = float(state.get("fos_yielding", 0.0) or 0.0)
    deflection = float(state.get("deflection", 0.0) or 0.0)
    return (
        softplus(target_fos - fos_b)
        + softplus(target_fos - fos_y)
        + softplus((deflection / max(max_deflection, 1e-6)) - 1.0)
    )


def compute_potential(
    state: dict,
    *,
    initial_mass: float,
    alpha: float = 5.0,
    target_fos: float = 1.5,
    max_deflection: float = 0.01,
) -> float:
    mass = float(state.get("mass", 0.0) or 0.0)
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
