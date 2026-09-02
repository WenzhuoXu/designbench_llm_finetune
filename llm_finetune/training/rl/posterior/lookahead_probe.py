"""Measuring whether a policy has internalised the lookahead.

The training loop has been logging a quantity called rho since turn 0. It is
``argmax_i Phi_H(i) == argmax_i reward(i)`` over a GRPO group
(``grpo_trainer.py:345``), which is an algebraic identity when the reward is a
monotone function of Phi_H and a 1/K coin flip otherwise. It cannot rise, and it
never measured what the design doc's section 3.6 defines: agreement between the
policy's choice at a state and the lookahead's choice at that same state.

This module measures the right thing, and two useful things the doc did not ask
for. At a visited state s the probe enumerates candidate actions, simulates each
one step, scores the successors with the design program's potential, and compares
that ranking against the action the policy actually took:

    rho          1 if the policy's action is potential-optimal among the candidates
    regret       Phi(best candidate) - Phi(policy's successor), in potential units
    rank_frac    fraction of candidates strictly better than the policy's choice
    margin       Phi(best) - Phi(second best)          -- Delta_rank in section 3.3
    sigma        spread of candidate potentials        -- sigma_G in section 3.3
    reliable     margin > 2 * gamma^(D+1) * sigma      -- the ranking-reliability gate

rho is binary and saturating; **regret** is the quantity that actually tracks
internalisation, because it keeps reporting how far off the policy is after rho
has stopped moving. And ``reliable`` is the decision variable for *when a search
budget is worth spending at all*: where the top two candidates are separated by
less than the noise, the lookahead cannot rank them, so paying for it buys
nothing.

Domain-agnostic: the caller injects an action enumerator and a one-step
transition, and supplies the DesignProgram. Nothing here knows about trusses.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from llm_finetune.training.rl.posterior.potential import (
    DesignProgram,
    compute_potential_v2,
)

log = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    n_candidates: int
    rho: float
    regret: float
    rank_frac: float
    margin: float
    sigma: float
    reliable: bool
    policy_phi: float
    best_phi: float
    baseline_phi: float = 0.0
    best_action: str = ""
    extras: dict = field(default_factory=dict)

    def as_metrics(self, prefix: str = "lookahead") -> dict[str, float]:
        return {
            f"{prefix}/rho": float(self.rho),
            f"{prefix}/regret": float(self.regret),
            f"{prefix}/rank_frac": float(self.rank_frac),
            f"{prefix}/margin": float(self.margin),
            f"{prefix}/sigma": float(self.sigma),
            f"{prefix}/reliable": float(bool(self.reliable)),
            f"{prefix}/n_candidates": float(self.n_candidates),
        }


def probe_state(
    *,
    program: DesignProgram,
    candidates: Sequence[str],
    transition: Callable[[str], Optional[dict]],
    policy_next_state: Optional[dict],
    alpha: float = 5.0,
    tau: float = 0.05,
    gamma: float = 0.99,
    depth: int = 1,
    objective_weight: float = 1.0,
) -> Optional[ProbeResult]:
    """Score ``candidates`` one step ahead and locate the policy's action in that order.

    ``transition(action)`` returns the successor state dict, or None if the action
    is invalid. ``policy_next_state`` is the successor the policy actually reached
    (already computed by the rollout, so it costs nothing extra).
    Returns None when fewer than two candidates could be evaluated.
    """
    phi_kwargs = dict(alpha=alpha, tau=tau, objective_weight=objective_weight)

    scored: list[tuple[str, float]] = []
    for action in candidates:
        try:
            nxt = transition(action)
        except Exception:  # a broken candidate must not kill a training step
            continue
        if not nxt:
            continue
        scored.append((action, compute_potential_v2(nxt, program, **phi_kwargs)))

    if len(scored) < 2:
        return None

    scored.sort(key=lambda item: item[1], reverse=True)
    values = [v for _, v in scored]
    best_action, best_phi = scored[0]
    second_phi = values[1]
    mean = sum(values) / len(values)
    sigma = math.sqrt(sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1))

    if policy_next_state:
        policy_phi = compute_potential_v2(policy_next_state, program, **phi_kwargs)
    else:
        # No successor (unparsed or failed action): the policy forfeited the step,
        # which is a real and costly outcome, so score it at the worst candidate.
        policy_phi = values[-1]

    n_better = sum(1 for v in values if v > policy_phi + 1e-12)
    margin = best_phi - second_phi
    # Section 3.3: a depth-D lookahead can only be trusted to rank two actions
    # apart when their value gap exceeds twice the continuation-noise band.
    reliable = margin > 2.0 * (gamma ** (depth + 1)) * sigma

    return ProbeResult(
        n_candidates=len(scored),
        rho=1.0 if n_better == 0 else 0.0,
        regret=max(0.0, best_phi - policy_phi),
        rank_frac=n_better / len(values),
        margin=margin,
        sigma=sigma,
        reliable=bool(reliable),
        policy_phi=policy_phi,
        best_phi=best_phi,
        baseline_phi=mean,
        best_action=best_action,
    )


def counterfactual_advantage(result: "ProbeResult") -> float:
    """A^(1)(s, a) = Phi(s') - E_c[Phi(f(s, c))] -- the estimator the framework claims.

    The baseline is the mean potential of the candidate successors at the SAME
    state, so it depends on s but not on the action taken. Any such baseline is
    unbiased for the policy gradient, and this one is strictly more informative
    than the shaping term gamma*Phi(s') - Phi(s): it measures the action against
    what was ACHIEVABLE at that state rather than against the state itself, which
    normalises away how easy or hard the state was.

    This is also the reason it survives GRPO. TRL runs with scale_rewards='group',
    so any reward that is an affine function of a single per-trajectory quantity
    (such as Phi(s_H)) is collapsed by within-group mean-centring -- which is
    exactly what happened to the previous "tree advantage",
    gamma*Phi(s_H^k) - mean_j Phi(s_H^j), whose baseline is a group constant.
    Here the baseline varies with the rollout AND the step, so the summed
    advantage is not an affine image of Phi(s_H) and carries information no other
    term in the reward contains.
    """
    return result.policy_phi - result.baseline_phi


def aggregate(results: Iterable[Optional[ProbeResult]], prefix: str = "lookahead") -> dict[str, float]:
    """Mean of each probe field over the states measured this batch."""
    rows = [r for r in results if r is not None]
    if not rows:
        return {}
    n = len(rows)
    out = {
        f"{prefix}/rho": sum(r.rho for r in rows) / n,
        f"{prefix}/regret": sum(r.regret for r in rows) / n,
        f"{prefix}/rank_frac": sum(r.rank_frac for r in rows) / n,
        f"{prefix}/margin": sum(r.margin for r in rows) / n,
        f"{prefix}/sigma": sum(r.sigma for r in rows) / n,
        f"{prefix}/reliable_frac": sum(1.0 for r in rows if r.reliable) / n,
        f"{prefix}/n_states": float(n),
        f"{prefix}/n_candidates": sum(r.n_candidates for r in rows) / n,
        f"{prefix}/advantage": sum(counterfactual_advantage(r) for r in rows) / n,
    }
    return out


# ── truss adapter (the only domain-specific code in this file) ───────────────

def truss_candidate_actions(
    truss,
    bounds: dict,
    *,
    factors: Sequence[float] = (0.85, 1.15, 1.35, 1.7),
    params: Sequence[str] = ("r", "t"),
    max_candidates: int = 64,
    rng=None,
) -> list[str]:
    """Sizing actions inside the problem's own parameter bounds, subsampled.

    Subsampling keeps the probe's simulator cost bounded: the estimate of rho and
    regret is over a random subset of the action space, which is unbiased for the
    ranking statistics as long as the subset is drawn the same way every step.
    """
    actions: list[str] = []
    for mid, member in enumerate(getattr(truss, "members", []) or []):
        shape_params = getattr(getattr(member, "shape", None), "_params", None) or {}
        for name in params:
            cur = shape_params.get(name)
            if cur is None:
                continue
            lo, hi = bounds.get(name, (0.0, math.inf))
            for f in factors:
                nxt = float(cur) * f
                if nxt < lo or nxt > hi:
                    continue
                actions.append(f"SCALE_PARAM({mid}, {name}, {f})")
    if rng is not None and len(actions) > max_candidates:
        return rng.sample(actions, max_candidates)
    return actions[:max_candidates]


def truss_param_bounds(problem_spec: dict) -> dict:
    opt = (problem_spec or {}).get("optimization") or {}
    shape_params = opt.get("shape_params") or {}
    out = {}
    for name, rng in shape_params.items():
        if isinstance(rng, dict) and "min" in rng and "max" in rng:
            out[name] = (float(rng["min"]), float(rng["max"]))
    return out
