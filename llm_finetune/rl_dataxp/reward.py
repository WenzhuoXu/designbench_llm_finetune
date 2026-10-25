"""Episode reward: the evaluation outcome, nothing else.

R = 1 + speed_bonus * (horizon - t) / horizon   if the design first becomes feasible at turn t <= horizon
R = 0                                          otherwise

Feasibility is the simulator's verdict as the harness records it (``first_feasible_turn``),
the same field feasible@12 is computed from. No Phi term (it telescopes to a group constant
under GRPO's mean-centring; ledger Turns 10 and 22) and no format term (an unparsable reply
already spends a turn).
"""
from __future__ import annotations

from typing import Optional


def invalid_reason(ep: dict) -> Optional[str]:
    """Why an episode cannot be scored, or None. Such an episode is dropped from its group:
    a crash or a server error is the infrastructure's failure, not the policy's, and a design
    that is feasible at the start never asks the model anything."""
    if ep.get("crashed"):
        return "crashed"
    if ep.get("rl_error"):
        return "rl_error:%s" % ep["rl_error"]
    if ep.get("feasible_start"):
        return "feasible_start"
    if any(t.get("fail") == "error" for t in ep.get("turns", [])):
        return "server_error"
    return None


def episode_reward(ep: dict, horizon: int = 12, speed_bonus: float = 0.2) -> Optional[float]:
    """The reward of one harness episode, or None when ``invalid_reason`` is not None."""
    if invalid_reason(ep) is not None:
        return None
    t = ep.get("first_feasible_turn")
    if t is None or t > horizon:
        return 0.0
    return 1.0 + speed_bonus * (horizon - t) / horizon
