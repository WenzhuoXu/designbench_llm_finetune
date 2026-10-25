"""From a step's rollout records to training samples.

A record is one episode: the harness's episode dict plus ``calls``, the token-level record of
every model request the episode made (``harness.Recorder``). The harness asks the model once
per turn with a fresh system + user prompt, so each call is one sample:
(prompt_ids, completion_ids) exactly as vLLM saw and produced them, weighted by the advantage
of its episode. Episodes are grouped by problem (``group``): the G rollouts of one problem at
one step.
"""
from __future__ import annotations

import collections
import math
from typing import Optional

from .reward import episode_reward, invalid_reason

# harness fail codes that mean "the reply yielded no move" (dataxp_policy_eval_v3.parse_move*)
PARSE_FAILS = {"no_action", "unparsed", "think_unclosed", "bad_number"}


def group_advantages(rewards: list[float], scale: str = "group", eps: float = 1e-6) -> Optional[list[float]]:
    """GRPO advantages for one group, or None when the group carries no signal (fewer than two
    episodes, or every reward equal). scale='group' divides by the group's standard deviation
    (unbiased, as TRL does); 'none' only centres."""
    n = len(rewards)
    if n < 2:
        return None
    mean = sum(rewards) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / (n - 1))
    if std < eps:
        return None
    if scale == "group":
        return [(r - mean) / (std + eps) for r in rewards]
    if scale == "none":
        return [r - mean for r in rewards]
    raise ValueError("scale must be 'group' or 'none', got %r" % scale)


def build_step_samples(records: list[dict], horizon: int, speed_bonus: float,
                       scale: str = "group") -> tuple[list[dict], dict]:
    """(samples, stats). Every sample is a dict with prompt_ids, completion_ids, advantage,
    vllm_logprobs (or None), key, turn, variant, finish. Samples come in a deterministic order
    (records sorted by key, calls in turn order), so every rank builds the same list."""
    by_group = collections.defaultdict(list)
    for r in sorted(records, key=lambda r: r["key"]):
        by_group[r["group"]].append(r)

    st = collections.Counter()
    invalid = collections.Counter()
    fails = collections.Counter()
    finish = collections.Counter()
    per_var = collections.defaultdict(lambda: collections.Counter())
    rewards_all = []
    samples = []
    for g in sorted(by_group):
        recs = by_group[g]
        st["groups"] += 1
        valid, rewards = [], []
        for r in recs:
            ep = r["episode"]
            why = invalid_reason(ep)
            if why is None and r.get("rl_error"):
                why = "rl_error:%s" % r["rl_error"]
            if why is not None:
                invalid[why] += 1
                continue
            rew = episode_reward(ep, horizon, speed_bonus)
            valid.append(r)
            rewards.append(rew)
            v = per_var[r["variant"]]
            v["episodes"] += 1
            v["feasible"] += int(rew > 0)
            v["reward_x1e6"] += int(round(rew * 1e6))
            for t in ep.get("turns", []):
                v["turns"] += 1
                f = t.get("fail")
                if f:
                    fails[f.split(":")[0]] += 1
                    if f in PARSE_FAILS:
                        v["parse_fail"] += 1
                fin = t.get("finish")
                if fin:
                    finish[fin] += 1
                    if fin == "length":
                        v["at_cap"] += 1
        rewards_all += rewards
        adv = group_advantages(rewards, scale)
        if adv is None:
            st["groups_zero_var" if len(rewards) >= 2 else "groups_too_few"] += 1
            continue
        st["groups_used"] += 1
        for r, a in zip(valid, adv):
            calls = r.get("calls") or []
            n_turns = len(r["episode"].get("turns", []))
            if len(calls) != n_turns:
                raise RuntimeError("%s: %d model calls for %d turns; the loop guard must be off for RL"
                                   % (r["key"], len(calls), n_turns))
            for k, c in enumerate(calls):
                if not c.get("token_ids"):
                    continue                     # a call that produced no token (cannot train on it)
                samples.append({"prompt_ids": c["prompt_token_ids"], "completion_ids": c["token_ids"],
                                "advantage": float(a), "vllm_logprobs": c.get("logprobs"),
                                "key": r["key"], "turn": k, "variant": r["variant"],
                                "finish": c.get("finish")})
    n_tok = sum(len(s["completion_ids"]) for s in samples)
    stats = {
        "groups": st["groups"], "groups_used": st["groups_used"],
        "groups_zero_var": st["groups_zero_var"], "groups_too_few": st["groups_too_few"],
        "frac_groups_zero_var": st["groups_zero_var"] / max(1, st["groups"]),
        "episodes_invalid": dict(invalid),
        "reward_mean": sum(rewards_all) / len(rewards_all) if rewards_all else None,
        "feasible_rate": (sum(1 for x in rewards_all if x > 0) / len(rewards_all)) if rewards_all else None,
        "samples": len(samples), "completion_tokens": n_tok,
        "prompt_tokens": sum(len(s["prompt_ids"]) for s in samples),
        "fails": dict(fails), "finish": dict(finish),
        "per_variant": {
            v: {"episodes": c["episodes"], "feasible_rate": c["feasible"] / max(1, c["episodes"]),
                "reward_mean": c["reward_x1e6"] / 1e6 / max(1, c["episodes"]),
                "parse_rate": 1 - c["parse_fail"] / max(1, c["turns"]),
                "cap_share": c["at_cap"] / max(1, c["turns"]), "turns": c["turns"]}
            for v, c in sorted(per_var.items())},
    }
    turns = sum(c["turns"] for c in per_var.values())
    stats["parse_rate"] = 1 - sum(c["parse_fail"] for c in per_var.values()) / max(1, turns)
    return samples, stats


def shard_samples(samples: list[dict], world: int, rank: int) -> list[dict]:
    """This rank's share, balanced by tokens (longest first, to the least-loaded rank).
    Deterministic, so the ranks agree without communicating."""
    order = sorted(range(len(samples)),
                   key=lambda i: (-(len(samples[i]["prompt_ids"]) + len(samples[i]["completion_ids"])), i))
    load = [0] * world
    mine = []
    for i in order:
        r = min(range(world), key=lambda k: (load[k], k))
        load[r] += len(samples[i]["prompt_ids"]) + len(samples[i]["completion_ids"])
        if r == rank:
            mine.append(samples[i])
    return mine
