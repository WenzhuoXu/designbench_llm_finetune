"""dataxp GRPO: the pure-Python pieces (no torch, no harness)."""
import json
import math

import pytest

from llm_finetune.rl_dataxp.reward import episode_reward, invalid_reason
from llm_finetune.rl_dataxp.samples import build_step_samples, group_advantages, shard_samples
from llm_finetune.rl_dataxp.trainer import (DEFAULTS, ProblemSampler, apply_overrides, check_guards,
                                            fingerprint, load_config, step_episodes)


# ---------------------------------------------------------------- reward

def ep(t=None, **kw):
    d = {"first_feasible_turn": t, "feasible_start": False, "turns": [{"fail": None}] * (t or 12)}
    d.update(kw)
    return d


def test_reward_values():
    assert episode_reward(ep(3)) == pytest.approx(1 + 0.2 * 9 / 12)
    assert episode_reward(ep(12)) == pytest.approx(1.0)
    assert episode_reward(ep(None)) == 0.0
    assert episode_reward(ep(13)) == 0.0          # feasible after the horizon counts as unsolved
    assert episode_reward(ep(1), horizon=12, speed_bonus=0.0) == 1.0


def test_reward_late_success_beats_failure():
    assert min(episode_reward(ep(t)) for t in range(1, 13)) > episode_reward(ep(None))


def test_invalid_episodes():
    assert invalid_reason(ep(3, crashed=True)) == "crashed"
    assert invalid_reason(ep(3, feasible_start=True)) == "feasible_start"
    assert invalid_reason({"turns": [{"fail": "error"}]}) == "server_error"
    assert episode_reward(ep(3, crashed=True)) is None


# ---------------------------------------------------------------- advantages

def test_group_advantages():
    a = group_advantages([1.0, 0.0, 0.0, 1.0])
    assert sum(a) == pytest.approx(0, abs=1e-9)
    assert a[0] > 0 > a[1]
    std = math.sqrt(sum(x * x for x in a) / 3)
    assert std == pytest.approx(1.0, rel=1e-4)
    assert group_advantages([1.0, 1.0, 1.0]) is None          # no signal
    assert group_advantages([1.0]) is None
    assert group_advantages([1.0, 0.0], scale="none") == [0.5, -0.5]


def rec(key, group, t, n_turns, variant="circuit2", fail=None, **kw):
    turns = [{"fail": fail, "finish": "stop"} for _ in range(n_turns)]
    calls = [{"prompt_token_ids": [1, 2, 3, k], "token_ids": [7, 8, k], "logprobs": [-0.1, -0.2, -0.3],
              "finish": "stop"} for k in range(n_turns)]
    r = {"key": key, "group": group, "variant": variant,
         "episode": {"first_feasible_turn": t, "feasible_start": False, "turns": turns}, "calls": calls}
    r.update(kw)
    return r


def test_build_step_samples():
    recs = [rec("g0/m0", "g0", 2, 2), rec("g0/m1", "g0", None, 12),
            rec("g1/m0", "g1", None, 12), rec("g1/m1", "g1", None, 12),      # zero variance
            rec("g2/m0", "g2", 1, 1), rec("g2/m1", "g2", 5, 5, crashed=False),
            rec("g2/m2", "g2", None, 12)]
    recs[5]["episode"]["crashed"] = True                                       # dropped, group keeps 2
    samples, st = build_step_samples(recs, horizon=12, speed_bonus=0.2)
    assert st["groups"] == 3 and st["groups_used"] == 2 and st["groups_zero_var"] == 1
    assert st["episodes_invalid"] == {"crashed": 1}
    keys = {s["key"] for s in samples}
    assert keys == {"g0/m0", "g0/m1", "g2/m0", "g2/m2"}
    assert len(samples) == 2 + 12 + 1 + 12
    pos = [s for s in samples if s["key"] == "g0/m0"]
    assert all(s["advantage"] > 0 for s in pos) and [s["turn"] for s in pos] == [0, 1]
    assert pos[0]["completion_ids"] == [7, 8, 0]


def test_samples_require_one_call_per_turn():
    r = rec("g0/m0", "g0", 2, 2)
    r["calls"] = r["calls"][:1]
    with pytest.raises(RuntimeError, match="loop guard"):
        build_step_samples([r, rec("g0/m1", "g0", None, 3)], 12, 0.2)


def test_parse_rate_and_per_variant():
    recs = [rec("g0/m0", "g0", None, 4, fail="no_action"), rec("g0/m1", "g0", 2, 2)]
    _, st = build_step_samples(recs, 12, 0.2)
    assert st["parse_rate"] == pytest.approx(2 / 6)
    assert st["per_variant"]["circuit2"]["feasible_rate"] == 0.5


def test_shard_samples_partition_and_balance():
    samples = [{"prompt_ids": [0] * n, "completion_ids": [0] * 3} for n in (100, 90, 50, 40, 10, 5, 5)]
    parts = [shard_samples(samples, 3, r) for r in range(3)]
    assert sum(len(p) for p in parts) == len(samples)
    ids = sorted(id(s) for p in parts for s in p)
    assert ids == sorted(id(s) for s in samples)
    loads = [sum(len(s["prompt_ids"]) for s in p) for p in parts]
    assert max(loads) - min(loads) <= 100


# ---------------------------------------------------------------- sampler, episodes

def problems(nv=3, n=5):
    return {"v%d" % i: [{"problem_id": "v%d_%03d" % (i, k)} for k in range(n)] for i in range(nv)}


def test_sampler_deterministic_and_balanced():
    s1, s2 = ProblemSampler(problems(), 8, seed=1), ProblemSampler(problems(), 8, seed=1)
    for step in (0, 1, 7):
        assert s1.draw(step) == s2.draw(step)
    counts = {}
    for step in range(30):
        for v, _ in s1.draw(step):
            counts[v] = counts.get(v, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1
    assert ProblemSampler(problems(), 8, seed=2).draw(0) != s1.draw(0)


def test_sampler_covers_pool_before_repeating():
    s = ProblemSampler(problems(nv=1, n=10), 5, seed=0)
    first = [p["problem_id"] for _, p in s.draw(0) + s.draw(1)]
    assert len(set(first)) == 10


def test_step_episodes_groups():
    s = ProblemSampler(problems(), 4, seed=0)
    eps = step_episodes(s, 3, group_size=8, seed=0, n_servers=2)
    assert len(eps) == 32
    groups = {}
    for e in eps:
        groups.setdefault(e["group"], []).append(e)
    assert all(len(g) == 8 for g in groups.values())
    assert all(len({e["server"] for e in g}) == 1 for g in groups.values())     # a group stays on one server
    assert len({e["seed"] for e in eps}) == 32


# ---------------------------------------------------------------- guards, config

def hist(parse, kl):
    return {i: {"step": i, "parse_rate": p, "kl": k} for i, (p, k) in enumerate(zip(parse, kl))}


def test_guards():
    g = dict(DEFAULTS["guards"], stall_steps=12)
    assert check_guards(hist([0.9] * 10, [0.01 * i for i in range(10)]), g) is None
    assert "parse rate" in check_guards(hist([0.9] * 5 + [0.7] * 5, [0.01 * i for i in range(10)]), g)
    assert "KL" in check_guards(hist([0.9] * 10, [0.0] * 5 + [0.2] * 5), g)
    assert "not moved" in check_guards(hist([0.9] * 12, [0.05] * 12), g)
    assert check_guards(hist([0.9] * 12, [0.05] * 12), dict(g, enabled=False)) is None


def test_config_overrides_and_fingerprint(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("grpo:\n  lr: 2.0e-5\nrollout:\n  servers: []\n")
    cfg = load_config(str(p), ["grpo.kl_coef=0.01", "rollout.servers=[\"http://a/v1\"]", "run_dir=/x"])
    assert cfg["grpo"]["lr"] == 2e-5 and cfg["grpo"]["kl_coef"] == 0.01
    assert cfg["rollout"]["servers"] == ["http://a/v1"] and cfg["grpo"]["steps"] == 150
    f1 = fingerprint(cfg, "abc")
    assert fingerprint(apply_overrides(json.loads(json.dumps(cfg)), ["grpo.steps=300"]), "abc") == f1
    assert fingerprint(apply_overrides(json.loads(json.dumps(cfg)), ["grpo.lr=1e-4"]), "abc") != f1
    assert fingerprint(cfg, "abd") != f1


def test_crash_fraction_aborts():
    from llm_finetune.rl_dataxp.trainer import check_crashes
    recs = [{"episode": {"crashed": i < 2}} for i in range(8)]
    check_crashes(recs, 0.25, 0, "x")                        # 2 of 8: allowed
    recs[2]["episode"]["crashed"] = True
    with pytest.raises(RuntimeError, match="3 of 8 episodes crashed"):
        check_crashes(recs, 0.25, 0, "x")


def test_subprocess_rollout_checks_paths(tmp_path):
    from llm_finetune.rl_dataxp.trainer import SubprocessRollout
    script = tmp_path / "h.py"
    script.write_text("")
    (tmp_path / "code" / "dataxp").mkdir(parents=True)
    cfg = load_config(None, ["rollout.servers=[\"http://x/v1\"]", "rollout.python=%s" % script.as_posix(),
                             "harness.script=%s" % script.as_posix()])
    cfg["harness"]["argv"] = ["--code", str(tmp_path / "code")]
    SubprocessRollout(cfg, str(tmp_path / "run"))
    for argv, msg in ((["--code", "CODE_V2"], "--code"), (["--code", str(tmp_path / "code"), "--turns", "5"], "--turns")):
        cfg["harness"]["argv"] = argv
        with pytest.raises(SystemExit, match=msg):
            SubprocessRollout(cfg, str(tmp_path / "run"))


def test_turn_subsampling_is_unbiased_and_deterministic():
    from llm_finetune.rl_dataxp.samples import subsample_turns
    assert subsample_turns(5, None, "k", 0, 0) == ([0, 1, 2, 3, 4], 1.0)
    assert subsample_turns(3, 4, "k", 0, 0) == ([0, 1, 2], 1.0)
    pick, w = subsample_turns(12, 4, "g/m0", 7, 3)
    assert len(pick) == 4 and w == 3.0 and pick == subsample_turns(12, 4, "g/m0", 7, 3)[0]
    # E[sum over kept turns of w * x_t] = sum_t x_t
    x = [float(t * t + 1) for t in range(12)]
    est = []
    for s in range(4000):
        p, w = subsample_turns(12, 4, "g/m%d" % s, 0, 0)
        est.append(sum(w * x[t] for t in p))
    assert sum(est) / len(est) == pytest.approx(sum(x), rel=0.02)


def test_build_step_samples_with_max_turns():
    recs = [rec("g0/m0", "g0", None, 9), rec("g0/m1", "g0", 2, 2)]
    samples, st = build_step_samples(recs, 12, 0.2, max_turns=3, seed=1, step=0)
    a = [s for s in samples if s["key"] == "g0/m0"]
    b = [s for s in samples if s["key"] == "g0/m1"]
    assert len(a) == 3 and all(s["weight"] == 3.0 for s in a)
    assert len(b) == 2 and all(s["weight"] == 1.0 for s in b)
    assert st["samples_full"] == 11 and st["samples"] == 5
