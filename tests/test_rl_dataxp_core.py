"""dataxp GRPO: the pure-Python pieces (no torch, no harness)."""
import argparse
import errno
import json
import math
import os
import pickle
import subprocess
import sys
import time

import pytest

from llm_finetune.rl_dataxp import diskio
from llm_finetune.rl_dataxp import rollout as RO
from llm_finetune.rl_dataxp.diskio import disk_full
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


def test_no_completion_cap_anywhere():
    """No reply cap is invented: the default request carries max_tokens null, and no config passes
    the harness a --max-tokens."""
    import os
    import yaml
    assert DEFAULTS["rollout"]["inject"]["max_tokens"] is None
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("dataxp_grpo_p2_datav4.yaml", "dataxp_grpo_p2_datav4_dx38mtR.yaml", "dataxp_grpo_p1_v2.yaml",
                 "dataxp_grpo_p2_bridges2.yaml"):
        cfg = yaml.safe_load(open(os.path.join(here, "configs", "rl", name), encoding="utf-8"))
        assert "--max-tokens" not in [str(x) for x in cfg["harness"]["argv"]], name
        assert cfg["rollout"]["inject"]["max_tokens"] is None, name


# ---------------------------------------------------------------- a full file system

class Clock:
    """Stands in for the time module inside diskio: sleep() moves the clock, nothing waits."""

    def __init__(self):
        self.t, self.sleeps = 0.0, 0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s
        self.sleeps += 1


def enospc():
    return OSError(errno.ENOSPC, "No space left on device")


def test_disk_full_is_recognised():
    assert disk_full(enospc()) and disk_full(OSError(errno.EDQUOT, "Disk quota exceeded"))
    assert not disk_full(OSError(errno.EACCES, "Disk quota exceeded"))         # the errno decides when there is one
    # what safetensors raised in job 167222: not an OSError, only the text is left
    assert disk_full(Exception("Error while serializing: I/O error: Disk quota exceeded (os error 122)"))
    assert disk_full(RuntimeError("write failed: No space left on device (os error 28)"))
    assert not disk_full(RuntimeError("PytorchStreamWriter failed writing file data/0: file write failed"))
    assert not disk_full(KeyError("optimizer"))


def test_retry_when_full_waits_then_gives_up(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(diskio, "time", clock)
    logs, calls = [], []

    def fn(fail):
        calls.append(1)
        if len(calls) <= fail:
            raise enospc()
        return "done"

    assert diskio.retry_when_full(lambda: fn(3), "x", 10, logs.append) == "done"
    assert clock.sleeps == 3 and len(calls) == 4
    assert "file system full while writing x" in logs[0] and "x written after waiting 3.0 min" in logs[-1]
    # nothing is logged, and nothing waits, when the write just works
    logs.clear(); calls.clear(); clock.sleeps = 0
    assert diskio.retry_when_full(lambda: fn(0), "x", 10, logs.append) == "done" and not logs and not clock.sleeps
    # the wait is bounded: 5 minutes = the first attempt and 5 more, one a minute
    calls.clear(); clock.sleeps = 0
    with pytest.raises(OSError):
        diskio.retry_when_full(lambda: fn(10 ** 9), "x", 5, logs.append)
    assert clock.sleeps == 5 and len(calls) == 6
    # 0 = fail at once; any other error is never retried
    calls.clear(); clock.sleeps = 0
    with pytest.raises(OSError):
        diskio.retry_when_full(lambda: fn(1), "x", 0, logs.append)
    assert len(calls) == 1 and clock.sleeps == 0

    def other():
        calls.append(1)
        raise ValueError("not the disk")

    calls.clear()
    with pytest.raises(ValueError):
        diskio.retry_when_full(other, "x", 10, logs.append)
    assert len(calls) == 1 and clock.sleeps == 0


def test_write_text_keeps_the_old_file_when_the_write_fails(tmp_path, monkeypatch):
    p = str(tmp_path / "state.json")
    diskio.write_text(p, "old")
    real = os.fsync
    with monkeypatch.context() as mp:
        mp.setattr(diskio.os, "fsync", lambda fd: (_ for _ in ()).throw(enospc()))
        with pytest.raises(OSError):
            diskio.write_text(p, "new, and longer")
    assert os.fsync is real and open(p).read() == "old"
    diskio.write_text(p, "new")
    assert open(p).read() == "new" and sorted(os.listdir(tmp_path)) == ["state.json"]


def test_write_errors_keeps_the_os_error():
    class Full:
        def write(self, b):
            raise enospc()

        def tell(self):
            return 7

    w = diskio.WriteErrors(Full())
    with pytest.raises(OSError):
        w.write(b"x")
    assert disk_full(w.error) and w.tell() == 7


def test_outfile_keeps_no_half_and_no_double_record(tmp_path, monkeypatch):
    p = str(tmp_path / "out.jsonl")
    with open(p, "w") as f:
        f.write(json.dumps({"key": "a"}) + "\n" + '{"key": "b", "calls": [1, 2')       # a job killed mid-write
    clock = Clock()
    monkeypatch.setattr(diskio, "time", clock)
    real, left = os.fsync, [2]

    def fsync(fd):                                   # the line reached the file, the flush to disk failed
        if left[0]:
            left[0] -= 1
            raise enospc()
        return real(fd)

    monkeypatch.setattr(RO.os, "fsync", fsync)
    with RO.OutFile(p, wait_min=10) as fo:
        fo.write({"key": "c"})
        fo.write({"key": "d"})
    assert [json.loads(l)["key"] for l in open(p)] == ["a", "c", "d"]
    assert clock.sleeps == 2 and RO.read_done(p) == {"a", "c", "d"}
    # no wait allowed: the error comes out, and what is in the file is still whole records
    left[0] = 1
    with RO.OutFile(p, wait_min=0) as fo:
        with pytest.raises(OSError):
            fo.write({"key": "e"})
        fo.write({"key": "f"})
    assert [json.loads(l)["key"] for l in open(p)] == ["a", "c", "d", "f"]


def test_outfile_waits_when_the_file_cannot_be_created(tmp_path, monkeypatch):
    """The first record of a step creates the file; on a full pool that is the call that fails
    (job 167223), and the repeat must not trip over the file not being there."""
    p = str(tmp_path / "step_00007.jsonl")
    clock = Clock()
    monkeypatch.setattr(diskio, "time", clock)
    real, left = open, [2]

    def full_open(path, mode="r", *a, **k):
        if "a" in mode and left[0]:
            left[0] -= 1
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return real(path, mode, *a, **k)

    monkeypatch.setattr(RO, "open", full_open, raising=False)
    with RO.OutFile(p, wait_min=10) as fo:
        assert clock.sleeps == 2 and os.path.getsize(p) == 0
        fo.write({"key": "a"})
        fo.write({"key": "b"})
    assert [json.loads(l)["key"] for l in open(p)] == ["a", "b"]
    # the same when the file goes missing between two records and cannot be made again at once
    os.remove(p)
    left[0], clock.sleeps = 1, 0
    with RO.OutFile(p, wait_min=10) as fo:
        fo.f.close()
        fo.f = None
        os.remove(p)
        left[0] = 1
        fo.write({"key": "c"})
    assert clock.sleeps == 2 and [json.loads(l)["key"] for l in open(p)] == ["c"]


def test_outfile_exists_after_a_run_without_records(tmp_path):
    p = str(tmp_path / "out.jsonl")
    with RO.OutFile(p):
        pass
    assert os.path.getsize(p) == 0 and RO.read_done(p) == set()
    assert RO.summarize(p) == {} and json.load(open(p + ".summary.json")) == {}


def test_a_child_that_hit_the_full_disk_is_run_again_not_recorded_as_crashed(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(diskio, "time", clock)
    full = (RO.DISK_FULL_RC, None, "Traceback ...\nOSError: [Errno 28] No space left on device: '/pool/tmp/tmpk3'")
    answers = []
    monkeypatch.setattr(RO, "run_child", lambda payload, env: answers.pop(0))
    # two failures on the full pool, then the episode runs: its result, no crash
    answers[:] = [full, full, (0, {"episode": {}}, "")]
    assert RO.run_episode({}, {}, "s7/g0/m1", wait_min=10) == ({"episode": {}}, None)
    assert clock.sleeps == 2 and not answers
    # any other failure is a crash after two attempts, as before, without waiting
    clock.sleeps = 0
    answers[:] = [(1, None, "Segmentation fault"), (139, None, "ValueError: singular matrix")]
    assert RO.run_episode({}, {}, "s7/g0/m2", wait_min=10) == (None, "ValueError: singular matrix")
    assert clock.sleeps == 0
    # the exit status decides, not the text: a warning that names a full device next to a crash
    noisy = ("UserWarning: [Errno 28] No space left on device.  joblib will operate in serial mode\n"
             "Fatal Python error: Segmentation fault")
    answers[:] = [(-11, None, noisy), (-11, None, noisy)]
    assert RO.run_episode({}, {}, "s7/g0/m4", wait_min=10) == (None, noisy)
    assert clock.sleeps == 0 and not answers
    # past the wait the error is raised: no crash record is written for the episode
    answers[:] = [full] * 20
    with pytest.raises(OSError) as e:
        RO.run_episode({}, {}, "s7/g0/m3", wait_min=3)
    assert disk_full(e.value) and clock.sleeps == 3


def test_the_child_reports_a_full_disk_by_its_exit_status():
    code = "\n".join([
        "import errno, sys",
        "import llm_finetune.rl_dataxp.harness as H",
        "from llm_finetune.rl_dataxp import rollout",
        "def run_one(payload):",
        "    print('a simulator prints: No space left on device')",
        "    if payload['kind'] == 'full':",
        "        try:",
        "            raise OSError(errno.EDQUOT, 'Disk quota exceeded')",
        "        except OSError as e:",
        "            raise RuntimeError('simulator failed') from e",
        "    if payload['kind'] == 'bug':",
        "        raise ValueError('singular matrix')",
        "    return {'kind': payload['kind']}",
        "H.run_one = run_one",
        "sys.exit(rollout.main(['_one_episode']))"])

    def child(kind):
        env = dict(os.environ, PYTHONPATH=RO.REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        return subprocess.run([sys.executable, "-c", code], input=pickle.dumps({"kind": kind}),
                              capture_output=True, env=env, cwd=RO.REPO_ROOT)

    pr = child("full")                               # wrapped by the simulator: still a full disk
    assert pr.returncode == RO.DISK_FULL_RC and b"Disk quota exceeded" in pr.stderr
    pr = child("bug")                                # a crash, although its output names a full device
    assert pr.returncode == 1 and b"singular matrix" in pr.stderr and b"No space left" in pr.stderr
    pr = child("ok")
    assert pr.returncode == 0 and pickle.loads(pr.stdout) == {"kind": "ok"}


def test_outfile_starts_over_when_the_file_was_removed(tmp_path, monkeypatch):
    """OUT gone in mid-run (a wiped pool) and then a failed write: the next attempt must not cut
    the new, shorter file at the old length (that pads it with NUL bytes no resume can read)."""
    p = str(tmp_path / "out.jsonl")
    clock = Clock()
    monkeypatch.setattr(diskio, "time", clock)
    real, left = os.fsync, [0]

    def fsync(fd):
        if left[0]:
            left[0] -= 1
            raise enospc()
        return real(fd)

    monkeypatch.setattr(RO.os, "fsync", fsync)
    with RO.OutFile(p, wait_min=10) as fo:
        fo.write({"key": "a", "calls": list(range(50))})
        fo.write({"key": "b", "calls": list(range(50))})
        fo.close()                                   # Windows does not remove an open file
        os.remove(p)
        left[0] = 2
        fo.write({"key": "c"})
        fo.write({"key": "d"})
    assert b"\0" not in open(p, "rb").read() and clock.sleeps == 2
    assert [json.loads(l)["key"] for l in open(p)] == ["c", "d"] and RO.read_done(p) == {"c", "d"}


@pytest.mark.parametrize("workers", [1, 2])
def test_rollout_stops_starting_episodes_once_one_failed_for_good(tmp_path, monkeypatch, workers):
    """The pool stays full past the wait: the episodes not started yet are dropped (each would wait
    as long again), every episode that still finishes is recorded, the error comes out, and the
    next run does only what is missing."""
    eps = [{"key": "e%d" % i, "group": 0, "member": i, "variant": "v", "spec": {"problem_id": "e%d" % i}}
           for i in range(8)]
    req = {"script": "x.py", "harness_argv": [], "servers": ["http://127.0.0.1:1/v1"], "model": "m",
           "temperature": 1.0, "turns": 3, "workers": workers, "episodes": eps}
    rq, out = str(tmp_path / "req.json"), str(tmp_path / "out.jsonl")
    with open(rq, "w") as f:
        json.dump(req, f)
    ns = argparse.Namespace(request=rq, out=out)
    good = (0, {"episode": {"turns": []}, "calls": [], "rl_error": None}, "")
    started = []

    def run_child(payload, env):                     # two workers: e0 is still running when e1 fails
        pid = payload["spec"]["problem_id"]
        started.append(pid)
        time.sleep(0.5 if pid == "e0" and workers > 1 else 0.05)
        return (RO.DISK_FULL_RC, None, "OSError: [Errno 28] No space left on device") if pid == "e1" else good

    monkeypatch.setattr(RO, "run_child", run_child)
    with pytest.raises(OSError) as e:
        RO.cmd_run(ns)
    assert disk_full(e.value)
    recorded = [json.loads(l)["key"] for l in open(out)]
    # nothing is started after e1, also not by the thread whose episode just failed
    assert recorded == ["e0"] and sorted(started) == ["e0", "e1"]
    first = list(started)
    monkeypatch.setattr(RO, "run_child", lambda payload, env: (started.append(payload["spec"]["problem_id"]), good)[1])
    assert RO.cmd_run(ns) == 0
    assert sorted(json.loads(l)["key"] for l in open(out)) == ["e%d" % i for i in range(8)]
    assert sorted(started[len(first):]) == sorted(set("e%d" % i for i in range(8)) - set(recorded))


@pytest.mark.parametrize("cut_to, d_len, keys", [(2, 5, "abde"), (2, 50, "abde"), (2, 200, "abde"), (0, 5, "de")])
def test_outfile_goes_by_the_file_when_someone_cut_it(tmp_path, monkeypatch, cut_to, d_len, keys):
    """OUT made shorter by someone else and then one failed write: the repeat must not cut the file
    at the length remembered from before (NUL padding, a cut record glued to the repeated one, or
    the record twice, depending on its length)."""
    p = str(tmp_path / "out.jsonl")
    monkeypatch.setattr(diskio, "time", Clock())
    real, left = os.fsync, [0]

    def fsync(fd):
        if left[0]:
            left[0] -= 1
            raise enospc()
        return real(fd)

    monkeypatch.setattr(RO.os, "fsync", fsync)
    with RO.OutFile(p, wait_min=10) as fo:
        for k in "abc":
            fo.write({"key": k, "calls": list(range(50))})
        with open(p, "r+b") as f:
            f.truncate(cut_to * (os.path.getsize(p) // 3))
        left[0] = 1
        fo.write({"key": "d", "calls": list(range(d_len))})
        fo.write({"key": "e"})
    assert b"\0" not in open(p, "rb").read()
    assert [json.loads(l)["key"] for l in open(p)] == list(keys) and RO.read_done(p) == set(keys)


def test_has_room(tmp_path, monkeypatch):
    d = str(tmp_path)
    assert diskio.has_room([d]) and diskio.has_room([]) and os.listdir(d) == []
    assert diskio.has_room([os.path.join(d, "not there")])           # not a full disk
    monkeypatch.setattr(diskio.os, "fsync", lambda fd: (_ for _ in ()).throw(enospc()))
    assert not diskio.has_room([os.path.join(d, "not there"), d]) and os.listdir(d) == []


def test_a_child_that_died_without_saying_why_on_a_full_disk_is_no_crash(monkeypatch):
    """numpy's short write, a simulator's own sub-process, SIGBUS from a memory map: the child's
    status is an ordinary failure, the file system it writes to has no room."""
    clock = Clock()
    monkeypatch.setattr(diskio, "time", clock)
    answers, room, asked = [], [], []
    monkeypatch.setattr(RO, "run_child", lambda payload, env: answers.pop(0))
    monkeypatch.setattr(RO, "has_room", lambda dirs: (asked.append(dirs), room.pop(0))[1])
    short = (1, None, "OSError: 50000 requested and 2032 written")
    answers[:] = [short, (-7, None, "Fatal Python error: Bus error"), (0, {"episode": {}}, "")]
    room[:] = [False, False]
    assert RO.run_episode({}, {}, "s7/g0/m1", wait_min=10, dirs=("/pool/tmp",)) == ({"episode": {}}, None)
    assert clock.sleeps == 2 and asked == [("/pool/tmp",)] * 2 and not answers
    # the same failures with room on the file system: a crash after two attempts, no waiting
    clock.sleeps = 0
    answers[:], room[:] = [short, short], [True, True]
    assert RO.run_episode({}, {}, "s7/g0/m2", wait_min=10, dirs=("/pool/tmp",)) == (None, short[2])
    assert clock.sleeps == 0
    # no room and no wait allowed (a trainer that does not send disk_full_wait_min): raised, no record
    answers[:], room[:] = [short], [False]
    with pytest.raises(OSError) as e:
        RO.run_episode({}, {}, "s7/g0/m3", wait_min=0, dirs=("/pool/tmp",))
    assert disk_full(e.value)
