"""Run one step's episodes against the vLLM servers; append-only and resumable.

    python -m llm_finetune.rl_dataxp.rollout run --request REQ.json --out OUT.jsonl

REQ.json (written by the trainer):
  script        path of dataxp_policy_eval_v3.py
  harness_argv  the evaluation flags (--code, --obs, --no-think, --max-tokens, --instruction-file,
                --param-line, --native-prompt, ...), without --model/--base-url/--temperature/--turns
  servers       ["http://127.0.0.1:PORT/v1", ...]
  model         served adapter name for this step
  temperature, turns, inject (extra request fields), workers, env (extra environment)
  episodes      [{key, group, member, variant, spec, seed, server}, ...]

Every episode runs in its own child process, as in the harness's cmd_episodes: the simulators
call native code that has segfaulted under threads (j81, j100). A crash is retried once, then
recorded as crashed (dropped from its group). Each finished episode is appended to OUT as one
line and flushed, so a requeued job reruns only the episodes that are missing. The run exits
with status 4 if any episode came back without token ids (the server did not honour
return_token_ids; training cannot use it).

Runs in the simulator env; no torch.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .diskio import DISK_FULL_RC, disk_full_cause, has_room, retry_when_full, write_text

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_done(path: str) -> set:
    """Keys already in OUT. A partial last line (a job killed mid-write) is cut off first."""
    if not os.path.exists(path):
        return set()
    with open(path, "rb") as f:
        data = f.read()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        with open(path, "r+b") as f:
            f.truncate(cut)
        data = data[:cut]
    keys = set()
    for line in data.splitlines():
        if line.strip():
            keys.add(json.loads(line)["key"])
    return keys


class OutFile:
    """OUT: one line per finished episode, on disk (fsync) before the next. A line that fails on a
    full file system is written again once there is room, for up to wait_min minutes; the file is
    first cut back to the last record known to be on disk, so it never holds half a record or the
    same record twice."""

    def __init__(self, path: str, wait_min: float = 0.0):
        self.path, self.wait_min, self.f, self.good = path, wait_min, None, None

    def _open(self) -> None:
        if self.good is None:
            read_done(self.path)                     # cuts the partial last line of a killed job
            self.good = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        elif os.path.exists(self.path):
            with open(self.path, "r+b") as f:        # drop what a failed attempt left
                f.truncate(self.good)
        else:                                        # creating it is what failed, or it was removed
            self.good = 0
        self.f = open(self.path, "ab")

    def _put(self, data: bytes) -> None:
        if self.f is None:
            self._open()
        try:
            self.f.write(data)
            self.f.flush()
            os.fsync(self.f.fileno())
        except BaseException:
            f, self.f = self.f, None
            try:
                f.close()
            except OSError:
                pass
            raise
        self.good += len(data)

    def _check(self) -> None:
        """Before a new record: when OUT is no longer the file, or the length, the last record
        left (cut, emptied, removed or replaced by someone else), go by what is on disk. Cutting
        it back to a length remembered from before would pad it with NUL bytes or leave a cut
        record, and no resumed step could read that."""
        if self.f is None:
            return
        try:
            st = os.stat(self.path)
            same = st.st_size == self.good and os.path.samestat(st, os.fstat(self.f.fileno()))
        except OSError:
            same = False
        if not same:
            self.close()
            self.good = None                         # _open reads the file again

    def write(self, rec: dict) -> None:
        data = (json.dumps(rec) + "\n").encode()
        self._check()
        retry_when_full(lambda: self._put(data), "episode %s" % rec.get("key"), self.wait_min,
                        log=lambda s: print(s, flush=True))

    def close(self) -> None:
        if self.f is not None:
            f, self.f = self.f, None
            f.close()

    def __enter__(self):                             # OUT exists from here on, also with no record
        retry_when_full(self._open, self.path, self.wait_min, log=lambda s: print(s, flush=True))
        return self

    def __exit__(self, *exc):
        self.close()


def payload_for(req: dict, ep: dict) -> dict:
    server = req["servers"][int(ep.get("server", 0)) % len(req["servers"])]
    argv = list(req["harness_argv"]) + [
        "--policy", "llm", "--model", req["model"], "--base-url", server,
        "--temperature", str(req["temperature"]), "--turns", str(req["turns"]),
        "--policy-label", req.get("policy_label") or req["model"], "--out", os.devnull]
    return {"script": req["script"], "argv": argv, "variant": ep["variant"], "spec": ep["spec"],
            "inject": req.get("inject") or {}, "seed": ep.get("seed"),
            "require_token_ids": req.get("require_token_ids", True)}


def run_child(payload: dict, env: dict, python: str = sys.executable) -> tuple[int, dict | None, str]:
    pr = subprocess.run([python, "-X", "faulthandler", "-m", "llm_finetune.rl_dataxp.rollout", "_one_episode"],
                        input=pickle.dumps(payload), capture_output=True, env=env, cwd=REPO_ROOT)
    if pr.returncode == 0:
        return 0, pickle.loads(pr.stdout), ""
    return pr.returncode, None, pr.stderr.decode(errors="replace")


def run_episode(payload: dict, env: dict, key: str, wait_min: float = 0.0,
                dirs: tuple = ()) -> tuple[dict | None, str | None]:
    """One episode in a child process: (result, None), or (None, end of its stderr) when two
    attempts failed. A child that stopped on a full file system (TMPDIR and the caches are on the
    shared pool; it says so with exit status DISK_FULL_RC) has not crashed: a crash record is
    final, a resumed step never reruns it. That child is run again once there is room, for up to
    wait_min minutes; past that the error is raised and the rollout fails without a record for
    the episode. The status decides, not the text of stderr: a warning there can name a full
    device (joblib does when /dev/shm is full) next to an unrelated crash. A child can also die
    on a full file system without saying so (has_room lists the measured cases), so after any
    other failure dirs, where the children write, are tested: no room there is a full disk too."""

    def child():
        rc, res, err = run_child(payload, env)
        if rc == DISK_FULL_RC or (rc != 0 and not has_room(dirs)):
            raise OSError(errno.ENOSPC, "child of %s (rc %d): %s" % (key, rc, err.strip()[-300:]))
        return rc, res, err

    err = ""
    for attempt in (1, 2):
        rc, res, err = retry_when_full(child, "episode %s (its child process)" % key, wait_min,
                                       log=lambda s: print(s, flush=True))
        if rc == 0:
            return res, None
        print("!!! %s: child exited rc=%d (attempt %d)\n%s" % (
            key, rc, attempt, "\n".join(l for l in err.splitlines() if "arning" not in l)[-3000:]), flush=True)
    return None, err[-2000:]


def cmd_run(a) -> int:
    req = json.load(open(a.request))
    wait_min = float(req.get("disk_full_wait_min") or 0)
    # read_done cuts the partial last line of a killed job: a write, and a full pool can fail it
    done = retry_when_full(lambda: read_done(a.out), a.out, wait_min, log=lambda s: print(s, flush=True))
    todo = [e for e in req["episodes"] if e["key"] not in done]
    env = dict(os.environ)
    # as the harness's validated runs: one thread per simulator (thread count can change BLAS round-off)
    # and PYTHONHASHSEED=0 (metrics_after's key order is set-derived). Forced, not defaulted: the
    # trainer's environment exports OMP_NUM_THREADS=4.
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = "1"
    env["PYTHONHASHSEED"] = "0"
    env.update({k: str(v) for k, v in (req.get("env") or {}).items()})
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    print("rollout: %d episodes (%d already done), model %s, T=%s, %d turns, %d servers, %d workers"
          % (len(req["episodes"]), len(done), req["model"], req["temperature"], req["turns"],
             len(req["servers"]), req.get("workers", 16)), flush=True)
    t0 = time.time()
    n_crash = n_noids = 0

    # where the children write: OUT's directory, their temporary files and caches
    dirs = tuple(dict.fromkeys(os.path.abspath(d) for d in (
        os.path.dirname(os.path.abspath(a.out)), env.get("TMPDIR") or tempfile.gettempdir(),
        env.get("XDG_CACHE_HOME")) if d and os.path.isdir(d)))
    stop = threading.Event()

    def one(ep):
        if stop.is_set():                            # left to the next run
            return None
        try:
            res, err = run_episode(payload_for(req, ep), env, ep["key"], wait_min, dirs)
        except BaseException:
            stop.set()                               # before this thread takes the next episode
            raise
        return ep, res, err

    failed = None
    with OutFile(a.out, wait_min) as fo, \
            ThreadPoolExecutor(int(req.get("workers", 16))) as ex:
        futs = [ex.submit(one, e) for e in todo]
        try:
            for fu in as_completed(futs):
                if fu.exception() is not None:
                    # an episode failed for good (the pool stayed full past the wait): the episodes
                    # not started yet would each wait as long again, so they are not started (stop);
                    # those already running are still recorded, and the error is raised at the end
                    failed = failed or fu.exception()
                    continue
                if fu.result() is None:
                    continue
                ep, res, err = fu.result()
                rec = {"key": ep["key"], "group": ep["group"], "member": ep.get("member"),
                       "variant": ep["variant"], "problem_id": ep["spec"].get("problem_id"),
                       "server": ep.get("server"), "seed": ep.get("seed")}
                if res is None:
                    n_crash += 1
                    rec.update(episode={"crashed": True, "turns": [], "first_feasible_turn": None,
                                        "problem_id": ep["spec"].get("problem_id"), "variant": ep["variant"]},
                               calls=[], rl_error=None, crash_tail=err)
                else:
                    rec.update(episode=res["episode"], calls=res["calls"], rl_error=res["rl_error"])
                    n_noids += int(res["rl_error"] == "no_token_ids")
                fo.write(rec)
        except BaseException:                        # a record that could not be written: stop too
            stop.set()
            raise
    if failed is not None:
        raise failed
    print("rollout done in %.0f s | crashed %d | without token ids %d" % (time.time() - t0, n_crash, n_noids),
          flush=True)
    return 4 if n_noids else 0


def cmd_eval(a) -> int:
    """Evaluation episodes through the same harness path as training: T=0, --turns 20 (the
    harness default), no completion cap (max_tokens null), one record per (problem, run) in the
    harness's own episode format (+ key/run), resumable by key. Then a feasible@12/@20 summary."""
    probs = []
    for path in a.problems:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if not a.variants or r["variant"] in a.variants:
                    probs.append(r)
    eps = []
    for run in range(1, a.runs + 1):
        for j, r in enumerate(probs):
            pid = r["spec"]["problem_id"]
            eps.append({"key": "%s/r%d/%s" % (a.label, run, pid), "group": "%s/r%d" % (a.label, run),
                        "member": run, "variant": r["variant"], "spec": r["spec"], "seed": None,
                        "server": j % len(a.servers)})
    req = {"script": a.script, "harness_argv": list(a.harness_argv), "servers": a.servers, "model": a.model,
           "policy_label": a.label, "temperature": 0.0, "turns": a.turns,
           "inject": {"max_tokens": None}, "require_token_ids": False, "workers": a.workers, "env": {},
           "episodes": eps, "disk_full_wait_min": a.disk_full_wait_min}
    req_path = a.out + ".request.json"
    retry_when_full(lambda: write_text(req_path, json.dumps(req)), "the evaluation request", a.disk_full_wait_min)
    ns = argparse.Namespace(request=req_path, out=a.out)
    rc = cmd_run(ns)
    summarize(a.out, a.disk_full_wait_min)
    return rc


def summarize(path: str, wait_min: float = 0.0) -> dict:
    import collections
    by = collections.defaultdict(lambda: collections.Counter())
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        ep = r["episode"]
        if ep.get("feasible_start"):
            continue
        c = by[r["variant"]]
        c["n"] += 1
        t = ep.get("first_feasible_turn")
        c["f12"] += int(t is not None and t <= 12)
        c["f20"] += int(t is not None and t <= 20)
        c["crashed"] += int(bool(ep.get("crashed")))
        turns = ep.get("turns", [])
        c["turns"] += len(turns)
        c["at_eos_or_ctx"] += sum(1 for x in turns if x.get("finish") == "length")
    out = {v: dict(c) for v, c in sorted(by.items())}
    print("%-11s %5s %8s %8s %8s %8s" % ("variant", "n", "feas@12", "feas@20", "crashed", "len-stop"))
    for v, c in out.items():
        n = max(1, c["n"])
        print("%-11s %5d %8.3f %8.3f %8d %8d" % (v, c["n"], c["f12"] / n, c["f20"] / n, c["crashed"], c["at_eos_or_ctx"]))
    retry_when_full(lambda: write_text(path + ".summary.json", json.dumps(out, indent=1)), "the summary", wait_min)
    return out


def _one_episode() -> None:
    """Hidden mode: one episode in this process; payload (pickle) on stdin, result on stdout."""
    import faulthandler
    faulthandler.enable()
    payload = pickle.load(sys.stdin.buffer)
    # stdout carries the pickle; anything a simulator prints (Python or native) goes to stderr
    sys.stdout.flush()
    real_stdout = os.dup(1)
    os.dup2(2, 1)
    try:
        from llm_finetune.rl_dataxp.harness import run_one
        res = run_one(payload)
    except Exception as e:
        if not disk_full_cause(e):
            raise
        import traceback
        traceback.print_exc()
        sys.exit(DISK_FULL_RC)                       # not a crash: the parent runs the episode again
    finally:
        sys.stdout.flush()
        os.dup2(real_stdout, 1)
    with os.fdopen(real_stdout, "wb") as out:
        out.write(pickle.dumps(res))


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_one_episode":
        _one_episode()
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--request", required=True)
    r.add_argument("--out", required=True)
    e = sub.add_parser("eval", help="evaluation episodes: T=0, no completion cap, resumable")
    e.add_argument("--script", required=True, help="dataxp_policy_eval_v3.py")
    e.add_argument("--problems", nargs="+", required=True, help="{variant, spec} files (harness format)")
    e.add_argument("--variants", nargs="*", default=None)
    e.add_argument("--servers", nargs="+", required=True)
    e.add_argument("--model", required=True, help="served name: base, or a LoRA adapter's name")
    e.add_argument("--label", required=True)
    e.add_argument("--runs", type=int, default=2)
    e.add_argument("--turns", type=int, default=20)
    e.add_argument("--workers", type=int, default=32)
    e.add_argument("--out", required=True)
    e.add_argument("--disk-full-wait-min", type=float, default=120.0,
                   help="how long a record that fails on a full file system is retried")
    e.add_argument("harness_argv", nargs=argparse.REMAINDER, help="after --: the harness flags (--code ...)")
    a = ap.parse_args(argv)
    if a.cmd == "eval":
        a.harness_argv = [x for x in a.harness_argv if x != "--"]
        return cmd_eval(a)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
