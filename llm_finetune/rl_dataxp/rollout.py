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
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def payload_for(req: dict, ep: dict) -> dict:
    server = req["servers"][int(ep.get("server", 0)) % len(req["servers"])]
    argv = list(req["harness_argv"]) + [
        "--policy", "llm", "--model", req["model"], "--base-url", server,
        "--temperature", str(req["temperature"]), "--turns", str(req["turns"]),
        "--policy-label", req.get("policy_label") or req["model"], "--out", os.devnull]
    return {"script": req["script"], "argv": argv, "variant": ep["variant"], "spec": ep["spec"],
            "inject": req.get("inject") or {}, "seed": ep.get("seed")}


def run_child(payload: dict, env: dict, python: str = sys.executable) -> tuple[int, dict | None, str]:
    pr = subprocess.run([python, "-X", "faulthandler", "-m", "llm_finetune.rl_dataxp.rollout", "_one_episode"],
                        input=pickle.dumps(payload), capture_output=True, env=env, cwd=REPO_ROOT)
    if pr.returncode == 0:
        return 0, pickle.loads(pr.stdout), ""
    return pr.returncode, None, pr.stderr.decode(errors="replace")


def cmd_run(a) -> int:
    req = json.load(open(a.request))
    done = read_done(a.out)
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

    def one(ep):
        payload = payload_for(req, ep)
        err = ""
        for attempt in (1, 2):
            rc, res, err = run_child(payload, env)
            if rc == 0:
                return ep, res, None
            print("!!! %s: child exited rc=%d (attempt %d)\n%s" % (
                ep["key"], rc, attempt, "\n".join(l for l in err.splitlines() if "arning" not in l)[-3000:]),
                flush=True)
        return ep, None, err[-2000:]

    with open(a.out, "a") as fo, ThreadPoolExecutor(int(req.get("workers", 16))) as ex:
        futs = [ex.submit(one, e) for e in todo]
        for fu in as_completed(futs):
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
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
    print("rollout done in %.0f s | crashed %d | without token ids %d" % (time.time() - t0, n_crash, n_noids),
          flush=True)
    return 4 if n_noids else 0


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
    a = ap.parse_args(argv)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
