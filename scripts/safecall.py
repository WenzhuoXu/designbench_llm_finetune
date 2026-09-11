"""A bounded wrapper around H.call, plus resume support for long evaluation runs.

The v7 held-out run stalled at 700 of 860 episodes: H.call has no timeout, one request hung,
and because results are consumed in order everything behind it stopped being written. Both
problems are worth fixing once rather than working around each time.

  call(...)        the same call, bounded. On timeout or error it retries once and then gives
                   up, returning "" -- a turn with no parseable proposal, which the planner
                   already handles as "no proposal beat doing nothing". A hung request costs
                   one turn instead of the run.

  done_keys(path)  the (problem_id, arm) pairs already in an output file, so a run can be
                   resumed instead of restarted.
"""
import sys, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts")
import api_grammar_2x2 as H

_ORIG_CALL = H.call          # bind before anyone rebinds H.call to this wrapper
_POOL = ThreadPoolExecutor(max_workers=64)
TIMEOUT_S = 120


def call(convo, system, model, region, token, timeout=TIMEOUT_S, attempts=2):
    for i in range(attempts):
        fut = _POOL.submit(_ORIG_CALL, convo, system, model, region, token)
        try:
            out = fut.result(timeout=timeout)
            if out:
                return out
        except FTimeout:
            fut.cancel()
        except Exception:
            pass
    return ""


def done_keys(path):
    p = Path(path)
    if not p.exists():
        return set()
    keys = set()
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("problem_id") is not None and r.get("arm"):
            keys.add((r["problem_id"], r["arm"]))
    return keys
