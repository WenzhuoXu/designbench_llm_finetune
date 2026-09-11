#!/usr/bin/env python
"""Independent verification of a gen_expert JSONL.

Reads the file back with json.loads, replays every recorded action from the untouched
problem spec through the real DesignBench executor, and checks that

  * every action executes (apply_candidate does not return None),
  * every recorded STRUCTURAL ANALYSIS RESULT turn is byte-identical to the observation
    that replay produces at that point,
  * the final state is feasible (FOS_buckling >= req, FOS_yielding >= req, mass <= max),
  * the message roles alternate as DesignBench SFT data does and the counts line up.

Nothing here reuses gen_expert's own state; it starts from the problem json.
"""
import json, sys, math
from pathlib import Path
from multiprocessing import Pool

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for _p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import probe15_matrix as PM
import api_grammar_2x2 as H
from llm_finetune.data.processors import designbench_prompt as DP
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals

PROB = DESIGNBENCH / "data" / "problems_hard"


def obs_text(state, truss):
    return DP.format_eval_result(state) + PM.build_table(truss, PM.true_fos(truss))


def check(rec):
    pid = rec["problem_id"]
    spec = json.load(open(PROB / (pid + ".json")))
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    msgs = rec["messages"]

    if msgs[0]["role"] != "system" or msgs[1]["role"] != "system":
        return pid, "bad_header"
    if msgs[1]["content"] != "INITIAL STATE ANALYSIS:\n" + obs_text(state, truss):
        return pid, "initial_obs_mismatch"
    if len(msgs) != 2 + 2 * len(rec["actions"]) + 1:
        return pid, "message_count"

    cur = truss
    for k, action in enumerate(rec["actions"]):
        a_msg, s_msg = msgs[2 + 2 * k], msgs[3 + 2 * k]
        if a_msg["role"] != "assistant" or s_msg["role"] != "system":
            return pid, "role_order@%d" % k
        if action not in a_msg["content"]:
            return pid, "action_not_in_turn@%d" % k
        nxt = H.apply_candidate(cur, action)
        if nxt is None:
            return pid, "apply_failed@%d" % k
        state = _analyze_truss(nxt, goals)
        want = "STRUCTURAL ANALYSIS RESULT:\n" + obs_text(state, nxt)
        if s_msg["content"] != want:
            return pid, "obs_mismatch@%d" % k
        cur = nxt

    if msgs[-1]["role"] != "assistant" or "<answer>" not in msgs[-1]["content"]:
        return pid, "bad_answer_turn"
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    mm = float(goals.get("maximum_mass", float("inf")))
    fb = float(state.get("fos_buckling", 0.0) or 0.0)
    fy = float(state.get("fos_yielding", 0.0) or 0.0)
    ms = float(state.get("mass", float("inf")) or float("inf"))
    if not (fb >= gb and fy >= gy and ms <= mm and state.get("is_feasible")):
        return pid, "final_infeasible fb=%.3f/%.2f fy=%.3f/%.2f m=%.2f/%.2f" % (fb, gb, fy, gy, ms, mm)
    return pid, "ok"


def _w(rec):
    try:
        return check(rec)
    except Exception as e:
        return rec.get("problem_id"), "error:%s" % str(e)[:140]


if __name__ == "__main__":
    path = sys.argv[1]
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 44
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    print("verifying %d records from %s" % (len(recs), path), flush=True)
    with Pool(procs) as pool:
        res = pool.map(_w, recs, chunksize=1)
    hist = {}
    for _, why in res:
        key = why.split()[0]
        hist[key] = hist.get(key, 0) + 1
    print("  " + ", ".join("%s=%d" % kv for kv in sorted(hist.items())))
    bad = [(p, w) for p, w in res if w != "ok"]
    for p, w in bad[:10]:
        print("  FAIL %s: %s" % (p, w))
    print("  ALL RECORDS VERIFIED" if not bad else "  %d FAILURES" % len(bad))
