"""
Planner v6: give the search a memory, so a bad line can be abandoned.

v5 reaches 0.7744 against FSD's 0.4930 on the held-out set, discordant 121-0. Its search is
greedy in one respect that has never been addressed: each turn commits to a single structure
and everything else is discarded forever. If a committed line turns out to be a dead end -- the
planner spends a turn on a change that looked good under rollout and then stops paying -- there
is no way back to the structure it came from.

Two changes, neither costing a model call:

  ARCHIVE   the best structure seen so far, by the same potential the search already uses, is
            kept. Nothing is lost merely because a later turn moved away from it.

  RESTART   if the current line fails to improve on the archive for two consecutive turns, the
            search returns to the archived structure and the planner is told it has done so.
            A stalled line stops consuming the remaining horizon.

The do-nothing branch still competes every turn, and the archive can only ever hold a structure
the search already preferred, so the arm still cannot fall below its own baseline. Proposal
count raised to 12 since widening has paid twice. Both controls in this same script.
"""
import os, sys, json, math, re, argparse, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals
from llm_finetune.data.processors import designbench_prompt as DP
import api_grammar_2x2 as H
from probe15_matrix import build_table, true_fos
from probe26_presentation import sign_test
from planner01 import fsd_pass, TOOL_RE, SONNET45
from planner03 import rollout, potential
from planner04 import TOOLS_DOC
from planner05 import step_search, FEEDBACK_DOC

MAX_STEPS = 8
NPROP = 12
STALL = 2


def episode(spec, arm, region, token, k, max_steps, nprop=NPROP):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    pid = spec.get("problem_id", "")
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))

    if arm == "fsd":
        ok, st, _ = rollout(truss, goals, max_steps)
        return {"problem_id": pid, "arm": arm, "feasible": ok}

    system = DP.TRUSS_SYSTEM_PROMPT + "\n\n" + (TOOLS_DOC % (nprop, nprop)) + FEEDBACK_DOC
    history, wins, props, restarts = [], 0, 0, 0
    best_truss, best_v, stall = None, -1e9, 0

    for turn in range(max_steps):
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "wins": wins, "props": props, "restarts": restarts}
        nxt = fsd_pass(truss, goals)
        if nxt is not truss:
            truss = nxt
            state = _analyze_truss(truss, goals)
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "wins": wins, "props": props, "restarts": restarts}

        v_now = potential(state, goals, gb, gy)
        if v_now > best_v + 1e-9:
            best_v, best_truss, stall = v_now, truss, 0
        else:
            stall += 1
        restarted = False
        if stall >= STALL and best_truss is not None and best_truss is not truss:
            truss = best_truss
            state = _analyze_truss(truss, goals)
            stall, restarts, restarted = 0, restarts + 1, True

        remaining = max_steps - turn - 1
        tbl = build_table(truss, true_fos(truss))
        head = ("The search has returned to the best structure it found earlier; the line you "
                "were on stopped paying. Try a different kind of change.\n"
                if restarted else "")
        obs = (head + "A fully-stressed sizing pass has just been applied. Resulting state:\n"
               + DP.format_eval_result(state) + tbl)
        convo = [{"role": "user", "content": [{"text":
                  DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                  + DP.format_eval_result(initial) + "\n\n" + obs}]}]
        for h in history:
            convo.append({"role": "assistant", "content": [{"text": h["t"]}]})
            convo.append({"role": "user", "content": [{"text": "[Simulation Result]\n" + h["o"]}]})
        text = H.call(convo, system, SONNET45, region, token) or ""
        calls = [(n, a) for (n, a) in TOOL_RE.findall(text)
                 if n not in ("FSD_ALL", "FSD_ON")][:nprop]
        props += len(calls)

        ok, chosen, note = step_search(truss, goals, calls, remaining, gb, gy)
        if ok:
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "wins": wins + 1, "props": props, "restarts": restarts}
        if chosen is not None:
            truss = chosen
            state = _analyze_truss(truss, goals)
            wins += 1
            v_new = potential(state, goals, gb, gy)
            if v_new > best_v + 1e-9:
                best_v, best_truss, stall = v_new, truss, 0
        history.append({"t": text.strip()[:1200],
                        "o": ("Search outcome: %s.\n" % note)
                             + DP.format_eval_result(state) + build_table(truss, true_fos(truss))})

    for cand in (truss, best_truss):
        if cand is None:
            continue
        ok, st, _ = rollout(cand, goals, 2)
        if ok:
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "wins": wins, "props": props, "restarts": restarts}
    return {"problem_id": pid, "arm": arm, "feasible": False,
            "wins": wins, "props": props, "restarts": restarts}


def work(t):
    spec, arm, region, token, k, ms = t
    try:
        return episode(spec, arm, region, token, k, ms)
    except Exception as e:
        return {"problem_id": spec.get("problem_id"), "arm": arm, "feasible": False,
                "err": str(e)[:200]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--arms", default="planner,fsd")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/planner06.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    arms = tuple(a.arms.split(","))
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]]
    tasks = [(s, arm, a.region, token, a.k, a.max_steps) for s in specs for arm in arms]
    print("problems %d-%d | horizon %d | episodes %d"
          % (a.start, a.start + a.n, a.max_steps, len(tasks)), flush=True)
    t0 = time.time()
    out = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w") as fh:
        for i, r in enumerate(ex.map(work, tasks)):
            out.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 40 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs\n" % (time.time() - t0))

    by = {arm: {r["problem_id"]: (1.0 if r["feasible"] else 0.0) for r in out if r["arm"] == arm}
          for arm in arms}
    pids = sorted(set.intersection(*[set(by[arm]) for arm in arms]))
    rate = {arm: sum(by[arm][p] for p in pids) / max(len(pids), 1) for arm in arms}
    print("paired on %d problems\n" % len(pids))
    for arm in arms:
        print("  %-9s %.4f" % (arm, rate[arm]))
    if "fsd" in arms and "planner" in arms:
        u, d, p = sign_test([by["fsd"][x] for x in pids], [by["planner"][x] for x in pids])
        print("\n  planner vs FSD control: %+.4f   discordant %d-%d   exact sign p = %.3g"
              % (rate["planner"] - rate["fsd"], u, d, p))
    pr = [r.get("props", 0) for r in out if r["arm"] == "planner"]
    w = [r.get("wins", 0) for r in out if r["arm"] == "planner"]
    rs = [r.get("restarts", 0) for r in out if r["arm"] == "planner"]
    if pr:
        print("\n  %.1f proposals/episode, %.1f accepted, %.2f restarts"
              % (sum(pr) / len(pr), sum(w) / len(w), sum(rs) / len(rs)))
