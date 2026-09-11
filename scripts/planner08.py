"""
Planner v8: spend the same budget on two independent attempts instead of one long one.

v5 and v7 both land on 0.7744 against FSD's 0.4930, discordant 121-0. Depth saturated at eight
turns, width at eight proposals, and the archive and margin search each added nothing at scale.
What paid was composition and telling the proposer what the search did. 97 of the 218 problems
FSD fails are still unconverted.

Every version so far has been one trajectory. The planner is stochastic, so a run that commits
early to an unproductive line has no way to sample a different opening -- and the restart
mechanism in v6 only ever returned to a structure on the same line, which is why it did
nothing.

So: two independent attempts of four turns each, from the original structure, and the episode
succeeds if either does. That is the SAME number of model calls as one eight-turn attempt, so
the comparison against every previous variant stays budget-matched; it trades depth, which is
saturated, for diversity, which has never been tried. The second attempt is given a different
proposal temperature so it does not retrace the first.

The FSD control is unchanged and runs in this same script.
"""
import os, sys, json, math, argparse, time
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

MAX_STEPS = 8          # total model-call budget per problem, split across attempts
ATTEMPTS = 2
NPROP = 8
NUDGE = ("\n\nThis is a fresh attempt from the original structure. A previous attempt did not "
         "reach feasibility, so do not open with the most obvious change -- try a different "
         "region of the structure or a different kind of move than you otherwise would.")


def one_attempt(spec, region, token, steps, nprop, variant):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    system = (DP.TRUSS_SYSTEM_PROMPT + "\n\n" + (TOOLS_DOC % (nprop, nprop)) + FEEDBACK_DOC
              + (NUDGE if variant else ""))
    history, props, wins = [], 0, 0
    for turn in range(steps):
        if state.get("is_feasible"):
            return True, props, wins
        nxt = fsd_pass(truss, goals)
        if nxt is not truss:
            truss = nxt
            state = _analyze_truss(truss, goals)
        if state.get("is_feasible"):
            return True, props, wins
        remaining = steps - turn - 1
        tbl = build_table(truss, true_fos(truss))
        obs = ("A fully-stressed sizing pass has just been applied. Resulting state:\n"
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
            return True, props, wins + 1
        if chosen is not None:
            truss = chosen
            state = _analyze_truss(truss, goals)
            wins += 1
        history.append({"t": text.strip()[:1200],
                        "o": ("Search outcome: %s.\n" % note)
                             + DP.format_eval_result(state) + build_table(truss, true_fos(truss))})
    ok, _, _ = rollout(truss, goals, 2)
    return ok, props, wins


def episode(spec, arm, region, token, k, max_steps, nprop=NPROP):
    pid = spec.get("problem_id", "")
    if arm == "fsd":
        truss, goals = _load_truss_and_goals(spec)
        ok, _, _ = rollout(truss, goals, max_steps)
        return {"problem_id": pid, "arm": arm, "feasible": ok}

    per = max(1, max_steps // ATTEMPTS)
    total_props, total_wins = 0, 0
    for att in range(ATTEMPTS):
        ok, props, wins = one_attempt(spec, region, token, per, nprop, att)
        total_props += props
        total_wins += wins
        if ok:
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "attempt": att + 1, "props": total_props, "wins": total_wins}
    return {"problem_id": pid, "arm": arm, "feasible": False,
            "attempt": ATTEMPTS, "props": total_props, "wins": total_wins}


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
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/planner08.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    arms = tuple(a.arms.split(","))
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]]
    tasks = [(s, arm, a.region, token, a.k, a.max_steps) for s in specs for arm in arms]
    print("problems %d-%d | %d attempts x %d turns | episodes %d"
          % (a.start, a.start + a.n, ATTEMPTS, a.max_steps // ATTEMPTS, len(tasks)), flush=True)
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
    pids = sorted(set.intersection(*[set(by[x]) for x in arms]))
    rate = {arm: sum(by[arm][p] for p in pids) / max(len(pids), 1) for arm in arms}
    print("paired on %d problems\n" % len(pids))
    for arm in arms:
        print("  %-9s %.4f" % (arm, rate[arm]))
    if "fsd" in arms and "planner" in arms:
        u, d, p = sign_test([by["fsd"][x] for x in pids], [by["planner"][x] for x in pids])
        print("\n  planner vs FSD control: %+.4f   discordant %d-%d   exact sign p = %.3g"
              % (rate["planner"] - rate["fsd"], u, d, p))
    a2 = [r.get("attempt") for r in out if r["arm"] == "planner" and r.get("feasible")]
    if a2:
        print("\n  successes resolved on attempt 1: %d, attempt 2: %d"
              % (sum(1 for x in a2 if x == 1), sum(1 for x in a2 if x == 2)))
