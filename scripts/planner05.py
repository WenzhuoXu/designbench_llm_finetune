"""
Planner v5: tell the proposer what the search did with its proposals.

v4 reaches 0.7442 against FSD's 0.4930 on the held-out set (discordant 108-0). Horizon 12
returned exactly the same numbers as horizon 8, so depth is saturated and the smoke that
suggested otherwise was two problems of noise.

The remaining slack is in the proposer. It emits about 30 candidates per episode, the search
evaluates them plus 21 pair compositions, and roughly 2 are accepted -- and the model is never
told which. Each turn it sees only the resulting state, so it cannot tell whether its structural
idea was chosen and improved on, or silently discarded in favour of doing nothing. It is
proposing blind into a selector it gets no feedback from.

Two changes:

  FEEDBACK  after each turn the planner is told exactly what happened to its proposals -- which
            one was selected, whether a pair was formed, or that none beat doing nothing. That
            is the one signal that lets it adapt within an episode.

  TRIPLES   a greedy third composition step, since pairs were worth roughly +10pp over singles.

Horizon fixed at 8. Both controls in this same script.
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
from planner01 import apply_tool, fsd_pass, TOOL_RE, SONNET45
from planner03 import rollout, potential
from planner04 import TOOLS_DOC

MAX_STEPS = 8
NPROP = 8

FEEDBACK_DOC = """

After each turn you are told what the search did with your proposals: which one it selected,
whether it combined two or three of them, or that none of them beat doing nothing. Use it. If
nothing you proposed was selected, your ideas were not helping -- change kind, change region,
be bolder. If one was selected, the state you now see already includes it."""


def step_search(truss, goals, calls, remaining, gb, gy):
    """Singles, then greedy pairs, then a greedy third. Returns (feasible, truss, note)."""
    base_ok, base_st, _ = rollout(truss, goals, remaining)
    if base_ok:
        return True, truss, "the structure was already on track"
    best_v = potential(base_st, goals, gb, gy)
    best_truss, best_note = None, None

    singles = []
    for nm, argstr in calls:
        try:
            cand = apply_tool(truss, goals, nm, argstr)
        except Exception:
            cand = None
        if cand is None or cand is truss:
            continue
        ok, st, _ = rollout(cand, goals, remaining)
        if ok:
            return True, cand, "%s(%s) reached feasibility" % (nm, argstr.strip())
        v = potential(st, goals, gb, gy)
        singles.append((v, cand, nm, argstr))
        if v > best_v + 1e-9:
            best_v, best_truss = v, cand
            best_note = "%s(%s) was selected" % (nm, argstr.strip())
    if not singles:
        return False, best_truss, best_note or "none of your proposals could be applied"

    singles.sort(key=lambda x: -x[0])
    anchor = singles[0][1]
    anchor_lbl = "%s(%s)" % (singles[0][2], singles[0][3].strip())
    pair_best = None
    for v0, _, nm, argstr in singles[1:]:
        try:
            cand = apply_tool(anchor, goals, nm, argstr)
        except Exception:
            cand = None
        if cand is None or cand is anchor:
            continue
        ok, st, _ = rollout(cand, goals, remaining)
        if ok:
            return True, cand, "%s then %s(%s) reached feasibility" % (anchor_lbl, nm, argstr.strip())
        v = potential(st, goals, gb, gy)
        if pair_best is None or v > pair_best[0]:
            pair_best = (v, cand, "%s then %s(%s)" % (anchor_lbl, nm, argstr.strip()))
        if v > best_v + 1e-9:
            best_v, best_truss = v, cand
            best_note = "%s then %s(%s) were combined" % (anchor_lbl, nm, argstr.strip())

    # greedy third step on top of the best pair
    if pair_best is not None:
        anchor2, lbl2 = pair_best[1], pair_best[2]
        for v0, _, nm, argstr in singles[1:]:
            try:
                cand = apply_tool(anchor2, goals, nm, argstr)
            except Exception:
                cand = None
            if cand is None or cand is anchor2:
                continue
            ok, st, _ = rollout(cand, goals, remaining)
            if ok:
                return True, cand, "%s then %s(%s) reached feasibility" % (lbl2, nm, argstr.strip())
            v = potential(st, goals, gb, gy)
            if v > best_v + 1e-9:
                best_v, best_truss = v, cand
                best_note = "%s then %s(%s) were combined" % (lbl2, nm, argstr.strip())
    return False, best_truss, best_note or "none of your proposals beat doing nothing"


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
    history, wins, props = [], 0, 0
    for turn in range(max_steps):
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True, "wins": wins, "props": props}
        nxt = fsd_pass(truss, goals)
        if nxt is not truss:
            truss = nxt
            state = _analyze_truss(truss, goals)
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True, "wins": wins, "props": props}

        remaining = max_steps - turn - 1
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
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "wins": wins + 1, "props": props}
        if chosen is not None:
            truss = chosen
            state = _analyze_truss(truss, goals)
            wins += 1
        history.append({"t": text.strip()[:1200],
                        "o": ("Search outcome: %s.\n" % note)
                             + DP.format_eval_result(state) + build_table(truss, true_fos(truss))})
    ok, st, _ = rollout(truss, goals, 1)
    return {"problem_id": pid, "arm": arm, "feasible": ok, "wins": wins, "props": props}


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
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/planner05.jsonl"))
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
    w = [r.get("wins", 0) for r in out if r["arm"] == "planner"]
    pr = [r.get("props", 0) for r in out if r["arm"] == "planner"]
    if w:
        print("\n  %.1f proposals/episode, %.1f accepted" % (sum(pr) / len(pr), sum(w) / len(w)))
