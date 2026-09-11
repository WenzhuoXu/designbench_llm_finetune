"""
Planner v4: widen the search over tool COMPOSITIONS, at no extra model cost.

v3 beats fully-stressed design by +15.1pp on the held-out set (0.6442 vs 0.4930, discordant
65-0). Its bottleneck is visible in its own counters: 14.1 proposals per episode, of which only
2.1 beat the do-nothing branch. The model is generating candidates faster than the search is
using them, and every candidate is evaluated alone.

Two changes, both free -- the rollouts are FEA, not model calls:

  WIDER   ask for more proposals per turn, and evaluate all of them.

  COMPOSED  evaluate PAIRS as well as singles. The library's whole premise is that tools
            compose, and v3 never tested a composition: it applied one tool and rolled out.
            Here the best single is held and every other proposal is tried on top of it, so a
            move that only pays once another has been made can be found. This is a greedy
            second step rather than a full product, which keeps the branch count linear.

The do-nothing branch still competes on equal terms, so the arm still cannot lose to its own
baseline. Both controls run in this same script.
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

MAX_STEPS = 8
NPROP = 8

TOOLS_DOC = """A fully-stressed sizing pass is applied automatically at the start of every turn.
It rescales every member towards its required factor of safety -- members below requirement
grow, members far above it shrink. You never need to ask for it.

Your job is what that rule cannot do. It only resizes existing members, so it can never change
the load paths; it applies one uniform rule, so it cannot spend margin unevenly on purpose; and
when it converges while still infeasible -- usually the mass cap binding with every member
already at its stressed size -- it will return the same structure forever.

Propose %d DIFFERENT candidate interventions, each on its own line as <tool>...</tool>:

  ADD_MEMBER(joint1, joint2)      add a load path
  REMOVE_MEMBER(i)                delete a member carrying little and costing mass
  MOVE_JOINT(joint, x, y)         reshape the geometry
  TRIM(fos_threshold, factor)     shrink everything above fos_threshold by factor (<1)
  SCALE([ids], factor)            override the sizing on specific members

Every proposal is applied to its own copy of the structure and carried forward by the sizing
rule to the end of the episode. Pairs are tried too: the best proposal is kept and each of the
others is applied on top of it, so a move that only pays off once another has been made can
still be found. A do-nothing branch competes alongside them all.

Because a proposal that turns out worse than doing nothing is simply discarded, a bad guess
costs you nothing and a timid list costs you the turn. Make them genuinely different from each
other -- vary which region of the structure you touch and which kind of move you make -- and
include at least two bold structural changes. Say in one sentence what is blocking feasibility,
strength or mass, then give your %d lines."""


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

    system = DP.TRUSS_SYSTEM_PROMPT + "\n\n" + (TOOLS_DOC % (nprop, nprop))
    history, wins, props, pairs = [], 0, 0, 0
    for turn in range(max_steps):
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "wins": wins, "props": props, "pairs": pairs}
        nxt = fsd_pass(truss, goals)
        if nxt is not truss:
            truss = nxt
            state = _analyze_truss(truss, goals)
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "wins": wins, "props": props, "pairs": pairs}

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

        base_ok, base_st, _ = rollout(truss, goals, remaining)
        if base_ok:
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "wins": wins, "props": props, "pairs": pairs}
        best_v = potential(base_st, goals, gb, gy)
        best_truss = None

        # --- singles
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
                return {"problem_id": pid, "arm": arm, "feasible": True,
                        "wins": wins + 1, "props": props, "pairs": pairs}
            v = potential(st, goals, gb, gy)
            singles.append((v, cand, nm, argstr))
            if v > best_v + 1e-9:
                best_v, best_truss = v, cand

        # --- pairs: hold the best single, try each other proposal on top of it
        if singles:
            singles.sort(key=lambda x: -x[0])
            anchor_v, anchor, anchor_nm, anchor_arg = singles[0]
            for v0, _, nm, argstr in singles[1:]:
                try:
                    cand = apply_tool(anchor, goals, nm, argstr)
                except Exception:
                    cand = None
                if cand is None or cand is anchor:
                    continue
                pairs += 1
                ok, st, _ = rollout(cand, goals, remaining)
                if ok:
                    return {"problem_id": pid, "arm": arm, "feasible": True,
                            "wins": wins + 1, "props": props, "pairs": pairs}
                v = potential(st, goals, gb, gy)
                if v > best_v + 1e-9:
                    best_v, best_truss = v, cand

        if best_truss is not None:
            truss = best_truss
            state = _analyze_truss(truss, goals)
            wins += 1
        history.append({"t": text.strip()[:1200],
                        "o": DP.format_eval_result(state) + build_table(truss, true_fos(truss))})
    ok, st, _ = rollout(truss, goals, 1)
    return {"problem_id": pid, "arm": arm, "feasible": ok,
            "wins": wins, "props": props, "pairs": pairs}


def work(t):
    spec, arm, region, token, k, ms = t
    try:
        return episode(spec, arm, region, token, k, ms)
    except Exception as e:
        return {"problem_id": spec.get("problem_id"), "arm": arm, "feasible": False,
                "err": str(e)[:200]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--arms", default="planner,fsd")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/planner04.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    arms = tuple(a.arms.split(","))
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]]
    tasks = [(s, arm, a.region, token, a.k, a.max_steps) for s in specs for arm in arms]
    print("problems %d-%d | arms %s | horizon %d | episodes %d"
          % (a.start, a.start + a.n, ",".join(arms), a.max_steps, len(tasks)), flush=True)
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
    pa = [r.get("pairs", 0) for r in out if r["arm"] == "planner"]
    if w:
        print("\n  %.1f proposals/episode, %.1f pair branches, %.1f accepted"
              % (sum(pr) / len(pr), sum(pa) / len(pa), sum(w) / len(w)))
