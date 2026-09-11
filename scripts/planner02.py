"""
Planner v2: fully-stressed design runs by default; the planner does what it cannot.

v1 handed the model FSD as one tool among several. That was worth +20.0pp over raw grammar
actions (0.5667 vs 0.3667, discordant 6-0) -- but it landed on 0.5667 against FSD's 0.6000 with
the two arms disagreeing on ONE problem in thirty. The planner had simply learned to call
FSD_ALL. Giving a model a strong tool and letting it choose reproduces the tool.

So the division of labour changes. FSD is no longer something the planner may call; it is
applied automatically at the start of every turn. The planner sees the post-FSD state and is
asked for what FSD structurally cannot do:

  - FSD cannot change topology. It only resizes what exists.
  - FSD cannot escape its own fixed point. When it converges while still infeasible -- usually
    because the mass cap binds and every member is already at its stressed size -- it will
    return the same answer forever.
  - FSD applies one rule uniformly and cannot spend margin unevenly on purpose.

The planner's tools are therefore the non-FSD ones, plus the ability to decline. An episode in
which the planner declines every turn is exactly the FSD baseline, so this arm starts from
FSD's trajectory and can only add to it. Both controls run in this same script.
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

MAX_STEPS = 4

TOOLS_DOC = """A fully-stressed sizing pass has ALREADY been applied to this structure this
turn, and will be applied again at the start of every following turn. It rescales every member
towards its required factor of safety: members below requirement grow, members far above it
shrink. You do not need to ask for it and you should not try to reproduce it.

Your job is what that rule cannot do:

  - it only resizes existing members, so it can never change the load paths
  - it applies one uniform rule, so it cannot deliberately spend margin unevenly
  - when it converges while still infeasible -- typically the mass cap binding with every
    member already at its stressed size -- it returns the same structure forever

Emit zero to two tool calls, each on its own line as <tool>...</tool>:

  ADD_MEMBER(joint1, joint2)      add a load path
  REMOVE_MEMBER(i)                delete a member that is carrying little and costing mass
  MOVE_JOINT(joint, x, y)         reshape the geometry
  TRIM(fos_threshold, factor)     shrink everything above fos_threshold by factor (<1)
  SCALE([ids], factor)            override the sizing on specific members

If the sizing pass is making progress on its own and you see nothing structural worth changing,
emit <tool>PASS()</tool> and let it work. Declining is a real option and often the right one.
Say in one sentence what is currently blocking feasibility -- strength or mass -- then act."""


def episode(spec, arm, region, token, k, max_steps):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    pid = spec.get("problem_id", "")
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))

    if arm == "fsd":
        for _ in range(max_steps):
            if state.get("is_feasible"):
                return {"problem_id": pid, "arm": arm, "feasible": True}
            nxt = fsd_pass(truss, goals)
            if nxt is truss:
                break
            truss = nxt
            state = _analyze_truss(truss, goals)
        return {"problem_id": pid, "arm": arm, "feasible": bool(state.get("is_feasible"))}

    system = DP.TRUSS_SYSTEM_PROMPT + "\n\n" + TOOLS_DOC
    history, acted, passed = [], 0, 0
    for _ in range(max_steps):
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "acted": acted, "passed": passed}
        # --- the default move, every turn, before the planner is consulted
        nxt = fsd_pass(truss, goals)
        if nxt is not truss:
            truss = nxt
            state = _analyze_truss(truss, goals)
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "acted": acted, "passed": passed}
        tbl = build_table(truss, true_fos(truss))
        obs = ("A fully-stressed sizing pass has just been applied. Resulting state:\n"
               + DP.format_eval_result(state) + tbl)
        convo = [{"role": "user", "content": [{"text":
                  DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                  + DP.format_eval_result(initial) + (tbl if not history else "")}]}]
        for h in history:
            convo.append({"role": "assistant", "content": [{"text": h["t"]}]})
            convo.append({"role": "user", "content": [{"text": "[Simulation Result]\n" + h["o"]}]})
        convo.append({"role": "user", "content": [{"text": obs}]}) if not history else None
        text = H.call(convo, system, SONNET45, region, token) or ""
        calls = [(n, a) for (n, a) in TOOL_RE.findall(text)
                 if n not in ("FSD_ALL", "FSD_ON")]
        if not calls:
            passed += 1
            history.append({"t": text.strip()[:1500], "o": obs})
            continue
        cur = truss
        for nm, argstr in calls[:2]:
            try:
                cand = apply_tool(cur, goals, nm, argstr)
            except Exception:
                cand = None
            if cand is not None:
                cur = cand
        if cur is truss:
            passed += 1
            history.append({"t": text.strip()[:1500], "o": obs})
            continue
        # keep the planner's edit only if it does not make things worse on the governing margin
        cand_state = _analyze_truss(cur, goals)
        B = float(goals.get("maximum_mass", float("inf")))

        def score(st):
            fb = float(st.get("fos_buckling", 0) or 0); fy = float(st.get("fos_yielding", 0) or 0)
            ms = float(st.get("mass", 9e9) or 9e9)
            return min(fb / gb, fy / gy) - (0.6 if ms > B else 0.0)
        if score(cand_state) >= score(state) - 1e-9:
            truss, state = cur, cand_state
            acted += 1
        else:
            passed += 1
        history.append({"t": text.strip()[:1500],
                        "o": DP.format_eval_result(state) + build_table(truss, true_fos(truss))})
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "acted": acted, "passed": passed}
    return {"problem_id": pid, "arm": arm, "feasible": bool(state.get("is_feasible")),
            "acted": acted, "passed": passed}


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
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/planner02.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    arms = tuple(a.arms.split(","))
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]]
    tasks = [(s, arm, a.region, token, a.k, a.max_steps) for s in specs for arm in arms]
    print("problems %d-%d | arms %s | episodes %d"
          % (a.start, a.start + a.n, ",".join(arms), len(tasks)), flush=True)
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
    ac = [r.get("acted", 0) for r in out if r["arm"] == "planner"]
    ps = [r.get("passed", 0) for r in out if r["arm"] == "planner"]
    if ac:
        print("\n  planner intervened on %.1f turns/episode, declined or was rejected on %.1f"
              % (sum(ac) / len(ac), sum(ps) / len(ps)))
