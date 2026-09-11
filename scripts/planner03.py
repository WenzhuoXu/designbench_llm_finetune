"""
Planner v3: the planner proposes, the tool library rolls out, the argmax decides.

v2 made fully-stressed sizing the automatic default and asked the planner only for what that
rule cannot do. It came out at 0.6000 against FSD's 0.6000, discordant 0-0 -- identical on every
problem. The planner did propose edits, on 0.8 turns per episode, and the accept-guard threw
them away: it required the governing margin not to fall immediately, and a structural change
costs mass now to pay later. The guard was rejecting precisely the moves worth making.

So acceptance stops being greedy. Each turn the planner proposes SEVERAL candidate
interventions. Every candidate is applied and then rolled forward to the horizon by the
procedural FSD loop -- no model calls, only analysis -- and so is the do-nothing branch. The
branch that reaches feasibility, or gets furthest under the potential, is kept.

That makes the model a proposer over a structured action space and the tool library the
evaluator, with the argmax over compositions settling what actually happens. A turn where no
proposal beats the do-nothing rollout leaves the FSD trajectory untouched, so the arm still
cannot lose to its own baseline.
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
NPROP = 4

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

Each proposal is applied to a separate copy of the structure and then carried forward by the
sizing rule to the end of the episode; the branch that does best is kept, and a do-nothing
branch competes alongside them. So propose genuinely different ideas rather than variations of
one, and include a bold structural change even if you are unsure -- a proposal that turns out
worse than doing nothing costs you nothing, because it will simply be discarded.

Say in one sentence what is blocking feasibility -- strength or mass -- then give your %d lines."""


def rollout(truss, goals, steps):
    """Carry a structure forward with the procedural sizing rule. No model calls."""
    st = _analyze_truss(truss, goals)
    cur = truss
    for _ in range(max(0, steps)):
        if st.get("is_feasible"):
            return True, st, cur
        nxt = fsd_pass(cur, goals)
        if nxt is cur:
            break
        cur = nxt
        st = _analyze_truss(cur, goals)
    return bool(st.get("is_feasible")), st, cur


def potential(st, goals, gb, gy):
    B = float(goals.get("maximum_mass", float("inf")))
    fb = float(st.get("fos_buckling", 0) or 0)
    fy = float(st.get("fos_yielding", 0) or 0)
    ms = float(st.get("mass", 9e9) or 9e9)
    strength = min(fb / gb, fy / gy)
    over = max(0.0, (ms / B) - 1.0) if (B and math.isfinite(B) and B > 0) else 0.0
    return min(strength, 1.0) - 0.75 * over


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

        # do-nothing branch competes on equal terms
        base_ok, base_st, base_truss = rollout(truss, goals, remaining)
        if base_ok:
            return {"problem_id": pid, "arm": arm, "feasible": True, "wins": wins, "props": props}
        best = (potential(base_st, goals, gb, gy), None, None)
        for nm, argstr in calls:
            try:
                cand = apply_tool(truss, goals, nm, argstr)
            except Exception:
                cand = None
            if cand is None or cand is truss:
                continue
            ok, st, rolled = rollout(cand, goals, remaining)
            if ok:
                return {"problem_id": pid, "arm": arm, "feasible": True,
                        "wins": wins + 1, "props": props}
            v = potential(st, goals, gb, gy)
            if v > best[0] + 1e-9:
                best = (v, cand, nm)
        if best[1] is not None:
            truss = best[1]
            state = _analyze_truss(truss, goals)
            wins += 1
        history.append({"t": text.strip()[:1200],
                        "o": DP.format_eval_result(state) + build_table(truss, true_fos(truss))})
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
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--arms", default="planner,fsd")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/planner03.jsonl"))
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
    w = [r.get("wins", 0) for r in out if r["arm"] == "planner"]
    pr = [r.get("props", 0) for r in out if r["arm"] == "planner"]
    if w:
        print("\n  proposals %.1f/episode, of which %.1f beat the do-nothing rollout"
              % (sum(pr) / len(pr), sum(w) / len(w)))
