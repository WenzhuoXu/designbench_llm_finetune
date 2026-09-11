"""
The planner: an LLM composing tools from the library, with fully-stressed design as ONE TOOL.

Everything tried before either let the model emit raw grammar actions (0.3326) or took the
model's chosen members and overwrote its sizing (0.3767) -- and both lose to simply running
fully-stressed design over every member (0.4698). The architecture was never actually built:
the model was never handed FSD as a tool it could CALL, alongside its own targeted edits, and
allowed to decide when to use which.

That is what this is. Each turn the planner sees the state and the per-member table and emits
a short plan of tool calls, executed in order:

  FSD_ALL(margin)            fully-stressed resize of every member
  FSD_ON([ids], margin)      fully-stressed resize of a chosen subset
  SCALE([ids], factor)       direct scaling where the planner disagrees with FSD
  TRIM(fos_threshold, f)     shrink members far above requirement, to free mass
  REMOVE_MEMBER(i) / ADD_MEMBER(j1, j2) / MOVE_JOINT(j, x, y)   topology, passed through

Controls run in the SAME script on the SAME problems, so nothing is compared to a remembered
number: fsd (the 0.4698 baseline) and llm (raw grammar, the 0.3326 baseline).
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
from probe27_decompose import fsd_factor
from probe26_presentation import sign_test

SONNET45 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
MAX_STEPS = 4
ARMS = ("planner", "fsd", "llm")

TOOL_RE = re.compile(
    r"(FSD_ALL|FSD_ON|SCALE|TRIM|REMOVE_MEMBER|ADD_MEMBER|MOVE_JOINT)\s*\(([^)]*)\)")

TOOLS_DOC = """You have a TOOL LIBRARY. Each turn, emit a short plan: one to three tool calls,
each on its own line wrapped in <tool>...</tool>. They are executed in the order you give.

  FSD_ALL(margin)
      Fully-stressed resize of EVERY member at once. Each member's radius is scaled by
      max( (1.5*margin/FOS_buckling)^(1/3), 1.5*margin/FOS_yielding ), clipped to [0.7, 2.0].
      Members below requirement grow; members far above it shrink, freeing mass.
      margin is a safety cushion, typically 1.0 to 1.15. This is a strong general move.

  FSD_ON([ids], margin)
      The same rule applied only to the members you list. Use when a global pass would
      disturb members you want left alone.

  SCALE([ids], factor)
      Multiply the radius of the listed members by factor, 0.7 to 2.0. Use when you disagree
      with what the fully-stressed rule would do for specific members.

  TRIM(fos_threshold, factor)
      Shrink every member whose governing FOS exceeds fos_threshold by factor (<1).
      Frees mass when the mass constraint is what is blocking feasibility.

  REMOVE_MEMBER(i)
  ADD_MEMBER(joint1, joint2)
  MOVE_JOINT(joint, x, y)
      Topology changes.

Think briefly about whether the binding problem is strength or mass, then emit your plan."""


def apply_tool(truss, goals, name, argstr):
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    nums = re.findall(r"-?\d+\.?\d*", argstr)
    ids = [int(x) for x in re.findall(r"\d+", (re.search(r"\[([^\]]*)\]", argstr) or
                                              re.match(r"", "")).group(1))] \
        if re.search(r"\[([^\]]*)\]", argstr) else []
    cur = truss
    if name in ("FSD_ALL", "FSD_ON"):
        tail = re.sub(r"\[[^\]]*\]", "", argstr)
        mnums = re.findall(r"-?\d+\.?\d*", tail)
        margin = float(mnums[0]) if mnums else 1.05
        margin = min(1.4, max(0.9, margin))
        tf = true_fos(cur)
        targets = ids if (name == "FSD_ON" and ids) else list(range(len(cur.members)))
        for i in targets:
            if i >= len(tf):
                continue
            f = fsd_factor(tf[i][0], tf[i][1], gb, gy, margin)
            if abs(f - 1.0) < 1e-4:
                continue
            nt = H.apply_candidate(cur, "SCALE_PARAM(%d, radius, %.4f)" % (i, f))
            if nt is not None:
                cur = nt
        return cur
    if name == "SCALE":
        tail = re.sub(r"\[[^\]]*\]", "", argstr)
        mnums = re.findall(r"-?\d+\.?\d*", tail)
        f = float(mnums[0]) if mnums else 1.1
        f = min(2.0, max(0.7, f))
        for i in ids:
            nt = H.apply_candidate(cur, "SCALE_PARAM(%d, radius, %.4f)" % (i, f))
            if nt is not None:
                cur = nt
        return cur
    if name == "TRIM":
        thr = float(nums[0]) if nums else 3.0
        f = float(nums[1]) if len(nums) > 1 else 0.85
        f = min(1.0, max(0.7, f))
        tf = true_fos(cur)
        for i in range(len(cur.members)):
            try:
                gov = min(tf[i][0] / gb, tf[i][1] / gy)
            except Exception:
                continue
            if math.isfinite(gov) and gov > thr:
                nt = H.apply_candidate(cur, "SCALE_PARAM(%d, radius, %.4f)" % (i, f))
                if nt is not None:
                    cur = nt
        return cur
    if name == "REMOVE_MEMBER" and nums:
        return H.apply_candidate(cur, "REMOVE_MEMBER(%d)" % int(float(nums[0]))) or cur
    if name == "ADD_MEMBER" and len(nums) >= 2:
        # the executor wants the shape with parenthesised params, not comma-separated
        return H.apply_candidate(
            cur, "ADD_MEMBER(%d, %d, 6061_T6_Aluminum, Pipe(r=0.030, t=0.004))"
            % (int(float(nums[0])), int(float(nums[1])))) or cur
    if name == "MOVE_JOINT" and len(nums) >= 3:
        # three coordinates; the executor ignores any old position and uses the new one
        j = int(float(nums[0]))
        z = float(nums[3]) if len(nums) >= 4 else 0.0
        return H.apply_candidate(cur, "MOVE_JOINT(%d, [%.4f, %.4f, %.4f])"
                                 % (j, float(nums[1]), float(nums[2]), z)) or cur
    return cur


def fsd_pass(truss, goals, margin=1.05):
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    tf = true_fos(truss)
    cur = truss
    for i in range(len(truss.members)):
        f = fsd_factor(tf[i][0], tf[i][1], gb, gy, margin)
        if abs(f - 1.0) < 1e-6:
            continue
        nt = H.apply_candidate(cur, "SCALE_PARAM(%d, radius, %.4f)" % (i, f))
        if nt is None:
            return cur
        cur = nt
    return cur


def episode(spec, arm, region, token, k, max_steps):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    pid = spec.get("problem_id", "")
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

    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    if arm == "planner":
        system = DP.TRUSS_SYSTEM_PROMPT + "\n\n" + TOOLS_DOC
    else:
        system = DP.TRUSS_SYSTEM_PROMPT + (
            f"\n\nEach turn, propose {k} DIFFERENT candidate moves rather than one. Think "
            f"briefly, then emit exactly {k} lines of the form <action>...</action>."
        ) + H.COMPOUND_RULE
    history = []
    ncalls = 0
    for _ in range(max_steps):
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True, "tools": ncalls}
        tbl = build_table(truss, true_fos(truss))
        obs = DP.format_eval_result(state) + tbl
        convo = [{"role": "user", "content": [{"text":
                  DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                  + DP.format_eval_result(initial) + (tbl if not history else "")}]}]
        for h in history:
            convo.append({"role": "assistant", "content": [{"text": h["t"]}]})
            convo.append({"role": "user", "content": [{"text":
                          "[Simulation Result]\nSTRUCTURAL ANALYSIS RESULT:\n" + h["o"]}]})
        text = H.call(convo, system, SONNET45, region, token) or ""

        if arm == "planner":
            calls = TOOL_RE.findall(text)
            if not calls:
                history.append({"t": text.strip()[:2000],
                                "o": obs + "\nNo tool call parsed. Emit <tool>NAME(args)</tool>."})
                continue
            cur = truss
            for nm, argstr in calls[:3]:
                try:
                    cur = apply_tool(cur, goals, nm, argstr) or cur
                except Exception:
                    pass
                ncalls += 1
            if cur is truss:
                history.append({"t": text.strip()[:2000], "o": obs})
                continue
            truss = cur
            state = _analyze_truss(truss, goals)
        else:
            cands = H.parse_candidates(text, k, True)
            if not cands:
                history.append({"t": text.strip()[:2000],
                                "o": obs + f"\nNo <action> parsed. Emit exactly {k} lines."})
                continue
            scored = []
            for c in cands:
                nt = H.apply_candidate(truss, c)
                if nt is not None:
                    scored.append((nt, _analyze_truss(nt, goals)))
            if not scored:
                history.append({"t": text.strip()[:2000], "o": obs})
                continue
            B = float(goals.get("maximum_mass", float("inf")))

            def key(t):
                st = t[1]
                fb = float(st.get("fos_buckling", 0) or 0); fy = float(st.get("fos_yielding", 0) or 0)
                ms = float(st.get("mass", 9e9) or 9e9)
                return min(fb / gb, fy / gy) - (0.5 if ms > B else 0.0)
            truss, state = max(scored, key=key)
        history.append({"t": text.strip()[:2000],
                        "o": DP.format_eval_result(state) + build_table(truss, true_fos(truss))})
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True, "tools": ncalls}
    return {"problem_id": pid, "arm": arm, "feasible": bool(state.get("is_feasible")),
            "tools": ncalls}


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
    ap.add_argument("--arms", default="planner,fsd,llm")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/planner01.jsonl"))
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
    print("paired on %d problems\n" % len(pids))
    rate = {arm: sum(by[arm][p] for p in pids) / max(len(pids), 1) for arm in arms}
    for arm in arms:
        print("  %-9s %.4f" % (arm, rate[arm]))
    if "fsd" in arms and "planner" in arms:
        u, d, p = sign_test([by["fsd"][x] for x in pids], [by["planner"][x] for x in pids])
        print("\n  planner vs FSD control: %+.4f   discordant %d-%d   exact sign p = %.3g"
              % (rate["planner"] - rate["fsd"], u, d, p))
    if "llm" in arms and "planner" in arms:
        u, d, p = sign_test([by["llm"][x] for x in pids], [by["planner"][x] for x in pids])
        print("  planner vs plain LLM  : %+.4f   discordant %d-%d   exact sign p = %.3g"
              % (rate["planner"] - rate["llm"], u, d, p))
    tc = [r.get("tools", 0) for r in out if r["arm"] == "planner"]
    if tc:
        print("\n  planner tool calls per episode: mean %.1f" % (sum(tc) / len(tc)))
