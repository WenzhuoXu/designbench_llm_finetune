"""
Planner v11: the model as a pre-filter, paying for itself in evaluations.

At matched proposal count the model is inert: mixed (8 model + 8 random) scores 0.8953 against
random16's 0.9023, discordant 18-15, p=0.728. Its eight suggestions are worth exactly eight more
random draws. But pool size IS the lever -- random8 to random16 is +4.4pp at p=0.0003 -- and the
thing that limits pool size is the rollout, since every candidate costs a full FSD continuation
in FEA calls.

That gives the model a job it can actually do better than sampling: decide what is worth
evaluating. Draw a large pool cheaply, let the model read the candidate list against the current
state, and roll out only the subset it picks. The model never sets a magnitude, never executes,
and never sees a rollout result -- it only spends the evaluation budget.

Arms, all in this script on the same problems, with rollout budget held equal between the first
two:

  prefilter   64 random candidates drawn, the model selects 16, those 16 are rolled out
  random16    16 random candidates drawn and rolled out          -- matched FEA budget
  random64    all 64 rolled out                                  -- what 4x the budget buys
  fsd         the sizing rule alone

If prefilter beats random16, the model is recognising good moves without evaluating them, which
is worth something no amount of sampling provides. If it ties, the model cannot tell a good
candidate from a bad one even when handed both.
"""
import os, sys, json, math, random, re, argparse, time
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
from planner01 import fsd_pass, SONNET45
from planner03 import rollout
from planner05 import step_search
from planner09 import random_proposals

MAX_STEPS = 8
POOL = 64
KEEP = 16
IDX_RE = re.compile(r"\d+")

SELECT_SYSTEM = """You are choosing which candidate modifications are worth simulating.

A fully-stressed sizing pass is applied automatically every turn: it rescales every member
towards its required factor of safety, growing those below requirement and shrinking those far
above it. It cannot change load paths, cannot spend margin unevenly on purpose, and once it
converges while still infeasible -- usually the mass cap binding -- it returns the same
structure forever.

You are shown the current state, the per-member table, and a numbered list of candidate moves.
Simulating a candidate is expensive, so only %d of them will be evaluated. Choose the %d you
judge most likely to move this structure towards satisfying every factor of safety within the
mass cap.

Reply with nothing but the %d numbers, comma separated. No explanation."""


def render_candidates(cands):
    return "\n".join("  %d. %s(%s)" % (i, nm, a.strip()) for i, (nm, a) in enumerate(cands))


def episode(spec, arm, region, token, k, max_steps):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    pid = spec.get("problem_id", "")
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))

    if arm == "fsd":
        ok, _, _ = rollout(truss, goals, max_steps)
        return {"problem_id": pid, "arm": arm, "feasible": ok}

    rng = random.Random(abs(hash(pid)) & 0xffffffff)
    system = SELECT_SYSTEM % (KEEP, KEEP, KEEP)
    picked_total, turns_used = 0, 0

    for turn in range(max_steps):
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "picked": picked_total, "turns": turns_used}
        nxt = fsd_pass(truss, goals)
        if nxt is not truss:
            truss = nxt
            state = _analyze_truss(truss, goals)
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "picked": picked_total, "turns": turns_used}

        remaining = max_steps - turn - 1
        if arm == "random16":
            chosen = random_proposals(truss, rng, KEEP)
        elif arm == "random64":
            chosen = random_proposals(truss, rng, POOL)
        else:
            pool = random_proposals(truss, rng, POOL)
            tbl = build_table(truss, true_fos(truss))
            body = ("A fully-stressed sizing pass has just been applied. Resulting state:\n"
                    + DP.format_eval_result(state) + tbl
                    + "\n\nCandidate moves:\n" + render_candidates(pool)
                    + "\n\nWhich %d should be simulated? Numbers only." % KEEP)
            convo = [{"role": "user", "content": [{"text":
                      DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                      + DP.format_eval_result(initial) + "\n\n" + body}]}]
            text = H.call(convo, system, SONNET45, region, token) or ""
            idx, seen = [], set()
            for m in IDX_RE.findall(text):
                v = int(m)
                if 0 <= v < len(pool) and v not in seen:
                    seen.add(v); idx.append(v)
                if len(idx) >= KEEP:
                    break
            if len(idx) < KEEP:                       # top up from the pool, in order
                for v in range(len(pool)):
                    if v not in seen:
                        idx.append(v); seen.add(v)
                    if len(idx) >= KEEP:
                        break
            chosen = [pool[v] for v in idx]
            picked_total += len(idx)
            turns_used += 1

        ok, sel, note = step_search(truss, goals, chosen, remaining, gb, gy)
        if ok:
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "picked": picked_total, "turns": turns_used}
        if sel is not None:
            truss = sel
            state = _analyze_truss(truss, goals)
    ok, _, _ = rollout(truss, goals, 2)
    return {"problem_id": pid, "arm": arm, "feasible": ok,
            "picked": picked_total, "turns": turns_used}


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
    ap.add_argument("--arms", default="prefilter,random16,random64,fsd")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/planner11.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    arms = tuple(a.arms.split(","))
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]]
    tasks = [(s, arm, a.region, token, a.k, a.max_steps) for s in specs for arm in arms]
    print("problems %d-%d | pool %d keep %d | arms %s | episodes %d"
          % (a.start, a.start + a.n, POOL, KEEP, ",".join(arms), len(tasks)), flush=True)
    t0 = time.time()
    out = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w") as fh:
        for i, r in enumerate(ex.map(work, tasks)):
            out.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 60 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs\n" % (time.time() - t0))

    by = {arm: {r["problem_id"]: (1.0 if r["feasible"] else 0.0) for r in out if r["arm"] == arm}
          for arm in arms}
    pids = sorted(set.intersection(*[set(by[x]) for x in arms]))
    rate = {arm: sum(by[arm][p] for p in pids) / max(len(pids), 1) for arm in arms}
    print("paired on %d problems\n" % len(pids))
    for arm in sorted(arms, key=lambda x: -rate[x]):
        print("  %-10s %.4f" % (arm, rate[arm]))
    if "fsd" in arms:
        for arm in arms:
            if arm == "fsd":
                continue
            u, d, p = sign_test([by["fsd"][x] for x in pids], [by[arm][x] for x in pids])
            print("\n  %-10s vs FSD control: %+.4f   discordant %d-%d   exact sign p = %.3g"
                  % (arm, rate[arm] - rate["fsd"], u, d, p))
    if "prefilter" in arms and "random16" in arms:
        u, d, p = sign_test([by["random16"][x] for x in pids], [by["prefilter"][x] for x in pids])
        print("\n  prefilter vs random16 at MATCHED rollout budget: %+.4f   discordant %d-%d   p = %.3g"
              % (rate["prefilter"] - rate["random16"], u, d, p))
