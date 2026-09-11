"""
The headline run, made reproducible.

Five tests have now put the model at or below uniform sampling from the tool library: sole
proposer (0.7674 vs 0.8465), pooled proposer (0.8953 vs 0.9023, p=0.728), pre-filter (0.8000 vs
0.8833), distribution guide (0.8333 vs 0.8333), and Opus as sole proposer (0.7833 vs 0.7833).
What wins is the tool library with rollout composition search and a potential argmax over
branches, which needs no model at all.

Before that number is used for anything it has to be reproducible, and it is not: every arm
using random_proposals seeded its generator from hash(pid), and Python randomises hash() per
process unless PYTHONHASHSEED is pinned. Within a run the arms are still comparable -- they
share the process -- but the same script run twice draws different candidates, which is not
acceptable for the headline result. zlib.crc32 is stable across processes and is what the rest
of this project already uses for per-problem seeding.

Three arms, one script, same problems, paired:

  planner   Sonnet 4.5 as sole proposer, the architecture the project set out to build
  search    the same search with uniform draws from the tool library, no model
  fsd       fully-stressed design alone, the standard heuristic
"""
import os, sys, json, math, random, zlib, argparse, time
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
from planner03 import rollout
from planner04 import TOOLS_DOC
from planner05 import step_search, FEEDBACK_DOC
from planner09 import random_proposals

MAX_STEPS = 8
NPROP = 16          # the pool size the held-out sweep settled on
NMODEL = 8


def seed_of(pid):
    """Stable across processes, unlike hash()."""
    return zlib.crc32(str(pid).encode()) & 0xffffffff


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

    rng = random.Random(seed_of(pid))
    system = DP.TRUSS_SYSTEM_PROMPT + "\n\n" + (TOOLS_DOC % (NMODEL, NMODEL)) + FEEDBACK_DOC
    history = []

    for turn in range(max_steps):
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True}
        nxt = fsd_pass(truss, goals)
        if nxt is not truss:
            truss = nxt
            state = _analyze_truss(truss, goals)
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True}

        remaining = max_steps - turn - 1
        if arm == "search":
            chosen = random_proposals(truss, rng, NPROP)
            text = ""
        else:
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
            chosen = [(nm, a) for (nm, a) in TOOL_RE.findall(text)
                      if nm not in ("FSD_ALL", "FSD_ON")][:NMODEL]

        ok, sel, note = step_search(truss, goals, chosen, remaining, gb, gy)
        if ok:
            return {"problem_id": pid, "arm": arm, "feasible": True}
        if sel is not None:
            truss = sel
            state = _analyze_truss(truss, goals)
        if arm != "search":
            history.append({"t": text.strip()[:1200],
                            "o": ("Search outcome: %s.\n" % note)
                                 + DP.format_eval_result(state)
                                 + build_table(truss, true_fos(truss))})
    ok, _, _ = rollout(truss, goals, 2)
    return {"problem_id": pid, "arm": arm, "feasible": ok}


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
    ap.add_argument("--arms", default="planner,search,fsd")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/planner13.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    arms = tuple(a.arms.split(","))
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]]
    tasks = [(s, arm, a.region, token, a.k, a.max_steps) for s in specs for arm in arms]
    print("problems %d-%d | pool %d | arms %s | episodes %d"
          % (a.start, a.start + a.n, NPROP, ",".join(arms), len(tasks)), flush=True)
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
        print("  %-8s %.4f" % (arm, rate[arm]))
    for arm in arms:
        if arm == "fsd":
            continue
        u, d, p = sign_test([by["fsd"][x] for x in pids], [by[arm][x] for x in pids])
        print("\n  %-8s vs FSD control: %+.4f   discordant %d-%d   exact sign p = %.3g"
              % (arm, rate[arm] - rate["fsd"], u, d, p))
    if "planner" in arms and "search" in arms:
        u, d, p = sign_test([by["search"][x] for x in pids], [by["planner"][x] for x in pids])
        print("\n  planner vs search (model vs no model): %+.4f   discordant %d-%d   p = %.3g"
              % (rate["planner"] - rate["search"], u, d, p))
