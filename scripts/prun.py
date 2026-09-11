"""Resumable runner for any planner variant, with bounded model calls.

Takes a planner module by name, patches its model call to the bounded one, skips
(problem, arm) pairs already present in the output file, appends the rest, and prints the
paired summary. Used so a stalled run costs the tail rather than the whole evaluation.

  python scripts/prun.py --module planner07 --n 430 --start 150 --max-steps 8 --out <file>
"""
import os, sys, json, argparse, importlib, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import api_grammar_2x2 as H
import safecall
from probe26_presentation import sign_test

H.call = safecall.call            # every planner module calls H.call; bound it once here

ap = argparse.ArgumentParser()
ap.add_argument("--module", required=True)
ap.add_argument("--n", type=int, default=430)
ap.add_argument("--start", type=int, default=150)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--max-steps", type=int, default=8)
ap.add_argument("--arms", default="planner,fsd")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
ap.add_argument("--out", required=True)
a = ap.parse_args()

mod = importlib.import_module(a.module)
mod.H.call = safecall.call        # the module holds its own reference

token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
    open(os.path.expanduser("~/.bedrock_token")).read().strip()
arms = tuple(a.arms.split(","))
specs = [json.load(open(f)) for f in
         sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]]
done = safecall.done_keys(a.out)
tasks = [(s, arm, a.region, token, a.k, a.max_steps)
         for s in specs for arm in arms
         if (s.get("problem_id"), arm) not in done]
print("%s | problems %d-%d | %d already done | %d to run"
      % (a.module, a.start, a.start + a.n, len(done), len(tasks)), flush=True)

t0 = time.time()
if tasks:
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "a") as fh:
        for i, r in enumerate(ex.map(mod.work, tasks)):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 40 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
print("wall %.0fs\n" % (time.time() - t0))

rows = [json.loads(l) for l in open(a.out) if l.strip()]
by = {arm: {r["problem_id"]: (1.0 if r["feasible"] else 0.0)
            for r in rows if r["arm"] == arm} for arm in arms}
pids = sorted(set.intersection(*[set(by[x]) for x in arms]))
rate = {arm: sum(by[arm][p] for p in pids) / max(len(pids), 1) for arm in arms}
print("paired on %d problems\n" % len(pids))
for arm in arms:
    print("  %-9s %.4f" % (arm, rate[arm]))
if "fsd" in arms and "planner" in arms:
    u, d, p = sign_test([by["fsd"][x] for x in pids], [by["planner"][x] for x in pids])
    print("\n  planner vs FSD control: %+.4f   discordant %d-%d   exact sign p = %.3g"
          % (rate["planner"] - rate["fsd"], u, d, p))
