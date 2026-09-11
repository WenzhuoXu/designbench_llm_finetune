"""
One benchmark, four arms, same problems, same run: the search against the best heuristic there is.

The portability run scored the generic search at 0.8814 against a generic heuristic at 0.4163.
That control is weaker than it should be: the generic size_pass moves every element by
(margin/m)^(1/3), while truss-native fully-stressed design uses the two-mechanism formula
max((1.5m/FOS_b)^(1/3), 1.5m/FOS_y) and reaches 0.4930 on the same problems. Scoring against
the weaker of two available heuristics inflates the margin, so this runs both.

  search        generic search + domain-registered tools (da_search2)
  fsd_native    truss-native fully-stressed design -- the strongest heuristic available here
  heur_generic  the generic sizing rule, for the portability comparison
  llm           Sonnet 4.5 proposing into the same search, kept live per standing rule

CPU arms run in a process pool, the LLM arm in a thread pool since it is I/O bound; both cover
the same problem list in the same invocation, so nothing is compared against a remembered
number.
"""
import os, sys, json, math, random, zlib, argparse, time
from pathlib import Path
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts"),
          str(PROJECT / "design_agent")):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
import da_search as S
import da_search2 as S2
from planner01 import fsd_pass
from planner03 import rollout as native_rollout
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals
from probe26_presentation import sign_test

STEPS = 8
NPROP = 16
CPU_ARMS = ("search", "fsd_native", "heur_generic")


def seed_of(x):
    return zlib.crc32(str(x).encode()) & 0xffffffff


def _cpu(t):
    key, arm = t
    try:
        if arm == "fsd_native":
            truss, goals = _load_truss_and_goals(json.load(open(key)))
            ok, _, _ = native_rollout(truss, goals, STEPS)
            return (arm, str(key), 1.0 if ok else 0.0)
        dom = get_domain("truss")
        st = dom.load(json.load(open(key)))
        rng = random.Random(seed_of(key))
        if arm == "heur_generic":
            y = S.heuristic_episode(dom, st, "r", STEPS)
        else:
            y = S2.search_episode(dom, st, "r", STEPS, rng, NPROP)
        return (arm, str(key), float(y))
    except Exception:
        return None


def _llm(t):
    key, region, token = t
    try:
        import planner13
        spec = json.load(open(key))
        r = planner13.episode(spec, "planner", region, token, 8, STEPS)
        return ("llm", str(key), 1.0 if r.get("feasible") else 0.0)
    except Exception:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--procs", type=int, default=44)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/da_bench.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    files = [str(f) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]]
    print("problems %d-%d | %d files | arms %s%s"
          % (a.start, a.start + a.n, len(files), ",".join(CPU_ARMS),
             "" if a.no_llm else ",llm"), flush=True)
    t0 = time.time()
    rows = []
    with Pool(a.procs) as pool:
        rows += [r for r in pool.map(_cpu, [(f, arm) for f in files for arm in CPU_ARMS],
                                     chunksize=2) if r]
    print("  cpu arms done, %.0fs" % (time.time() - t0), flush=True)
    if not a.no_llm:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            rows += [r for r in ex.map(_llm, [(f, a.region, token) for f in files]) if r]
        print("  llm arm done, %.0fs" % (time.time() - t0), flush=True)

    with open(a.out, "w") as fh:
        for arm, pid, y in rows:
            fh.write(json.dumps({"arm": arm, "problem_id": pid, "feasible": bool(y)}) + "\n")

    arms = sorted({r[0] for r in rows})
    by = {arm: {r[1]: r[2] for r in rows if r[0] == arm} for arm in arms}
    pids = sorted(set.intersection(*[set(by[x]) for x in arms]))
    rate = {arm: sum(by[arm][p] for p in pids) / max(len(pids), 1) for arm in arms}
    print("\npaired on %d problems\n" % len(pids))
    for arm in sorted(arms, key=lambda x: -rate[x]):
        print("  %-13s %.4f" % (arm, rate[arm]))
    print()
    for other in arms:
        if other == "search":
            continue
        u, d, p = sign_test([by[other][x] for x in pids], [by["search"][x] for x in pids])
        print("  search vs %-13s %+.4f   discordant %d-%d   exact sign p = %.3g"
              % (other + ":", rate["search"] - rate[other], u, d, p))
