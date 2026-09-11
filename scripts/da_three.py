"""
All three domains, fixed exponent against searched exponent, each with its own control.

The portable core carried the truss's response exponent, 3.0, as a hard-coded constant.
da_search3 makes it a searched parameter that the episode adopts when a SIZE_PASS candidate
wins. This asks whether that helps, hurts, or does nothing, in each domain, against that
domain's own heuristic in the same run:

  search_fixed    exponent pinned at 3.0        (the current implementation)
  search_learned  exponent searched per episode (da_search3)
  heuristic       the domain's own sizing rule

The truss should be indifferent -- 3.0 is its own number. The synthetic domain responds around
2.2 and had the weakest margin of the three. The pipe domain responds as the fifth power and had
to be worked around locally. If the framework is portable, the searched version should match the
fixed one where the constant happened to fit and beat it where it did not, with no per-domain
configuration anywhere.
"""
import os, sys, json, math, random, zlib, argparse, time, statistics
from pathlib import Path
from multiprocessing import Pool
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts"),
          str(PROJECT / "design_agent")):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
from da_synth import SynthDomain
from da_pipe import PipeDomain
import da_search3 as S3
from da_search import size_pass
from da_search2 import search_episode as search_fixed
from probe26_presentation import sign_test

POOL, HORIZON = 32, 12


def seed_of(x):
    return zlib.crc32(str(x).encode()) & 0xffffffff


def make(domain, key):
    if domain == "truss":
        d = get_domain("truss")
        return d, d.load(json.load(open(key))), "r"
    if domain == "synth":
        d = SynthDomain()
        return d, d.load({"seed": key, "n": 10}), "x"
    d = PipeDomain()
    return d, d.load({"seed": key, "n": 14}), "d"


def heuristic(dom, st, param, steps):
    cur = st
    for _ in range(steps):
        if dom.feasible(cur):
            return 1
        nxt = size_pass(dom, cur, param)
        if nxt is cur:
            break
        cur = nxt
    return 1 if dom.feasible(cur) else 0


def _work(t):
    domain, key, arm = t
    try:
        dom, st, param = make(domain, key)
        rng = random.Random(seed_of("%s%s" % (domain, key)))
        pid = "%s:%s" % (domain, key)
        if arm == "heuristic":
            return (domain, arm, pid, float(heuristic(dom, st, param, HORIZON)), 3.0, dom.calls)
        if arm == "search_fixed":
            y = search_fixed(dom, st, param, HORIZON, rng, POOL)
            return (domain, arm, pid, float(y), 3.0, dom.calls)
        y, exp = S3.search_episode(dom, st, param, HORIZON, rng, POOL)
        return (domain, arm, pid, float(y), exp, dom.calls)
    except Exception:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", type=int, default=430)
    ap.add_argument("--start", type=int, default=150)
    ap.add_argument("--ns", type=int, default=500)
    ap.add_argument("--np", type=int, default=500)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/da_three.jsonl"))
    a = ap.parse_args()
    files = [str(f) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.nt]]
    arms = ("search_learned", "search_fixed", "heuristic")
    jobs = ([("truss", f, arm) for f in files for arm in arms]
            + [("synth", i, arm) for i in range(a.ns) for arm in arms]
            + [("pipe", i, arm) for i in range(a.np) for arm in arms])
    print("%d jobs across three domains" % len(jobs), flush=True)
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = [r for r in pool.map(_work, jobs, chunksize=4) if r]
    print("done, %.0fs\n" % (time.time() - t0))
    with open(a.out, "w") as fh:
        for dm, arm, pid, y, e, c in rows:
            fh.write(json.dumps({"domain": dm, "arm": arm, "problem_id": pid,
                                 "feasible": bool(y), "exp": e, "calls": c}) + "\n")

    for dm in ("truss", "synth", "pipe"):
        sub = [r for r in rows if r[0] == dm]
        if not sub:
            continue
        by = {arm: {r[2]: r[3] for r in sub if r[1] == arm} for arm in arms}
        pids = sorted(set.intersection(*[set(by[x]) for x in arms]))
        if not pids:
            continue
        rate = {arm: sum(by[arm][p] for p in pids) / len(pids) for arm in arms}
        cost = {arm: statistics.mean([r[5] for r in sub if r[1] == arm]) for arm in arms}
        print("--- %s (paired on %d) ---" % (dm, len(pids)))
        for arm in arms:
            print("  %-15s %.4f   mean evals %7.0f" % (arm, rate[arm], cost[arm]))
        for arm in ("search_learned", "search_fixed"):
            u, d, p = sign_test([by["heuristic"][x] for x in pids], [by[arm][x] for x in pids])
            print("    %-15s vs heuristic: %+.4f  %d-%d  p=%.2g"
                  % (arm, rate[arm] - rate["heuristic"], u, d, p))
        u, d, p = sign_test([by["search_fixed"][x] for x in pids],
                            [by["search_learned"][x] for x in pids])
        print("    learned vs fixed exponent: %+.4f  discordant %d-%d  p=%.3g"
              % (rate["search_learned"] - rate["search_fixed"], u, d, p))
        ex = [r[4] for r in sub if r[1] == "search_learned"]
        if ex:
            ex_sorted = sorted(ex)
            print("    exponent the search settled on: median %.2f (quartiles %.2f to %.2f)\n"
                  % (statistics.median(ex), ex_sorted[len(ex) // 4], ex_sorted[3 * len(ex) // 4]))
