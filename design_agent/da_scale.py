"""Does the gap close with instance size?

The standing claim -- asserted twice, never tested -- is that DesignBench is too small for a
planner to matter, because everything in an instance is enumerable inside one model call's
worth of simulator time. problems_hard stratifies into three size tiers (10-11, 15-16, 20-21
members), so the claim is directly checkable: if the planner's deficit against the installed
closed form NARROWS as instances grow, the substrate argument holds and larger instances are
the path. If it is flat or widens, the argument is dead and so is the direction.
"""
import sys, os, json, glob, time, statistics, collections
from math import comb
from pathlib import Path
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
import da_run as R

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
TIERS = {"small 10-11": (10, 11), "mid 15-16": (15, 16), "large 20-21": (20, 21)}
ARMS = ["base", "installed", "induced"]
BUDGET = 120


def nmem(s):
    t = s.get("topology")
    return len(t.get("members") or []) if isinstance(t, dict) else (len(t) if isinstance(t, list) else 0)


def _w(t):
    spec, arm = t
    try:
        s, c = R.run_arm(spec, arm, sim_budget=BUDGET)
        return spec["problem_id"], arm, s, c
    except Exception:
        return spec["problem_id"], arm, None, 0


def _wl(t):
    spec, model, region, tok = t
    try:
        s, c, l = R.run_planner(spec, model, region, tok, sim_budget=BUDGET)
        return spec["problem_id"], "planner", s, c, l
    except Exception:
        return spec["problem_id"], "planner", None, 0, 0


def mc(a, b, keys):
    hi = sum(1 for k in keys if a.get(k) and not b.get(k))
    lo = sum(1 for k in keys if b.get(k) and not a.get(k))
    d = hi + lo
    p = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
    return hi, lo, 100 * (hi - lo) / max(len(keys), 1), p


if __name__ == "__main__":
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    use_llm = "--llm" in sys.argv
    allf = sorted((DB / "data/problems_hard").glob("*.json"))
    pool_by_tier = collections.defaultdict(list)
    for f in allf:
        try:
            s = json.load(open(f))
        except Exception:
            continue
        m = nmem(s)
        for name, (lo, hi) in TIERS.items():
            if lo <= m <= hi:
                pool_by_tier[name].append(s)
    tok = model = region = None
    if use_llm:
        tok = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
            open(os.path.expanduser("~/.bedrock_token")).read().strip()
        model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        region = os.environ.get("AWS_REGION", "us-west-2")

    print("scaling test, %d problems/tier, simulator budget %d/episode\n" % (per, BUDGET))
    for name in TIERS:
        specs = pool_by_tier[name][:per]
        t0 = time.time()
        with Pool(40) as pool:
            res = pool.map(_w, [(s, a) for s in specs for a in ARMS], chunksize=1)
        solved = collections.defaultdict(dict)
        for pid, arm, s, c in res:
            if s is not None:
                solved[arm][pid] = s
        keys = sorted(set.intersection(*[set(solved[a]) for a in ARMS]))
        line = "  %-12s n=%-3d " % (name, len(keys))
        for a in ARMS:
            line += "%s %.3f  " % (a[:9], statistics.mean(solved[a][k] for k in keys))
        if use_llm:
            with ThreadPoolExecutor(max_workers=20) as ex:
                pres = list(ex.map(_wl, [(s, model, region, tok) for s in specs]))
            ps = {pid: s for pid, _, s, _, _ in pres if s is not None}
            kk = sorted(set(ps) & set(keys))
            line += "planner %.3f  " % statistics.mean(ps[k] for k in kk)
            hi, lo, d, p = mc(ps, solved["installed"], kk)
            line += "| planner-installed %+5.1fpp p=%.1e" % (d, p)
        else:
            hi, lo, d, p = mc(solved["induced"], solved["installed"], keys)
            line += "| induced-installed %+5.1fpp p=%.1e" % (d, p)
        hi2, lo2, d2, p2 = mc(solved["installed"], solved["base"], keys)
        line += " | installed-base %+5.1fpp" % d2
        print(line + "   (%.0fs)" % (time.time() - t0), flush=True)
