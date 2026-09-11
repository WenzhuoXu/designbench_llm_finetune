"""Is the installed exponent winning as physics, or as damping?

Measured per-element elasticity has median 2.32. If the best exponent is near 2.32, the
closed form is doing physics. If the optimum sits well above it, the exponent is a step-size
knob and the whole "fit the true response" framing is misdirected.
"""
import sys, json, statistics, collections, time
from math import comb
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
EXPS = [1.0, 1.5, 2.0, 2.32, 3.0, 4.0, 6.0, 9.0, 15.0]


def run(t):
    spec, e = t
    try:
        dom = get_domain("truss")
        st = dom.load(spec)
        param = dom.params[0]
        for _ in range(20):
            if dom.feasible(st):
                return spec["problem_id"], e, 1
            if dom.calls >= 120:
                break
            op = T.INSTALLED_OP(e)
            build = lambda target: T.apply_op(
                dom, st, op, list(range(dom.n_elements(st))), param, target)
            nx, v, _ = T.inner_optimise(dom, st, build, {"target": (1.0, 1.8)}, budget=18)
            if v <= T.phi_rho(dom, st):
                break
            st = nx
        return spec["problem_id"], e, (1 if dom.feasible(st) else 0)
    except Exception:
        return spec["problem_id"], e, None


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    specs = [json.load(open(f)) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:n]]
    t0 = time.time()
    with Pool(44) as pool:
        res = pool.map(run, [(s, e) for s in specs for e in EXPS], chunksize=1)
    by = collections.defaultdict(dict)
    for pid, e, s in res:
        if s is not None:
            by[e][pid] = s
    keys = sorted(set.intersection(*[set(by[e]) for e in EXPS]))
    print("exponent sweep, n=%d, %.0fs   (measured elasticity median 2.32)" % (len(keys), time.time() - t0))
    best = None
    for e in EXPS:
        r = statistics.mean(by[e][k] for k in keys)
        mark = ""
        if abs(e - 2.32) < 1e-6:
            mark = "  <- measured elasticity"
        if e == 3.0:
            mark = "  <- the installed closed form"
        print("  exponent %5.2f   solved %.3f%s" % (e, r, mark))
        if best is None or r > best[1]:
            best = (e, r)
    print("\n  best exponent: %.2f at %.3f" % best)
    b = by[best[0]]
    for e in (2.32, 3.0):
        hi = sum(1 for k in keys if b[k] and not by[e][k])
        lo = sum(1 for k in keys if by[e][k] and not b[k])
        d = hi + lo
        p = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        print("  best vs exponent %.2f : disc %3d-%-3d  %+5.1fpp  p=%.2e"
              % (e, hi, lo, 100 * (hi - lo) / len(keys), p))
