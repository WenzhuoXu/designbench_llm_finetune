"""
Where does the truss sit on the coupling axis, and does the law predict its measured value?

The synthetic sweep gives the value of correct addressing as a function of coupling c. If the
truss's coupling can be measured independently, the law makes a QUANTITATIVE prediction about
a domain it was never fitted to, and the truss's own measured value (+26.0pp, correctly
addressed per-element factors vs base, disc 52-0) either lands on the curve or it does not.

Coupling is estimated from the simulator directly. Perturb element j by delta and record the
log-response of EVERY element's own margin:

    own_j   = d log(margin_j) / d log(x_j)          the element's response to itself
    cross_j = mean over i != j of |d log(margin_i) / d log(x_j)|

In the synthetic model at coupling c with n elements,
    own   = k(1-c) + k*c/n
    cross = k*c/n
so r = cross/own = (c/n) / (1 - c + c/n), which inverts to c = r*n / (1 + r*(n-1)).

No parameter is tuned here: r is measured on the truss, c falls out, and the synthetic sweep
supplies the prediction.
"""
import sys, json, math, statistics, collections, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
DELTA = 0.10


def coupling_of(spec):
    try:
        dom = get_domain("truss")
        st = dom.load(spec)
        param = dom.params[0]
        m0 = dom.element_margins(st)
        n = dom.n_elements(st)
        if not m0 or n < 3:
            return None
        owns, crosses = [], []
        for j in range(n):
            b = dom.get(st, j, param)
            if not b or b <= 0 or not (0 < m0[j] < float("inf")):
                continue
            nx = dom.clone(st)
            dom.set(nx, j, param, b * math.exp(DELTA))
            m1 = dom.element_margins(nx)
            if not m1:
                continue
            o = None; cs = []
            for i in range(n):
                a, c = m0[i], m1[i]
                if not (0 < a < float("inf")) or not (0 < c < float("inf")):
                    continue
                e = (math.log(c) - math.log(a)) / DELTA
                if i == j:
                    o = abs(e)
                else:
                    cs.append(abs(e))
            if o is not None and o > 1e-6 and cs:
                owns.append(o); crosses.append(statistics.mean(cs))
        if not owns:
            return None
        own = statistics.median(owns); cross = statistics.median(crosses)
        r = cross / own if own > 0 else None
        if r is None:
            return None
        c_hat = r * n / (1 + r * (n - 1))
        return {"n": n, "own": own, "cross": cross, "r": r, "c": min(max(c_hat, 0.0), 1.0)}
    except Exception:
        return None


def _w(f):
    try:
        return coupling_of(json.load(open(f)))
    except Exception:
        return None


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    files = sorted((DB / "data/problems_hard").glob("*.json"))[:N]
    t0 = time.time()
    with Pool(40) as pool:
        res = [r for r in pool.map(_w, [str(f) for f in files], chunksize=1) if r]
    cs = sorted(r["c"] for r in res)
    owns = sorted(r["own"] for r in res)
    crs = sorted(r["cross"] for r in res)
    q = lambda a, f: a[int(f * (len(a) - 1))]
    print("truss coupling, measured from the simulator, n=%d problems, %.0fs" % (len(res), time.time() - t0))
    print("  own-elasticity   median %.3f  (p25 %.3f, p75 %.3f)" % (q(owns, .5), q(owns, .25), q(owns, .75)))
    print("  cross-elasticity median %.3f  (p25 %.3f, p75 %.3f)" % (q(crs, .5), q(crs, .25), q(crs, .75)))
    print("  cross/own ratio  median %.3f" % (q(crs, .5) / q(owns, .5)))
    print("\n  estimated coupling c: median %.3f | p10 %.3f | p25 %.3f | p75 %.3f | p90 %.3f"
          % (q(cs, .5), q(cs, .1), q(cs, .25), q(cs, .75), q(cs, .9)))
    print("\n  synthetic sweep, value of correct addressing:")
    print("     c=0.00 +42.8pp | c=0.20 +47.8 | c=0.40 +38.8 | c=0.60 +24.0 | c=0.80 +15.5 | c=0.95 +3.2 | c=1.00 null")
    print("  truss measured value of correct addressing: +26.0pp (disc 52-0, p=4.4e-16)")
    cm = q(cs, .5)
    grid = [(0.0, 42.8), (0.2, 47.8), (0.4, 38.8), (0.6, 24.0), (0.8, 15.5), (0.95, 3.2), (1.0, 1.8)]
    lo = max([g for g in grid if g[0] <= cm], key=lambda g: g[0])
    hi = min([g for g in grid if g[0] >= cm], key=lambda g: g[0])
    if hi[0] == lo[0]:
        pred = lo[1]
    else:
        w = (cm - lo[0]) / (hi[0] - lo[0])
        pred = lo[1] + w * (hi[1] - lo[1])
    print("\n  => at the measured coupling c=%.3f the law predicts %+.1fpp; the truss measures +26.0pp"
          % (cm, pred))
