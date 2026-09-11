"""
Dose-response inside the real domain.

Two aggregate numbers agreeing is weak evidence. The law's core prediction is a MONOTONE
relationship: the more coupled a system, the less correct addressing is worth. That is
testable per problem on the truss, where coupling is measurable from the simulator and the
addressing value is measurable from the arms.

For each problem: estimate coupling c from cross- vs own-elasticity, then run the three arms
(truthful / permuted / none). Bin by c and report the addressing value per bin.

Prediction: the truthful-minus-permuted gap decreases with c. If it is flat, the law's shape
does not hold inside the domain and the aggregate agreement was a coincidence of two numbers.
"""
import sys, json, math, statistics, collections, time
from math import comb
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_locate import coupling_of
from da_match import episode

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")


def _w(f):
    try:
        spec = json.load(open(f))
    except Exception:
        return None
    cc = coupling_of(spec)
    if not cc:
        return None
    out = {"pid": spec["problem_id"], "c": cc["c"], "n": cc["n"],
           "own": cc["own"], "cross": cc["cross"]}
    for arm in ("truthful", "permuted", "none"):
        try:
            out[arm] = episode(spec, arm)
        except Exception:
            return None
    return out


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    files = [str(f) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:N]]
    t0 = time.time()
    with Pool(40) as pool:
        rows = [r for r in pool.map(_w, files, chunksize=1) if r]
    print("dose-response, n=%d problems, %.0fs" % (len(rows), time.time() - t0))
    rows.sort(key=lambda r: r["c"])
    B = 5
    per = max(1, len(rows) // B)
    print("\n  coupling bin        n    c(med)  truthful  permuted  none  | addressing value")
    xs, ys = [], []
    for b in range(B):
        chunk = rows[b * per: (b + 1) * per] if b < B - 1 else rows[(B - 1) * per:]
        if len(chunk) < 8:
            continue
        cmed = statistics.median(r["c"] for r in chunk)
        t = statistics.mean(r["truthful"] for r in chunk)
        p_ = statistics.mean(r["permuted"] for r in chunk)
        nn = statistics.mean(r["none"] for r in chunk)
        hi = sum(1 for r in chunk if r["truthful"] and not r["permuted"])
        lo = sum(1 for r in chunk if r["permuted"] and not r["truthful"])
        d = hi + lo
        pv = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        gap = 100 * (t - p_)
        xs.append(cmed); ys.append(gap)
        print("  %d  c<=%.3f  %-4d %.3f   %.3f     %.3f     %.3f | %+5.1fpp (disc %3d-%-3d p=%.1e)"
              % (b + 1, max(r["c"] for r in chunk), len(chunk), cmed, t, p_, nn, gap, hi, lo, pv))

    # Spearman between per-problem coupling and per-problem addressing benefit
    ok = [r for r in rows if r["truthful"] != r["permuted"] or True]
    n = len(ok)
    cs = [r["c"] for r in ok]
    gs = [r["truthful"] - r["permuted"] for r in ok]

    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[o[k]] = a
            i = j + 1
        return rk
    rc, rg = rank(cs), rank(gs)
    mc_, mg = sum(rc) / n, sum(rg) / n
    num = sum((rc[i] - mc_) * (rg[i] - mg) for i in range(n))
    den = (sum((rc[i] - mc_) ** 2 for i in range(n)) * sum((rg[i] - mg) ** 2 for i in range(n))) ** .5
    rho = num / den if den else 0.0
    z = rho * math.sqrt(n - 1)
    print("\n  Spearman(coupling, addressing benefit) = %+.3f  (n=%d, z=%.2f)" % (rho, n, z))
    print("  prediction: NEGATIVE -- more coupling, less benefit from correct addressing")
    if len(xs) >= 3:
        print("  bin trend: %s" % " -> ".join("%+.1f" % y for y in ys))
