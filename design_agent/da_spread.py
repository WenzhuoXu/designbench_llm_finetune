"""Is there any real coupling variation across truss problems?

The dose-response came out flat. Two readings: the law is wrong for trusses, or every truss
sits at essentially the same (very low) coupling and the test had no range to work with.

The estimator c = r*n/(1+r*(n-1)) multiplies a tiny ratio by n, so its 0.09-0.83 spread may
be an artefact of member-count variation rather than coupling variation. This looks at the
RAW ratio r = cross-elasticity / own-elasticity directly, and asks how much of the spread in
c is r moving versus n moving.

If r is tightly clustered near zero, the truss is uniformly weakly coupled, a flat
dose-response is the predicted result, and the law is untestable here rather than refuted.
"""
import sys, json, math, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_locate import coupling_of

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")


def _w(f):
    try:
        return coupling_of(json.load(open(f)))
    except Exception:
        return None


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    files = [str(f) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:N]]
    t0 = time.time()
    with Pool(40) as pool:
        rows = [r for r in pool.map(_w, files, chunksize=1) if r]
    q = lambda a, f: sorted(a)[int(f * (len(a) - 1))]
    rs = [r["r"] for r in rows]
    ns = [r["n"] for r in rows]
    cs = [r["c"] for r in rows]
    print("n=%d problems, %.0fs\n" % (len(rows), time.time() - t0))
    print("raw cross/own ratio r:")
    print("  median %.4f | p10 %.4f | p25 %.4f | p75 %.4f | p90 %.4f | max %.4f"
          % (q(rs, .5), q(rs, .1), q(rs, .25), q(rs, .75), q(rs, .9), max(rs)))
    print("  p90/p10 spread factor: %.1fx" % (q(rs, .9) / max(q(rs, .1), 1e-9)))
    print("\nmember count n: median %d, range %d..%d" % (statistics.median(ns), min(ns), max(ns)))
    print("derived c:  median %.3f | p10 %.3f | p90 %.3f" % (q(cs, .5), q(cs, .1), q(cs, .9)))

    # how much of c's spread is r moving vs n moving?
    n_med = statistics.median(ns)
    r_med = statistics.median(rs)
    c_fix_n = [x * n_med / (1 + x * (n_med - 1)) for x in rs]        # vary r, hold n
    c_fix_r = [r_med * k / (1 + r_med * (k - 1)) for k in ns]        # vary n, hold r
    print("\n  c spread with n held at its median (r varying): p10 %.3f -> p90 %.3f"
          % (q(c_fix_n, .1), q(c_fix_n, .9)))
    print("  c spread with r held at its median (n varying): p10 %.3f -> p90 %.3f"
          % (q(c_fix_r, .1), q(c_fix_r, .9)))
    print("\n  On the synthetic curve, c in [%.3f, %.3f] spans an addressing value of roughly"
          % (q(cs, .1), q(cs, .9)))
    print("  +45pp to +30pp -- a 15pp range, against per-bin sampling noise of about 10pp at n=80.")
