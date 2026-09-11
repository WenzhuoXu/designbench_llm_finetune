"""
Pre-registered confirmatory test.

Exploratory analysis over five candidate moderators found exactly one: the interaction of
initial deficit with ordering fidelity, at z=2.54. That sits on the Bonferroni line for five
tests, so it is suggestive and not established.

This is the confirmation. ONE hypothesis, stated before the run:

    H1: in outcome ~ rho + deficit + rho:deficit, the rho:deficit coefficient is POSITIVE
        with cluster-robust z > 2.

Held-out problems 200-400, disjoint from the 0-200 that produced the hypothesis. No other
moderator is fitted, so there is no multiplicity to correct. A negative result retires the
deficit account exactly as the coupling account was retired.
"""
import sys, json, math, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_moderate import _w, z
from da_fit2 import irls

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")

if __name__ == "__main__":
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    files = [str(f) for f in sorted((DB / "data/problems_hard").glob("*.json"))[lo:hi]]
    print("CONFIRMATORY: held-out problems %d-%d (%d files)" % (lo, hi, len(files)))
    print("H1: rho:deficit coefficient positive, cluster z > 2. One test.\n")
    t0 = time.time()
    with Pool(40) as pool:
        packs = [r for r in pool.map(_w, files, chunksize=1) if r]
    rows = [x for pk in packs for x in pk]
    print("%d episodes over %d problems, %.0fs" % (len(rows), len(packs), time.time() - t0))
    rho = [r[4] for r in rows]
    y = [float(r[3]) for r in rows]
    ids = [r[0] for r in rows]
    dfc = z([r[5]["deficit"] for r in rows])
    X = [[1.0, rho[i], dfc[i], rho[i] * dfc[i]] for i in range(len(rows))]
    b, se, cl, ll = irls(X, y, ids)
    zi = b[3] / max(cl[3], 1e-9)
    print("\n  rho                %+7.3f  (cluster z=%6.2f)" % (b[1], b[1] / max(cl[1], 1e-9)))
    print("  deficit            %+7.3f  (cluster z=%6.2f)" % (b[2], b[2] / max(cl[2], 1e-9)))
    print("  rho:deficit        %+7.3f  (cluster z=%6.2f)   <- the pre-registered term" % (b[3], zi))
    print("\n  discovery sample gave rho:deficit = +2.643 (z=2.54)")
    verdict = "CONFIRMED" if (b[3] > 0 and zi > 2.0) else "NOT CONFIRMED"
    print("  H1 %s" % verdict)
    if b[3] > 0 and zi > 2.0:
        print("  => difficulty moderates how much ordering fidelity is worth, on held-out data.")
    else:
        print("  => retire the deficit account, as the coupling account was retired.")
