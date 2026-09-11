"""
Pre-registered test of the second moderator, which the mechanism experiment implies.

da_why showed that when shrinking is forbidden and the budget is slack -- the arm where
mis-ranking a safe element is harmless -- the critical-set coefficient OVERTAKES the global
one in the truss (+0.464 against +0.174), and that arm is also by far the easiest (base 0.670
against 0.145-0.402). That suggests the two fidelity statistics divide by DIFFICULTY rather
than by mechanism: near feasibility only a few elements bind and the critical set is what
matters; far from feasibility most elements need action and the whole ordering matters.

The first moderator is already confirmed held-out: the rho slope RISES with initial deficit
(+3.099, cluster z=4.48). This registers the complementary prediction.

  H1: in  outcome ~ rho + ovq + deficit + rho:deficit + ovq:deficit,
      the ovq:deficit coefficient is NEGATIVE with cluster |z| > 2.

Problems 200-400, disjoint from the 0-150 that produced the mechanism result. One test on the
held-out set; the 0-200 fit is reported as exploratory only. If the confirmation fails, the
difficulty account of the topk result is retired.
"""
import sys, json, math, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
from da_moderate import moderators, z, CELLS
from da_topk import episode
from da_fit2 import irls

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")


def _w(f):
    try:
        dom = get_domain("truss")
        spec = json.load(open(f))
        st = dom.load(spec)
        mods = moderators(dom, st)
        if mods is None:
            return None
        out = []
        for mode, lv in CELLS:
            d2 = get_domain("truss")
            s2 = d2.load(spec)
            o, acc = episode(d2, s2, mode, lv, hash(f) & 0xffff)
            r = {"id": f, "y": float(o)}
            for kk in ("rho", "ovq"):
                r[kk] = statistics.mean(acc[kk]) if acc[kk] else 1.0
            r["deficit"] = mods["deficit"]
            out.append(r)
        return out
    except Exception:
        return None


def run(files, label):
    with Pool(40) as pool:
        packs = [r for r in pool.map(_w, files, chunksize=1) if r]
    rows = [x for pk in packs for x in pk]
    n = len(rows)
    y = [r["y"] for r in rows]
    ids = [r["id"] for r in rows]
    rho = [r["rho"] for r in rows]
    ovq = [r["ovq"] for r in rows]
    dfc = z([r["deficit"] for r in rows])
    X = [[1.0, rho[i], ovq[i], dfc[i], rho[i] * dfc[i], ovq[i] * dfc[i]] for i in range(n)]
    b, se, cl, ll = irls(X, y, ids)
    zz = [b[i] / max(cl[i], 1e-9) for i in range(len(b))]
    print("\n%s: %d episodes over %d problems, base %.3f" % (label, n, len(packs), sum(y) / n))
    for i, nm in enumerate(["intercept", "rho", "crit-quartile", "deficit",
                            "rho:deficit", "crit:deficit"]):
        print("  %-16s %+7.3f   cluster z = %6.2f" % (nm, b[i], zz[i]))
    return b, zz


if __name__ == "__main__":
    allf = sorted((DB / "data/problems_hard").glob("*.json"))
    t0 = time.time()
    print("H1 (pre-registered): crit:deficit NEGATIVE, cluster |z| > 2, on problems 200-400.")
    print("Reported second so the exploratory fit cannot be mistaken for the test.")
    b0, z0 = run([str(f) for f in allf[0:200]], "EXPLORATORY  problems 0-200")
    b1, z1 = run([str(f) for f in allf[200:400]], "CONFIRMATORY problems 200-400")
    print("\n  %.0fs total" % (time.time() - t0))
    ok = (b1[5] < 0 and abs(z1[5]) > 2.0)
    print("\n  pre-registered term crit:deficit = %+.3f (z=%.2f)  ->  %s"
          % (b1[5], z1[5], "CONFIRMED" if ok else "NOT CONFIRMED"))
    print("  replication of moderator 1, rho:deficit = %+.3f (z=%.2f); held-out value was +3.099 (z=4.48)"
          % (b1[4], z1[4]))
    if not ok:
        print("  => retire the difficulty account of the critical-set result.")
