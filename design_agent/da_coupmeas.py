"""
Break the circularity: measure coupling directly, then see whether it predicts the truss.

The scoping law says induction pays under heterogeneity and weak coupling, and I said the truss
ties because it "has enough coupling to sit at the crossover". That was read off the gain it
produced, so it explains nothing -- the coupling was inferred from the very number it was meant
to predict.

A coupling statistic that any domain can supply, from the Domain interface alone:

    perturb element i, record the response of its own margin and of every other margin
    coupling = mean_j!=i |dlog g_j|  /  ( |dlog g_i| + mean_j!=i |dlog g_j| )

Zero when elements are independent, rising as a perturbation spills onto its neighbours. It
needs no ground truth and costs one evaluation per element.

The estimator is calibrated first against the synthetic domain, where the true coupling is set
by hand, so its scale is known rather than assumed. Then it is measured on the truss. The
prediction is a number fixed before looking at the truss's induction result: if the measured
coupling lands near the 0.45 crossover, the tie is predicted; if it lands low, the completeness
account is wrong and the truss tie needs a different explanation again.
"""
import sys, json, math, random, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
from da_disp2 import Coupled, COUPLINGS

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
DELTA = 0.10


def coupling_of(dom, st, param, get_margins, n, delta=DELTA):
    base = get_margins(st)
    if not base or n < 3:
        return None
    own, cross = [], []
    for i in range(n):
        b = dom.get(st, i, param)
        if not b or base[i] <= 0:
            continue
        nx = dom.clone(st)
        dom.set(nx, i, param, b * (1 + delta))
        m = get_margins(nx)
        if not m:
            continue
        try:
            o = abs(math.log(max(m[i], 1e-9) / max(base[i], 1e-9)))
        except Exception:
            continue
        cs = []
        for j in range(n):
            if j == i or base[j] <= 0 or m[j] <= 0:
                continue
            cs.append(abs(math.log(m[j] / base[j])))
        if not cs or o <= 0:
            continue
        own.append(o)
        cross.append(statistics.mean(cs))
    if not own:
        return None
    O, C = statistics.mean(own), statistics.mean(cross)
    return C / (O + C) if (O + C) > 0 else None


def _synth(t):
    seed, c = t
    try:
        dom = Coupled()
        st = dom.load(seed, c)
        return (c, coupling_of(dom, st, "x", dom.element_margins, st["n"]))
    except Exception:
        return None


def _truss(f):
    try:
        dom = get_domain("truss")
        st = dom.load(json.load(open(f)))
        return coupling_of(dom, st, "r", dom.element_margins, dom.n_elements(st))
    except Exception:
        return None


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    files = [str(x) for x in sorted((DB / "data/problems_hard").glob("*.json"))[:250]]
    t0 = time.time()
    with Pool(44) as pool:
        cs = [r for r in pool.map(_synth, [(s, c) for s in range(N) for c in COUPLINGS],
                                   chunksize=8) if r and r[1] is not None]
        tr = [r for r in pool.map(_truss, files, chunksize=2) if r is not None]
    print("calibration on the synthetic domain, where true coupling is set by hand (%.0fs)\n"
          % (time.time() - t0))
    print("  true c    measured statistic")
    table = []
    for c in COUPLINGS:
        v = [x[1] for x in cs if x[0] == c]
        if not v:
            continue
        m = statistics.median(v)
        table.append((c, m))
        print("  %5.2f       %.3f" % (c, m))
    md = statistics.median(tr)
    lo = sorted(tr)[len(tr) // 4]
    hi = sorted(tr)[3 * len(tr) // 4]
    print("\n  TRUSS measured statistic: median %.3f  (interquartile %.3f to %.3f, n=%d)"
          % (md, lo, hi, len(tr)))
    # invert the calibration curve to an implied c
    implied = None
    for i in range(1, len(table)):
        c0, m0 = table[i - 1]
        c1, m1 = table[i]
        if (m0 - md) * (m1 - md) <= 0 and m1 != m0:
            implied = c0 + (md - m0) * (c1 - c0) / (m1 - m0)
            break
    if implied is None:
        implied = table[-1][0] if md > table[-1][1] else table[0][0]
    print("  implied coupling for the truss: c = %.2f" % implied)
    print("\n  The completeness account put the induction crossover at c = 0.45 and the truss's")
    print("  measured induction gain at -0.005. Independent measurement now says c = %.2f." % implied)
