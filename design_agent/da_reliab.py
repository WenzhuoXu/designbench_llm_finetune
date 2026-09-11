"""
Is the truss's exponent heterogeneity real, or is it measurement noise?

Everything else has been ruled out. Induction ties on the truss whether its measurement is
charged to the budget or given away free, so it is not a cost problem; the truss's
independently measured coupling is 0.28, where the dose-response predicts a +11pp win, so it
is not coupling. And only 11 of 400 problems even discriminate between the arms, meaning the
induced and constant operators are making nearly the same moves.

One assumption has gone unchecked. The dispersion account treats the measured spread of
own-elasticities -- 0.750 on the truss -- as heterogeneity to exploit. But that number is the
spread of a MEASUREMENT, and a measurement of a coupled structure by finite difference may be
mostly artefact. If the per-element exponents do not reproduce across step sizes, there is no
real heterogeneity there, induction has nothing to recover, and it should tie exactly as it
does. The synthetic domain, where the exponents are exact, would then be the special case.

Test: measure each element's exponent at two step sizes and correlate them within each problem.
A real per-element property reproduces; noise does not. The synthetic domain is the control and
should come back near 1.
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
from da_synth import SynthDomain

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
CAP = 50.0


def spear(a, b):
    n = len(a)
    if n < 4:
        return None
    def rk(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0] * n
        for p_, i in enumerate(o):
            r[i] = p_
        return r
    ra, rb = rk(a), rk(b)
    m = (n - 1) / 2
    num = sum((ra[i] - m) * (rb[i] - m) for i in range(n))
    den = (sum((ra[i] - m) ** 2 for i in range(n)) * sum((rb[i] - m) ** 2 for i in range(n))) ** .5
    return num / den if den else None


def expo(dom, st, param, delta):
    base = dom.element_margins(st)
    n = dom.n_elements(st)
    out = []
    for i in range(n):
        b = dom.get(st, i, param)
        if not b or not base or base[i] is None:
            out.append(None); continue
        try:
            b0 = min(float(base[i]), CAP)
            if not math.isfinite(b0) or b0 <= 0:
                out.append(None); continue
        except Exception:
            out.append(None); continue
        nx = dom.clone(st)
        dom.set(nx, i, param, b * (1 + delta))
        m = dom.element_margins(nx)
        try:
            m0 = min(float(m[i]), CAP)
            e = math.log(max(m0, 1e-9) / b0) / math.log(1 + delta)
        except Exception:
            out.append(None); continue
        out.append(e if math.isfinite(e) else None)
    return out


def _truss(f):
    try:
        dom = get_domain("truss")
        st = dom.load(json.load(open(f)))
        a = expo(dom, st, "r", 0.05)
        b = expo(dom, st, "r", 0.15)
        ok = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
        if len(ok) < 4:
            return None
        av = [a[i] for i in ok]; bv = [b[i] for i in ok]
        pos = [x for x in av if x > 0.2]
        return (spear(av, bv),
                statistics.pstdev([math.log(x) for x in pos]) if len(pos) > 2 else None)
    except Exception:
        return None


def _synth(seed):
    try:
        dom = SynthDomain()
        st = dom.load({"seed": seed, "n": 10})
        a = expo(dom, st, "x", 0.05)
        b = expo(dom, st, "x", 0.15)
        ok = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
        if len(ok) < 4:
            return None
        av = [a[i] for i in ok]; bv = [b[i] for i in ok]
        pos = [x for x in av if x > 0.2]
        return (spear(av, bv),
                statistics.pstdev([math.log(x) for x in pos]) if len(pos) > 2 else None)
    except Exception:
        return None


if __name__ == "__main__":
    files = [str(x) for x in sorted((DB / "data/problems_hard").glob("*.json"))[:300]]
    t0 = time.time()
    with Pool(44) as pool:
        tr = [r for r in pool.map(_truss, files, chunksize=2) if r and r[0] is not None]
        sy = [r for r in pool.map(_synth, list(range(300)), chunksize=4) if r and r[0] is not None]
    print("exponent reproducibility across step sizes 0.05 and 0.15, %.0fs\n" % (time.time() - t0))
    for name, g in (("TRUSS", tr), ("SYNTHETIC (control, exact exponents)", sy)):
        rs = [x[0] for x in g]
        ds = [x[1] for x in g if x[1] is not None]
        print("  %-38s n=%d" % (name, len(g)))
        print("    within-problem rank correlation of the two measurements:")
        print("      median %+.3f   interquartile %+.3f to %+.3f"
              % (statistics.median(rs), sorted(rs)[len(rs) // 4], sorted(rs)[3 * len(rs) // 4]))
        print("      share of problems above +0.5: %.2f" % (sum(1 for x in rs if x > 0.5) / len(rs)))
        if ds:
            print("    measured dispersion sd(log e): median %.3f" % statistics.median(ds))
        print()
    print("  If the truss correlation is low, its measured dispersion is largely artefact and")
    print("  there is no real per-element heterogeneity for induction to exploit.")
