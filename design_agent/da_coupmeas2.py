"""
The coupling measurement again, with non-finite margins handled.

The first pass returned a NaN median on the truss: members carrying no force have infinite
factors of safety, so log(inf/inf) poisons the statistic for most problems. Margins are clamped
at 50 here, which is the convention already used elsewhere in this work, and every element
whose response cannot be formed as a finite log-ratio is dropped rather than silently counted.

The calibration also showed the raw statistic is heavily compressed -- true coupling 0.70 reads
as 0.159 -- so the inversion through the calibration curve is doing real work and is reported
alongside the raw number.
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
from da_disp2 import Coupled, COUPLINGS

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
DELTA, CAP = 0.10, 50.0


def clean(v):
    out = []
    for x in v:
        try:
            x = float(x)
        except (TypeError, ValueError):
            out.append(None); continue
        if not math.isfinite(x) or x <= 0:
            out.append(CAP if (isinstance(x, float) and x == float("inf")) else None)
        else:
            out.append(min(x, CAP))
    return out


def coupling_of(dom, st, param, margins, n, delta=DELTA):
    base = clean(margins(st))
    if not base or n < 3:
        return None
    own, cross = [], []
    for i in range(n):
        b = dom.get(st, i, param)
        if not b or base[i] is None:
            continue
        nx = dom.clone(st)
        dom.set(nx, i, param, b * (1 + delta))
        m = clean(margins(nx))
        if not m or m[i] is None:
            continue
        o = abs(math.log(m[i] / base[i]))
        cs = [abs(math.log(m[j] / base[j])) for j in range(n)
              if j != i and base[j] is not None and m[j] is not None]
        cs = [x for x in cs if math.isfinite(x)]
        if not cs or not math.isfinite(o):
            continue
        own.append(o)
        cross.append(statistics.mean(cs))
    if len(own) < 3:
        return None
    O, C = statistics.mean(own), statistics.mean(cross)
    if not math.isfinite(O) or not math.isfinite(C) or (O + C) <= 0:
        return None
    return C / (O + C)


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
        tr = [r for r in pool.map(_truss, files, chunksize=2)
              if r is not None and math.isfinite(r)]
    print("calibration on the synthetic domain (%.0fs)\n" % (time.time() - t0))
    print("  true c    measured statistic")
    table = []
    for c in COUPLINGS:
        v = [x[1] for x in cs if x[0] == c]
        if v:
            m = statistics.median(v)
            table.append((c, m))
            print("  %5.2f       %.4f" % (c, m))
    if not tr:
        print("\n  no usable truss measurements")
        sys.exit(0)
    md = statistics.median(tr)
    print("\n  TRUSS: median %.4f   interquartile %.4f to %.4f   (n=%d usable of %d)"
          % (md, sorted(tr)[len(tr) // 4], sorted(tr)[3 * len(tr) // 4], len(tr), len(files)))
    implied = None
    for i in range(1, len(table)):
        c0, m0 = table[i - 1]; c1, m1 = table[i]
        if (m0 - md) * (m1 - md) <= 0 and m1 != m0:
            implied = c0 + (md - m0) * (c1 - c0) / (m1 - m0)
            break
    if implied is None:
        implied = float("inf") if md > table[-1][1] else 0.0
    print("  implied coupling: c = %s"
          % ("above the calibrated range" if implied == float("inf") else "%.2f" % implied))
    print("\n  Crossover for induction is c = 0.45; the truss's measured induction gain is -0.005.")
    print("  If the implied c is far from 0.45, the completeness account does not explain the truss.")
