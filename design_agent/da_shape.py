"""
Is it the SHAPE of coupling rather than its average? Matched-average test.

Induction wins +15.7pp in a decoupled synthetic domain and nothing on the truss, and coupling
magnitude, measurement cost and measurement noise have all been ruled out as explanations. The
remaining hypothesis is that the average cross-response is the wrong summary: the synthetic
domain spreads a perturbation thinly across every element, while a truss routes it down a few
load paths, and a mean cannot tell a diffuse field from a sparse one.

The control is built into the design rather than bolted on afterwards. Two domains:

  diffuse   g_i = own_i^(1-c) * (geometric mean of all)^c
  sparse    g_i = own_i^(1-c) * (geometric mean of two ring neighbours)^c

The coupling parameter is swept in both and the measured statistic recorded, so the two shapes
can be matched on the SAME measured average coupling -- set to the truss's own 0.0373 -- rather
than on their nominal parameters. Only then is induction run.

If the two shapes give the same induction gain at matched measured coupling, shape is
irrelevant and this hypothesis dies with the other three. If sparse gives a markedly lower
gain, the averaging statistic is blind to the thing that matters and the truss is explained.
"""
import sys, math, random, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune"):
    if p not in sys.path:
        sys.path.insert(0, p)
import da_tools as T

STEPS, N_ELEM, SHRINK, SIGMA = 6, 10, 0.4, 0.75
CS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8)
TRUSS_STAT = 0.0373


class Shaped:
    params = ("x",)

    def __init__(self):
        self.calls = 0

    def load(self, seed, c, shape):
        r = random.Random(seed)
        n = N_ELEM
        return {"a": [math.exp(r.gauss(-0.35, 0.55)) for _ in range(n)],
                "k": [max(0.3, r.lognormvariate(0.80, SIGMA)) for _ in range(n)],
                "w": [math.exp(r.gauss(0.0, 0.4)) for _ in range(n)],
                "x": [1.0] * n, "n": n, "c": c, "shape": shape,
                "B": sum(math.exp(r.gauss(0.0, 0.4)) for _ in range(n)) * r.uniform(1.5, 3.2)}

    def clone(self, st):
        d = dict(st); d["x"] = list(st["x"]); return d

    def n_elements(self, st):
        return st["n"]

    def get(self, st, i, p):
        return st["x"][i]

    def set(self, st, i, p, v):
        st["x"][i] = min(4.0, max(0.05, float(v)))

    def element_margins(self, st):
        self.calls += 1
        n = st["n"]
        own = [st["a"][i] * st["x"][i] ** st["k"][i] for i in range(n)]
        c = st["c"]
        if c <= 0:
            return own
        lg = [math.log(max(v, 1e-9)) for v in own]
        out = []
        for i in range(n):
            if st["shape"] == "diffuse":
                nb = sum(lg) / n
            else:
                nb = (lg[(i - 1) % n] + lg[(i + 1) % n]) / 2.0    # two load paths
            out.append(math.exp((1 - c) * lg[i] + c * nb))
        return out

    def global_margins(self, st):
        return [min(self.element_margins(st))]

    def budget_ratio(self, st):
        return sum(st["w"][i] * st["x"][i] ** 2 for i in range(st["n"])) / st["B"]

    def feasible(self, st):
        return min(self.element_margins(st)) >= 1.0 and self.budget_ratio(st) <= 1.0


def coupling_stat(dom, st, delta=0.10):
    base = dom.element_margins(st)
    n = st["n"]
    own, cross = [], []
    for i in range(n):
        b = st["x"][i]
        nx = dom.clone(st)
        dom.set(nx, i, "x", b * (1 + delta))
        m = dom.element_margins(nx)
        try:
            o = abs(math.log(m[i] / base[i]))
            cs = [abs(math.log(m[j] / base[j])) for j in range(n) if j != i]
        except Exception:
            continue
        if math.isfinite(o) and o > 0:
            own.append(o); cross.append(statistics.mean(cs))
    if not own:
        return None
    O, C = statistics.mean(own), statistics.mean(cross)
    return C / (O + C) if (O + C) > 0 else None


def ep(seed, c, shape, mode):
    dom = Shaped()
    st = dom.load(seed, c, shape)
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1
        em = dom.element_margins(st)
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(st["n"]):
                d = math.log(max(tgt, 1e-9) / max(em[i], 1e-9))
                lf = d / 3.0 if mode == "fixed" else d * SHRINK / max(st["k"][i], 0.2)
                dom.set(nx, i, "x", st["x"][i] * min(2.0, max(0.7, math.exp(lf))))
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def _stat(t):
    seed, c, shape = t
    try:
        d = Shaped()
        return (shape, c, coupling_stat(d, d.load(seed, c, shape)))
    except Exception:
        return None


def _run(t):
    seed, c, shape, mode = t
    try:
        return (shape, c, mode, ep(seed, c, shape, mode))
    except Exception:
        return None


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    t0 = time.time()
    with Pool(44) as pool:
        st = [r for r in pool.map(_stat, [(s, c, sh) for s in range(120) for c in CS
                                          for sh in ("diffuse", "sparse")], chunksize=8)
              if r and r[2] is not None]
    print("step 1 -- calibrate each shape's measured coupling (%.0fs)\n" % (time.time() - t0))
    print("  nominal c    diffuse stat    sparse stat")
    cal = {"diffuse": [], "sparse": []}
    for c in CS:
        row = []
        for sh in ("diffuse", "sparse"):
            v = [x[2] for x in st if x[0] == sh and x[1] == c]
            m = statistics.median(v) if v else float("nan")
            cal[sh].append((c, m)); row.append(m)
        print("  %5.2f        %.4f          %.4f" % (c, row[0], row[1]))

    def invert(tab, target):
        for i in range(1, len(tab)):
            c0, m0 = tab[i - 1]; c1, m1 = tab[i]
            if (m0 - target) * (m1 - target) <= 0 and m1 != m0:
                return c0 + (target - m0) * (c1 - c0) / (m1 - m0)
        return None
    cd = invert(cal["diffuse"], TRUSS_STAT)
    csp = invert(cal["sparse"], TRUSS_STAT)
    print("\n  matching both shapes to the truss's measured 0.0373:")
    print("    diffuse needs c = %s" % ("n/a" if cd is None else "%.3f" % cd))
    print("    sparse  needs c = %s" % ("n/a" if csp is None else "%.3f" % csp))
    if cd is None or csp is None:
        print("  cannot match; stopping")
        sys.exit(0)

    jobs = []
    for sh, c in (("diffuse", cd), ("sparse", csp)):
        for m in ("fixed", "induced"):
            jobs += [(s, c, sh, m) for s in range(N)]
    with Pool(44) as pool:
        got = [r for r in pool.map(_run, jobs, chunksize=8) if r]
    print("\nstep 2 -- induction at MATCHED measured coupling, %d seeds each\n" % N)
    print("  shape      nominal c    fixed 3.0    induced    gain")
    for sh, c in (("diffuse", cd), ("sparse", csp)):
        a = [x[3] for x in got if x[0] == sh and x[2] == "fixed"]
        b = [x[3] for x in got if x[0] == sh and x[2] == "induced"]
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        print("  %-9s  %.3f        %.3f        %.3f      %+.3f" % (sh, c, ma, mb, mb - ma))
    print("\n  truss reference at this measured coupling: gain -0.005 (discordant 3-4, p=1)")
    print("  Shape matters only if the two rows differ.")
