"""
A synthetic domain, and the permutation control run in it.

The surviving result -- that a per-item signal's value is carried by its item-to-value
CORRESPONDENCE rather than by its values -- was shown with no model in the loop, but only on
trusses. Its whole point is that it is a claim about signal structure, so it should reproduce
in any domain satisfying the two admissibility conditions: per-element attribution, and a
local inverse. This domain has both and no physics whatsoever.

  element i has parameter x_i > 0
  its own margin      g_i = a_i * x_i ** k_i          (monotone in its own parameter)
  the budget          c   = sum_i w_i * x_i ** 2
  feasible when min_i g_i >= 1 and c <= B

a_i, k_i, w_i and B are drawn per instance. k_i is drawn to span the same range measured on
the truss (median ~2.3, a long tail below 1.5), so the difficulty profile is comparable
without importing anything truss-specific.
"""
import sys, math, random, statistics, collections, time
from math import comb
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from da_domain import Domain, REGISTRY
import da_tools as T


class SynthDomain(Domain):
    name = "synth"
    params = ("x",)

    def __init__(self, n=16, seed=0):
        self.n = n
        self.rng = random.Random(seed)
        self.calls = 0

    def load(self, spec):
        r = random.Random(spec["seed"])
        n = spec.get("n", self.n)
        a = [math.exp(r.gauss(-0.35, 0.55)) for _ in range(n)]
        # exponents spanning the range measured on the truss: median ~2.3, tail below 1.5
        k = [max(0.3, r.lognormvariate(0.80, 0.55)) for _ in range(n)]
        w = [math.exp(r.gauss(0.0, 0.4)) for _ in range(n)]
        x = [1.0] * n
        st = {"a": a, "k": k, "w": w, "x": x, "B": float("inf"), "n": n}

        # Price the budget off the LIGHTEST requirement-feasible design rather than a multiple
        # of total weight. Iterate the sizing rule at margin 1.00 to convergence: that design is
        # about as light as the requirement permits, so a pass at a safety margin oversizes and
        # exceeds the cap. Both constraints then bind and the search has a trade-off to make.
        # Drawn independently, the cap almost never bound and the sizing rule solved outright.
        probe = list(x)
        for _ in range(60):
            g = [a[i] * probe[i] ** k[i] for i in range(n)]
            if min(g) >= 0.999 and max(g) <= 1.06:
                break
            for i in range(n):
                m = g[i] if g[i] > 1e-9 else 1e-9
                probe[i] = min(4.0, max(0.05,
                                        probe[i] * min(1.5, max(0.7, (1.0 / m) ** (1.0 / max(k[i], 0.3))))))
        st["B"] = 1.04 * sum(w[i] * probe[i] ** 2 for i in range(n))
        return st

    def clone(self, st):
        d = dict(st); d["x"] = list(st["x"]); return d

    def evaluate(self, st):
        self.calls += 1
        return {}

    def n_elements(self, st): return st["n"]
    def get(self, st, i, p): return st["x"][i]

    def set(self, st, i, p, v):
        st["x"][i] = min(4.0, max(0.05, float(v)))

    def bounds(self, st, p): return (0.05, 4.0)

    def element_margins(self, st):
        self.calls += 1
        return [st["a"][i] * st["x"][i] ** st["k"][i] for i in range(st["n"])]

    def global_margins(self, st):
        return [min(self.element_margins(st))]

    def budget_ratio(self, st):
        c = sum(st["w"][i] * st["x"][i] ** 2 for i in range(st["n"]))
        return c / st["B"]

    def feasible(self, st):
        return min(self.element_margins(st)) >= 1.0 and self.budget_ratio(st) <= 1.0


REGISTRY["synth"] = SynthDomain


def run(spec, arm, exponent=3.0, steps=20, budget=200):
    dom = SynthDomain()
    st = dom.load(spec)
    rng = random.Random(spec["seed"] ^ 0xABCD)
    n = dom.n_elements(st)
    perm = list(range(n))
    if arm == "permuted":
        rng.shuffle(perm)
    for _ in range(steps):
        if dom.feasible(st):
            return 1
        if dom.calls >= budget:
            break
        em = dom.element_margins(st)
        # the signal the actor consumes: correct values, possibly at the wrong addresses
        shown = [em[perm[i]] for i in range(n)] if arm == "permuted" else em
        op = T.INSTALLED_OP(exponent)
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(n):
                f = min(2.0, max(0.7, op(shown[i], i, tgt)))
                dom.set(nx, i, "x", st["x"][i] * f)
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    specs = [{"seed": s, "n": 16} for s in range(N)]
    ARMS = ["truthful", "permuted", "none"]
    t0 = time.time()
    solved = collections.defaultdict(dict)
    for sp in specs:
        for arm in ARMS:
            if arm == "none":
                # no per-element signal at all: one shared factor, chosen by the same sweep
                dom = SynthDomain(); st = dom.load(sp)
                ok = 0
                for _ in range(20):
                    if dom.feasible(st):
                        ok = 1; break
                    if dom.calls >= 200:
                        break
                    best, bv = None, T.phi_rho(dom, st)
                    for f in [0.7 * (2.0 / 0.7) ** (j / 15) for j in range(16)]:
                        nx = dom.clone(st)
                        for i in range(dom.n_elements(st)):
                            dom.set(nx, i, "x", st["x"][i] * f)
                        v = T.phi_rho(dom, nx)
                        if v > bv:
                            best, bv = nx, v
                    if best is None:
                        break
                    st = best
                solved[arm][sp["seed"]] = ok or (1 if dom.feasible(st) else 0)
            else:
                solved[arm][sp["seed"]] = run(sp, arm)
    keys = sorted(solved["truthful"])
    print("synthetic domain -- no physics, no simulator. n=%d instances, %.0fs"
          % (len(keys), time.time() - t0))
    for a in ARMS:
        print("  %-9s solved %.3f" % (a, statistics.mean(solved[a][k] for k in keys)))

    def mc(x, y):
        hi = sum(1 for k in keys if solved[x][k] and not solved[y][k])
        lo = sum(1 for k in keys if solved[y][k] and not solved[x][k])
        d = hi + lo
        p = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        return hi, lo, 100 * (hi - lo) / len(keys), p
    print()
    for x, y, q in (("truthful", "none", "correct per-element signal vs none"),
                    ("permuted", "none", "PERMUTED per-element signal vs none"),
                    ("truthful", "permuted", "the cost of wrong addressing")):
        hi, lo, d, p = mc(x, y)
        print("  %-9s vs %-9s disc %4d-%-4d  %+5.1fpp  p=%.2e   %s" % (x, y, hi, lo, d, p, q))
