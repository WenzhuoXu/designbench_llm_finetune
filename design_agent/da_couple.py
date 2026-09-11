"""
How much does correct addressing matter, as a function of how coupled the system is?

The synthetic result is open to one obvious objection: in a domain where each element's
margin depends only on its own parameter, misaddressing destroys the signal by construction.
Real systems couple -- a truss couples through the load path -- so the question is not
whether addressing matters but WHERE it stops mattering.

Make coupling a parameter and sweep it:

    g_i = a_i * (x_i ** k_i) ** (1-c) * (geomean_j x_j ** k_j) ** c

  c = 0  every element's margin is its own business; addressing is maximally informative
  c = 1  every element has the same margin; there is no per-element structure to address,
         so a permutation is a no-op and correct addressing must be worth nothing

If the value of correct addressing decays smoothly to zero as c goes to 1, the claim stops
being "addressing matters" and becomes a quantitative statement about decoupling, with the
truss sitting somewhere on the curve. That is the version a reviewer cannot wave away.
"""
import sys, math, random, statistics, collections, time
from math import comb
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from da_domain import Domain
import da_tools as T


class CoupledDomain(Domain):
    name = "coupled"
    params = ("x",)

    def __init__(self, c=0.0):
        self.c = c
        self.calls = 0

    def load(self, spec):
        r = random.Random(spec["seed"])
        n = spec["n"]
        return {"n": n,
                "a": [math.exp(r.gauss(-0.35, 0.55)) for _ in range(n)],
                "k": [max(0.3, r.lognormvariate(0.80, 0.55)) for _ in range(n)],
                "w": [math.exp(r.gauss(0.0, 0.4)) for _ in range(n)],
                "x": [1.0] * n,
                "B": None}

    def prepare(self, st, r):
        st["B"] = sum(st["w"]) * r.uniform(1.5, 3.2)
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
        n, c = st["n"], self.c
        own = [st["x"][i] ** st["k"][i] for i in range(n)]
        gm = math.exp(sum(math.log(max(o, 1e-9)) for o in own) / n)
        return [st["a"][i] * (own[i] ** (1 - c)) * (gm ** c) for i in range(n)]

    def global_margins(self, st): return [min(self.element_margins(st))]

    def budget_ratio(self, st):
        return sum(st["w"][i] * st["x"][i] ** 2 for i in range(st["n"])) / st["B"]

    def feasible(self, st):
        return min(self.element_margins(st)) >= 1.0 and self.budget_ratio(st) <= 1.0


def episode(spec, c, arm, steps=20, budget=200):
    dom = CoupledDomain(c)
    st = dom.prepare(dom.load(spec), random.Random(spec["seed"]))
    n = dom.n_elements(st)
    rng = random.Random(spec["seed"] ^ 0xABCD)
    perm = list(range(n))
    if arm == "permuted":
        rng.shuffle(perm)
    op = T.INSTALLED_OP(3.0)
    for _ in range(steps):
        if dom.feasible(st):
            return 1
        if dom.calls >= budget:
            break
        em = dom.element_margins(st)
        shown = ([em[perm[i]] for i in range(n)] if arm == "permuted"
                 else em if arm == "truthful" else None)
        best, bv = None, T.phi_rho(dom, st)
        if arm == "none":
            for f in [0.7 * (2.0 / 0.7) ** (j / 15) for j in range(16)]:
                nx = dom.clone(st)
                for i in range(n):
                    dom.set(nx, i, "x", st["x"][i] * f)
                v = T.phi_rho(dom, nx)
                if v > bv:
                    best, bv = nx, v
        else:
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
    CS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    specs = [{"seed": s, "n": 16} for s in range(N)]
    t0 = time.time()
    print("coupling sweep, n=%d instances per cell" % N)
    print("  c      truthful  permuted  none  | correct-addressing value (truthful-permuted)")
    for c in CS:
        r = {a: {sp["seed"]: episode(sp, c, a) for sp in specs}
             for a in ("truthful", "permuted", "none")}
        ks = list(r["truthful"])
        t = statistics.mean(r["truthful"][k] for k in ks)
        p_ = statistics.mean(r["permuted"][k] for k in ks)
        nn = statistics.mean(r["none"][k] for k in ks)
        hi = sum(1 for k in ks if r["truthful"][k] and not r["permuted"][k])
        lo = sum(1 for k in ks if r["permuted"][k] and not r["truthful"][k])
        d = hi + lo
        pv = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        hi2 = sum(1 for k in ks if r["permuted"][k] and not r["none"][k])
        lo2 = sum(1 for k in ks if r["none"][k] and not r["permuted"][k])
        d2 = hi2 + lo2
        pv2 = 1.0 if d2 == 0 else min(1.0, 2 * sum(comb(d2, i) for i in range(0, min(hi2, lo2) + 1)) / 2 ** d2)
        note = ""
        if c == 0.0:
            note = "  <- fully decoupled"
        if c == 1.0:
            note = "  <- fully coupled: permutation is a no-op"
        print("  %.2f    %.3f     %.3f    %.3f | %+5.1fpp (disc %3d-%-3d p=%.1e) | "
              "permuted-vs-none %+5.1fpp (p=%.1e)%s"
              % (c, t, p_, nn, 100 * (hi - lo) / len(ks), hi, lo, pv,
                 100 * (hi2 - lo2) / len(ks), pv2, note))
    print("\n%.0fs" % (time.time() - t0))
