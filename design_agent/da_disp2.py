"""
Not heterogeneity -- completeness. Does coupling destroy the value of induction?

The dispersion sweep gave a clean monotone curve: induction is worth nothing below sigma~0.4
and +27pp at sigma=0.75. It also falsified the prediction it was built to test. The truss's
measured own-elasticity dispersion is 0.750, squarely in the region where induction should be
worth +27pp, and induction ties there (-0.5pp, discordant 3-4, p=1). So heterogeneity is
necessary and not sufficient, and the truss result needs a different explanation.

The candidate is that induction recovers an OWN-elasticity, which is a complete model of the
response only when elements are decoupled. In the synthetic domain g_i depends on x_i alone,
so the measurement is the whole truth. In a truss, resizing a member redistributes forces and
changes other members' margins, so an own-elasticity is a partial model however precisely it
is measured -- and a precise partial model can be worse than a crude uniform one.

This holds dispersion at the truss's own value, 0.75, and sweeps coupling instead:

    g_i = a_i * (x_i^k_i)^(1-c) * (geometric mean of all x_j^k_j)^c

c=0 is the decoupled domain where induction wins by +27pp. If completeness is the mechanism,
the gain should fall towards zero as c rises, and the truss's tie is then explained by its
coupling rather than by anything about its exponents.
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
COUPLINGS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.7)


class Coupled:
    """The synthetic domain with a tunable amount of cross-element coupling."""
    params = ("x",)

    def __init__(self):
        self.calls = 0

    def load(self, seed, c):
        r = random.Random(seed)
        n = N_ELEM
        return {"a": [math.exp(r.gauss(-0.35, 0.55)) for _ in range(n)],
                "k": [max(0.3, r.lognormvariate(0.80, SIGMA)) for _ in range(n)],
                "w": [math.exp(r.gauss(0.0, 0.4)) for _ in range(n)],
                "x": [1.0] * n, "n": n, "c": c,
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
        own = [st["a"][i] * st["x"][i] ** st["k"][i] for i in range(st["n"])]
        c = st["c"]
        if c <= 0:
            return own
        gm = math.exp(sum(math.log(max(v, 1e-9)) for v in own) / st["n"])
        return [max(v, 1e-9) ** (1 - c) * gm ** c for v in own]

    def global_margins(self, st):
        return [min(self.element_margins(st))]

    def budget_ratio(self, st):
        return sum(st["w"][i] * st["x"][i] ** 2 for i in range(st["n"])) / st["B"]

    def feasible(self, st):
        return min(self.element_margins(st)) >= 1.0 and self.budget_ratio(st) <= 1.0


def ep(seed, c, mode):
    dom = Coupled()
    st = dom.load(seed, c)
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1
        em = dom.element_margins(st)
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(st["n"]):
                b = st["x"][i]
                d = math.log(max(tgt, 1e-9) / max(em[i], 1e-9))
                # induction recovers the OWN exponent; under coupling that is a partial model
                lf = d / 3.0 if mode == "fixed" else d * SHRINK / max(st["k"][i], 0.2)
                dom.set(nx, i, "x", b * min(2.0, max(0.7, math.exp(lf))))
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def _w(t):
    seed, c, mode = t
    try:
        return (c, mode, ep(seed, c, mode))
    except Exception:
        return None


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    t0 = time.time()
    jobs = [(s, c, m) for s in range(N) for c in COUPLINGS for m in ("fixed", "induced")]
    with Pool(44) as pool:
        got = [r for r in pool.map(_w, jobs, chunksize=8) if r]
    print("%d episodes, dispersion held at the truss's own 0.75, %.0fs\n"
          % (len(got), time.time() - t0))
    print("  coupling c    fixed 3.0    induced    gain from induction")
    for c in COUPLINGS:
        a = [x[2] for x in got if x[0] == c and x[1] == "fixed"]
        b = [x[2] for x in got if x[0] == c and x[1] == "induced"]
        if not a or not b:
            continue
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        print("  %5.2f         %.3f        %.3f      %+.3f" % (c, ma, mb, mb - ma))
    print("\n  truss reference: dispersion 0.75, induction gain -0.005 (discordant 3-4, p=1)")
    print("  Completeness account: the gain should fall towards zero as coupling rises.")
