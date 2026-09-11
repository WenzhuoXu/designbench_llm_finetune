"""
Is "addressing" a construct, or just information loss with a name?

The permutation control holds the multiset of values, the amplitude, the format and the
token count fixed and destroys only the item-to-value mapping. That is what makes it a claim
about correspondence. But in-loop, at matched rank correlation between shown and true, address
corruption and value corruption were indistinguishable -- a null I reported. If that holds
with no model either, the honest claim shrinks from "correspondence carries the value" to
"the ordering carries it, and permutation is one way to destroy the ordering".

The synthetic domain lets both corruptions be dialled and matched exactly:

  perm(rho)   permute a fraction rho of the shown values among themselves
  noise(sig)  multiply each shown value by exp(N(0, sig))

Both are scored on the SAME axis -- Spearman between the shown ordering and the true one --
so the comparison is at matched information loss rather than at matched dial setting.

If the two lie on one curve, the distinctive claim is narrower than I have been stating, and
the honest version is: no marginal statistic of the signal carries its value, since permutation
preserves every one of them and removes everything.
"""
import sys, math, random, statistics, collections, time
from math import comb
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from da_synth import SynthDomain
import da_tools as T


def spearman(a, b):
    n = len(a)
    def rk(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for p_, i in enumerate(o):
            r[i] = p_
        return r
    ra, rb = rk(a), rk(b)
    m = (n - 1) / 2
    num = sum((ra[i] - m) * (rb[i] - m) for i in range(n))
    den = (sum((ra[i] - m) ** 2 for i in range(n)) * sum((rb[i] - m) ** 2 for i in range(n))) ** .5
    return num / den if den else 0.0


def corrupt(vals, mode, level, rng):
    n = len(vals)
    out = list(vals)
    if mode == "perm" and level > 0:
        k = max(2, int(round(level * n)))
        idx = rng.sample(range(n), min(k, n))
        pool = [out[i] for i in idx]
        rng.shuffle(pool)
        for i, v in zip(idx, pool):
            out[i] = v
    elif mode == "noise" and level > 0:
        out = [v * math.exp(rng.gauss(0, level)) for v in out]
    return out


def episode(spec, mode, level, steps=20, budget=200):
    dom = SynthDomain()
    st = dom.load(spec)
    rng = random.Random(spec["seed"] ^ 0x5EED)
    op = T.INSTALLED_OP(3.0)
    n = dom.n_elements(st)
    rhos = []
    for _ in range(steps):
        if dom.feasible(st):
            return 1, rhos
        if dom.calls >= budget:
            break
        em = dom.element_margins(st)
        shown = corrupt(em, mode, level, rng)
        rhos.append(spearman(em, shown))
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
    return (1 if dom.feasible(st) else 0), rhos


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    specs = [{"seed": s, "n": 16} for s in range(N)]
    CELLS = ([("perm", r) for r in (0.0, 0.15, 0.3, 0.5, 0.7, 1.0)] +
             [("noise", s) for s in (0.15, 0.3, 0.5, 0.8, 1.3, 2.5)])
    t0 = time.time()
    print("matched-information comparison, synthetic domain, n=%d instances per cell\n" % N)
    print("  mode    level   mean rank corr(shown, true)   solved")
    pts = []
    for mode, lv in CELLS:
        outs, rr = [], []
        for sp in specs:
            o, rh = episode(sp, mode, lv)
            outs.append(o); rr += rh
        rho = statistics.mean(rr) if rr else 1.0
        rate = statistics.mean(outs)
        pts.append((mode, lv, rho, rate))
        print("  %-6s  %.2f    %+.3f                    %.3f" % (mode, lv, rho, rate))

    print("\n  nearest-neighbour pairs at matched rank correlation:")
    perms = [p for p in pts if p[0] == "perm"]
    for p in [x for x in pts if x[0] == "noise"]:
        near = min(perms, key=lambda q: abs(q[2] - p[2]))
        print("    noise %.2f (corr %+.3f, %.3f)  vs  perm %.2f (corr %+.3f, %.3f)"
              "   -> address penalty %+.3f"
              % (p[1], p[2], p[3], near[1], near[2], near[3], near[3] - p[3]))
    print("\n  If the address penalties are near zero, the two corruptions lie on one curve and")
    print("  the honest claim is about ordering, not addressing specifically.")
    print("  %.0fs" % (time.time() - t0))
