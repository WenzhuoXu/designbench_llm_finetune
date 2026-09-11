"""
Does everything collapse onto one curve in ordering fidelity?

The address framing was withdrawn: at matched rank correlation, permutation and value noise
cost the same. The positive version of that is a law -- outcome should be a function of the
rank correlation between the shown ordering and the true one, and NOTHING else. Corruption
type should add no explanatory power, and neither should the domain.

This runs the same matched sweep on the truss (model-free) that was run synthetically, pools
both domains, fits outcome ~ rho, and asks whether corruption type or domain improves the fit.

If one curve fits all four combinations, the retraction becomes a result: ordering fidelity is
sufficient, and the way it was destroyed is irrelevant.
"""
import sys, json, math, random, statistics, collections, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
from da_synth import SynthDomain
from da_matched_info import corrupt, spearman
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
BUDGET = 200
CELLS = ([("perm", r) for r in (0.0, 0.15, 0.3, 0.5, 0.7, 1.0)] +
         [("noise", s) for s in (0.15, 0.3, 0.5, 0.8, 1.3, 2.5)])


def episode(dom, st, mode, level, seed, steps=20):
    rng = random.Random(seed)
    op = T.INSTALLED_OP(3.0)
    n = dom.n_elements(st)
    rhos = []
    for _ in range(steps):
        if dom.feasible(st):
            return 1, rhos
        if dom.calls >= BUDGET:
            break
        em = dom.element_margins(st)
        if not em:
            break
        shown = corrupt(em, mode, level, rng)
        rhos.append(spearman(em, shown))
        best, bv = None, T.phi_rho(dom, st)
        param = dom.params[0]
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(n):
                b = dom.get(st, i, param)
                if not b:
                    continue
                f = min(2.0, max(0.7, op(shown[i], i, tgt)))
                dom.set(nx, i, param, b * f)
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return (1 if dom.feasible(st) else 0), rhos


def _truss(t):
    f, mode, lv = t
    try:
        dom = get_domain("truss")
        st = dom.load(json.load(open(f)))
        o, rh = episode(dom, st, mode, lv, hash(f) & 0xffff)
        return ("truss", mode, lv, o, statistics.mean(rh) if rh else 1.0)
    except Exception:
        return None


def _synth(t):
    seed, mode, lv = t
    try:
        dom = SynthDomain()
        st = dom.load({"seed": seed, "n": 16})
        o, rh = episode(dom, st, mode, lv, seed ^ 0x5EED)
        return ("synth", mode, lv, o, statistics.mean(rh) if rh else 1.0)
    except Exception:
        return None


def logistic_fit(xs, ys, extra=None, iters=400):
    """outcome ~ sigmoid(b0 + b1*rho [+ b2*extra]); returns coefficients and log-likelihood."""
    k = 2 + (1 if extra is not None else 0)
    b = [0.0] * k
    for _ in range(iters):
        g = [0.0] * k
        for i in range(len(xs)):
            f = [1.0, xs[i]] + ([extra[i]] if extra is not None else [])
            z = sum(b[j] * f[j] for j in range(k))
            p_ = 1 / (1 + math.exp(-max(-30, min(30, z))))
            e = ys[i] - p_
            for j in range(k):
                g[j] += e * f[j]
        for j in range(k):
            b[j] += 0.02 * g[j] / len(xs)
    ll = 0.0
    for i in range(len(xs)):
        f = [1.0, xs[i]] + ([extra[i]] if extra is not None else [])
        z = sum(b[j] * f[j] for j in range(k))
        p_ = min(1 - 1e-9, max(1e-9, 1 / (1 + math.exp(-max(-30, min(30, z))))))
        ll += ys[i] * math.log(p_) + (1 - ys[i]) * math.log(1 - p_)
    return b, ll


if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    files = [str(f) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    t0 = time.time()
    with Pool(44) as pool:
        rt = [r for r in pool.map(_truss, [(f, m, l) for f in files for (m, l) in CELLS],
                                  chunksize=1) if r]
        rs = [r for r in pool.map(_synth, [(s, m, l) for s in range(NS) for (m, l) in CELLS],
                                  chunksize=1) if r]
    rows = rt + rs
    print("collapse test, %d truss + %d synthetic observations, %.0fs"
          % (len(rt), len(rs), time.time() - t0))

    by = collections.defaultdict(list)
    for dm, mode, lv, o, rho in rows:
        by[(dm, mode, lv)].append((rho, o))
    print("\n  domain  mode    level   rho     solved   n")
    for k in sorted(by):
        v = by[k]
        print("  %-7s %-6s  %.2f    %+.3f   %.3f    %d"
              % (k[0], k[1], k[2], statistics.mean(x for x, _ in v),
                 statistics.mean(y for _, y in v), len(v)))

    xs = [r[4] for r in rows]; ys = [r[3] for r in rows]
    b1, ll1 = logistic_fit(xs, ys)
    b2, ll2 = logistic_fit(xs, ys, extra=[1.0 if r[1] == "perm" else 0.0 for r in rows])
    b3, ll3 = logistic_fit(xs, ys, extra=[1.0 if r[0] == "truss" else 0.0 for r in rows])
    print("\n  outcome ~ rho only          : b1=%+.3f  logL=%.1f" % (b1[1], ll1))
    print("  + corruption-type indicator : coef=%+.3f  logL=%.1f  |  LR chi2=%.2f (1 df)"
          % (b2[2], ll2, 2 * (ll2 - ll1)))
    print("  + domain indicator          : coef=%+.3f  logL=%.1f  |  LR chi2=%.2f (1 df)"
          % (b3[2], ll3, 2 * (ll3 - ll1)))
    print("\n  chi2 above 3.84 means the extra term is doing real work at p<0.05;")
    print("  below it, ordering fidelity alone is sufficient and the curve collapses.")
