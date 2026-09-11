"""
Repairing the law: two signal properties instead of one, tested by extrapolation.

Ordering fidelity is necessary and not sufficient -- an arm with rho=1.000 and synthetic
magnitudes loses 28pp. Every earlier sweep moved ordering and magnitude together, so rho was
standing in for both and no fit could have separated them.

This varies them independently by construction:

  ORDERING   a permutation applied to the assignment, at five strengths -> controls rho
  MAGNITUDE  the shown values are interpolated in log space between the true multiset and a
             canonical ladder, at five strengths -> controls how right the magnitudes are,
             while leaving the assignment untouched

Two measurable coordinates, both computable by an observer who can see the true signal:

  rho = spearman(true, shown)                                assignment fidelity
  mu  = pearson(log sorted(shown), log sorted(true))         magnitude fidelity, assignment-free

The test is extrapolation, not fit quality. The model is fitted on INTERIOR cells only and then
asked to predict the four corner arms it never saw -- true, rank-only, wrong-scale, permuted --
whose outcomes are already measured. A two-parameter law that predicts corners from interiors
is worth having; one that only fits its own grid is not.
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
from da_synth import SynthDomain
from da_fit2 import irls
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
CAP, STEPS, BUDGET = 50.0, 6, 200
PERMS = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
LADDER = (0.5, 3.0)


def spearman(a, b):
    n = len(a)
    if n < 3:
        return None
    def rk(v):
        o = sorted(range(n), key=lambda i: v[i]); r = [0] * n
        for p_, i in enumerate(o):
            r[i] = p_
        return r
    ra, rb = rk(a), rk(b)
    m = (n - 1) / 2
    num = sum((ra[i] - m) * (rb[i] - m) for i in range(n))
    den = (sum((ra[i] - m) ** 2 for i in range(n)) * sum((rb[i] - m) ** 2 for i in range(n))) ** .5
    return num / den if den else None


def pearson(a, b):
    n = len(a)
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = sum((x - ma) ** 2 for x in a) ** .5
    vb = sum((x - mb) ** 2 for x in b) ** .5
    if va * vb == 0:
        return 1.0
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (va * vb)


def clean(em):
    out = []
    for x in em:
        try:
            x = min(float(x), CAP)
        except Exception:
            x = CAP
        out.append(x if (math.isfinite(x) and x > 0) else CAP)
    return out


def make_signal(em, pfrac, alpha, rng, ladder=LADDER):
    """pfrac controls the assignment, alpha controls the magnitudes, independently."""
    v = clean(em)
    n = len(v)
    order = sorted(range(n), key=lambda i: v[i])
    # magnitudes: blend the true sorted multiset toward a canonical ladder in log space
    lo, hi = ladder
    srt = sorted(v)
    mags = []
    for r in range(n):
        lt = math.log(srt[r])
        ll = math.log(lo * (hi / lo) ** (r / max(n - 1, 1)))
        mags.append(math.exp((1 - alpha) * lt + alpha * ll))
    shown = [0.0] * n
    for r, i in enumerate(order):
        shown[i] = mags[r]                        # rank-preserving assignment so far
    # assignment: permute a fraction of the positions
    idx = list(range(n))
    k = int(round(pfrac * n))
    if k >= 2:
        pick = rng.sample(idx, k)
        vals = [shown[i] for i in pick]
        rng.shuffle(vals)
        for i, val in zip(pick, vals):
            shown[i] = val
    return shown, v


def run(dom, st, param, pfrac, alpha, seed, ladder=LADDER):
    rng = random.Random(seed)
    op = T.INSTALLED_OP(3.0)
    rr, mm = [], []
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1, rr, mm
        if getattr(dom, "calls", 0) >= BUDGET:
            break
        em = dom.element_margins(st)
        if not em:
            break
        shown, v = make_signal(em, pfrac, alpha, rng, ladder)
        r = spearman(v, shown)
        if r is not None:
            rr.append(r)
        mm.append(pearson([math.log(x) for x in sorted(shown)],
                          [math.log(x) for x in sorted(v)]))
        n = dom.n_elements(st)
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(n):
                b = dom.get(st, i, param)
                if not b:
                    continue
                dom.set(nx, i, param, b * min(2.0, max(0.7, op(shown[i], i, tgt))))
            val = T.phi_rho(dom, nx)
            if val > bv:
                best, bv = nx, val
        if best is None:
            break
        st = best
    return (1 if dom.feasible(st) else 0), rr, mm


def _job(t):
    kind, key, pf, al, ladder, tag = t
    try:
        if kind == "truss":
            dom = get_domain("truss"); st = dom.load(json.load(open(key))); pm = "r"
            sd = hash(key) & 0xffff
        else:
            dom = SynthDomain(); st = dom.load({"seed": key, "n": 10}); pm = "x"
            sd = key ^ 0x5EED
        y, rr, mm = run(dom, st, pm, pf, al, sd, ladder)
        if not rr:
            return None
        return (kind, tag, str(key), float(y), statistics.mean(rr), statistics.mean(mm))
    except Exception:
        return None


CORNERS = [("true", 0.0, 0.0, LADDER), ("rank_only", 0.0, 1.0, LADDER),
           ("wrong_scale", 0.0, 1.0, (0.05, 40.0)), ("permuted", 1.0, 0.0, LADDER)]

if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    files = [str(x) for x in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    grid = [(pf, al) for pf in PERMS for al in ALPHAS]
    jobs = []
    for pf, al in grid:
        jobs += [("truss", f, pf, al, LADDER, "grid") for f in files]
        jobs += [("synth", s, pf, al, LADDER, "grid") for s in range(NS)]
    for nm, pf, al, ld in CORNERS:
        jobs += [("truss", f, pf, al, ld, nm) for f in files]
        jobs += [("synth", s, pf, al, ld, nm) for s in range(NS)]
    t0 = time.time()
    with Pool(44) as pool:
        got = [r for r in pool.map(_job, jobs, chunksize=4) if r]
    print("%d episodes, %.0fs\n" % (len(got), time.time() - t0))

    for dm in ("truss", "synth"):
        sub = [x for x in got if x[0] == dm]
        grid_rows = [x for x in sub if x[1] == "grid"]
        # interior only: drop cells at the extremes of BOTH knobs
        interior = [x for x in grid_rows if 0.0 < x[4] < 0.999 and 0.0 < x[5] < 0.999]
        if len(interior) < 300:
            interior = grid_rows
        y = [x[3] for x in interior]
        ids = [x[2] for x in interior]
        X = [[1.0, x[4], x[5], x[4] * x[5]] for x in interior]
        b, se, cl, ll = irls(X, y, ids)
        print("--- %s: fitted on %d INTERIOR episodes ---" % (dm, len(interior)))
        for i, nm in enumerate(["intercept", "rho (ordering)", "mu (magnitude)", "rho x mu"]):
            print("  %-16s %+7.3f   cluster z = %6.2f" % (nm, b[i], b[i] / max(cl[i], 1e-9)))
        Xr = [[1.0, x[4]] for x in interior]
        br, _, _, llr = irls(Xr, y, ids)
        print("  adding mu and the interaction to rho alone: LR chi2 = %.1f (2 df)" % (2 * (ll - llr)))
        print("\n  extrapolation to corner arms never used in the fit:")
        print("    arm            rho     mu     predicted   observed   error")
        for nm, _, _, _ in CORNERS:
            c = [x for x in sub if x[1] == nm]
            if not c:
                continue
            rho = statistics.mean([x[4] for x in c])
            mu = statistics.mean([x[5] for x in c])
            z = b[0] + b[1] * rho + b[2] * mu + b[3] * rho * mu
            pred = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            obs = sum(x[3] for x in c) / len(c)
            print("    %-12s %6.3f %6.3f    %.3f       %.3f      %+.3f"
                  % (nm, rho, mu, pred, obs, pred - obs))
        print()
