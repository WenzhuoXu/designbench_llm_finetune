"""
What moderates the slope on ordering fidelity?

The remaining gap is an account of WHEN fidelity matters most. Three attempts to transfer the
synthetic coupling law failed, but all three asked the wrong question: they tried to predict
the LEVEL of the addressing benefit. In the GLM framework the right question is whether an
instance property moderates the SLOPE on rho -- an interaction, not a main effect.

Fit, on truss episodes pooled over all twelve corruption cells:

    outcome ~ rho + m + rho:m

for each candidate moderator m, measured once per problem and standardised:

    coupling      cross-elasticity / own-elasticity, the estimator that failed before
    n_elements    problem size
    slack         1 - budget_ratio at the start state
    deficit       how far the worst element starts below requirement
    elas_spread   p90/p10 of own-elasticities across elements

A significant rho:m term means m tells you how much ordering fidelity is worth on that
instance, which is the account that has been missing. Errors are clustered by problem, since
each contributes twelve episodes.
"""
import sys, json, math, statistics, collections, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
from da_collapse import episode, CELLS
from da_fit2 import irls
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
DELTA = 0.10


def moderators(dom, st):
    n = dom.n_elements(st)
    em = dom.element_margins(st)
    if not em:
        return None
    param = dom.params[0]
    owns, crosses, els = [], [], []
    for j in range(min(n, 12)):
        b = dom.get(st, j, param)
        if not b or b <= 0 or not (0 < em[j] < float("inf")):
            continue
        nx = dom.clone(st)
        dom.set(nx, j, param, b * math.exp(DELTA))
        m1 = dom.element_margins(nx)
        if not m1:
            continue
        o, cs = None, []
        for i in range(n):
            a, c = em[i], m1[i]
            if not (0 < a < float("inf")) or not (0 < c < float("inf")):
                continue
            e = (math.log(c) - math.log(a)) / DELTA
            if i == j:
                o = abs(e)
            else:
                cs.append(abs(e))
        if o and o > 1e-6:
            owns.append(o); els.append(o)
            if cs:
                crosses.append(statistics.mean(cs))
    if not owns:
        return None
    own = statistics.median(owns)
    cross = statistics.median(crosses) if crosses else 0.0
    e = sorted(els)
    return {"coupling": cross / own if own else 0.0,
            "n_elements": float(n),
            "slack": 1.0 - dom.budget_ratio(st),
            "deficit": math.log(max(1.0 / max(min(em), 1e-6), 1.0)),
            "elas_spread": e[int(.9 * (len(e) - 1))] / max(e[int(.1 * (len(e) - 1))], 1e-6)}


def _w(f):
    try:
        dom = get_domain("truss")
        spec = json.load(open(f))
        st = dom.load(spec)
        mods = moderators(dom, st)
        if mods is None:
            return None
        out = []
        for mode, lv in CELLS:
            d2 = get_domain("truss")
            s2 = d2.load(spec)
            o, rh = episode(d2, s2, mode, lv, hash(f) & 0xffff)
            out.append((f, mode, lv, o, statistics.mean(rh) if rh else 1.0, mods))
        return out
    except Exception:
        return None


def z(vals):
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) or 1.0
    return [(v - m) / s for v in vals]


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    files = [str(f) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:N]]
    t0 = time.time()
    with Pool(40) as pool:
        packs = [r for r in pool.map(_w, files, chunksize=1) if r]
    rows = [x for pk in packs for x in pk]
    print("moderator analysis: %d episodes over %d problems, %.0fs"
          % (len(rows), len(packs), time.time() - t0))
    rho = [r[4] for r in rows]
    y = [float(r[3]) for r in rows]
    ids = [r[0] for r in rows]
    base, sb, cb, _ = irls([[1.0, rho[i]] for i in range(len(rows))], y, ids)
    print("\n  baseline   rho coefficient %+.3f (cluster z=%.2f)" % (base[1], base[1] / max(cb[1], 1e-9)))
    print("\n  moderator      main effect        interaction rho:m      verdict")
    for key in ("coupling", "n_elements", "slack", "deficit", "elas_spread"):
        raw = [r[5][key] for r in rows]
        mv = z([math.log(v) if key in ("coupling", "elas_spread") and v > 0 else v for v in raw])
        X = [[1.0, rho[i], mv[i], rho[i] * mv[i]] for i in range(len(rows))]
        b, s_, c_, _ = irls(X, y, ids)
        zi = b[3] / max(c_[3], 1e-9)
        print("  %-13s %+7.3f (z=%5.2f)   %+7.3f (z=%5.2f)    %s"
              % (key, b[2], b[2] / max(c_[2], 1e-9), b[3], zi,
                 "MODERATES" if abs(zi) > 2.0 else "no"))
    print("\n  A significant interaction is the missing account of when fidelity matters most.")
