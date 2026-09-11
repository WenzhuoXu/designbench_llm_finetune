"""
Why is the law about the WHOLE ordering rather than the critical few?

da_topk found that global Spearman beats every critical-set statistic, and survives their
inclusion under cell fixed effects while they do not survive its. That is not obvious: only
the worst element can make a design infeasible, so naively the ordering of the safe elements
should be irrelevant.

The proposed mechanism is the BUDGET. The action is a global reallocation -- every element is
rescaled by a factor read off its shown margin -- and under a resource cap, material given to
an element that did not need it is material taken from one that did. On that account a
mis-ranked safe element is harmful only because the budget binds.

That yields a falsifiable prediction: SLACKEN THE BUDGET and the weight should move from
global fidelity to critical-set fidelity. If the two coefficients instead sit still as the
cap is relaxed, the budget account is wrong and global fidelity matters for some other reason.

Slack is applied as a multiplier on the resource cap: the truss maximum_mass, the synthetic B.
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
from da_fit2 import irls
from da_topk import episode, stats

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
CELLS = ([("perm", r) for r in (0.0, 0.3, 0.7, 1.0)] +
         [("noise", s) for s in (0.3, 0.8, 2.5)])
SLACK = (0.85, 1.0, 1.3, 1.8, 2.6, 4.0)


def _truss(t):
    f, mode, lv, sl = t
    try:
        dom = get_domain("truss")
        spec = json.load(open(f))
        st = dom.load(spec)
        mm = st["goals"].get("maximum_mass")
        if not mm:
            return None
        st["goals"] = dict(st["goals"]); st["goals"]["maximum_mass"] = mm * sl
        o, acc = episode(dom, st, mode, lv, hash(f) & 0xffff)
        r = {"domain": "truss", "mode": mode, "level": lv, "slack": sl,
             "y": float(o), "id": f}
        for kk in ("rho", "ov1", "ov3", "ovq", "sp3"):
            r[kk] = statistics.mean(acc[kk]) if acc[kk] else 1.0
        return r
    except Exception:
        return None


def _synth(t):
    seed, mode, lv, sl = t
    try:
        dom = SynthDomain()
        st = dom.load({"seed": seed, "n": 16})
        st["B"] = st["B"] * sl
        o, acc = episode(dom, st, mode, lv, seed ^ 0x5EED)
        r = {"domain": "synth", "mode": mode, "level": lv, "slack": sl,
             "y": float(o), "id": "s%d" % seed}
        for kk in ("rho", "ov1", "ov3", "ovq", "sp3"):
            r[kk] = statistics.mean(acc[kk]) if acc[kk] else 1.0
        return r
    except Exception:
        return None


def within(rows, crit="ovq"):
    """outcome ~ rho + crit + cell fixed effects, on one slack level."""
    n = len(rows)
    y = [r["y"] for r in rows]
    ids = [r["id"] for r in rows]
    cells = sorted({(r["mode"], r["level"]) for r in rows})
    cix = {c: i for i, c in enumerate(cells)}
    D = [[1.0 if cix[(r["mode"], r["level"])] == j else 0.0 for j in range(1, len(cells))]
         for r in rows]
    X = [[1.0, rows[i]["rho"], rows[i][crit]] + D[i] for i in range(n)]
    b, se, cl, ll = irls(X, y, ids)
    return b[1], b[1] / max(cl[1], 1e-9), b[2], b[2] / max(cl[2], 1e-9)


if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    files = [str(f) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    t0 = time.time()
    jt = [(f, m, l, s) for f in files for (m, l) in CELLS for s in SLACK]
    js = [(i, m, l, s) for i in range(NS) for (m, l) in CELLS for s in SLACK]
    with Pool(44) as pool:
        rt = [r for r in pool.map(_truss, jt, chunksize=4) if r]
        rs = [r for r in pool.map(_synth, js, chunksize=4) if r]
    rows = rt + rs
    print("%d episodes (%d truss, %d synth), %.0fs" % (len(rows), len(rt), len(rs), time.time() - t0))

    print("\nPrediction: as the cap is relaxed, the rho coefficient should FALL and the")
    print("critical-set coefficient should RISE. Flat coefficients falsify the budget account.\n")
    for dm in ("truss", "synth", "both"):
        sub0 = rows if dm == "both" else [r for r in rows if r["domain"] == dm]
        if len(sub0) < 200:
            continue
        print("--- %s ---" % dm)
        print("  slack   n     base    mean rho   rho coef  (z)     crit-quartile coef  (z)")
        for sl in SLACK:
            sub = [r for r in sub0 if r["slack"] == sl]
            if len(sub) < 100:
                continue
            base = sum(r["y"] for r in sub) / len(sub)
            if base < 0.02 or base > 0.98:
                print("  %5.2f  %4d   %.3f    %.3f      (degenerate outcome, not fitted)"
                      % (sl, len(sub), base, statistics.mean([r["rho"] for r in sub])))
                continue
            try:
                br, zr, bc, zc = within(sub)
                print("  %5.2f  %4d   %.3f    %.3f    %+7.3f (%5.2f)      %+7.3f (%5.2f)"
                      % (sl, len(sub), base, statistics.mean([r["rho"] for r in sub]), br, zr, bc, zc))
            except Exception as e:
                print("  %5.2f  %4d   %.3f    fit failed: %s" % (sl, len(sub), base, e))
        print()

    print("--- formal test: does slack moderate the rho slope? ---")
    fit = [r for r in rows if 0.0 < r["slack"] <= 4.0]
    y = [r["y"] for r in fit]
    ids = [r["id"] for r in fit]
    ls = [math.log(r["slack"]) for r in fit]
    ms = statistics.mean(ls)
    ls = [v - ms for v in ls]
    rho = [r["rho"] for r in fit]
    ovq = [r["ovq"] for r in fit]
    X = [[1.0, rho[i], ovq[i], ls[i], rho[i] * ls[i], ovq[i] * ls[i]] for i in range(len(fit))]
    b, se, cl, ll = irls(X, y, ids)
    lbl = ["intercept", "rho", "crit-quartile", "log slack", "rho x log slack", "crit x log slack"]
    for i, nm in enumerate(lbl):
        print("  %-18s %+7.3f   cluster z = %6.2f" % (nm, b[i], b[i] / max(cl[i], 1e-9)))
    print("\n  budget account predicts: 'rho x log slack' NEGATIVE, 'crit x log slack' POSITIVE.")
