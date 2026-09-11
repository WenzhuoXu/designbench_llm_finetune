"""
Which ordering is the law about -- the whole ordering, or the critical few?

The fitted law uses global Spearman between the shown per-element margins and the true ones.
But feasibility is a min-margin condition: only the worst elements can bind. If outcome is
really governed by whether the signal identifies the CRITICAL SET, then the law is directly
prescriptive -- it says concentrate a probe budget on the elements that might be worst. If
instead global fidelity is what matters, no concentration is possible and the prescription
is much weaker.

Every fidelity statistic is driven by the corruption level, so marginal comparisons are
confounded. The decisive fit therefore carries CELL FIXED EFFECTS: within a single
(corruption type, level) cell, does critical-set fidelity still predict outcome once global
fidelity is controlled for, and vice versa?
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
from da_fit2 import irls
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
BUDGET = 200
CELLS = ([("perm", r) for r in (0.0, 0.15, 0.3, 0.5, 0.7, 1.0)] +
         [("noise", s) for s in (0.15, 0.3, 0.5, 0.8, 1.3, 2.5)])


def bottom_k(v, k):
    """Indices of the k smallest values -- low margin means close to violation."""
    return set(sorted(range(len(v)), key=lambda i: v[i])[:k])


def topk_overlap(em, shown, k):
    """Share of the true critical set that the shown signal also puts in its critical set."""
    k = max(1, min(k, len(em)))
    return len(bottom_k(em, k) & bottom_k(shown, k)) / float(k)


def topk_spearman(em, shown, k):
    """Rank agreement restricted to the true critical set (ordering WITHIN the few that bind)."""
    k = max(2, min(k, len(em)))
    idx = sorted(bottom_k(em, k))
    return spearman([em[i] for i in idx], [shown[i] for i in idx])


def stats(em, shown):
    n = len(em)
    q = max(1, n // 4)
    return {
        "rho": spearman(em, shown),
        "ov1": topk_overlap(em, shown, 1),
        "ov3": topk_overlap(em, shown, 3),
        "ovq": topk_overlap(em, shown, q),
        "sp3": topk_spearman(em, shown, 3),
    }


def episode(dom, st, mode, level, seed, steps=20):
    """Identical to the fitted episode loop; only the recording is extended."""
    rng = random.Random(seed)
    op = T.INSTALLED_OP(3.0)
    n = dom.n_elements(st)
    acc = collections.defaultdict(list)
    for _ in range(steps):
        if dom.feasible(st):
            return 1, acc
        if dom.calls >= BUDGET:
            break
        em = dom.element_margins(st)
        if not em:
            break
        shown = corrupt(em, mode, level, rng)
        for kk, vv in stats(em, shown).items():
            acc[kk].append(vv)
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
    return (1 if dom.feasible(st) else 0), acc


def _pack(dm, mode, lv, o, acc):
    r = {"domain": dm, "mode": mode, "level": lv, "y": float(o)}
    for kk in ("rho", "ov1", "ov3", "ovq", "sp3"):
        r[kk] = statistics.mean(acc[kk]) if acc[kk] else 1.0
    return r


def _truss(t):
    f, mode, lv = t
    try:
        dom = get_domain("truss")
        st = dom.load(json.load(open(f)))
        o, acc = episode(dom, st, mode, lv, hash(f) & 0xffff)
        r = _pack("truss", mode, lv, o, acc); r["id"] = f
        return r
    except Exception:
        return None


def _synth(t):
    seed, mode, lv = t
    try:
        dom = SynthDomain()
        st = dom.load({"seed": seed, "n": 16})
        o, acc = episode(dom, st, mode, lv, seed ^ 0x5EED)
        r = _pack("synth", mode, lv, o, acc); r["id"] = "s%d" % seed
        return r
    except Exception:
        return None


def corr(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    if va * vb == 0:
        return 0.0
    return sum((a[i] - ma) * (b[i] - mb) for i in range(len(a))) / (va * vb)


NAMES = {"rho": "global Spearman (fitted law)",
         "ov1": "worst element identified",
         "ov3": "critical-3 set overlap",
         "ovq": "critical-quartile overlap",
         "sp3": "ordering within critical 3"}

if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    files = [str(f) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    t0 = time.time()
    with Pool(44) as pool:
        rt = [r for r in pool.map(_truss, [(f, m, l) for f in files for (m, l) in CELLS], chunksize=1) if r]
        rs = [r for r in pool.map(_synth, [(s, m, l) for s in range(NS) for (m, l) in CELLS], chunksize=1) if r]
    rows = rt + rs
    y = [r["y"] for r in rows]
    ids = [r["id"] for r in rows]
    n = len(rows)
    print("%d episodes (%d truss, %d synthetic), %.0fs, base rate %.3f"
          % (n, len(rt), len(rs), time.time() - t0, sum(y) / n))

    print("\n--- 1. each fidelity statistic on its own (same df, so logL is comparable) ---")
    print("  statistic                        coef    cluster z      logL")
    marg = {}
    for kk in ("rho", "ov1", "ov3", "ovq", "sp3"):
        v = [r[kk] for r in rows]
        b, se, cl, ll = irls([[1.0, v[i]] for i in range(n)], y, ids)
        marg[kk] = (b[1], b[1] / max(cl[1], 1e-9), ll)
        print("  %-30s %+7.3f    %7.2f   %9.1f" % (NAMES[kk], b[1], b[1] / max(cl[1], 1e-9), ll))
    best = max(marg, key=lambda k_: marg[k_][2])
    print("  best marginal fit: %s" % NAMES[best])

    print("\n--- 2. how collinear are they? (corruption level drives all of them) ---")
    keys = ("rho", "ov1", "ov3", "ovq", "sp3")
    print("        " + "".join("%8s" % k_ for k_ in keys))
    for a in keys:
        print("  %-6s" % a + "".join("%8.3f" % corr([r[a] for r in rows], [r[b_] for r in rows]) for b_ in keys))

    print("\n--- 3. decisive test: both terms, WITH cell fixed effects ---")
    print("    (within a corruption cell, which statistic still moves the outcome?)")
    cells = sorted({(r["mode"], r["level"]) for r in rows})
    cix = {c: i for i, c in enumerate(cells)}
    D = [[1.0 if cix[(r["mode"], r["level"])] == j else 0.0 for j in range(1, len(cells))] for r in rows]
    for kk in ("ov1", "ov3", "ovq", "sp3"):
        X = [[1.0, rows[i]["rho"], rows[i][kk]] + D[i] for i in range(n)]
        b, se, cl, ll = irls(X, y, ids)
        Xr = [[1.0, rows[i]["rho"]] + D[i] for i in range(n)]
        br, ser, clr, llr = irls(Xr, y, ids)
        Xr2 = [[1.0, rows[i][kk]] + D[i] for i in range(n)]
        b2, se2, cl2, ll2 = irls(Xr2, y, ids)
        print("\n  global rho  vs  %s" % NAMES[kk])
        print("    joint:  rho %+6.3f (z=%5.2f)   %s %+6.3f (z=%5.2f)"
              % (b[1], b[1] / max(cl[1], 1e-9), kk, b[2], b[2] / max(cl[2], 1e-9)))
        print("    adding %s to rho:  LR chi2 = %6.2f" % (kk, 2 * (ll - llr)))
        print("    adding rho to %s:  LR chi2 = %6.2f" % (kk, 2 * (ll - ll2)))
