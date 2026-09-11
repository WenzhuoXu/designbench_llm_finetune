"""
Why is the law about the whole ordering when only the worst element can bind?

Budget competition was the obvious answer and it is falsified: on the probability scale the
rho effect is flat as the resource cap is relaxed (truss Spearman +0.03), and where it moves
at all it moves the wrong way (synth +0.83). Relaxing the budget does not make the ordering
of the safe elements matter less.

The remaining account needs no budget. The action rescales EVERY element by a factor read off
its own shown margin, and the factor is allowed to SHRINK (the clip floor is 0.7). So a safe
element that is shown too healthy gets shrunk, and can be shrunk into violation. On that
account mis-ranking a safe element does not merely waste effort -- it MANUFACTURES a new
critical element. Errors anywhere in the ordering create violations anywhere.

The test: forbid shrinking. With the clip floor at 1.0 the action can only grow, so a
mis-ranked safe element can be over-grown but never pushed into violation. Crossed with
budget slack, which removes the other channel:

  grow-only + slack budget  ->  mis-ranking a safe element should cost nothing at all,
                                so the rho effect should COLLAPSE and only the critical
                                set should still matter.

If rho still dominates there, both mechanical accounts are wrong.
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
from da_matched_info import corrupt
from da_fit2 import irls
from da_topk import stats
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
BUDGET = 200
CELLS = ([("perm", r) for r in (0.0, 0.3, 0.7, 1.0)] +
         [("noise", s) for s in (0.3, 0.8, 2.5)])
ARMS = [(0.7, 1.0), (0.7, 4.0), (1.0, 1.0), (1.0, 4.0)]   # (clip floor, budget slack)


def episode(dom, st, mode, level, seed, clip_lo, steps=20):
    """The fitted episode loop with the clip floor exposed."""
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
                f = min(2.0, max(clip_lo, op(shown[i], i, tgt)))
                dom.set(nx, i, param, b * f)
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return (1 if dom.feasible(st) else 0), acc


def _pack(dm, mode, lv, clip, sl, o, acc, rid):
    r = {"domain": dm, "mode": mode, "level": lv, "clip": clip, "slack": sl,
         "y": float(o), "id": rid}
    for kk in ("rho", "ov1", "ov3", "ovq", "sp3"):
        r[kk] = statistics.mean(acc[kk]) if acc[kk] else 1.0
    return r


def _truss(t):
    f, mode, lv, clip, sl = t
    try:
        dom = get_domain("truss")
        st = dom.load(json.load(open(f)))
        mm = st["goals"].get("maximum_mass")
        if not mm:
            return None
        st["goals"] = dict(st["goals"]); st["goals"]["maximum_mass"] = mm * sl
        o, acc = episode(dom, st, mode, lv, hash(f) & 0xffff, clip)
        return _pack("truss", mode, lv, clip, sl, o, acc, f)
    except Exception:
        return None


def _synth(t):
    seed, mode, lv, clip, sl = t
    try:
        dom = SynthDomain()
        st = dom.load({"seed": seed, "n": 16})
        st["B"] = st["B"] * sl
        o, acc = episode(dom, st, mode, lv, seed ^ 0x5EED, clip)
        return _pack("synth", mode, lv, clip, sl, o, acc, "s%d" % seed)
    except Exception:
        return None


def fit_ame(rows, crit="ovq"):
    n = len(rows)
    y = [r["y"] for r in rows]
    ids = [r["id"] for r in rows]
    cells = sorted({(r["mode"], r["level"]) for r in rows})
    cix = {c: i for i, c in enumerate(cells)}
    D = [[1.0 if cix[(r["mode"], r["level"])] == j else 0.0 for j in range(1, len(cells))]
         for r in rows]
    X = [[1.0, rows[i]["rho"], rows[i][crit]] + D[i] for i in range(n)]
    b, se, cl, ll = irls(X, y, ids)
    w = 0.0
    for i in range(n):
        z = max(-30.0, min(30.0, sum(b[j] * X[i][j] for j in range(len(b)))))
        p = 1.0 / (1.0 + math.exp(-z))
        w += p * (1 - p)
    w /= n
    return (b[1] * w, b[1] / max(cl[1], 1e-9),
            b[2] * w, b[2] / max(cl[2], 1e-9), sum(y) / n)


LBL = {(0.7, 1.0): "shrink allowed, budget tight   (the fitted setting)",
       (0.7, 4.0): "shrink allowed, budget slack   (budget channel removed)",
       (1.0, 1.0): "grow only,      budget tight   (violation channel removed)",
       (1.0, 4.0): "grow only,      budget slack   (BOTH channels removed)"}

if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    files = [str(f) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    t0 = time.time()
    jt = [(f, m, l, c, s) for f in files for (m, l) in CELLS for (c, s) in ARMS]
    js = [(i, m, l, c, s) for i in range(NS) for (m, l) in CELLS for (c, s) in ARMS]
    with Pool(44) as pool:
        rt = [r for r in pool.map(_truss, jt, chunksize=4) if r]
        rs = [r for r in pool.map(_synth, js, chunksize=4) if r]
    rows = rt + rs
    with open("/ocean/projects/mch250030p/wxu7/llm_finetune/results/da_why_rows.jsonl",
              "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("%d episodes (%d truss, %d synth), %.0fs" % (len(rows), len(rt), len(rs), time.time() - t0))
    print("\nPrediction of the 'shrinking manufactures violations' account: the rho effect")
    print("should collapse in the last arm, where mis-ranking a safe element is harmless.\n")
    for dm in ("truss", "synth"):
        print("--- %s ---" % dm)
        print("  arm                                                n    base    AME rho   (z)      AME crit  (z)")
        for arm in ARMS:
            sub = [r for r in rows if r["domain"] == dm and (r["clip"], r["slack"]) == arm]
            if len(sub) < 100:
                continue
            try:
                ar, zr, ac, zc, base = fit_ame(sub)
                print("  %-48s %4d   %.3f   %+7.4f (%5.2f)   %+7.4f (%5.2f)"
                      % (LBL[arm], len(sub), base, ar, zr, ac, zc))
            except Exception as e:
                print("  %-48s fit failed: %s" % (LBL[arm], e))
        print()
