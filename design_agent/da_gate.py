"""
Gate on a FITTED predictor rather than a hand-designed statistic.

The information audit came back in two voices. Mutual information says the agent's observables
carry 22.8% of the bit the oracle policy needs, which reads as very little. A classifier fitted
on the same observables predicts that bit out of sample at AUC 0.81, accuracy 0.72 against a
0.52 majority -- which reads as a usable predictor. Both are right; 0.23 bits is consistent
with good-but-imperfect discrimination.

So the earlier conclusion was wrong. The information is not absent. What failed was the
extraction: prediction-error and probe-rank were single hand-built statistics, worth 5.8% and
5.5% of the bit on their own, while the fitted combination of all of them is worth far more.

This builds the policy the audit actually licenses. Features are PREFIX aggregates -- computed
only from turns already taken -- so training and deployment see the same thing, and the
decision can be revised every turn. probe_r is excluded so the policy costs no evaluations at
all: every feature comes from observations the loop already makes.

Coefficients are fitted on problems 0-200 and then frozen. Held-out problems 200-400 carry one
comparison against the better fixed policy, at matched budget.
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
from da_matched_info import corrupt, spearman
from da_adapt import sign_test, MIX, BUDGET
from da_fit2 import irls
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
THETA = 0.8
# free features only: no probing, no extra evaluations
FEATS = ["pred_err", "disp", "minshown", "budget", "nelem", "turn"]
AGG = ["_now", "_mean", "_last"]


def featurise(hist, cur):
    """prefix aggregates: only turns already taken, plus this turn's directly visible state"""
    f = {}
    for k in FEATS:
        f[k + "_now"] = cur[k]
        v = [h[k] for h in hist] or [cur[k]]
        f[k + "_mean"] = statistics.mean(v)
        f[k + "_last"] = v[-1]
    return f


def observables(dom, st, shown, n, ti, last_err):
    lg = [math.log(max(v, 1e-9)) for v in shown]
    return {"pred_err": last_err if last_err is not None else 0.0,
            "disp": statistics.pstdev(lg) if len(lg) > 1 else 0.0,
            "minshown": min(shown),
            "budget": dom.budget_ratio(st),
            "nelem": float(n),
            "turn": float(ti)}


COLS = [k + a for k in FEATS for a in AGG]


def episode(dom, st, mode, level, seed, policy, coef=None, norm=None, steps=20, collect=None):
    rng = random.Random(seed)
    op = T.INSTALLED_OP(3.0)
    n = dom.n_elements(st)
    hist, last_err = [], None
    for ti in range(steps):
        if dom.feasible(st):
            return 1
        if dom.calls >= BUDGET:
            break
        em = dom.element_margins(st)
        if not em:
            break
        shown = corrupt(em, mode, level, rng)
        param = dom.params[0]
        rho = spearman(em, shown)
        cur = observables(dom, st, shown, n, ti, last_err)
        fv = featurise(hist, cur)
        if collect is not None and rho is not None:
            collect.append((fv, 1.0 if rho >= THETA else 0.0))
        if policy == "shrink":
            clip_lo = 0.7
        elif policy == "grow":
            clip_lo = 1.0
        elif policy == "oracle":
            clip_lo = 0.7 if (rho is not None and rho >= THETA) else 1.0
        else:
            x = [1.0] + [(fv[c] - norm[c][0]) / norm[c][1] for c in COLS]
            z = sum(coef[j] * x[j] for j in range(len(coef)))
            clip_lo = 0.7 if 1.0 / (1.0 + math.exp(-max(-30, min(30, z)))) >= 0.5 else 1.0
        best, bv, bfac = None, T.phi_rho(dom, st), None
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            fac = []
            for i in range(n):
                b = dom.get(st, i, param)
                if not b:
                    fac.append(1.0); continue
                f = min(2.0, max(clip_lo, op(shown[i], i, tgt)))
                fac.append(f)
                dom.set(nx, i, param, b * f)
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv, bfac = nx, v, fac
        if best is None:
            break
        hist.append(cur)
        if bfac is not None:
            try:
                pred = min(shown[i] * (bfac[i] ** 3.0) for i in range(n))
                act = min(dom.global_margins(best))
                last_err = abs(math.log(pred / act)) if (pred > 0 and act > 0) else last_err
            except Exception:
                pass
        st = best
    return 1 if dom.feasible(st) else 0


def _train(t):
    f, mode, lv = t
    try:
        dom = get_domain("truss")
        st = dom.load(json.load(open(f)))
        acc = []
        episode(dom, st, mode, lv, hash(f) & 0xffff, "shrink", collect=acc)
        return acc
    except Exception:
        return []


def _eval(t):
    f, mode, lv, coef, norm = t
    try:
        out = {"id": f}
        for arm in ("shrink", "grow", "oracle", "gated"):
            dom = get_domain("truss")
            st = dom.load(json.load(open(f)))
            out[arm] = episode(dom, st, mode, lv, hash(f) & 0xffff, arm, coef, norm)
        return out
    except Exception:
        return None


if __name__ == "__main__":
    allf = sorted((DB / "data/problems_hard").glob("*.json"))
    tr_f = [str(f) for f in allf[0:200]]
    te_f = [str(f) for f in allf[200:400]]
    t0 = time.time()
    with Pool(44) as pool:
        packs = pool.map(_train, [(f, m, l) for f in tr_f for (m, l) in MIX], chunksize=2)
    data = [x for pk in packs for x in pk]
    print("training turns: %d  (from problems 0-200)" % len(data))
    norm = {}
    for c in COLS:
        v = [d[0][c] for d in data]
        norm[c] = (statistics.mean(v), statistics.pstdev(v) or 1.0)
    X = [[1.0] + [(d[0][c] - norm[c][0]) / norm[c][1] for c in COLS] for d in data]
    y = [d[1] for d in data]
    coef, se, cl, ll = irls(X, y)
    pr = []
    for i in range(len(X)):
        z = sum(coef[j] * X[i][j] for j in range(len(coef)))
        pr.append(1.0 / (1.0 + math.exp(-max(-30, min(30, z)))))
    ins = sum(1 for i in range(len(y)) if (pr[i] >= .5) == bool(y[i])) / len(y)
    print("  in-sample accuracy %.4f  (base %.4f)" % (ins, max(sum(y) / len(y), 1 - sum(y) / len(y))))

    with Pool(44) as pool:
        rows = [r for r in pool.map(
            _eval, [(f, m, l, coef, norm) for f in te_f for (m, l) in MIX], chunksize=2) if r]
    print("\nHELD-OUT problems 200-400: %d episodes per arm, %.0fs, matched budget"
          % (len(rows), time.time() - t0))
    rate = {a: sum(r[a] for r in rows) / len(rows) for a in ("shrink", "grow", "oracle", "gated")}
    for a in ("shrink", "grow", "oracle", "gated"):
        print("  %-8s %.4f" % (a, rate[a]))
    bf = max(("shrink", "grow"), key=lambda a: rate[a])
    u, d, p = sign_test([r[bf] for r in rows], [r["gated"] for r in rows])
    gain = rate["gated"] - rate[bf]
    orc = rate["oracle"] - rate[bf]
    print("\n  gated vs best fixed (%s): %+.4f  discordant %d-%d  exact sign p = %.3g"
          % (bf, gain, u, d, p))
    print("  oracle vs best fixed     : %+.4f" % orc)
    if orc > 0:
        print("  share of oracle gain recovered: %.0f%%" % (100 * gain / orc))
    print("\n  %s" % ("GATED POLICY WINS out of sample" if (gain > 0 and p < 0.05)
                      else "no significant held-out gain"))
