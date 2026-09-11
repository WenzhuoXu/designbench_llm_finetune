"""
Is the gap an effort failure or an information failure?

Eight interventions have now failed to cash in the value the law identifies, each at matched
budget with a held-out or pre-registered endpoint. The obvious reviewer objection is that a
ninth, cleverer estimator would work. This asks the question that settles it rather than
adding a ninth: how much information about ordering fidelity is present in what the agent can
actually observe?

The oracle policy needs one bit -- is rho above the threshold or below it. So the ceiling on
ANY gating policy, however clever, is the mutual information between the agent's observables
and that bit. If that is near zero, no estimator exists, and the eight failures are one fact
rather than eight.

Observables, all computable by the agent with no ground truth:
  pred_err   |log(predicted governing margin / observed)|, the da_adapt2 estimator
  probe_r    the probe-based rank estimate, the da_probe estimator
  disp       dispersion of the shown margins (std of log)
  minshown   the smallest shown margin
  budget     resource ratio
  nelem      element count
  turn       turn index

Mutual information is positively biased at finite sample, badly so in high dimension, so every
figure is reported against a PERMUTATION NULL: the same estimator applied after shuffling rho,
which measures exactly the bias. What matters is the excess over that null, not the raw value.
Miller-Madow correction is applied on top.
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
from da_probe import probe_fidelity
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
BUDGET = 400
THETA = 0.8
CELLS = ([("perm", r) for r in (0.0, 0.15, 0.3, 0.5, 0.7, 1.0)] +
         [("noise", s) for s in (0.15, 0.3, 0.5, 0.8, 1.3, 2.5)])
FEATS = ["pred_err", "probe_r", "disp", "minshown", "budget", "nelem", "turn"]


def episode(dom, st, mode, level, seed, steps=12):
    """Run the fitted loop, recording the true rho and the agent's observables each turn."""
    rng = random.Random(seed)
    op = T.INSTALLED_OP(3.0)
    n = dom.n_elements(st)
    rows, last_err = [], None
    for ti in range(steps):
        if dom.feasible(st) or dom.calls >= BUDGET:
            break
        em = dom.element_margins(st)
        if not em:
            break
        shown = corrupt(em, mode, level, rng)
        param = dom.params[0]
        rho = spearman(em, shown)
        if rho is None:
            break
        pr = probe_fidelity(dom, st, shown, 6, param)
        lg = [math.log(max(v, 1e-9)) for v in shown]
        obs = {"pred_err": last_err if last_err is not None else 0.0,
               "probe_r": pr if pr is not None else 0.0,
               "disp": statistics.pstdev(lg) if len(lg) > 1 else 0.0,
               "minshown": min(shown),
               "budget": dom.budget_ratio(st),
               "nelem": float(n),
               "turn": float(ti)}
        best, bv, bfac = None, T.phi_rho(dom, st), None
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            fac = []
            for i in range(n):
                b = dom.get(st, i, param)
                if not b:
                    fac.append(1.0); continue
                f = min(2.0, max(0.7, op(shown[i], i, tgt)))
                fac.append(f)
                dom.set(nx, i, param, b * f)
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv, bfac = nx, v, fac
        if best is None:
            break
        rows.append((obs, rho))
        if bfac is not None:
            try:
                pred = min(shown[i] * (bfac[i] ** 3.0) for i in range(n))
                act = min(dom.global_margins(best))
                last_err = abs(math.log(pred / act)) if (pred > 0 and act > 0) else last_err
            except Exception:
                pass
        st = best
    return rows


def _truss(t):
    f, mode, lv = t
    try:
        dom = get_domain("truss")
        st = dom.load(json.load(open(f)))
        return [(o, r, "truss") for o, r in episode(dom, st, mode, lv, hash(f) & 0xffff)]
    except Exception:
        return []


def _synth(t):
    seed, mode, lv = t
    try:
        dom = SynthDomain()
        st = dom.load({"seed": seed, "n": 16})
        return [(o, r, "synth") for o, r in episode(dom, st, mode, lv, seed ^ 0x5EED)]
    except Exception:
        return []


def qbin(vals, nb):
    """quantile bins, so each feature is discretised on its own distribution"""
    sv = sorted(vals)
    cuts = [sv[min(len(sv) - 1, int(len(sv) * j / nb))] for j in range(1, nb)]
    out = []
    for v in vals:
        k = 0
        for c in cuts:
            if v > c:
                k += 1
        out.append(k)
    return out


def mi_mm(xs, ys):
    """mutual information in bits, Miller-Madow corrected"""
    n = len(xs)
    jx = collections.Counter(zip(xs, ys))
    cx = collections.Counter(xs)
    cy = collections.Counter(ys)
    h = lambda c: -sum(v / n * math.log2(v / n) for v in c.values() if v)
    raw = h(cx) + h(cy) - h(jx)
    mm = (len(cx) - 1) * (len(cy) - 1) / (2.0 * n * math.log(2))
    return max(0.0, raw - mm)


def joint_key(cols, i):
    return tuple(c[i] for c in cols)


if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    NB = 4
    files = [str(f) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    t0 = time.time()
    with Pool(44) as pool:
        rt = pool.map(_truss, [(f, m, l) for f in files for (m, l) in CELLS], chunksize=2)
        rs = pool.map(_synth, [(s, m, l) for s in range(NS) for (m, l) in CELLS], chunksize=2)
    rows = [x for grp in (rt + rs) for x in grp]
    print("%d turn-level observations, %.0fs" % (len(rows), time.time() - t0))

    for dom_sel in ("truss", "synth", "both"):
        sub = rows if dom_sel == "both" else [r for r in rows if r[2] == dom_sel]
        if len(sub) < 500:
            continue
        rho = [r[1] for r in sub]
        y = [1 if v >= THETA else 0 for v in rho]
        base = sum(y) / len(y)
        Hy = -(base * math.log2(base) + (1 - base) * math.log2(1 - base)) if 0 < base < 1 else 0.0
        print("\n=== %s: n=%d turns, P(rho>=%.1f)=%.3f, H(bit)=%.4f bits ==="
              % (dom_sel, len(sub), THETA, base, Hy))
        cols = {f: qbin([r[0][f] for r in sub], NB) for f in FEATS}
        rng = random.Random(0)
        print("  observable    I(obs;bit)   permutation null      excess     %% of the bit")
        for f in FEATS:
            i_real = mi_mm(cols[f], y)
            nulls = []
            for _ in range(12):
                yp = y[:]; rng.shuffle(yp)
                nulls.append(mi_mm(cols[f], yp))
            nu = statistics.mean(nulls)
            exc = max(0.0, i_real - nu)
            print("  %-12s  %8.5f    %8.5f          %8.5f    %5.1f%%"
                  % (f, i_real, nu, exc, 100 * exc / Hy if Hy else 0))
        # all observables jointly
        jt = [joint_key([cols[f] for f in FEATS], i) for i in range(len(sub))]
        i_j = mi_mm(jt, y)
        nulls = []
        for _ in range(12):
            yp = y[:]; rng.shuffle(yp)
            nulls.append(mi_mm(jt, yp))
        nu = statistics.mean(nulls)
        exc = max(0.0, i_j - nu)
        print("  %-12s  %8.5f    %8.5f          %8.5f    %5.1f%%   <- ceiling on any gating policy"
              % ("ALL JOINT", i_j, nu, exc, 100 * exc / Hy if Hy else 0))
        print("  (the joint null is large because the joint alphabet is %d cells on %d points;"
              % (len(set(jt)), len(sub)))
        print("   the excess over that null is the interpretable quantity)")
