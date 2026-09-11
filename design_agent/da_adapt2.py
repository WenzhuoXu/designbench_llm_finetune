"""
The deployable version: gate action aggression on fidelity ESTIMATED FROM OBSERVABLES.

The oracle arm is told the true rank correlation and beats the best fixed policy by +3.4pp in
the truss (disc 100-19, p=1.8e-14). It is not deployable -- an agent does not have the true
ordering. This builds the estimator it would actually have.

What the agent may legitimately use: its per-element signal is possibly corrupted, but the
GLOBAL analysis result is not -- total mass and the governing margin come back from the
solver whatever the per-element reporting does. So the agent can score its own signal by
prediction error: from the shown margins and the factors it applied, predict where the
governing margin should land, then compare against where it actually landed.

    r_hat = exp( -mean_t | log( predicted governing margin / observed governing margin ) | )

r_hat needs no ground truth, costs no extra evaluations (the move is evaluated anyway), and
is defined for any domain implementing global_margins -- so it ports with the interface.

Policy: shrink while r_hat >= theta, grow only otherwise. The first turn has no history and
takes a default. theta and the default are chosen on problems 0-200 and then applied
unchanged to held-out problems 200-400.
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
from da_matched_info import corrupt, spearman
from da_adapt import sign_test, MIX, BUDGET
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
GRID = [(th, df) for th in (0.55, 0.7, 0.8, 0.9) for df in ("shrink", "grow")]


def episode(dom, st, mode, level, seed, policy, theta=0.8, default="shrink", steps=20):
    """policy: 'shrink' | 'grow' | 'oracle' | 'est'."""
    rng = random.Random(seed)
    op = T.INSTALLED_OP(3.0)
    n = dom.n_elements(st)
    errs = []
    for _ in range(steps):
        if dom.feasible(st):
            return 1
        if dom.calls >= BUDGET:
            break
        em = dom.element_margins(st)
        if not em:
            break
        shown = corrupt(em, mode, level, rng)
        if policy == "shrink":
            clip_lo = 0.7
        elif policy == "grow":
            clip_lo = 1.0
        elif policy == "oracle":
            clip_lo = 0.7 if spearman(em, shown) >= theta else 1.0
        else:
            if not errs:
                clip_lo = 0.7 if default == "shrink" else 1.0
            else:
                r_hat = math.exp(-sum(errs) / len(errs))
                clip_lo = 0.7 if r_hat >= theta else 1.0
        best, bv, bfac = None, T.phi_rho(dom, st), None
        param = dom.params[0]
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            fac = []
            for i in range(n):
                b = dom.get(st, i, param)
                if not b:
                    fac.append(1.0)
                    continue
                f = min(2.0, max(clip_lo, op(shown[i], i, tgt)))
                fac.append(f)
                dom.set(nx, i, param, b * f)
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv, bfac = nx, v, fac
        if best is None:
            break
        if policy == "est" and bfac is not None:
            # what the shown signal said the governing margin would become ...
            pred = min(shown[i] * (bfac[i] ** 3.0) for i in range(n))
            try:
                act = min(dom.global_margins(best))       # ... against what the solver reports
                if pred > 0 and act > 0:
                    errs.append(abs(math.log(pred / act)))
            except Exception:
                pass
        st = best
    return 1 if dom.feasible(st) else 0


def _run(t):
    f, mode, lv, arms = t
    try:
        out = {"id": f, "cell": "%s%s" % (mode, lv)}
        for key, pol, th, df in arms:
            dom = get_domain("truss")
            st = dom.load(json.load(open(f)))
            out[key] = episode(dom, st, mode, lv, hash(f) & 0xffff, pol, th, df)
        return out
    except Exception:
        return None


ARMS = ([("shrink", "shrink", 0, ""), ("grow", "grow", 0, ""),
         ("oracle", "oracle", 0.8, "")] +
        [("est_%.2f_%s" % (th, df), "est", th, df) for (th, df) in GRID])


def report(rows, keys, label):
    n = len(rows)
    rates = {k: sum(r[k] for r in rows) / n for k in keys}
    print("\n%s  (n=%d paired episodes)" % (label, n))
    for k in keys:
        print("  %-16s %.4f" % (k, rates[k]))
    return rates


if __name__ == "__main__":
    allf = sorted((DB / "data/problems_hard").glob("*.json"))
    disc = [str(f) for f in allf[0:200]]
    conf = [str(f) for f in allf[200:400]]
    keys = [a[0] for a in ARMS]
    t0 = time.time()
    with Pool(44) as pool:
        rd = [r for r in pool.map(_run, [(f, m, l, ARMS) for f in disc for (m, l) in MIX], chunksize=2) if r]
        rc = [r for r in pool.map(_run, [(f, m, l, ARMS) for f in conf for (m, l) in MIX], chunksize=2) if r]
    print("%d discovery + %d held-out episodes per arm, %.0fs, matched budget %d"
          % (len(rd), len(rc), time.time() - t0, BUDGET))

    rates_d = report(rd, keys, "DISCOVERY problems 0-200 (used to choose theta and the default)")
    best_fixed_d = max(("shrink", "grow"), key=lambda a: rates_d[a])
    est_keys = [k for k in keys if k.startswith("est_")]
    pick = max(est_keys, key=lambda k: rates_d[k])
    print("\n  best fixed policy on discovery : %s (%.4f)" % (best_fixed_d, rates_d[best_fixed_d]))
    print("  chosen estimator setting       : %s (%.4f)" % (pick, rates_d[pick]))
    print("  oracle ceiling on discovery    : %.4f" % rates_d["oracle"])

    print("\n" + "=" * 78)
    print("HELD-OUT problems 200-400. One comparison, decided above: %s vs %s" % (pick, best_fixed_d))
    rates_c = report(rc, ["shrink", "grow", "oracle", pick], "held-out")
    u, d, p = sign_test([r[best_fixed_d] for r in rc], [r[pick] for r in rc])
    gain = rates_c[pick] - rates_c[best_fixed_d]
    print("\n  estimator vs best fixed : %+.4f   discordant %d-%d   exact sign p = %.3g" % (gain, u, d, p))
    orc = rates_c["oracle"] - rates_c[best_fixed_d]
    print("  oracle vs best fixed    : %+.4f" % orc)
    if orc > 0:
        print("  share of the oracle gain recovered from observables: %.0f%%" % (100 * gain / orc))
    print("\n  VERDICT: %s" % ("estimator beats the best fixed policy out of sample"
                               if (gain > 0 and p < 0.05) else
                               "no held-out gain -- the estimator does not carry the oracle result"))
