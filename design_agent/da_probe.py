"""
Final method attempt: estimate signal fidelity by PROBING, not by inference from a summary.

The previous estimator scored the signal by how well it predicted the governing margin. That
is a min-statistic -- coarse, and confounded with error in the installed exponent. It
recovered 31% of the oracle gain and missed significance (p=0.171).

A probe measures the thing directly. Perturb one element and watch the governing margin: it
responds when that element is near binding and not otherwise. Doing this for a few elements
spread across the SHOWN ordering gives a direct read on whether that ordering is real:

    probe k elements spanning the shown ordering, one perturbation each
    r_hat = -spearman(shown margins of the probed, measured response of the probed)

A faithful signal puts its low-margin elements where the response is large, so r_hat is high.
This needs no ground truth, uses only global_margins and set() -- so it ports with the Domain
interface -- and it is the planner composing one tool to decide how to use another.

Probes are charged against the same evaluation budget as everything else, so the comparison
stays matched: an arm that probes takes those evaluations away from its own search.

PRE-REGISTERED: H1 is that the probe-gated policy beats the better fixed policy on held-out
problems 200-400, exact sign test p < 0.05. Settings are chosen on 0-200 and then frozen.
If this fails, the deployable-intervention question is settled negatively and the contribution
is the law plus the systematic failure of recoverable interventions.
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
GRID = [(k, th) for k in (4, 6, 8) for th in (0.3, 0.5, 0.7)]


def probe_fidelity(dom, st, shown, k, param, delta=1.15):
    """Perturb k elements spanning the shown ordering; correlate shown margin with response."""
    n = dom.n_elements(st)
    order = sorted(range(n), key=lambda i: shown[i])
    if n < 3:
        return None
    k = max(3, min(k, n))
    idx = [order[round(j * (n - 1) / (k - 1))] for j in range(k)]
    idx = sorted(set(idx))
    if len(idx) < 3:
        return None
    try:
        g0 = min(dom.global_margins(st))
    except Exception:
        return None
    if not g0 or g0 <= 0:
        return None
    resp = []
    for i in idx:
        b = dom.get(st, i, param)
        if not b:
            resp.append(0.0)
            continue
        nx = dom.clone(st)
        dom.set(nx, i, param, b * delta)
        try:
            g = min(dom.global_margins(nx))
        except Exception:
            g = g0
        resp.append(math.log(max(g, 1e-9) / g0))
    if max(resp) - min(resp) < 1e-12:
        return 0.0
    return -spearman([shown[i] for i in idx], resp)


def episode(dom, st, mode, level, seed, policy, k=6, theta=0.5, steps=20):
    rng = random.Random(seed)
    op = T.INSTALLED_OP(3.0)
    n = dom.n_elements(st)
    clip_lo = 0.7
    probed = False
    for _ in range(steps):
        if dom.feasible(st):
            return 1
        if dom.calls >= BUDGET:
            break
        em = dom.element_margins(st)
        if not em:
            break
        shown = corrupt(em, mode, level, rng)
        param = dom.params[0]
        if policy == "shrink":
            clip_lo = 0.7
        elif policy == "grow":
            clip_lo = 1.0
        elif policy == "oracle":
            clip_lo = 0.7 if spearman(em, shown) >= 0.8 else 1.0
        elif policy == "probe" and not probed:
            probed = True
            r = probe_fidelity(dom, st, shown, k, param)      # charged to the budget
            clip_lo = 0.7 if (r is not None and r >= theta) else 1.0
        best, bv = None, T.phi_rho(dom, st)
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
    return 1 if dom.feasible(st) else 0


ARMS = ([("shrink", "shrink", 0, 0), ("grow", "grow", 0, 0), ("oracle", "oracle", 0, 0)] +
        [("probe_%d_%.1f" % (k, th), "probe", k, th) for (k, th) in GRID])


def _run(t):
    f, mode, lv = t
    try:
        out = {"id": f, "cell": "%s%s" % (mode, lv)}
        for key, pol, k, th in ARMS:
            dom = get_domain("truss")
            st = dom.load(json.load(open(f)))
            out[key] = episode(dom, st, mode, lv, hash(f) & 0xffff, pol, k or 6, th or 0.5)
        return out
    except Exception:
        return None


if __name__ == "__main__":
    allf = sorted((DB / "data/problems_hard").glob("*.json"))
    disc = [str(f) for f in allf[0:200]]
    conf = [str(f) for f in allf[200:400]]
    keys = [a[0] for a in ARMS]
    t0 = time.time()
    with Pool(44) as pool:
        rd = [r for r in pool.map(_run, [(f, m, l) for f in disc for (m, l) in MIX], chunksize=2) if r]
        rc = [r for r in pool.map(_run, [(f, m, l) for f in conf for (m, l) in MIX], chunksize=2) if r]
    print("%d discovery + %d held-out episodes per arm, %.0fs, budget %d incl. probes"
          % (len(rd), len(rc), time.time() - t0, BUDGET))

    rd_r = {k: sum(r[k] for r in rd) / len(rd) for k in keys}
    print("\nDISCOVERY problems 0-200")
    for k in keys:
        print("  %-14s %.4f" % (k, rd_r[k]))
    best_fixed = max(("shrink", "grow"), key=lambda a: rd_r[a])
    pick = max([k for k in keys if k.startswith("probe_")], key=lambda k: rd_r[k])
    print("\n  best fixed policy : %s (%.4f)" % (best_fixed, rd_r[best_fixed]))
    print("  chosen probe arm  : %s (%.4f)" % (pick, rd_r[pick]))
    print("  oracle ceiling    : %.4f" % rd_r["oracle"])

    print("\n" + "=" * 74)
    print("HELD-OUT 200-400. One pre-registered comparison: %s vs %s" % (pick, best_fixed))
    rc_r = {k: sum(r[k] for r in rc) / len(rc) for k in keys}
    for k in ("shrink", "grow", "oracle", pick):
        print("  %-14s %.4f" % (k, rc_r[k]))
    u, d, p = sign_test([r[best_fixed] for r in rc], [r[pick] for r in rc])
    gain = rc_r[pick] - rc_r[best_fixed]
    orc = rc_r["oracle"] - rc_r[best_fixed]
    print("\n  probe-gated vs best fixed : %+.4f   discordant %d-%d   exact sign p = %.3g"
          % (gain, u, d, p))
    print("  oracle vs best fixed      : %+.4f" % orc)
    if orc > 0:
        print("  share of oracle gain recovered : %.0f%%" % (100 * gain / orc))
    print("\n  H1 %s" % ("SUPPORTED: a deployable fidelity-gated policy beats the best fixed policy"
                         if (gain > 0 and p < 0.05) else
                         "REJECTED: no deployable gain. The negative pattern stands."))
