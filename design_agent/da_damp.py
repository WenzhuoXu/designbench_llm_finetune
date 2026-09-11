"""
Does tool induction win once damping is separated from measurement?

"Induced operators lose to a hardcoded constant" is one of the oldest results here and it shut
down the induction half of the project. It was never explained. The exponent experiment now
supplies a candidate explanation: the hardcoded 3.0 is not a physical match but a step-size
limiter, since f = (target/g)^(1/k) with a small k overshoots straight to the clip. If that is
right, the comparison was never measurement-vs-constant at all. It was damped-vs-undamped, and
the induced arm lost because using a true exponent means taking an undamped step.

The confound is separable. Cross the exponent source with an explicit trust region on the log
step, so both arms are damped identically and only the exponent differs:

    f = exp( clip( log (target/g)^(1/k), -cap, +cap ) )

  exponent source : fixed 3.0   vs   the element's true k_i (what induction would recover)
  cap             : none, 0.30, 0.15, 0.08, 0.04 in log units

The prior result predicts fixed 3.0 wins at every cap. The damping account predicts the true
exponent wins once the cap is tight enough to equalise step size, because it then supplies the
correct RELATIVE allocation across elements without the overshoot.
"""
import sys, math, random, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_synth import SynthDomain
from da_domain import get_domain
import da_tools as T

STEPS = 6
N_ELEM = 10
CAPS = (None, 0.30, 0.15, 0.08, 0.04)
SRC = ("fixed", "true_k")


def step_factor(tgt, g, e, cap):
    lf = math.log(max(tgt, 1e-9) / max(g, 1e-9)) / max(e, 0.2)
    if cap is not None:
        lf = max(-cap, min(cap, lf))
    return min(2.0, max(0.7, math.exp(lf)))


def episode_synth(seed, src, cap):
    dom = SynthDomain()
    st = dom.load({"seed": seed, "n": N_ELEM})
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1
        em = dom.element_margins(st)
        if not em:
            break
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(dom.n_elements(st)):
                b = dom.get(st, i, "x")
                if not b:
                    continue
                e = st["k"][i] if src == "true_k" else 3.0
                dom.set(nx, i, "x", b * step_factor(tgt, em[i], e, cap))
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def measured_exponents(dom, st, param, delta=0.10):
    """What induction would actually recover: own-elasticity by finite difference."""
    n = dom.n_elements(st)
    base = dom.element_margins(st)
    out = []
    for i in range(n):
        b = dom.get(st, i, param)
        if not b or not base or base[i] <= 0:
            out.append(3.0); continue
        nx = dom.clone(st)
        dom.set(nx, i, param, b * (1.0 + delta))
        m = dom.element_margins(nx)
        try:
            e = math.log(max(m[i], 1e-9) / base[i]) / math.log(1.0 + delta)
        except Exception:
            e = 3.0
        out.append(e if e > 0.2 else 3.0)
    return out


def episode_truss(f, src, cap):
    dom = get_domain("truss")
    import json
    st = dom.load(json.load(open(f)))
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1
        em = dom.element_margins(st)
        if not em:
            break
        ex = measured_exponents(dom, st, "r") if src == "true_k" else None
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(dom.n_elements(st)):
                b = dom.get(st, i, "r")
                if not b:
                    continue
                e = ex[i] if ex else 3.0
                dom.set(nx, i, "r", b * step_factor(tgt, em[i], e, cap))
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def _s(t):
    seed, src, cap = t
    try:
        return (src, cap, episode_synth(seed, src, cap))
    except Exception:
        return None


def _t(t):
    f, src, cap = t
    try:
        return (src, cap, episode_truss(f, src, cap))
    except Exception:
        return None


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    NT = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
    files = [str(x) for x in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    t0 = time.time()
    with Pool(44) as pool:
        gs = [r for r in pool.map(_s, [(s, sc, c) for s in range(N) for sc in SRC for c in CAPS],
                                   chunksize=4) if r]
        gt = [r for r in pool.map(_t, [(f, sc, c) for f in files for sc in SRC for c in CAPS],
                                   chunksize=2) if r]
    print("%.0fs\n" % (time.time() - t0))
    for name, got, n in (("ABSTRACT DOMAIN (true exponents known)", gs, N),
                         ("TRUSS (exponents measured by finite difference, as induction would)", gt, len(files))):
        print("--- %s ---" % name)
        print("  log-step cap     fixed 3.0    induced      difference")
        for c in CAPS:
            a = [x[2] for x in got if x[0] == "fixed" and x[1] == c]
            b = [x[2] for x in got if x[0] == "true_k" and x[1] == c]
            if not a or not b:
                continue
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            print("  %-14s   %.3f        %.3f       %+.3f%s"
                  % ("none" if c is None else "%.2f" % c, ma, mb, mb - ma,
                     "   <- induction wins" if mb > ma else ""))
        print()
    print("Prior result: induction loses at the default (uncapped) setting. The damping account")
    print("predicts that ordering reverses once the cap equalises step size.")
