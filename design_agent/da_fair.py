"""
Was the whole contrast a control asymmetry? Induction against a TUNED constant.

Six mechanisms have been eliminated. Before proposing a seventh, check the comparison itself.

On the truss, 3.0 is not an arbitrary number: an earlier sweep peaked there, and it coincides
with the measured median exponent of 3.094. So the truss comparison pits induction against a
constant tuned for that domain. In the synthetic domain 3.0 was carried over unchanged while
the true exponents sit at 2.199 -- so induction there may simply have been beating a constant
that was wrong for the domain, and none of the +15.7pp need reflect per-element information at
all.

The fair test sweeps the fixed exponent in the synthetic domain, takes the best, and gives
induction that as its opponent. Both arms also get their own best uniform shrink, so neither is
handicapped by a scale chosen for the other. Tuning is done on seeds 0-299 and the winner
re-run on unseen seeds 300-799, so a tuned baseline cannot win by overfitting either.

If induction's advantage survives against a properly tuned constant, the contrast with the
truss is real and unexplained. If it collapses, six hypotheses were chasing an artefact of my
own comparison.
"""
import sys, math, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_synth import SynthDomain
from da_adapt import sign_test
import da_tools as T

STEPS = 6
EXPS = (3.0, 5.0, 7.0, 10.0, 14.0, 20.0, 30.0, 45.0)
SHRINKS = (1.0, 0.8, 0.6, 0.4, 0.25)


def ep(seed, mode, e_or_s):
    dom = SynthDomain()
    st = dom.load({"seed": seed, "n": 10})
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
                d = math.log(max(tgt, 1e-9) / max(em[i], 1e-9))
                if mode == "fixed":
                    lf = d / e_or_s                       # tuned constant exponent
                else:
                    lf = d * e_or_s / max(st["k"][i], 0.2)  # induced, tuned shrink
                dom.set(nx, i, "x", b * min(2.0, max(0.7, math.exp(lf))))
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def _w(t):
    seed, mode, v = t
    try:
        return (mode, v, seed, ep(seed, mode, v))
    except Exception:
        return None


if __name__ == "__main__":
    t0 = time.time()
    tune = list(range(300))
    jobs = [(s, "fixed", e) for s in tune for e in EXPS] + \
           [(s, "induced", sh) for s in tune for sh in SHRINKS]
    with Pool(44) as pool:
        got = [r for r in pool.map(_w, jobs, chunksize=8) if r]
    print("TUNING on seeds 0-299 (%.0fs)\n" % (time.time() - t0))
    print("  fixed exponent sweep")
    bf, bfv = None, -1
    for e in EXPS:
        v = [x[3] for x in got if x[0] == "fixed" and x[1] == e]
        m = sum(v) / len(v)
        print("    e = %.1f      %.3f%s" % (e, m, "   <- best" if m > bfv else ""))
        if m > bfv:
            bf, bfv = e, m
    print("\n  induced, uniform shrink sweep")
    bi, biv = None, -1
    for sh in SHRINKS:
        v = [x[3] for x in got if x[0] == "induced" and x[1] == sh]
        m = sum(v) / len(v)
        print("    s = %.2f     %.3f%s" % (sh, m, "   <- best" if m > biv else ""))
        if m > biv:
            bi, biv = sh, m
    print("\n  best tuned constant: e=%.1f at %.3f" % (bf, bfv))
    print("  best induced       : s=%.2f at %.3f" % (bi, biv))

    held = list(range(300, 800))
    jobs = [(s, "fixed", bf) for s in held] + [(s, "induced", bi) for s in held]
    with Pool(44) as pool:
        g2 = [r for r in pool.map(_w, jobs, chunksize=8) if r]
    fa = {x[2]: x[3] for x in g2 if x[0] == "fixed"}
    ia = {x[2]: x[3] for x in g2 if x[0] == "induced"}
    pids = sorted(set(fa) & set(ia))
    a = [fa[p] for p in pids]; b = [ia[p] for p in pids]
    u, d, p = sign_test(a, b)
    print("\nHELD-OUT seeds 300-799, both settings frozen (n=%d)" % len(pids))
    print("  tuned constant e=%.1f   %.4f" % (bf, sum(a) / len(a)))
    print("  induced s=%.2f          %.4f" % (bi, sum(b) / len(b)))
    print("  gain %+.4f   discordant %d-%d   exact p = %.3g"
          % (sum(b) / len(b) - sum(a) / len(a), u, d, p))
    print("\n  Against the UNTUNED 3.0 the measured gain was +15.7pp.")
    print("  If that shrinks towards zero here, the truss contrast was a control asymmetry.")
