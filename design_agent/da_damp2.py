"""
If equal step size does not rescue induction, is the harm in the SPREAD of steps?

Capping the log step failed to reverse the induction result at any cap, so the induced
operator's problem is not that its steps are too big. What remains is that they are too
UNEQUAL. The fixed exponent gives every element log(target/g_i)/3 -- a uniform contraction
that shrinks all steps by the same factor and preserves their relative sizes. The induced
operator gives log(target/g_i)/k_i, so elements with a small exponent move far more than
others. A cap does not fix that; it clips the largest steps, which destroys their ordering
instead of preserving it.

Two arms separate scale from spread, both applied to the induced steps:

  uniform shrink  lf_i * s          keeps the 1/k_i spread, changes only the scale
  spread removal  lf_i normalised to the mean 1/k_i, i.e. exactly the fixed operator
                  rescaled -- keeps the scale, removes the spread

If scale is the problem, uniform shrink should recover the fixed operator's performance at
some s. If spread is the problem, it should not, at any s.
"""
import sys, math, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_synth import SynthDomain
import da_tools as T

STEPS, N_ELEM = 6, 10
SHRINK = (1.0, 0.6, 0.4, 0.25, 0.15)


def episode(seed, mode, s):
    dom = SynthDomain()
    st = dom.load({"seed": seed, "n": N_ELEM})
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1
        em = dom.element_margins(st)
        if not em:
            break
        n = dom.n_elements(st)
        inv = [1.0 / max(st["k"][i], 0.2) for i in range(n)]
        mean_inv = statistics.mean(inv)
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(n):
                b = dom.get(st, i, "x")
                if not b:
                    continue
                d = math.log(max(tgt, 1e-9) / max(em[i], 1e-9))
                if mode == "fixed":
                    lf = d / 3.0
                elif mode == "shrink":
                    lf = d * inv[i] * s            # induced spread, scaled
                else:
                    lf = d * mean_inv * s          # induced scale, spread removed
                dom.set(nx, i, "x", b * min(2.0, max(0.7, math.exp(lf))))
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def _w(t):
    seed, mode, s = t
    try:
        return (mode, s, episode(seed, mode, s))
    except Exception:
        return None


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    jobs = [(x, "fixed", 1.0) for x in range(N)]
    for s in SHRINK:
        jobs += [(x, "shrink", s) for x in range(N)]
        jobs += [(x, "nospread", s) for x in range(N)]
    t0 = time.time()
    with Pool(44) as pool:
        got = [r for r in pool.map(_w, jobs, chunksize=8) if r]
    fx = [x[2] for x in got if x[0] == "fixed"]
    base = sum(fx) / len(fx)
    print("abstract domain, %d seeds, %.0fs" % (N, time.time() - t0))
    print("\n  fixed exponent 3.0 (reference)            %.3f\n" % base)
    print("  scale s    induced, spread kept    induced scale, spread removed")
    for s in SHRINK:
        a = [x[2] for x in got if x[0] == "shrink" and x[1] == s]
        b = [x[2] for x in got if x[0] == "nospread" and x[1] == s]
        print("  %5.2f         %.3f                     %.3f"
              % (s, sum(a) / len(a), sum(b) / len(b)))
    print("\n  Scale account: 'spread kept' should reach %.3f at some s." % base)
    print("  Spread account: only the 'spread removed' column should.")
