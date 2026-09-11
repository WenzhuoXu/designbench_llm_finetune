"""
Is the truss's local elasticity simply not extrapolable to the step the operator takes?

Four accounts of why induction wins in the synthetic domain and ties on the truss have failed:
coupling magnitude, measurement cost, measurement noise, and coupling shape. The noise test
compared step sizes 0.05 and 0.15 and found perfect reproducibility -- but both are small, so
it established only that the LOCAL derivative is stable, which was never the question.

A truss margin is min(FOS_buckling, FOS_yielding)/1.5, a minimum over two mechanisms with
different dependence on radius. Buckling and yielding do not scale alike, so the binding
mechanism can switch part-way through a step. A local exponent would then be exactly right in
the limit and wrong for the factor of 1.3 or 1.7 the operator actually applies. The synthetic
domain has a single power law per element, so its local exponent is its global one by
construction -- which makes it the control rather than a second test.

Two parts:
  (a) diagnostic -- compare the local exponent against the EFFECTIVE exponent over a large
      step, log(g(1.5x)/g(x))/log(1.5), on both domains
  (b) the fix -- if they diverge, induction should work on the truss when the exponent is
      measured at the step size actually used. That arm is run and charged nothing, so a tie
      cannot be blamed on budget.
"""
import sys, json, math, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
from da_synth import SynthDomain
from da_adapt import sign_test
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
CAP, STEPS, SHRINK, BUDGET = 50.0, 6, 0.6, 200


def eff_exp(dom, st, param, i, factor):
    base = dom.element_margins(st)
    b = dom.get(st, i, param)
    if not b:
        return None
    try:
        b0 = min(float(base[i]), CAP)
    except Exception:
        return None
    if not math.isfinite(b0) or b0 <= 0:
        return None
    nx = dom.clone(st)
    dom.set(nx, i, param, b * factor)
    m = dom.element_margins(nx)
    try:
        m0 = min(float(m[i]), CAP)
        e = math.log(max(m0, 1e-9) / b0) / math.log(factor)
    except Exception:
        return None
    return e if math.isfinite(e) else None


def _diag_truss(f):
    try:
        dom = get_domain("truss")
        st = dom.load(json.load(open(f)))
        out = []
        for i in range(dom.n_elements(st)):
            a = eff_exp(dom, st, "r", i, 1.10)
            b = eff_exp(dom, st, "r", i, 1.50)
            if a and b and a > 0.2 and b > 0.2:
                out.append((a, b))
        return out
    except Exception:
        return []


def _diag_synth(seed):
    try:
        dom = SynthDomain()
        st = dom.load({"seed": seed, "n": 10})
        out = []
        for i in range(dom.n_elements(st)):
            a = eff_exp(dom, st, "x", i, 1.10)
            b = eff_exp(dom, st, "x", i, 1.50)
            if a and b and a > 0.2 and b > 0.2:
                out.append((a, b))
        return out
    except Exception:
        return []


def truss_ep(f, mode):
    """mode: fixed | local (delta 1.10) | largestep (delta 1.50). Exponents not charged."""
    dom = get_domain("truss")
    st = dom.load(json.load(open(f)))
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1
        if dom.calls >= BUDGET:
            break
        em = dom.element_margins(st)
        if not em:
            break
        n = dom.n_elements(st)
        ex = None
        if mode != "fixed":
            before = dom.calls
            fac = 1.10 if mode == "local" else 1.50
            ex = [eff_exp(dom, st, "r", i, fac) or 3.0 for i in range(n)]
            ex = [e if e > 0.2 else 3.0 for e in ex]
            dom.calls = before
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(n):
                b = dom.get(st, i, "r")
                if not b:
                    continue
                try:
                    e0 = min(float(em[i]), CAP)
                except Exception:
                    continue
                d = math.log(max(tgt, 1e-9) / max(e0, 1e-9))
                lf = d / 3.0 if ex is None else d * SHRINK / max(ex[i], 0.2)
                dom.set(nx, i, "r", b * min(2.0, max(0.7, math.exp(lf))))
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def _run(t):
    f, mode = t
    try:
        return (f, mode, truss_ep(f, mode))
    except Exception:
        return None


if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    files = [str(x) for x in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    t0 = time.time()
    with Pool(44) as pool:
        tt = [x for g in pool.map(_diag_truss, files[:250], chunksize=2) for x in g]
        ss = [x for g in pool.map(_diag_synth, list(range(250)), chunksize=4) for x in g]
    print("(a) local exponent (step 1.10) vs effective exponent over a large step (1.50)\n")
    for name, g in (("TRUSS", tt), ("SYNTHETIC (single power law: control)", ss)):
        if not g:
            continue
        rat = [b / a for a, b in g if a > 0]
        print("  %-38s elements=%d" % (name, len(g)))
        print("    median local %.3f   median large-step %.3f"
              % (statistics.median([a for a, _ in g]), statistics.median([b for _, b in g])))
        print("    ratio large/local: median %.3f   interquartile %.3f to %.3f"
              % (statistics.median(rat), sorted(rat)[len(rat) // 4], sorted(rat)[3 * len(rat) // 4]))
        print("    share of elements where the ratio is outside 0.8-1.25: %.2f\n"
              % (sum(1 for r in rat if r < 0.8 or r > 1.25) / len(rat)))

    with Pool(44) as pool:
        got = [r for r in pool.map(_run, [(f, m) for f in files
                                          for m in ("fixed", "local", "largestep")],
                                    chunksize=2) if r]
    by = {m: {x[0]: x[2] for x in got if x[1] == m} for m in ("fixed", "local", "largestep")}
    pids = sorted(set.intersection(*[set(v) for v in by.values()]))
    print("(b) induction on the truss with the exponent measured at each step size")
    print("    %d problems, exponents given free\n" % len(pids))
    rate = {m: sum(by[m][p] for p in pids) / len(pids) for m in by}
    for m in ("fixed", "local", "largestep"):
        print("  %-11s %.4f" % (m, rate[m]))
    print()
    for m in ("local", "largestep"):
        u, d, p = sign_test([by["fixed"][x] for x in pids], [by[m][x] for x in pids])
        print("  %-11s vs fixed: %+.4f   discordant %d-%d   exact p = %.3g"
              % (m, rate[m] - rate["fixed"], u, d, p))
    print("\n  %.0fs" % (time.time() - t0))
