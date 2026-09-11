"""
When is tool induction worth its cost? A dose-response on response heterogeneity.

Induction beats the hardcoded constant by +15.7pp on held-out instances of the abstract domain
and by nothing at all on the truss (-0.5pp, discordant 3-4, p=1). The truss result is not a
failure of the method: only seven of 250 problems even disagreed, meaning the two operators
made nearly the same moves. Truss own-elasticities cluster near 3, so a constant 3.0 already
IS the induced operator and there is nothing left to discover. The synthetic exponents are
lognormal with real spread, so there is.

If that is the mechanism, the payoff should be a function of one measurable property of the
domain -- the dispersion of the true per-element exponents -- and should vanish at zero
dispersion. This sweeps that dispersion directly, holding everything else fixed, and locates
the truss on the resulting curve by measuring its own dispersion the same way.

A domain-level quantity that predicts whether a tool is worth building is more useful than
either single verdict, and it is what the portability layer was for.
"""
import sys, json, math, random, statistics, time
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

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
STEPS, N_ELEM, SHRINK = 6, 10, 0.4
SIGMAS = (0.0, 0.15, 0.30, 0.45, 0.55, 0.75, 1.0)


def load_sigma(seed, sigma, n=N_ELEM):
    """da_synth's instance with the exponent dispersion under our control."""
    r = random.Random(seed)
    a = [math.exp(r.gauss(-0.35, 0.55)) for _ in range(n)]
    k = [max(0.3, r.lognormvariate(0.80, sigma)) for _ in range(n)]
    w = [math.exp(r.gauss(0.0, 0.4)) for _ in range(n)]
    B = sum(w) * r.uniform(1.5, 3.2)
    return {"a": a, "k": k, "w": w, "x": [1.0] * n, "B": B, "n": n}


def ep(seed, sigma, mode):
    dom = SynthDomain()
    st = load_sigma(seed, sigma)
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
                lf = d / 3.0 if mode == "fixed" else d * SHRINK / max(st["k"][i], 0.2)
                dom.set(nx, i, "x", b * min(2.0, max(0.7, math.exp(lf))))
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def _w(t):
    seed, sigma, mode = t
    try:
        return (sigma, mode, ep(seed, sigma, mode))
    except Exception:
        return None


def truss_dispersion(f, delta=0.10):
    """sd of log own-elasticity on one truss, measured the way induction would."""
    try:
        dom = get_domain("truss")
        st = dom.load(json.load(open(f)))
        em = dom.element_margins(st)
        if not em:
            return None
        out = []
        for i in range(dom.n_elements(st)):
            b = dom.get(st, i, "r")
            if not b or em[i] <= 0:
                continue
            nx = dom.clone(st)
            dom.set(nx, i, "r", b * (1 + delta))
            m = dom.element_margins(nx)
            try:
                e = math.log(max(m[i], 1e-9) / em[i]) / math.log(1 + delta)
            except Exception:
                continue
            if e > 0.2:
                out.append(math.log(e))
        return statistics.pstdev(out) if len(out) > 2 else None
    except Exception:
        return None


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    t0 = time.time()
    jobs = [(s, sg, m) for s in range(N) for sg in SIGMAS for m in ("fixed", "induced")]
    files = [str(x) for x in sorted((DB / "data/problems_hard").glob("*.json"))[:200]]
    with Pool(44) as pool:
        got = [r for r in pool.map(_w, jobs, chunksize=8) if r]
        disp = [d for d in pool.map(truss_dispersion, files, chunksize=2) if d is not None]
    print("%d synthetic episodes + %d trusses measured, %.0fs\n" % (len(got), len(disp), time.time() - t0))
    print("  exponent dispersion    fixed 3.0    induced (shrink %.2f)    gain" % SHRINK)
    for sg in SIGMAS:
        a = [x[2] for x in got if x[0] == sg and x[1] == "fixed"]
        b = [x[2] for x in got if x[0] == sg and x[1] == "induced"]
        if not a or not b:
            continue
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        star = "  <-" if mb - ma > 0.05 else ""
        print("  sigma = %.2f            %.3f        %.3f                 %+.3f%s"
              % (sg, ma, mb, mb - ma, star))
    md = statistics.median(disp)
    print("\n  measured truss exponent dispersion: median sd(log k) = %.3f over %d problems"
          % (md, len(disp)))
    print("  (interquartile %.3f to %.3f)"
          % (sorted(disp)[len(disp) // 4], sorted(disp)[3 * len(disp) // 4]))
    print("\n  The account predicts a gain near zero at low dispersion, rising with it, and")
    print("  predicts the truss sits at the low end -- which is why induction ties there.")
