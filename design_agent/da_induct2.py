"""
Validate the reversal: does induction still beat the constant on held-out instances, and does
it survive the truss, where the exponents must be MEASURED and the measurement costs budget?

In the abstract domain, induced per-element exponents with a uniform scalar shrink reach 0.715
against the hardcoded constant's 0.570. That reverses one of the oldest negatives here, so it
gets the same discipline as everything else rather than a headline.

  1. The shrink was chosen on seeds 0-199. Re-run on seeds 200-599, unseen.
  2. On the truss the true exponents do not exist to be read off. Induction must spend two
     evaluations per element on a finite difference, and those come out of the SAME budget the
     constant arm gets to spend on search. That is the comparison that matters and the one the
     original result was making.

Truss protocol: sweep the shrink on problems 0-150, freeze it, test on problems 150-400 with
an exact paired sign test. One comparison.
"""
import sys, json, math, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_synth import SynthDomain
from da_domain import get_domain
from da_adapt import sign_test
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
STEPS = 6
BUDGET = 200
SHRINK = (0.6, 0.4, 0.25)


def synth_ep(seed, mode, s):
    dom = SynthDomain()
    st = dom.load({"seed": seed, "n": 10})
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1
        em = dom.element_margins(st)
        if not em:
            break
        n = dom.n_elements(st)
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(n):
                b = dom.get(st, i, "x")
                if not b:
                    continue
                d = math.log(max(tgt, 1e-9) / max(em[i], 1e-9))
                lf = d / 3.0 if mode == "fixed" else d * s / max(st["k"][i], 0.2)
                dom.set(nx, i, "x", b * min(2.0, max(0.7, math.exp(lf))))
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def truss_ep(f, mode, s):
    """Budget is shared: the induced arm pays for its own finite differences."""
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
        if mode == "induced":
            ex = []
            for i in range(n):
                b = dom.get(st, i, "r")
                if not b or em[i] <= 0 or dom.calls >= BUDGET:
                    ex.append(3.0); continue
                nx = dom.clone(st)
                dom.set(nx, i, "r", b * 1.10)
                m = dom.element_margins(nx)          # charged to the budget
                try:
                    e = math.log(max(m[i], 1e-9) / em[i]) / math.log(1.10)
                except Exception:
                    e = 3.0
                ex.append(e if e > 0.2 else 3.0)
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(n):
                b = dom.get(st, i, "r")
                if not b:
                    continue
                d = math.log(max(tgt, 1e-9) / max(em[i], 1e-9))
                lf = d / 3.0 if ex is None else d * s / ex[i]
                dom.set(nx, i, "r", b * min(2.0, max(0.7, math.exp(lf))))
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def _s(t):
    seed, mode, s = t
    try:
        return (mode, s, synth_ep(seed, mode, s))
    except Exception:
        return None


def _t(t):
    f, mode, s = t
    try:
        return (f, mode, s, truss_ep(f, mode, s))
    except Exception:
        return None


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(x, "fixed", 0.0) for x in range(200, 600)]
    for s in SHRINK:
        jobs += [(x, "induced", s) for x in range(200, 600)]
    with Pool(44) as pool:
        gs = [r for r in pool.map(_s, jobs, chunksize=8) if r]
    fx = [x[2] for x in gs if x[0] == "fixed"]
    print("1. ABSTRACT DOMAIN, held-out seeds 200-599 (shrink chosen on 0-199)")
    print("   fixed exponent 3.0        %.3f" % (sum(fx) / len(fx)))
    for s in SHRINK:
        v = [x[2] for x in gs if x[0] == "induced" and x[1] == s]
        print("   induced, shrink %.2f      %.3f   %+.3f" % (s, sum(v) / len(v), sum(v) / len(v) - sum(fx) / len(fx)))

    allf = sorted((DB / "data/problems_hard").glob("*.json"))
    disc = [str(x) for x in allf[0:150]]
    conf = [str(x) for x in allf[150:400]]
    jobs = [(f, "fixed", 0.0) for f in disc] + [(f, "induced", s) for f in disc for s in SHRINK]
    with Pool(44) as pool:
        gd = [r for r in pool.map(_t, jobs, chunksize=2) if r]
    fx = [x[3] for x in gd if x[1] == "fixed"]
    bd = sum(fx) / len(fx)
    print("\n2. TRUSS, exponents measured by finite difference, charged to the same budget")
    print("   discovery problems 0-150")
    print("   fixed exponent 3.0        %.3f" % bd)
    best_s, best_v = None, -1
    for s in SHRINK:
        v = [x[3] for x in gd if x[1] == "induced" and x[2] == s]
        m = sum(v) / len(v)
        print("   induced, shrink %.2f      %.3f   %+.3f" % (s, m, m - bd))
        if m > best_v:
            best_s, best_v = s, m

    jobs = [(f, "fixed", 0.0) for f in conf] + [(f, "induced", best_s) for f in conf]
    with Pool(44) as pool:
        gc = [r for r in pool.map(_t, jobs, chunksize=2) if r]
    fa = {x[0]: x[3] for x in gc if x[1] == "fixed"}
    ia = {x[0]: x[3] for x in gc if x[1] == "induced"}
    pids = sorted(set(fa) & set(ia))
    a = [fa[p] for p in pids]
    b = [ia[p] for p in pids]
    u, d, pv = sign_test(a, b)
    print("\n   HELD-OUT problems 150-400, one comparison, shrink frozen at %.2f" % best_s)
    print("   fixed %.4f   induced %.4f   %+.4f   discordant %d-%d   exact p = %.3g"
          % (sum(a) / len(a), sum(b) / len(b), sum(b) / len(b) - sum(a) / len(a), u, d, pv))
    print("\n   %.0fs total" % (time.time() - t0))
