"""
Is the truss tie about coupling, or about what the measurement COSTS?

Independently measured, the truss's coupling implies c = 0.28, where the synthetic dose-
response predicts induction should win by about +11pp. It ties. So coupling does not explain
the truss, and the account failed the first test that was not circular.

The likely culprit is a control failure in my own design. The synthetic sweeps read each
element's exponent off the instance for free. The truss arm measures it by finite difference
and is charged one evaluation per element per turn against the same 200-call budget the
constant arm spends on search -- for a twenty-member truss over six turns that is most of the
budget. The two experiments were never comparable.

Separating cost from information, all at the same nominal budget:

  charged_each_turn   measure every turn, pay for it            (what was run; ties)
  charged_once        measure at turn 1 only, reuse thereafter  (cost amortised)
  free                measure every turn, not charged           (information without cost)

If 'free' wins and 'charged_each_turn' does not, the truss tie is a budget artifact and the
scope condition is about measurement cost, not coupling. If even 'free' ties, the information
genuinely does not help on trusses and coupling is not the reason either.
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
from da_adapt import sign_test
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
STEPS, BUDGET, SHRINK = 6, 200, 0.6
ARMS = ("fixed", "charged_each_turn", "charged_once", "free")


def exponents(dom, st, em, charge):
    n = dom.n_elements(st)
    out = []
    before = dom.calls
    for i in range(n):
        b = dom.get(st, i, "r")
        if not b or not em[i] or em[i] <= 0 or not math.isfinite(em[i]):
            out.append(3.0); continue
        nx = dom.clone(st)
        dom.set(nx, i, "r", b * 1.10)
        m = dom.element_margins(nx)
        try:
            e = math.log(max(min(m[i], 50.0), 1e-9) / min(em[i], 50.0)) / math.log(1.10)
        except Exception:
            e = 3.0
        out.append(e if e > 0.2 else 3.0)
    if not charge:
        dom.calls = before          # information supplied without paying for it
    return out


def episode(f, arm):
    dom = get_domain("truss")
    st = dom.load(json.load(open(f)))
    cached = None
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1
        if dom.calls >= BUDGET:
            break
        em = dom.element_margins(st)
        if not em:
            break
        ex = None
        if arm == "charged_each_turn":
            ex = exponents(dom, st, em, True)
        elif arm == "free":
            ex = exponents(dom, st, em, False)
        elif arm == "charged_once":
            if cached is None:
                cached = exponents(dom, st, em, True)
            ex = cached
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(dom.n_elements(st)):
                b = dom.get(st, i, "r")
                if not b:
                    continue
                d = math.log(max(tgt, 1e-9) / max(min(em[i], 50.0), 1e-9))
                lf = d / 3.0 if ex is None else d * SHRINK / max(ex[i] if i < len(ex) else 3.0, 0.2)
                dom.set(nx, i, "r", b * min(2.0, max(0.7, math.exp(lf))))
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def _w(t):
    f, arm = t
    try:
        return (f, arm, episode(f, arm))
    except Exception:
        return None


if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    files = [str(x) for x in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    t0 = time.time()
    with Pool(44) as pool:
        got = [r for r in pool.map(_w, [(f, a) for f in files for a in ARMS], chunksize=2) if r]
    by = {a: {x[0]: x[2] for x in got if x[1] == a} for a in ARMS}
    pids = sorted(set.intersection(*[set(by[a]) for a in ARMS]))
    print("truss, %d problems paired across all arms, %.0fs, nominal budget %d\n"
          % (len(pids), time.time() - t0, BUDGET))
    rate = {a: sum(by[a][p] for p in pids) / len(pids) for a in ARMS}
    for a in ARMS:
        print("  %-20s %.4f" % (a, rate[a]))
    print()
    for a in ARMS[1:]:
        u, d, p = sign_test([by["fixed"][x] for x in pids], [by[a][x] for x in pids])
        print("  %-20s vs fixed: %+.4f   discordant %d-%d   exact p = %.3g"
              % (a, rate[a] - rate["fixed"], u, d, p))
    print("\n  Coupling predicted +0.11 for the truss at its measured c=0.28.")
