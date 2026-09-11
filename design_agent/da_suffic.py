"""
Is ordering SUFFICIENT, not merely necessary?

The law says outcome tracks the rank correlation between the shown per-element signal and the
true one. Every measurement so far has established necessity: destroy the ordering and the
outcome collapses. A reviewer is entitled to answer that degrading a signal degrades
performance and ask what has been learned.

The sharp form of the claim is sufficiency. Replace the values entirely -- keep only their
ORDER, and synthesise fresh numbers from a canonical ladder that carries no information about
the true magnitudes at all -- and see whether the outcome is preserved. If it is, the values
are doing no work beyond ranking the elements, which is a much stronger and more surprising
statement than "ordering matters", and it is what makes the law a claim about signal structure
rather than about information in general.

Arms, model-free, in both domains:

  true          the real per-element margins
  rank_only     values discarded; rank r mapped onto a fixed geometric ladder
  rank_wrong    the same ladder over a deliberately wrong range, to test scale sensitivity
  rank_coarse   only the ordering into quartiles survives, ties broken arbitrarily
  permuted      the known-bad control: real values, destroyed ordering

The law predicts rank_only is indistinguishable from true, and permuted collapses. If rank_only
falls well short of true, the values carry information beyond their order and the law as stated
is too strong.
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
from da_synth import SynthDomain
from da_adapt import sign_test
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
CAP, STEPS, BUDGET = 50.0, 6, 200
ARMS = ("true", "rank_only", "rank_wrong", "rank_coarse", "permuted")


def transform(em, arm, rng):
    n = len(em)
    v = []
    for x in em:
        try:
            x = min(float(x), CAP)
        except Exception:
            x = CAP
        v.append(x if (math.isfinite(x) and x > 0) else CAP)
    if arm == "true":
        return v
    if arm == "permuted":
        p = v[:]; rng.shuffle(p); return p
    order = sorted(range(n), key=lambda i: v[i])
    out = [0.0] * n
    if arm == "rank_only":
        lo, hi = 0.5, 3.0                 # a canonical ladder, unrelated to the real values
    elif arm == "rank_wrong":
        lo, hi = 0.05, 40.0               # deliberately the wrong range
    else:
        q = [0.6, 0.95, 1.4, 2.2]         # quartile buckets only
        for r, i in enumerate(order):
            out[i] = q[min(3, (r * 4) // max(n, 1))]
        return out
    for r, i in enumerate(order):
        out[i] = lo * (hi / lo) ** (r / max(n - 1, 1))
    return out


def run(dom, st, param, arm, seed):
    rng = random.Random(seed)
    op = T.INSTALLED_OP(3.0)
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1
        if getattr(dom, "calls", 0) >= BUDGET:
            break
        em = dom.element_margins(st)
        if not em:
            break
        shown = transform(em, arm, rng)
        n = dom.n_elements(st)
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(n):
                b = dom.get(st, i, param)
                if not b:
                    continue
                dom.set(nx, i, param, b * min(2.0, max(0.7, op(shown[i], i, tgt))))
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def _t(t):
    f, arm = t
    try:
        dom = get_domain("truss")
        st = dom.load(json.load(open(f)))
        return ("truss", arm, f, run(dom, st, "r", arm, hash(f) & 0xffff))
    except Exception:
        return None


def _s(t):
    seed, arm = t
    try:
        dom = SynthDomain()
        st = dom.load({"seed": seed, "n": 10})
        return ("synth", arm, "s%d" % seed, run(dom, st, "x", arm, seed ^ 0x5EED))
    except Exception:
        return None


if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    files = [str(x) for x in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    t0 = time.time()
    with Pool(44) as pool:
        gt = [r for r in pool.map(_t, [(f, a) for f in files for a in ARMS], chunksize=2) if r]
        gs = [r for r in pool.map(_s, [(s, a) for s in range(NS) for a in ARMS], chunksize=4) if r]
    print("%.0fs\n" % (time.time() - t0))
    for name, g in (("TRUSS", gt), ("SYNTHETIC", gs)):
        by = {a: {x[2]: x[3] for x in g if x[1] == a} for a in ARMS}
        pids = sorted(set.intersection(*[set(by[a]) for a in ARMS]))
        if not pids:
            continue
        rate = {a: sum(by[a][p] for p in pids) / len(pids) for a in ARMS}
        print("--- %s (n=%d paired) ---" % (name, len(pids)))
        for a in ARMS:
            print("  %-12s %.4f" % (a, rate[a]))
        print()
        for a in ARMS[1:]:
            u, d, pv = sign_test([by["true"][x] for x in pids], [by[a][x] for x in pids])
            print("  %-12s vs true: %+.4f   discordant %d-%d   exact p = %.3g"
                  % (a, rate[a] - rate["true"], u, d, pv))
        print()
    print("Sufficiency holds if rank_only is indistinguishable from true while permuted collapses.")
