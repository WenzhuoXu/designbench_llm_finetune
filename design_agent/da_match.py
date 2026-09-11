"""
Apples to apples: run the truss under the synthetic's exact arm definitions.

The law predicted +44.6pp at the truss's measured coupling and the truss "measured" +26.0pp,
but those two numbers are not the same quantity. The synthetic compares

    truthful  per-element signal, correct addressing
    permuted  same values, wrong addressing
    none      NO per-element signal at all: one shared factor, chosen by the same sweep

whereas the truss's +26.0pp was measured against a coarse enumerator that already has
single-element moves in it -- so it is not a no-signal baseline, and a stronger baseline
mechanically shrinks the incremental value of addressing. This runs the three synthetic arms
verbatim on the truss so the comparison is of one quantity.
"""
import sys, json, math, statistics, collections, time, random, zlib
from math import comb
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
BUDGET = 200


def episode(spec, arm, exponent=3.0, steps=20):
    dom = get_domain("truss")
    st = dom.load(spec)
    param = dom.params[0]
    n = dom.n_elements(st)
    rng = random.Random(zlib.crc32(spec["problem_id"].encode()))
    perm = list(range(n))
    if arm == "permuted":
        rng.shuffle(perm)
    op = T.INSTALLED_OP(exponent)
    for _ in range(steps):
        if dom.feasible(st):
            return 1
        if dom.calls >= BUDGET:
            break
        best, bv = None, T.phi_rho(dom, st)
        if arm == "none":
            # no per-element signal: one shared factor, same sweep resolution
            for f in [0.7 * (2.0 / 0.7) ** (j / 15) for j in range(16)]:
                nx = dom.clone(st)
                for i in range(n):
                    b = dom.get(st, i, param)
                    if b:
                        dom.set(nx, i, param, b * f)
                v = T.phi_rho(dom, nx)
                if v > bv:
                    best, bv = nx, v
        else:
            em = dom.element_margins(st)
            if not em:
                break
            shown = [em[perm[i]] for i in range(n)] if arm == "permuted" else em
            for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
                nx = dom.clone(st)
                for i in range(n):
                    b = dom.get(st, i, param)
                    if not b:
                        continue
                    f = min(2.0, max(0.7, op(shown[i], i, tgt)))
                    dom.set(nx, i, param, b * f)
                v = T.phi_rho(dom, nx)
                if v > bv:
                    best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def _w(t):
    spec, arm = t
    try:
        return spec["problem_id"], arm, episode(spec, arm)
    except Exception:
        return spec["problem_id"], arm, None


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    specs = [json.load(open(f)) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:N]]
    ARMS = ["truthful", "permuted", "none"]
    t0 = time.time()
    with Pool(40) as pool:
        res = pool.map(_w, [(s, a) for s in specs for a in ARMS], chunksize=1)
    solved = collections.defaultdict(dict)
    for pid, arm, s in res:
        if s is not None:
            solved[arm][pid] = s
    keys = sorted(set.intersection(*[set(solved[a]) for a in ARMS]))
    print("truss under the synthetic's arm definitions, n=%d, %.0fs" % (len(keys), time.time() - t0))
    for a in ARMS:
        print("  %-9s %.3f" % (a, statistics.mean(solved[a][k] for k in keys)))

    def mc(x, y):
        hi = sum(1 for k in keys if solved[x][k] and not solved[y][k])
        lo = sum(1 for k in keys if solved[y][k] and not solved[x][k])
        d = hi + lo
        p = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        return hi, lo, 100 * (hi - lo) / len(keys), p
    print()
    for x, y, q in (("truthful", "permuted", "value of correct addressing"),
                    ("truthful", "none", "value of a per-element signal at all"),
                    ("permuted", "none", "cost of a MISaddressed signal")):
        hi, lo, d, p = mc(x, y)
        print("  %-9s vs %-9s %+5.1fpp  disc %3d-%-3d p=%.2e   %s" % (x, y, d, hi, lo, p, q))
    print("\n  synthetic at c=0.27 (interpolated): value of correct addressing +44.6pp,"
          " none-arm base 0.06-0.09")
    print("  the law transfers iff these two lines agree once the baselines are matched")
