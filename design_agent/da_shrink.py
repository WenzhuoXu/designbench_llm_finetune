"""Shrinkage sweep.

The exponent sweep showed a single-peaked optimum at ~2.3-3.0, i.e. the measured elasticity,
so the exponent is doing physics rather than damping. Yet a single POOLED exponent (0.525)
beat a PER-ELEMENT fit (0.425). If per-element estimates were unbiased and useful that could
not happen, so the estimates must be too noisy to pay for their variance.

Empirical-Bayes fix: k_i(lam) = exp( lam*log(k_pooled) + (1-lam)*log(k_i) ).
  lam = 0 -> per-element (reproduces the failing arm)
  lam = 1 -> pooled      (reproduces the winning arm)
The pooled value is the median of THIS state's own probes -- nothing is imported from the
earlier corpus measurement, so there is no leakage.

Predicted ordering if the noise story is right: shrunk >= pooled > per-element, with the
optimum at an interior lambda.
"""
import sys, json, math, statistics, collections, time
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
LAMS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]


def shrunk_op(probes, lam, floor=0.3):
    good = {i: v for i, v in probes.items()
            if v is not None and math.isfinite(v) and v >= floor}
    if not good:
        return T.INSTALLED_OP(3.0), None
    pooled = statistics.median(good.values())
    k = {i: math.exp(lam * math.log(pooled) + (1 - lam) * math.log(v))
         for i, v in good.items()}

    def op(margin_now, i, target):
        e = k.get(i, pooled)
        if margin_now is None or margin_now <= 0 or not math.isfinite(margin_now):
            return 1.0
        try:
            return (target / margin_now) ** (1.0 / e)
        except (ValueError, ZeroDivisionError, OverflowError):
            return 1.0
    return op, pooled


def run(t):
    spec, lam = t
    try:
        dom = get_domain("truss")
        st = dom.load(spec)
        param = dom.params[0]
        pooled_seen = []
        for _ in range(20):
            if dom.feasible(st):
                return spec["problem_id"], lam, 1, pooled_seen
            if dom.calls >= 120:
                break
            n = dom.n_elements(st)
            probes = T.PROBE(dom, st, list(range(n)), param)
            op, pooled = shrunk_op(probes, lam)
            if pooled:
                pooled_seen.append(pooled)
            build = lambda target: T.apply_op(dom, st, op, list(range(n)), param, target)
            nx, v, _ = T.inner_optimise(dom, st, build, {"target": (1.0, 1.8)}, budget=18)
            if v <= T.phi_rho(dom, st):
                break
            st = nx
        return spec["problem_id"], lam, (1 if dom.feasible(st) else 0), pooled_seen
    except Exception:
        return spec["problem_id"], lam, None, []


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    specs = [json.load(open(f)) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:n]]
    t0 = time.time()
    with Pool(44) as pool:
        res = pool.map(run, [(s, l) for s in specs for l in LAMS], chunksize=1)
    by = collections.defaultdict(dict); pooled_all = []
    for pid, lam, s, ps in res:
        if s is not None:
            by[lam][pid] = s
        pooled_all += ps
    keys = sorted(set.intersection(*[set(by[l]) for l in LAMS]))
    print("shrinkage sweep, n=%d, %.0fs" % (len(keys), time.time() - t0))
    if pooled_all:
        pa = sorted(pooled_all)
        print("  pooled exponent seen in-run: median %.2f  p10 %.2f  p90 %.2f  (n=%d states)"
              % (pa[len(pa)//2], pa[int(.1*len(pa))], pa[int(.9*len(pa))], len(pa)))
    best = None
    for l in LAMS:
        r = statistics.mean(by[l][k] for k in keys)
        tag = {0.0: "  <- per-element (the failing arm)",
               1.0: "  <- pooled median"}.get(l, "")
        print("  lambda %.2f   solved %.3f%s" % (l, r, tag))
        if best is None or r > best[1]:
            best = (l, r)
    print("\n  best lambda %.2f at %.3f" % best)

    def mc(a, b):
        hi = sum(1 for k in keys if a[k] and not b[k])
        lo = sum(1 for k in keys if b[k] and not a[k])
        d = hi + lo
        p = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        return hi, lo, 100 * (hi - lo) / len(keys), p
    for l in (0.0, 1.0):
        hi, lo, d, p = mc(by[best[0]], by[l])
        print("  best vs lambda %.2f : disc %3d-%-3d  %+5.1fpp  p=%.2e" % (l, hi, lo, d, p))
    hi, lo, d, p = mc(by[1.0], by[0.0])
    print("  pooled vs per-element : disc %3d-%-3d  %+5.1fpp  p=%.2e" % (hi, lo, d, p))
