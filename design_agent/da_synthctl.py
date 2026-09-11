"""
Matched procedural control for the abstract-domain LLM run.

The LLM slope in the abstract domain is +3.476 and I compared it to the model-free +7.14 to
claim the model consumes ordering information at 49% of a solver's sensitivity. That
comparison is not matched: +7.14 was fitted on a different configuration (n=16 elements, a
different step budget, pooled with the truss), while the LLM ran on n=10 with six turns.
Slopes are configuration-dependent, so the ratio as stated is not supported.

This runs the procedural loop on EXACTLY the configuration the LLM saw -- same n, same seeds
0..99, same five permutation levels, same six turns -- and fits the same model. Only then is
the ratio a like-for-like statement.
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
from da_matched_info import corrupt, spearman
from da_fit2 import irls
import da_tools as T

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
STEPS = 6
N_ELEM = 10


def episode(seed, level):
    """Procedural: each turn rescale every element from the shown margins, keep the best target."""
    dom = SynthDomain()
    st = dom.load({"seed": seed, "n": N_ELEM})
    rng = random.Random(seed ^ 0xC0FFEE)
    op = T.INSTALLED_OP(3.0)
    rhos = []
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1, rhos
        em = dom.element_margins(st)
        if not em:
            break
        shown = corrupt(em, "perm", level, rng)
        r = spearman(em, shown)
        if r is not None:
            rhos.append(r)
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(dom.n_elements(st)):
                b = dom.get(st, i, "x")
                if not b:
                    continue
                f = min(2.0, max(0.7, op(shown[i], i, tgt)))
                dom.set(nx, i, "x", b * f)
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return (1 if dom.feasible(st) else 0), rhos


def _w(t):
    seed, lv = t
    try:
        o, rh = episode(seed, lv)
        return {"seed": seed, "level": lv, "y": float(o),
                "rho": statistics.mean(rh) if rh else 1.0}
    except Exception:
        return None


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    t0 = time.time()
    with Pool(44) as pool:
        rows = [r for r in pool.map(_w, [(s, lv) for s in range(N) for lv in LEVELS],
                                    chunksize=4) if r]
    print("procedural control, identical configuration: %d episodes, %.0fs"
          % (len(rows), time.time() - t0))
    print("\n  level   n     mean rho   feasible")
    for lv in LEVELS:
        sub = [r for r in rows if r["level"] == lv]
        print("  %5.2f  %4d    %.3f      %.3f"
              % (lv, len(sub), statistics.mean([r["rho"] for r in sub]),
                 sum(r["y"] for r in sub) / len(sub)))
    y = [r["y"] for r in rows]
    X = [[1.0, r["rho"]] for r in rows]
    ids = [str(r["seed"]) for r in rows]
    b, se, cl, ll = irls(X, y, ids)
    slope = b[1]
    print("\n  procedural slope on this configuration: %+.3f  (cluster z = %.2f)"
          % (slope, slope / max(cl[1], 1e-9)))
    print("  LLM slope on the same configuration   : +3.476")
    if slope > 0:
        print("  matched ratio, LLM / procedural       : %.0f%%" % (100 * 3.476 / slope))
    print("\n  (the previously quoted 49%% used the unmatched +7.14 and is superseded)")
