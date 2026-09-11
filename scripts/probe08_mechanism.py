"""
EXPERIMENT 8 -- what the +36.3pp macro effect actually is.

The enumerator's action set (search_ladder.SCALE_FACTORS = 0.85,1.1,1.25,1.5,2.0):
  - per-member single-param scaling at 5 factors        (arity 1, heterogeneous trivially)
  - SCALE_MULTI_PARAM over ALL members at 5 factors     (arity n, HOMOGENEOUS)
  - the FSD macro                                       (arity n, HETEROGENEOUS, 6 decimals)
So the enumerator already has arity n. What the macro adds is heterogeneity + resolution.
That confound has never been separated ("not identified by any measurement on the board").

Four arms, depth 1, Phi-argmax selection, 20 turns, same problems, paired:
  A  base          5 factors, arity-1 + homogeneous-global      [the 0.381 control]
  B  fine          31 factors, same structure                   [RESOLUTION only]
  C  blind-hetero  base + random per-member factor VECTORS,
                   matched candidate budget to D, no closed form [HETEROGENEITY only]
  D  macro         base + the FSD macro                          [the 0.744 arm]

Prediction under the amplitude account (probe04-07): B ~ A (resolution on a shared factor
buys little), C >> A (heterogeneity is the mechanism), C approaching D (the closed form is
one way to get heterogeneity, not the source of the effect).
If C ~ A then heterogeneity is NOT the mechanism and the closed form is doing the work.
"""
import sys, json, math, copy, random, argparse, time
from pathlib import Path
from multiprocessing import Pool
from math import comb
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals

BASE_F = (0.85, 1.1, 1.25, 1.5, 2.0)
FINE_F = tuple(math.exp(math.log(0.7) + i * (math.log(2.0) - math.log(0.7)) / 30) for i in range(31))


def mr(m):
    par = getattr(getattr(m, "shape", None), "_params", None) or {}
    v = par.get("r")
    return float(v) if isinstance(v, (int, float)) else None


def sr(m, v):
    par = getattr(getattr(m, "shape", None), "_params", None)
    if par is not None:
        par["r"] = v


def phi(st, goals, m0):
    """Selector: hinge on both constraints plus a mass term. Same shape as the project's Phi_v2."""
    gb = float(goals.get("minimum_fos_buckling", 1.5)); gy = float(goals.get("minimum_fos_yielding", 1.5))
    B = float(goals.get("maximum_mass", float("inf")))
    fb = float(st.get("fos_buckling", 0.0) or 0.0); fy = float(st.get("fos_yielding", 0.0) or 0.0)
    ms = float(st.get("mass", 9e9) or 9e9)
    v = 0.0
    v += 5.0 * min(0.0, fb / gb - 1.0)
    v += 5.0 * min(0.0, fy / gy - 1.0)
    if math.isfinite(B) and B > 0:
        v += 5.0 * min(0.0, 1.0 - ms / B)
    v += -0.05 * (ms / m0 if m0 > 0 else 0.0)
    return v


def apply_vec(truss, base, vec, lo, hi):
    nt = copy.deepcopy(truss)
    for i, f in vec.items():
        sr(nt.members[i], min(hi, max(lo, base[i] * f)))
    return nt


def candidates(truss, base, n, factors, mode, gb, gy, rng, budget):
    """Yield candidate factor-vectors for one turn."""
    out = []
    for i in range(n):
        for f in factors:
            out.append({i: f})
    for f in factors:
        out.append({i: f for i in range(n)})               # homogeneous global
    if mode == "hetero":
        for _ in range(budget):                            # blind heterogeneous vectors
            out.append({i: math.exp(rng.uniform(math.log(0.7), math.log(2.0))) for i in range(n)})
    if mode == "macro":
        for mgn in (1.0, 1.05, 1.15, 1.35):
            vec = {}
            for i, m in enumerate(truss.members):
                try:
                    fb = float(getattr(m, "fos_buckling", float("inf")) or float("inf"))
                    fy = float(getattr(m, "fos_yielding", float("inf")) or float("inf"))
                except Exception:
                    fb = fy = float("inf")
                fb = fb if fb == fb else float("inf"); fy = fy if fy == fy else float("inf")
                nb = (gb * mgn / fb) ** (1.0 / 3.0) if math.isfinite(fb) and fb > 0 else 0.7
                ny = (gy * mgn / fy) if math.isfinite(fy) and fy > 0 else 0.7
                vec[i] = min(2.0, max(0.7, max(nb, ny)))
            out.append(vec)
    return out


def episode(spec, arm, seed, max_steps=20):
    try:
        truss, goals = _load_truss_and_goals(spec)
        st = _analyze_truss(truss, goals)
    except Exception:
        return None
    sp = ((spec.get("optimization") or {}).get("shape_params") or {})
    rb = sp.get("r") or {}
    lo = float(rb.get("min", 1e-6)); hi = float(rb.get("max", 1e9))
    n = len(truss.members)
    if any(mr(m) is None or (mr(m) or 0) <= 0 for m in truss.members):
        return None
    gb = float(goals.get("minimum_fos_buckling", 1.5)); gy = float(goals.get("minimum_fos_yielding", 1.5))
    m0 = float(st.get("mass", 1.0) or 1.0)
    rng = random.Random(seed)
    factors = FINE_F if arm == "B" else BASE_F
    mode = {"A": "base", "B": "base", "C": "hetero", "D": "macro"}[arm]
    sims = 0
    for step in range(max_steps):
        if bool(st.get("is_feasible")):
            return {"solved": 1, "steps": step, "sims": sims}
        base = [mr(m) for m in truss.members]
        cands = candidates(truss, base, n, factors, mode, gb, gy, rng, budget=4)
        best = None
        for vec in cands:
            nt = apply_vec(truss, base, vec, lo, hi)
            try:
                s2 = _analyze_truss(nt, goals)
            except Exception:
                continue
            sims += 1
            v = phi(s2, goals, m0)
            if best is None or v > best[0]:
                best = (v, nt, s2)
        if best is None:
            break
        truss, st = best[1], best[2]
    return {"solved": 1 if bool(st.get("is_feasible")) else 0, "steps": max_steps, "sims": sims}


def work(t):
    fp, arm, seed = t
    try:
        r = episode(json.load(open(fp)), arm, seed)
    except Exception:
        r = None
    return (Path(fp).stem, arm, r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()
    files = [str(p) for p in sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[:a.limit]]
    tasks = [(f, arm, 7) for f in files for arm in ("A", "B", "C", "D")]
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(work, tasks, chunksize=1)
    print("wall %.0fs for %d episodes" % (time.time() - t0, len(tasks)))
    by = {}
    for pid, arm, r in res:
        if r is not None:
            by.setdefault(pid, {})[arm] = r
    keep = [p for p, d in by.items() if len(d) == 4]
    N = len(keep)
    LAB = {"A": "base: 5 factors, arity-1 + homogeneous global   [the control]",
           "B": "fine: 31 factors, same structure                [RESOLUTION only]",
           "C": "blind heterogeneous vectors, no closed form     [HETEROGENEITY only]",
           "D": "base + the FSD macro                            [closed form]"}
    print("\n=== depth-1, Phi-selected, 20 turns, paired, n=%d problems ===" % N)
    for arm in "ABCD":
        s = sum(by[p][arm]["solved"] for p in keep) / N
        sm = sum(by[p][arm]["sims"] for p in keep) / N
        print("  arm %s  solved %.3f   mean sims/episode %6.0f   %s" % (arm, s, sm, LAB[arm]))

    def mc(x, y):
        hi = sum(1 for p in keep if by[p][x]["solved"] and not by[p][y]["solved"])
        lo = sum(1 for p in keep if by[p][y]["solved"] and not by[p][x]["solved"])
        d = hi + lo
        pv = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        return hi, lo, 100 * (hi - lo) / N, pv
    print()
    for x, y in (("B", "A"), ("C", "A"), ("D", "A"), ("C", "B"), ("D", "C")):
        hi, lo, d, pv = mc(x, y)
        print("  %s vs %s : disc %3d-%-3d  delta %+5.1fpp  exact McNemar p=%.2e" % (x, y, hi, lo, d, pv))
