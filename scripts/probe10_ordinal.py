"""
EXPERIMENT 10 -- does the direction have to be CARDINAL, or is ORDINAL enough?

probe08 refuted my reading of probe04. Blind heterogeneity (random per-element vectors)
buys only +5.0pp over base, while the computed macro buys +23.3pp -- so it is not
heterogeneity per se, it is CORRECTLY DIRECTED heterogeneity. probe04's arm P was the
closed form, so "per-element" and "per-element and correct" were confounded there. Mine.

The question that now decides everything: how much of the direction does a policy need?

  A  base                      5 factors, arity-1 + homogeneous global      [control]
  D  macro                     computed per-element factors                 [cardinal, exact]
  F  ordinal                   f_i depends ONLY on element i's RANK by margin, not its
                               value; amplitude chosen by an oracle sweep    [ORDINAL only]
  G  cardinal-scrambled        the exact per-element factors, randomly PERMUTED across
                               elements, amplitude preserved                 [right values,
                                                                              wrong addresses]

F isolates whether reading a per-element table and ordering it -- which the policy
demonstrably can do (92.7% element identification) -- is sufficient, or whether the
cardinal magnitudes must be computed.

G is the simulator-side analogue of the placebo: same values, wrong addresses. If G
collapses to base or below, address correctness is the binding requirement, and that is
the same mechanism the -35.6pp shuffled-table arm measured on the API side.
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


def mr(m):
    par = getattr(getattr(m, "shape", None), "_params", None) or {}
    v = par.get("r")
    return float(v) if isinstance(v, (int, float)) else None


def sr(m, v):
    par = getattr(getattr(m, "shape", None), "_params", None)
    if par is not None:
        par["r"] = v


def phi(st, goals, m0):
    gb = float(goals.get("minimum_fos_buckling", 1.5)); gy = float(goals.get("minimum_fos_yielding", 1.5))
    B = float(goals.get("maximum_mass", float("inf")))
    fb = float(st.get("fos_buckling", 0.0) or 0.0); fy = float(st.get("fos_yielding", 0.0) or 0.0)
    ms = float(st.get("mass", 9e9) or 9e9)
    v = 5.0 * min(0.0, fb / gb - 1.0) + 5.0 * min(0.0, fy / gy - 1.0)
    if math.isfinite(B) and B > 0:
        v += 5.0 * min(0.0, 1.0 - ms / B)
    return v - 0.05 * (ms / m0 if m0 > 0 else 0.0)


def fos(m, a):
    try:
        v = float(getattr(m, a, float("inf")) or float("inf"))
    except Exception:
        v = float("inf")
    return v if v == v else float("inf")


def apply_vec(truss, base, vec, lo, hi):
    nt = copy.deepcopy(truss)
    for i, f in vec.items():
        sr(nt.members[i], min(hi, max(lo, base[i] * f)))
    return nt


def macro_vec(truss, gb, gy, mgn):
    vec = {}
    for i, m in enumerate(truss.members):
        fb, fy = fos(m, "fos_buckling"), fos(m, "fos_yielding")
        nb = (gb * mgn / fb) ** (1.0 / 3.0) if math.isfinite(fb) and fb > 0 else 0.7
        ny = (gy * mgn / fy) if math.isfinite(fy) and fy > 0 else 0.7
        vec[i] = min(2.0, max(0.7, max(nb, ny)))
    return vec


def candidates(truss, base, n, arm, gb, gy, rng):
    out = []
    for i in range(n):
        for f in BASE_F:
            out.append({i: f})
    for f in BASE_F:
        out.append({i: f for i in range(n)})
    if arm == "D":
        for mgn in (1.0, 1.05, 1.15, 1.35):
            out.append(macro_vec(truss, gb, gy, mgn))
    elif arm == "F":
        # ORDINAL: rank elements by their own margin, assign a monotone factor by RANK only
        marg = sorted(range(n), key=lambda i: min(fos(truss.members[i], "fos_buckling") / gb,
                                                  fos(truss.members[i], "fos_yielding") / gy))
        for spread in (1.2, 1.5, 2.0, 2.857):        # oracle sweep over the ordinal amplitude
            lo_f = math.sqrt(1.0 / spread); hi_f = math.sqrt(spread)
            vec = {}
            for rank, i in enumerate(marg):
                t = rank / max(n - 1, 1)             # 0 = worst element, 1 = best
                vec[i] = min(2.0, max(0.7, math.exp(math.log(hi_f) + t * (math.log(lo_f) - math.log(hi_f)))))
            out.append(vec)
    elif arm == "G":
        # CARDINAL VALUES, WRONG ADDRESSES: exact factors, permuted across elements
        for mgn in (1.0, 1.05, 1.15, 1.35):
            v = macro_vec(truss, gb, gy, mgn)
            vals = [v[i] for i in range(n)]
            rng.shuffle(vals)
            out.append({i: vals[i] for i in range(n)})
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
    sims = 0
    for step in range(max_steps):
        if bool(st.get("is_feasible")):
            return {"solved": 1, "steps": step, "sims": sims}
        base = [mr(m) for m in truss.members]
        best = None
        for vec in candidates(truss, base, n, arm, gb, gy, rng):
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
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--procs", type=int, default=120)
    a = ap.parse_args()
    files = [str(p) for p in sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[:a.limit]]
    tasks = [(f, arm, 7) for f in files for arm in ("A", "D", "F", "G")]
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
    LAB = {"A": "base                                      [control]",
           "D": "computed per-element factors              [cardinal, exact]",
           "F": "factor by element RANK only, oracle spread [ORDINAL only]",
           "G": "exact factors, permuted across elements   [right values, wrong addresses]"}
    print("\n=== depth-1, Phi-selected, 20 turns, paired, n=%d ===" % N)
    for arm in ("A", "D", "F", "G"):
        s = sum(by[p][arm]["solved"] for p in keep) / N
        sm = sum(by[p][arm]["sims"] for p in keep) / N
        print("  arm %s  solved %.3f   sims/ep %6.0f   %s" % (arm, s, sm, LAB[arm]))

    def mc(x, y):
        hi = sum(1 for p in keep if by[p][x]["solved"] and not by[p][y]["solved"])
        lo = sum(1 for p in keep if by[p][y]["solved"] and not by[p][x]["solved"])
        d = hi + lo
        pv = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        return hi, lo, 100 * (hi - lo) / N, pv
    print()
    for x, y in (("F", "A"), ("D", "A"), ("G", "A"), ("D", "F"), ("F", "G"), ("D", "G")):
        hi, lo, d, pv = mc(x, y)
        print("  %s vs %s : disc %3d-%-3d delta %+5.1fpp  exact McNemar p=%.2e" % (x, y, hi, lo, d, pv))
