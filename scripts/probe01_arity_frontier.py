"""
EXPERIMENT 1 -- the arity-feasibility frontier.

First-principles question: at an episode start state, what is REACHABLE by a single
sizing action, as a function of how many elements the action touches?

For each problem:
  - compute the start state and the per-member margins
  - order members by criticality (ascending min(fos_b, fos_y))
  - for nested supports S_k = the k most critical members, k = 1..n
      sweep a uniform log-magnitude over the grammar's own clamp range [0.7, 2.0]
      record max over t of (min global margin ratio) subject to mass <= limit
  - also run the two-sided FSD macro (per-member factors) for reference
  - report: the MINIMUM ARITY k* at which a feasible single action exists, or None

Endpoint: distribution of k* over problems. This is the empirical version of the
arity claim and it is measured, not assumed.

CPU only. ~1.4 ms per FEA.
"""
import sys, json, math, copy, time, argparse
from pathlib import Path
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals


def member_r(m):
    par = getattr(getattr(m, "shape", None), "_params", None) or {}
    v = par.get("r")
    return float(v) if isinstance(v, (int, float)) else None


def set_member_r(m, val):
    par = getattr(getattr(m, "shape", None), "_params", None)
    if par is None:
        return False
    par["r"] = val
    return True


def bounds_of(spec):
    sp = ((spec.get("optimization") or {}).get("shape_params") or {})
    rb = sp.get("r") or {}
    lo = float(rb.get("min", 1e-6)); hi = float(rb.get("max", 1e9))
    return lo, hi


def margins(st, goals):
    """ratio of achieved to required, per constraint; >=1 means satisfied."""
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    fb = float(st.get("fos_buckling", 0.0) or 0.0)
    fy = float(st.get("fos_yielding", 0.0) or 0.0)
    return min(fb / gb if gb > 0 else 9e9, fy / gy if gy > 0 else 9e9)


def run_problem(path, n_grid=29):
    spec = json.load(open(path))
    try:
        truss, goals = _load_truss_and_goals(spec)
    except Exception:
        return None
    lo, hi = bounds_of(spec)
    B = float(goals.get("maximum_mass", float("inf")))
    st0 = _analyze_truss(truss, goals)
    base_r = [member_r(m) for m in truss.members]
    if any(r is None or r <= 0 for r in base_r):
        return None
    n = len(truss.members)

    # per-member criticality from the START state
    crit = []
    for i, m in enumerate(truss.members):
        fb = float(getattr(m, "fos_buckling", float("inf")) or float("inf"))
        fy = float(getattr(m, "fos_yielding", float("inf")) or float("inf"))
        crit.append((min(fb, fy), i))
    crit.sort()
    order = [i for _, i in crit]

    m0 = float(st0.get("mass", 0.0))
    marg0 = margins(st0, goals)
    # deficit set at the start: members whose own margin is below the requirement
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    D = [i for (c, i) in crit if c < max(gb, gy)]

    factors = [math.exp(t) for t in
               [math.log(0.70) + k * (math.log(2.0) - math.log(0.70)) / (n_grid - 1)
                for k in range(n_grid)]]

    rows = []
    kstar = None
    for k in range(1, n + 1):
        S = order[:k]
        best = None
        for f in factors:
            nt = copy.deepcopy(truss)
            okall = True
            for i in S:
                v = base_r[i] * f
                if v < lo or v > hi:
                    okall = False
                    break
                set_member_r(nt.members[i], v)
            if not okall:
                continue
            st = _analyze_truss(nt, goals)
            mm = margins(st, goals)
            ms = float(st.get("mass", 9e9))
            feas = bool(st.get("is_feasible"))
            cand = (mm, ms, f, feas)
            # objective: maximise margin subject to the budget
            if ms <= B:
                if best is None or mm > best[0]:
                    best = cand
        if best is None:
            rows.append({"k": k, "reach": None})
            continue
        rows.append({"k": k, "reach": round(best[0], 5), "mass": round(best[1], 3),
                     "f": round(best[2], 4), "feasible": best[3]})
        if best[3] and kstar is None:
            kstar = k

    # two-sided FSD macro reference (per-member factors, one action)
    nt = copy.deepcopy(truss)
    okall = True
    for i, m in enumerate(truss.members):
        fb = float(getattr(m, "fos_buckling", float("inf")) or float("inf"))
        fy = float(getattr(m, "fos_yielding", float("inf")) or float("inf"))
        nb = (gb * 1.05 / fb) ** (1.0 / 3.0) if math.isfinite(fb) and fb > 0 else 0.7
        ny = (gy * 1.05 / fy) if math.isfinite(fy) and fy > 0 else 0.7
        f = min(2.0, max(0.7, max(nb, ny)))
        v = min(hi, max(lo, base_r[i] * f))
        set_member_r(nt.members[i], v)
    st_fsd = _analyze_truss(nt, goals)

    return {
        "problem_id": spec.get("problem_id"), "n": n,
        "mass0": round(m0, 3), "B": round(B, 3), "over_budget_at_start": m0 > B,
        "margin0": round(marg0, 5), "under_perf_at_start": marg0 < 1.0,
        "n_deficit": len(D), "n_inf_fosb": sum(1 for m in truss.members
                                               if not math.isfinite(float(getattr(m, "fos_buckling", float("inf")) or float("inf")))),
        "rows": rows, "kstar": kstar,
        "fsd": {"margin": round(margins(st_fsd, goals), 5),
                "mass": round(float(st_fsd.get("mass", 9e9)), 3),
                "feasible": bool(st_fsd.get("is_feasible"))},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="/ocean/projects/mch250030p/wxu7/llm_finetune/analysis_out/probe01.jsonl")
    a = ap.parse_args()
    files = sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[:a.limit]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok = 0
    with open(a.out, "w") as fh:
        for i, p in enumerate(files):
            r = run_problem(p)
            if r is None:
                continue
            ok += 1
            fh.write(json.dumps(r) + "\n")
            if (i + 1) % 10 == 0:
                print("  %d/%d  %.1fs" % (i + 1, len(files), time.time() - t0), flush=True)
    print("done: %d problems, %.1f s" % (ok, time.time() - t0))
