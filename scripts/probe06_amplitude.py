"""
EXPERIMENT 6 -- the amplitude-calibration curve.

probe04: per-member DIFFERENTIATION of the magnitude is worth +17.3pp over any shared
         factor (p=4.4e-16); support choice under a shared factor is worth 0.0 (p=1.00).
probe05: the policy DOES differentiate -- median within-candidate spread 1.833 -- but the
         problem requires 2.857 (the full legal clamp span, [0.700, 2.000] median range).
         In log terms the policy emits log(1.833)/log(2.857) = 0.577 of the required span.

Hypothesis: the policy has approximately the right PATTERN across members at the wrong
AMPLITUDE. Test it by taking the required per-member log-factor vector and scaling its
amplitude by gamma, holding direction and centre fixed:

    f_gamma(i) = exp( c + gamma * ( log f*(i) - c ) ),    c = mean_i log f*(i)

gamma = 1.0 is the closed form. gamma = 0.577 is the policy's measured amplitude.
If feasibility falls steeply between gamma = 1 and gamma = 0.58, then the policy's deficit
is amplitude calibration on a vector-valued action -- not support, not arity, not the
precision of any single number.

Reports the full curve, plus the centre-only control (gamma = 0, a pure shared factor).
"""
import sys, json, math, copy, statistics, argparse
from pathlib import Path
from math import comb
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals

GAMMAS = [0.0, 0.2, 0.4, 0.577, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6]


def mr(m):
    par = getattr(getattr(m, "shape", None), "_params", None) or {}
    v = par.get("r")
    return float(v) if isinstance(v, (int, float)) else None


def sr(m, v):
    par = getattr(getattr(m, "shape", None), "_params", None)
    if par is not None:
        par["r"] = v


def run(spec):
    try:
        truss, goals = _load_truss_and_goals(spec)
        _analyze_truss(truss, goals)
    except Exception:
        return None
    sp = ((spec.get("optimization") or {}).get("shape_params") or {})
    rb = sp.get("r") or {}
    lo = float(rb.get("min", 1e-6)); hi = float(rb.get("max", 1e9))
    base = [mr(m) for m in truss.members]
    if any(r is None or r <= 0 for r in base):
        return None
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))

    def val(m, a):
        try:
            v = float(getattr(m, a, float("inf")) or float("inf"))
        except Exception:
            v = float("inf")
        return v if v == v else float("inf")

    # required per-member factor, UNCLAMPED, at the best of a small margin sweep
    out = {"problem_id": spec.get("problem_id"), "n": len(truss.members)}
    best = {g: 0 for g in GAMMAS}
    for mgn in (1.0, 1.05, 1.15, 1.3):
        raw = []
        for m in truss.members:
            fb, fy = val(m, "fos_buckling"), val(m, "fos_yielding")
            nb = (gb * mgn / fb) ** (1.0 / 3.0) if math.isfinite(fb) and fb > 0 else 0.7
            ny = (gy * mgn / fy) if math.isfinite(fy) and fy > 0 else 0.7
            raw.append(max(nb, ny))
        L = [math.log(max(r, 1e-9)) for r in raw]
        c = sum(L) / len(L)
        for g in GAMMAS:
            nt = copy.deepcopy(truss)
            for i, m in enumerate(nt.members):
                f = math.exp(c + g * (L[i] - c))
                f = min(2.0, max(0.7, f))                 # the grammar's own clamp
                sr(m, min(hi, max(lo, base[i] * f)))
            try:
                if bool(_analyze_truss(nt, goals).get("is_feasible")):
                    best[g] = 1
            except Exception:
                pass
    for g in GAMMAS:
        out["g%.3f" % g] = best[g]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()
    files = sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[:a.limit]
    rows = []
    for fp in files:
        try:
            r = run(json.load(open(fp)))
        except Exception:
            r = None
        if r:
            rows.append(r)
    N = len(rows)
    print("=== AMPLITUDE-CALIBRATION CURVE, n=%d problems, one action from the start state ===" % N)
    print("gamma = fraction of the REQUIRED per-member log-differentiation actually emitted")
    print()
    print("  gamma   P(feasible)   note")
    for g in GAMMAS:
        note = ""
        if g == 0.0: note = "<- pure shared factor, no differentiation"
        if abs(g - 0.577) < 1e-6: note = "<- THE POLICY'S MEASURED AMPLITUDE"
        if g == 1.0: note = "<- the closed form"
        print("  %5.3f   %6.3f        %s" % (g, sum(r["g%.3f" % g] for r in rows) / N, note))

    def mc(x, y):
        hi = sum(1 for r in rows if r[x] and not r[y])
        lo = sum(1 for r in rows if r[y] and not r[x])
        d = hi + lo
        p = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        return hi, lo, 100 * (hi - lo) / N, p
    print()
    for x, y in (("g1.000", "g0.577"), ("g1.000", "g0.000"), ("g0.577", "g0.000"),
                 ("g1.300", "g1.000"), ("g0.850", "g0.577")):
        hi, lo, d, p = mc(x, y)
        print("  %s vs %s : disc %3d-%-3d  delta %+5.1fpp  exact McNemar p=%.2e" % (x, y, hi, lo, d, p))
