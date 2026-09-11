"""
EXPERIMENT 7 -- the same amplitude curve with NO physics.

probe06 measured the amplitude-calibration curve against a reference direction f* taken
from the closed-form structural inverse (cube root for buckling, linear for yielding).
That makes the reference a truss fact, which is exactly the dependency that has to go.

Here the reference direction is obtained from the SIMULATOR ALONE, by finite differences:
for each element i, perturb its own parameter by +-delta in log space, measure the response
of that element's OWN margin, fit a local log-log elasticity e_i, and invert:

    f_fd(i) = ( required_i / current_i ) ** (1 / e_i)

No exponent is assumed; e_i is measured. Cost is 2 FEA per element per problem (~30 calls),
i.e. nothing. If the amplitude curve reproduces against f_fd, the finding is about the
policy's emission and not about structural mechanics.

Also reports the measured elasticity distribution -- the model asserts r^4 and the project
measured a realized ~r^1 at the system level; this is the per-element truth.
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

GAMMAS = [0.0, 0.4, 0.577, 0.7, 0.85, 1.0, 1.15, 1.3]


def mr(m):
    par = getattr(getattr(m, "shape", None), "_params", None) or {}
    v = par.get("r")
    return float(v) if isinstance(v, (int, float)) else None


def sr(m, v):
    par = getattr(getattr(m, "shape", None), "_params", None)
    if par is not None:
        par["r"] = v


def own(m, gb, gy):
    """this element's own margin ratio: min over its own constraints of achieved/required."""
    def v(a):
        try:
            x = float(getattr(m, a, float("inf")) or float("inf"))
        except Exception:
            x = float("inf")
        return x if x == x else float("inf")
    return min(v("fos_buckling") / gb, v("fos_yielding") / gy)


def run(spec, delta=0.10):
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
    n = len(truss.members)
    m0 = [own(m, gb, gy) for m in truss.members]

    # --- FD elasticity of each element's OWN margin wrt its OWN parameter ---
    elas = []
    for i in range(n):
        es = []
        for s in (+delta, -delta):
            nt = copy.deepcopy(truss)
            sr(nt.members[i], min(hi, max(lo, base[i] * math.exp(s))))
            try:
                _analyze_truss(nt, goals)
            except Exception:
                continue
            m1 = own(nt.members[i], gb, gy)
            if math.isfinite(m0[i]) and math.isfinite(m1) and m0[i] > 0 and m1 > 0:
                es.append((math.log(m1) - math.log(m0[i])) / s)
        if es:
            elas.append(sum(es) / len(es))
        else:
            elas.append(None)

    # --- FD-inverted required factor per element ---
    raw = []
    for i in range(n):
        e = elas[i]
        if e is None or not math.isfinite(e) or abs(e) < 0.15:
            raw.append(0.7)              # unresponsive element: shrink, no physics assumed
            continue
        need = 1.0 / m0[i] if math.isfinite(m0[i]) and m0[i] > 0 else 2.0
        raw.append(max(1e-3, need ** (1.0 / e)))
    L = [math.log(max(r, 1e-9)) for r in raw]
    c = sum(L) / len(L)

    out = {"problem_id": spec.get("problem_id"), "n": n,
           "elas": [round(e, 3) for e in elas if e is not None and math.isfinite(e)]}
    for g in GAMMAS:
        hit = 0
        for mgn in (1.0, 1.05, 1.15, 1.3):
            nt = copy.deepcopy(truss)
            for i, m in enumerate(nt.members):
                f = math.exp(c + g * (L[i] - c)) * (mgn ** (g if g > 0 else 0.0))
                f = min(2.0, max(0.7, f))
                sr(m, min(hi, max(lo, base[i] * f)))
            try:
                if bool(_analyze_truss(nt, goals).get("is_feasible")):
                    hit = 1
                    break
            except Exception:
                pass
        out["g%.3f" % g] = hit
    # spread of the FD-required vector, for the gamma the policy actually emits
    fs = [min(2.0, max(0.7, r)) for r in raw]
    out["req_spread"] = max(fs) / min(fs) if min(fs) > 0 else None
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
    allel = [e for r in rows for e in r["elas"]]
    allel.sort()
    q = lambda f: allel[int(f * (len(allel) - 1))]
    print("=== MEASURED PER-ELEMENT ELASTICITY, d log(own margin)/d log(own radius) ===")
    print("  n=%d elements over %d problems" % (len(allel), N))
    print("  median %.3f | p10 %.3f | p25 %.3f | p75 %.3f | p90 %.3f" % (q(.5), q(.1), q(.25), q(.75), q(.9)))
    print("  P(e in [2.5,3.5]) = %.3f   (a cube law)" % (sum(1 for e in allel if 2.5 <= e <= 3.5) / len(allel)))
    print("  P(e in [3.5,4.5]) = %.3f   (the r^4 the model asserts)" % (sum(1 for e in allel if 3.5 <= e <= 4.5) / len(allel)))
    print("  P(e < 1.5)        = %.3f" % (sum(1 for e in allel if e < 1.5) / len(allel)))
    rs = sorted(r["req_spread"] for r in rows if r.get("req_spread"))
    print("\n  FD-required per-element factor spread: median %.3f (closed-form reference gave 2.857)"
          % rs[len(rs) // 2])
    print()
    print("=== AMPLITUDE CURVE AGAINST THE FD REFERENCE (no physics), n=%d ===" % N)
    print("  gamma   P(feasible)")
    for g in GAMMAS:
        note = ""
        if g == 0.0: note = "  <- pure shared factor"
        if abs(g - 0.577) < 1e-6: note = "  <- the policy's measured amplitude"
        if g == 1.0: note = "  <- full FD-inverted step"
        print("  %5.3f   %6.3f%s" % (g, sum(r["g%.3f" % g] for r in rows) / N, note))

    def mc(x, y):
        hi = sum(1 for r in rows if r[x] and not r[y])
        lo = sum(1 for r in rows if r[y] and not r[x])
        d = hi + lo
        p = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        return hi, lo, 100 * (hi - lo) / N, p
    print()
    for x, y in (("g1.000", "g0.577"), ("g1.000", "g0.000"), ("g0.850", "g0.577"), ("g1.300", "g1.000")):
        hi, lo, d, p = mc(x, y)
        print("  %s vs %s : disc %3d-%-3d delta %+5.1fpp  p=%.2e" % (x, y, hi, lo, d, p))
