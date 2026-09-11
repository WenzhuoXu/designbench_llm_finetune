"""
EXPERIMENT 12 -- the causal decomposition, and the fix it implies.

Established:
  probe02  the policy already plays the reachability-optimal ARITY (median |S|=5).
  probe11  the policy's DIRECTION is good: Spearman rho(quoted margin, emitted factor)
           mean -0.657, median -0.886, 58.6% at rho <= -0.8, 10.9% wrong-signed.
  probe05  the policy's AMPLITUDE is 0.577 of what is required.
  probe07  the amplitude curve peaks at 1.0 and is steep at 0.577 (+21pp to reach 1.0),
           and it reproduces with a finite-difference reference, no physics.
  probe10  ordinal direction + an ORACLE AMPLITUDE SWEEP recovers 18.0 of the macro's
           26.0pp; exact values at permuted addresses are worth +2.5pp (p=0.125, null).
  A7       the policy's K candidates are near-duplicates in direction AND amplitude
           (pairwise Jaccard 0.674, modal-element multiplier spread only 1.15x).

So the policy has the direction and the arity, has the wrong amplitude, and -- the part
that matters -- spends its whole candidate budget re-sampling the direction it already has
instead of varying the amplitude it does not.

Four arms isolate that claim. All use a rank-based direction; none uses a closed form.
  A   base                          the 5-factor enumerator                  [control]
  P   policy-like                   direction at policy quality (rho ~ -0.66),
                                    FIXED amplitude gamma = 0.577, K near-duplicates
                                                                             [the policy]
  S   policy-like + AMPLITUDE SWEEP same direction quality, same K budget, but K spent
                                    on distinct amplitudes                   [THE FIX]
  F   perfect ordinal + sweep       rho = -1                                 [direction ceiling]

S vs P is the intervention: identical information, identical budget, the candidate budget
reallocated from direction jitter to amplitude. F vs S prices the residual direction error.
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
K = 8                      # matched candidate budget for P and S
POLICY_GAMMA = 0.577       # measured in probe05
NOISE = 0.55               # rank-noise giving Spearman ~ -0.66 (calibrated below)


def mr(m):
    par = getattr(getattr(m, "shape", None), "_params", None) or {}
    v = par.get("r")
    return float(v) if isinstance(v, (int, float)) else None


def sr(m, v):
    par = getattr(getattr(m, "shape", None), "_params", None)
    if par is not None:
        par["r"] = v


def fos(m, a):
    try:
        v = float(getattr(m, a, float("inf")) or float("inf"))
    except Exception:
        v = float("inf")
    return v if v == v else float("inf")


def phi(st, goals, m0):
    gb = float(goals.get("minimum_fos_buckling", 1.5)); gy = float(goals.get("minimum_fos_yielding", 1.5))
    B = float(goals.get("maximum_mass", float("inf")))
    fb = float(st.get("fos_buckling", 0.0) or 0.0); fy = float(st.get("fos_yielding", 0.0) or 0.0)
    ms = float(st.get("mass", 9e9) or 9e9)
    v = 5.0 * min(0.0, fb / gb - 1.0) + 5.0 * min(0.0, fy / gy - 1.0)
    if math.isfinite(B) and B > 0:
        v += 5.0 * min(0.0, 1.0 - ms / B)
    return v - 0.05 * (ms / m0 if m0 > 0 else 0.0)


def direction(truss, n, gb, gy, noise, rng):
    """Rank elements by own margin, with multiplicative rank noise. Returns t in [0,1]
    per element, 0 = act on hardest. No cardinal magnitude used."""
    key = []
    for i, m in enumerate(truss.members):
        marg = min(fos(m, "fos_buckling") / gb, fos(m, "fos_yielding") / gy)
        marg = min(marg, 50.0)
        key.append((math.log(max(marg, 1e-6)) + (rng.gauss(0, noise) if noise > 0 else 0.0), i))
    key.sort()
    t = {}
    for rank, (_, i) in enumerate(key):
        t[i] = rank / max(n - 1, 1)
    return t


def vec_from(t, spread, n):
    lo_f = math.sqrt(1.0 / spread); hi_f = math.sqrt(spread)
    return {i: min(2.0, max(0.7, math.exp(math.log(hi_f) + t[i] * (math.log(lo_f) - math.log(hi_f)))))
            for i in range(n)}


def apply_vec(truss, base, vec, lo, hi):
    nt = copy.deepcopy(truss)
    for i, f in vec.items():
        sr(nt.members[i], min(hi, max(lo, base[i] * f)))
    return nt


REQ_SPREAD = 2.857          # measured in probe05
SWEEP = [REQ_SPREAD ** g for g in (0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)]


def candidates(truss, base, n, arm, gb, gy, rng):
    out = []
    for i in range(n):
        for f in BASE_F:
            out.append({i: f})
    for f in BASE_F:
        out.append({i: f for i in range(n)})
    if arm == "A":
        return out
    if arm == "P":
        # policy: K near-duplicate candidates -- direction resampled, amplitude fixed
        for _ in range(K):
            t = direction(truss, n, gb, gy, NOISE, rng)
            out.append(vec_from(t, REQ_SPREAD ** POLICY_GAMMA, n))
    elif arm == "S":
        # the fix: ONE direction of the same quality, K distinct amplitudes
        t = direction(truss, n, gb, gy, NOISE, rng)
        for s in SWEEP:
            out.append(vec_from(t, s, n))
    elif arm == "F":
        t = direction(truss, n, gb, gy, 0.0, rng)
        for s in SWEEP:
            out.append(vec_from(t, s, n))
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
            return {"solved": 1, "sims": sims}
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
    return {"solved": 1 if bool(st.get("is_feasible")) else 0, "sims": sims}


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
    ap.add_argument("--procs", type=int, default=60)
    a = ap.parse_args()

    # calibrate NOISE -> Spearman, reported so the arm's direction quality is auditable
    rng = random.Random(0)
    def spear(xs, ys):
        n = len(xs)
        rk = lambda v: [sorted(range(n), key=lambda i: v[i]).index(i) for i in range(n)]
        rx, ry = rk(xs), rk(ys)
        mx = my = (n - 1) / 2
        num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
        d = (sum((rx[i]-mx)**2 for i in range(n)) * sum((ry[i]-my)**2 for i in range(n))) ** .5
        return num/d if d else 0.0
    rr = []
    for _ in range(400):
        n = 16
        true = [rng.gauss(0, 1) for _ in range(n)]
        obs = [true[i] + rng.gauss(0, NOISE) for i in range(n)]
        rr.append(spear(true, obs))
    print("direction-quality calibration: NOISE=%.2f -> mean Spearman vs truth %+.3f "
          "(policy measured %+.3f mean / %+.3f median)" % (NOISE, sum(rr)/len(rr), -0.657, -0.886))

    files = [str(p) for p in sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[:a.limit]]
    tasks = [(f, arm, 11) for f in files for arm in ("A", "P", "S", "F")]
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
    LAB = {"A": "base enumerator                                  [control]",
           "P": "policy-like: direction resampled, FIXED gamma=0.577  [the policy]",
           "S": "same direction quality, K spent on AMPLITUDES     [THE FIX]",
           "F": "perfect ordinal direction + sweep                 [direction ceiling]"}
    print("\n=== depth-1, Phi-selected, 20 turns, paired, n=%d, K=%d for P and S ===" % (N, K))
    for arm in ("A", "P", "S", "F"):
        s = sum(by[p][arm]["solved"] for p in keep) / N
        sm = sum(by[p][arm]["sims"] for p in keep) / N
        print("  arm %s  solved %.3f  sims/ep %6.0f  %s" % (arm, s, sm, LAB[arm]))

    def mc(x, y):
        hi = sum(1 for p in keep if by[p][x]["solved"] and not by[p][y]["solved"])
        lo = sum(1 for p in keep if by[p][y]["solved"] and not by[p][x]["solved"])
        d = hi + lo
        pv = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        return hi, lo, 100 * (hi - lo) / N, pv
    print()
    for x, y in (("P", "A"), ("S", "A"), ("S", "P"), ("F", "S"), ("F", "A")):
        hi, lo, d, pv = mc(x, y)
        print("  %s vs %s : disc %3d-%-3d delta %+5.1fpp  exact McNemar p=%.2e" % (x, y, hi, lo, d, pv))
