"""
EXPERIMENT 3 -- the magnitude swap. THE decisive counterfactual.

probe02 established that the policy already plays the reachability-optimal ARITY
(median |S| = 5 vs measured argmax 5; 65.6% in the k=3-6 plateau; only 5.0% singletons).
So arity is not the deficit. What remains is the magnitude.

Design: hold the SUPPORT fixed at what a correct policy would choose (the k most
critical members, k drawn from the band the policy actually plays), hold the STATE
fixed, and vary ONLY where the multiplier comes from:

  arm L  : K draws from the policy's own EMPIRICAL emitted-factor distribution
           (extracted from the traced corpus, 39k+ emitted factors)
  arm U  : K draws uniform on the grammar's clamp range [0.7, 2.0]   (control:
           is arm L worse than chance, or just imprecise?)
  arm O  : the oracle magnitude -- best single factor on a dense grid (upper bound)
  arm P  : per-member oracle factors (the two-sided closed form's shape, oracle-tuned)

Matched K per arm. Endpoint: P(a feasible candidate appears among the K).
CPU only.
"""
import sys, json, math, copy, time, random, re, glob, argparse, collections
from pathlib import Path
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals


def empirical_factors():
    """Marginal distribution of emitted scale factors from the traced corpus."""
    pats = glob.glob(str(PROJECT / "results/api_guidance/traces_sonnet*.jsonl"))
    vals = []
    rx = re.compile(r"(?:radius|thickness|r|t)\s*:\s*([0-9]*\.?[0-9]+)|,\s*(?:radius|thickness|r|t)\s*,\s*([0-9]*\.?[0-9]+)\s*\)")
    for p in pats:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for c in (r.get("candidates") or []):
                a = c.get("action") or ""
                for m in rx.finditer(a):
                    v = m.group(1) or m.group(2)
                    try:
                        f = float(v)
                    except (TypeError, ValueError):
                        continue
                    if 0.2 <= f <= 5.0:
                        vals.append(f)
    return vals


def member_r(m):
    par = getattr(getattr(m, "shape", None), "_params", None) or {}
    v = par.get("r")
    return float(v) if isinstance(v, (int, float)) else None


def set_r(m, v):
    par = getattr(getattr(m, "shape", None), "_params", None)
    if par is None:
        return False
    par["r"] = v
    return True


def apply_factors(truss, base_r, S, facs, lo, hi):
    nt = copy.deepcopy(truss)
    for i, f in zip(S, facs):
        set_r(nt.members[i], min(hi, max(lo, base_r[i] * f)))
    return nt


def run(spec, FAC, K, rng, kband=(3, 6)):
    try:
        truss, goals = _load_truss_and_goals(spec)
    except Exception:
        return None
    sp = ((spec.get("optimization") or {}).get("shape_params") or {})
    rb = sp.get("r") or {}
    lo = float(rb.get("min", 1e-6)); hi = float(rb.get("max", 1e9))
    base_r = [member_r(m) for m in truss.members]
    if any(r is None or r <= 0 for r in base_r):
        return None
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    try:
        _analyze_truss(truss, goals)      # per-member FOS only exists after analysis
    except Exception:
        return None

    def _fb(m):
        try:
            v = float(getattr(m, "fos_buckling", float("inf")) or float("inf"))
        except Exception:
            v = float("inf")
        return v if v == v else float("inf")

    def _fy(m):
        try:
            v = float(getattr(m, "fos_yielding", float("inf")) or float("inf"))
        except Exception:
            v = float("inf")
        return v if v == v else float("inf")

    crit = sorted(((min(_fb(m), _fy(m)), i) for i, m in enumerate(truss.members)))
    order = [i for _, i in crit]
    n = len(truss.members)
    k = min(rng.randint(kband[0], kband[1]), n)
    S = order[:k]

    def feas(nt):
        try:
            return bool(_analyze_truss(nt, goals).get("is_feasible"))
        except Exception:
            return False

    out = {}
    # arm L: policy's own empirical factor distribution, one factor per member per candidate
    hit = 0
    for _ in range(K):
        facs = [FAC[rng.randrange(len(FAC))] for _ in S]
        if feas(apply_factors(truss, base_r, S, facs, lo, hi)):
            hit = 1; break
    out["L"] = hit
    # arm U: uniform on the clamp range
    hit = 0
    for _ in range(K):
        facs = [math.exp(rng.uniform(math.log(0.7), math.log(2.0))) for _ in S]
        if feas(apply_factors(truss, base_r, S, facs, lo, hi)):
            hit = 1; break
    out["U"] = hit
    # arm O: oracle single factor on a dense grid, same support (grid size = K*8 for a fair-ish sweep)
    G = 40
    hit = 0
    for j in range(G):
        f = math.exp(math.log(0.7) + j * (math.log(2.0) - math.log(0.7)) / (G - 1))
        if feas(apply_factors(truss, base_r, S, [f] * len(S), lo, hi)):
            hit = 1; break
    out["O"] = hit
    # arm P: per-member oracle factors, all members (closed-form shape, oracle margin sweep)
    hit = 0
    for mgn in (1.0, 1.02, 1.05, 1.10, 1.15, 1.25, 1.35, 1.5):
        nt = copy.deepcopy(truss)
        for i, m in enumerate(truss.members):
            fb = _fb(m); fy = _fy(m)
            nb = (gb * mgn / fb) ** (1.0 / 3.0) if math.isfinite(fb) and fb > 0 else 0.7
            ny = (gy * mgn / fy) if math.isfinite(fy) and fy > 0 else 0.7
            f = min(2.0, max(0.7, max(nb, ny)))
            set_r(nt.members[i], min(hi, max(lo, base_r[i] * f)))
        if feas(nt):
            hit = 1; break
    out["P"] = hit
    out["k"] = k; out["n"] = n
    out["problem_id"] = spec.get("problem_id")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(PROJECT / "analysis_out/probe03.jsonl"))
    a = ap.parse_args()
    FAC = empirical_factors()
    print("empirical emitted factors: n=%d" % len(FAC))
    c = collections.Counter(round(f, 2) for f in FAC)
    print("  top 10:", c.most_common(10))
    print("  P(f<1)=%.3f  P(f>1)=%.3f  median=%.3f"
          % (sum(1 for f in FAC if f < 1) / len(FAC), sum(1 for f in FAC if f > 1) / len(FAC),
             sorted(FAC)[len(FAC) // 2]))
    rng = random.Random(a.seed)
    files = sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[:a.limit]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); rows = []
    with open(a.out, "w") as fh:
        for i, p in enumerate(files):
            try:
                r = run(json.load(open(p)), FAC, a.K, rng)
            except Exception:
                r = None
            if r is None:
                continue
            rows.append(r); fh.write(json.dumps(r) + "\n")
            if (i + 1) % 50 == 0:
                print("  %d/%d %.0fs" % (i + 1, len(files), time.time() - t0), flush=True)
    N = len(rows)
    print("\n=== P(a feasible candidate appears), matched support and state, n=%d, K=%d ===" % (N, a.K))
    for arm, lab in (("L", "policy's own empirical factor distribution"),
                     ("U", "uniform on the clamp range [0.7,2.0]"),
                     ("O", "oracle single factor, same support"),
                     ("P", "per-member closed form, oracle margin")):
        print("  arm %s  %5.3f   %s" % (arm, sum(r[arm] for r in rows) / N, lab))
    from math import comb
    def mc(x, y):
        hi = sum(1 for r in rows if r[x] and not r[y]); lo = sum(1 for r in rows if r[y] and not r[x])
        d = hi + lo
        p = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        return hi, lo, p
    print()
    for x, y in (("O", "L"), ("O", "U"), ("L", "U"), ("P", "L"), ("O", "P")):
        hi, lo, p = mc(x, y)
        print("  %s vs %s : disc %d-%d, delta %+.1fpp, exact McNemar p=%.2e"
              % (x, y, hi, lo, 100 * (hi - lo) / N, p))
