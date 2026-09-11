"""
EXPERIMENT 4 -- probe03 with the confound fixed, plus the decomposition that matters.

probe03's arm L sampled the policy's MARGINAL factor distribution and applied it to the
most-critical support. That destroys the policy's own conditional structure (it directs
reductions at over-built members and increases at deficit members), so arm L was
uncharitable. Fixed here.

Arms, all at MATCHED STATE, K=8 candidates each:
  Lup   policy's UP-SCALING factors only (f>1), applied to the critical support   [charitable]
  Ltwo  policy-shaped two-sided: up-factors on deficit members, down-factors on
        over-built members, both drawn from the policy's own empirical halves     [most charitable]
  U     uniform on the clamp range, same support as Lup                           [chance control]
  Uspr  uniform, but per-member independent, all members two-sided                [chance, matched shape to Ltwo]
  O     oracle SINGLE shared factor on the critical support                       [best shared magnitude]
  P     per-member closed form, oracle margin sweep, all members                  [per-member structure]

The contrast that matters is P vs O: it isolates the value of PER-MEMBER magnitude
differentiation from the value of getting one shared number right. The policy's measured
within-turn modal-element multiplier spread is 1.15x, i.e. near-shared.
"""
import sys, json, math, copy, time, random, re, glob, argparse, collections
from pathlib import Path
from math import comb
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals

RX = re.compile(r"(?:radius|thickness|r|t)\s*:\s*([0-9]*\.?[0-9]+)|,\s*(?:radius|thickness|r|t)\s*,\s*([0-9]*\.?[0-9]+)\s*\)")


def empirical():
    vals = []
    for p in glob.glob(str(PROJECT / "results/api_guidance/traces_sonnet*.jsonl")):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            for c in (json.loads(line).get("candidates") or []):
                for m in RX.finditer(c.get("action") or ""):
                    v = m.group(1) or m.group(2)
                    try:
                        f = float(v)
                    except (TypeError, ValueError):
                        continue
                    if 0.2 <= f <= 5.0:
                        vals.append(f)
    return [f for f in vals if f > 1.0], [f for f in vals if f < 1.0]


def mr(m):
    par = getattr(getattr(m, "shape", None), "_params", None) or {}
    v = par.get("r")
    return float(v) if isinstance(v, (int, float)) else None


def sr(m, v):
    par = getattr(getattr(m, "shape", None), "_params", None)
    if par is not None:
        par["r"] = v


def run(spec, UP, DN, K, rng):
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

    def f_(m, a):
        try:
            v = float(getattr(m, a, float("inf")) or float("inf"))
        except Exception:
            v = float("inf")
        return v if v == v else float("inf")

    crit = sorted(((min(f_(m, "fos_buckling"), f_(m, "fos_yielding")), i)
                   for i, m in enumerate(truss.members)))
    order = [i for _, i in crit]
    n = len(truss.members)
    k = min(rng.randint(3, 6), n)
    S = order[:k]
    D = [i for (c, i) in crit if c < max(gb, gy)]            # deficit members
    OV = [i for (c, i) in crit if c >= max(gb, gy)]          # over-built members

    def feas(fac):
        nt = copy.deepcopy(truss)
        for i, f in fac.items():
            sr(nt.members[i], min(hi, max(lo, base[i] * f)))
        try:
            return bool(_analyze_truss(nt, goals).get("is_feasible"))
        except Exception:
            return False

    out = {"problem_id": spec.get("problem_id"), "n": n, "k": k, "nD": len(D)}
    pick = lambda pool: pool[rng.randrange(len(pool))]
    U = lambda: math.exp(rng.uniform(math.log(0.7), math.log(2.0)))

    out["Lup"] = int(any(feas({i: pick(UP) for i in S}) for _ in range(K)))
    out["Ltwo"] = int(any(feas({**{i: pick(UP) for i in D}, **{i: pick(DN) for i in OV}})
                          for _ in range(K)))
    out["U"] = int(any(feas({i: U() for i in S}) for _ in range(K)))
    out["Uspr"] = int(any(feas({i: U() for i in range(n)}) for _ in range(K)))
    G = 40
    out["O"] = int(any(feas({i: math.exp(math.log(0.7) + j * (math.log(2.0) - math.log(0.7)) / (G - 1))
                             for i in S}) for j in range(G)))
    hitP = 0
    for mgn in (1.0, 1.02, 1.05, 1.10, 1.15, 1.25, 1.35, 1.5):
        fac = {}
        for i, m in enumerate(truss.members):
            fb, fy = f_(m, "fos_buckling"), f_(m, "fos_yielding")
            nb = (gb * mgn / fb) ** (1.0 / 3.0) if math.isfinite(fb) and fb > 0 else 0.7
            ny = (gy * mgn / fy) if math.isfinite(fy) and fy > 0 else 0.7
            fac[i] = min(2.0, max(0.7, max(nb, ny)))
        if feas(fac):
            hitP = 1
            break
    out["P"] = hitP
    # Ofull: oracle SINGLE shared factor applied to ALL members (isolates support from magnitude)
    out["Ofull"] = int(any(feas({i: math.exp(math.log(0.7) + j * (math.log(2.0) - math.log(0.7)) / (G - 1))
                                 for i in range(n)}) for j in range(G)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(PROJECT / "analysis_out/probe04.jsonl"))
    a = ap.parse_args()
    UP, DN = empirical()
    print("empirical factors: %d up (median %.3f), %d down (median %.3f)"
          % (len(UP), sorted(UP)[len(UP)//2], len(DN), sorted(DN)[len(DN)//2]))
    rng = random.Random(a.seed)
    files = sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[:a.limit]
    rows = []
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        for i, p in enumerate(files):
            try:
                r = run(json.load(open(p)), UP, DN, a.K, rng)
            except Exception:
                r = None
            if r:
                rows.append(r); fh.write(json.dumps(r) + "\n")
            if (i + 1) % 100 == 0:
                print("  %d/%d %.0fs" % (i + 1, len(files), time.time() - t0), flush=True)
    N = len(rows)
    LAB = {"Lup": "policy up-factors on the critical support   [charitable]",
           "Ltwo": "policy two-sided: up on deficit, down on over-built [most charitable]",
           "U": "uniform, same support as Lup                [chance]",
           "Uspr": "uniform, all members, two-sided             [chance, matched shape]",
           "O": "oracle SHARED factor, critical support",
           "Ofull": "oracle SHARED factor, all members",
           "P": "per-member closed form, oracle margin"}
    print("\n=== P(a feasible candidate appears in K=%d), matched state, n=%d ===" % (a.K, N))
    for arm in ("Lup", "Ltwo", "U", "Uspr", "O", "Ofull", "P"):
        print("  %-6s %5.3f   %s" % (arm, sum(r[arm] for r in rows) / N, LAB[arm]))

    def mc(x, y):
        hi = sum(1 for r in rows if r[x] and not r[y])
        lo = sum(1 for r in rows if r[y] and not r[x])
        d = hi + lo
        p = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        return hi, lo, 100 * (hi - lo) / N, p
    print()
    for x, y in (("Ltwo", "Lup"), ("Ltwo", "Uspr"), ("Lup", "U"), ("O", "Ltwo"),
                 ("P", "Ltwo"), ("P", "O"), ("P", "Ofull"), ("Ofull", "O")):
        hi, lo, d, p = mc(x, y)
        print("  %-5s vs %-5s : disc %3d-%-3d  delta %+5.1fpp  p=%.2e" % (x, y, hi, lo, d, p))
