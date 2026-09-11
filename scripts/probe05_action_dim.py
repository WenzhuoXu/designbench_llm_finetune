"""
EXPERIMENT 5 -- effective action dimensionality.

probe04 result: at matched state, the ENTIRE headroom is per-member differentiation of
the magnitude, not the support and not the shared magnitude.
    oracle shared factor, critical support   0.073
    oracle shared factor, ALL members        0.077   (vs previous: +0.3pp, p=1.00, NULL)
    per-member factors, oracle margin        0.250   (+17.3pp over Ofull, p=4.4e-16)
So: which members you pick barely matters under a shared factor; what matters is that
different members get DIFFERENT factors.

This experiment measures, free from the traced corpus and then against the requirement:
  (a) the WITHIN-CANDIDATE spread of the factors the policy emits, per candidate
  (b) the spread the problem actually REQUIRES at that state (from the per-member inverse)
  (c) the gap -- the policy's effective action dimensionality vs the problem's

If (a) is concentrated near zero while (b) is wide, the policy is emitting an essentially
one-dimensional action into a problem that needs an n-dimensional one, and that -- not
arity, not support choice, not the precision of any single number -- is the bottleneck.
"""
import sys, json, math, re, glob, statistics, collections
from pathlib import Path
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals

PAIR = re.compile(r"SCALE_PARAM\(\s*(\d+)\s*,\s*\w+\s*,\s*([0-9]*\.?[0-9]+)\s*\)")
MULTI = re.compile(r"SCALE_MULTI_PARAM\(\s*\[([0-9,\s]*)\]\s*,\s*\[([^\]]*)\]")
KV = re.compile(r"(\w+)\s*:\s*([0-9]*\.?[0-9]+)")


def factors_of(action):
    """member_id -> list of factors, for one candidate action (possibly a macro)."""
    d = collections.defaultdict(list)
    for m in PAIR.finditer(action):
        d[int(m.group(1))].append(float(m.group(2)))
    for m in MULTI.finditer(action):
        ids = [int(x) for x in m.group(1).replace(" ", "").split(",") if x.isdigit()]
        fs = [float(v) for _, v in KV.findall(m.group(2))]
        for i in ids:
            for f in fs:
                d[i].append(f)
    return d


def spread(vals):
    """max/min ratio of a set of positive factors; 1.0 = perfectly shared."""
    vs = [v for v in vals if v > 0]
    return (max(vs) / min(vs)) if len(vs) >= 2 else None


print("=== (a) WITHIN-CANDIDATE FACTOR SPREAD, from the traced corpus ===")
sp_all, sp_by_k = [], collections.defaultdict(list)
ncand = 0
for p in glob.glob(str(PROJECT / "results/api_guidance/traces_sonnet*.jsonl")):
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        for c in (json.loads(line).get("candidates") or []):
            d = factors_of(c.get("action") or "")
            if len(d) < 2:
                continue
            per = [statistics.median(v) for v in d.values()]
            s = spread(per)
            if s is None:
                continue
            ncand += 1
            sp_all.append(s); sp_by_k[min(len(d), 8)].append(s)
sp_all.sort()
q = lambda a, f: a[int(f * (len(a) - 1))]
print("candidates with >=2 distinct members: %d" % ncand)
print("within-candidate factor spread (max/min):")
print("   median %.3f | p25 %.3f | p75 %.3f | p90 %.3f | p99 %.3f | max %.2f"
      % (q(sp_all, .5), q(sp_all, .25), q(sp_all, .75), q(sp_all, .90), q(sp_all, .99), sp_all[-1]))
for thr in (1.0, 1.05, 1.2, 1.5, 2.0):
    print("   P(spread <= %.2f) = %.4f" % (thr, sum(1 for s in sp_all if s <= thr) / len(sp_all)))
print("   by support size:")
for k in sorted(sp_by_k):
    v = sorted(sp_by_k[k])
    if len(v) < 30:
        continue
    print("     |S|=%-2s n=%-5d median spread %.3f  P(<=1.05)=%.3f" %
          (k if k < 8 else "8+", len(v), v[len(v)//2], sum(1 for s in v if s <= 1.05)/len(v)))

print()
print("=== (b) THE SPREAD THE PROBLEM REQUIRES, at episode start ===")
req_sp, req_detail = [], []
files = sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[:300]
for fp in files:
    spec = json.load(open(fp))
    try:
        truss, goals = _load_truss_and_goals(spec)
        _analyze_truss(truss, goals)
    except Exception:
        continue
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    fs = []
    for m in truss.members:
        try:
            fb = float(getattr(m, "fos_buckling", float("inf")) or float("inf"))
            fy = float(getattr(m, "fos_yielding", float("inf")) or float("inf"))
        except Exception:
            continue
        fb = fb if fb == fb else float("inf")
        fy = fy if fy == fy else float("inf")
        nb = (gb / fb) ** (1.0 / 3.0) if math.isfinite(fb) and fb > 0 else 0.7
        ny = (gy / fy) if math.isfinite(fy) and fy > 0 else 0.7
        fs.append(min(2.0, max(0.7, max(nb, ny))))
    s = spread(fs)
    if s:
        req_sp.append(s); req_detail.append((min(fs), max(fs)))
req_sp.sort()
print("problems: %d" % len(req_sp))
print("required per-member factor spread (max/min), clamped to the grammar's own [0.7,2.0]:")
print("   median %.3f | p25 %.3f | p75 %.3f | p90 %.3f" % (q(req_sp, .5), q(req_sp, .25), q(req_sp, .75), q(req_sp, .90)))
for thr in (1.05, 1.2, 1.5, 2.0):
    print("   P(required spread > %.2f) = %.4f" % (thr, sum(1 for s in req_sp if s > thr) / len(req_sp)))
lo_ = statistics.median([a for a, _ in req_detail]); hi_ = statistics.median([b for _, b in req_detail])
print("   median required factor range: [%.3f, %.3f]  (0.7 = shrink to the clamp, 2.0 = grow to it)" % (lo_, hi_))

print()
print("=== (c) THE GAP ===")
print("policy   median within-candidate spread : %.3f" % q(sp_all, .5))
print("problem  median required spread          : %.3f" % q(req_sp, .5))
print("ratio of required to emitted             : %.2fx" % (q(req_sp, .5) / max(q(sp_all, .5), 1e-9)))
print("P(policy spread <= 1.05)                 : %.4f   <- effectively a SHARED scalar" % (sum(1 for s in sp_all if s <= 1.05)/len(sp_all)))
print("P(required spread  > 1.50)               : %.4f" % (sum(1 for s in req_sp if s > 1.5)/len(req_sp)))
