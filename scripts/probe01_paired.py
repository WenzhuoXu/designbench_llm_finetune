"""Paired (within-problem) analysis of the arity frontier. Fixes the subset confound:
each problem contributes to every k it defines, and comparisons are made within problem."""
import json, sys, collections, statistics
from math import comb
path = sys.argv[1]
R = [json.loads(l) for l in open(path) if l.strip()]
print("problems:", len(R))

def sign_test(a, b):
    """paired: count problems where a>b vs b>a; exact two-sided binomial."""
    hi = lo = 0
    for x, y in zip(a, b):
        if x is None or y is None: continue
        if x > y: hi += 1
        elif y > x: lo += 1
    d = hi + lo
    if d == 0: return hi, lo, 1.0
    p = 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / (2 ** d)
    return hi, lo, min(p, 1.0)

print()
print("=== 1. START STATE (n=%d) ===" % len(R))
up = sum(r["under_perf_at_start"] for r in R); ob = sum(r["over_budget_at_start"] for r in R)
bo = sum(r["under_perf_at_start"] and r["over_budget_at_start"] for r in R)
oo = sum((not r["under_perf_at_start"]) and r["over_budget_at_start"] for r in R)
print("under-performance %d/%d (%.3f) | over-budget %d (%.3f) | both %d (%.3f) | over-budget ONLY %d"
      % (up, len(R), up/len(R), ob, ob/len(R), bo, bo/len(R), oo))
nd = [r["n_deficit"] for r in R]
print("deficit |D|: median %d  mean %.2f  |D|>=2 in %d/%d (%.3f)  |D|>=3 in %d"
      % (statistics.median(nd), sum(nd)/len(nd), sum(1 for x in nd if x>=2), len(nd),
         sum(1 for x in nd if x>=2)/len(nd), sum(1 for x in nd if x>=3)))

print()
print("=== 2. MINIMUM ARITY k* FOR A ONE-STEP FEASIBLE ACTION ===")
c = collections.Counter(r["kstar"] for r in R)
solv = sum(v for k,v in c.items() if k is not None)
print("solvable in ONE uniform action: %d/%d = %.3f" % (solv, len(R), solv/len(R)))
print("k*=1 (a single element ever suffices): %d  -> %.4f of all problems" % (c.get(1,0), c.get(1,0)/len(R)))
print("k*=2 : %d | k* in 3..5 : %d | k* >= 6 : %d"
      % (c.get(2,0), sum(c.get(k,0) for k in (3,4,5)), sum(v for k,v in c.items() if k is not None and k>=6)))
ks = [r["kstar"] for r in R if r["kstar"] is not None]
if ks: print("k* among solvable: median %d  min %d  max %d" % (statistics.median(ks), min(ks), max(ks)))

print()
print("=== 3. PAIRED REACH vs ARITY (within problem; only problems defining both k) ===")
def reach_at(r, k):
    for row in r["rows"]:
        if row["k"] == k: return row.get("reach")
    return None
print(" k_a  k_b    a>b   b>a      p        median(reach_a - reach_b)")
for (ka, kb) in [(1,2),(2,3),(3,4),(3,6),(6,9),(1,3),(3,'D'),(1,'D')]:
    A=[];B=[]
    for r in R:
        kb2 = min(max(r["n_deficit"],1), r["n"]) if kb=='D' else kb
        ka2 = min(max(r["n_deficit"],1), r["n"]) if ka=='D' else ka
        x = reach_at(r, ka2); y = reach_at(r, kb2)
        if x is not None and y is not None: A.append(x); B.append(y)
    if len(A) < 10: continue
    hi, lo, p = sign_test(A, B)
    d = statistics.median([x-y for x,y in zip(A,B)])
    print(" %-4s %-4s  %4d  %4d   %.2e   %+.4f   (n=%d)" % (ka, kb, hi, lo, p, d, len(A)))

print()
print("=== 4. WHERE IS THE OPTIMAL ARITY, WITHIN PROBLEM? ===")
best_k = []
for r in R:
    rows = [(row["k"], row.get("reach")) for row in r["rows"] if row.get("reach") is not None]
    if not rows: continue
    best_k.append(max(rows, key=lambda t: t[1])[0])
cb = collections.Counter(best_k)
print("argmax-arity distribution over %d problems:" % len(best_k))
for k in sorted(cb): print("   k=%-3d %3d (%.3f)" % (k, cb[k], cb[k]/len(best_k)))
print("median argmax arity: %d ; fraction with argmax==1: %.4f ; argmax<=2: %.4f"
      % (statistics.median(best_k), cb.get(1,0)/len(best_k), (cb.get(1,0)+cb.get(2,0))/len(best_k)))

print()
print("=== 5. TWO-SIDED MACRO vs BEST UNIFORM ACTION, PAIRED, SAME STATE ===")
u = [1 if r["kstar"] is not None else 0 for r in R]
f = [1 if r["fsd"]["feasible"] else 0 for r in R]
hi, lo, p = sign_test(u, f)
print("uniform-any-arity feasible %d/%d = %.3f | macro feasible %d = %.3f"
      % (sum(u), len(R), sum(u)/len(R), sum(f), sum(f)/len(R)))
print("paired: uniform-only %d, macro-only %d, exact p=%.2e" % (hi, lo, p))
mrg_u = [max((row.get("reach") or -9) for row in r["rows"]) for r in R]
mrg_f = [r["fsd"]["margin"] for r in R]
hi2, lo2, p2 = sign_test(mrg_u, mrg_f)
print("reach margin paired: uniform better %d, macro better %d, p=%.2e; median diff %+.4f"
      % (hi2, lo2, p2, statistics.median([a-b for a,b in zip(mrg_u,mrg_f)])))
mb = [r["fsd"]["mass"]/r["B"] for r in R if r["B"]>0]
print("macro mass/budget: median %.3f ; over budget in %d/%d" % (statistics.median(mb), sum(1 for x in mb if x>1.0), len(mb)))
