import json, sys, collections, statistics
path = sys.argv[1] if len(sys.argv) > 1 else "/ocean/projects/mch250030p/wxu7/llm_finetune/analysis_out/probe01.jsonl"
R = [json.loads(l) for l in open(path) if l.strip()]
print("problems:", len(R))
print()
print("START STATE")
print("  under-performance at start : %d/%d" % (sum(r["under_perf_at_start"] for r in R), len(R)))
print("  over-budget at start       : %d/%d" % (sum(r["over_budget_at_start"] for r in R), len(R)))
print("  both                       : %d/%d" % (sum(r["under_perf_at_start"] and r["over_budget_at_start"] for r in R), len(R)))
print("  over-budget ONLY           : %d/%d" % (sum((not r["under_perf_at_start"]) and r["over_budget_at_start"] for r in R), len(R)))
ns = [r["n"] for r in R]; nd = [r["n_deficit"] for r in R]; ninf = [r["n_inf_fosb"] for r in R]
print("  members n        : median %d  range %d-%d" % (statistics.median(ns), min(ns), max(ns)))
print("  deficit set |D|  : median %d  mean %.2f  |D|>=2 in %d/%d" % (statistics.median(nd), sum(nd)/len(nd), sum(1 for x in nd if x >= 2), len(nd)))
print("  members with fos_b = inf (tension) : median %d of %d" % (statistics.median(ninf), statistics.median(ns)))
print()
print("MINIMUM ARITY k* FOR A FEASIBLE SINGLE ACTION (uniform scaling of the k most critical members)")
c = collections.Counter(r["kstar"] for r in R)
tot = len(R)
solvable = sum(v for k, v in c.items() if k is not None)
for k in sorted([x for x in c if x is not None]):
    print("  k* = %-3d : %3d problems (%.1f%%)" % (k, c[k], 100 * c[k] / tot))
print("  k* = None: %3d problems (%.1f%%)  <- no uniform-scaling action of ANY arity is feasible" % (c[None], 100 * c[None] / tot))
print("  solvable by ONE uniform action: %d/%d = %.3f" % (solvable, tot, solvable / tot))
print("  of those, k*=1 (single element suffices): %d = %.3f of solvable" % (c.get(1, 0), c.get(1, 0) / solvable if solvable else 0))
print()
print("REACHABLE MARGIN AS ARITY GROWS (median over problems, margin ratio; >=1.0 satisfies performance)")
by_k = collections.defaultdict(list); feas_k = collections.defaultdict(int); cnt_k = collections.defaultdict(int)
for r in R:
    for row in r["rows"]:
        if row.get("reach") is None:
            continue
        by_k[row["k"]].append(row["reach"]); cnt_k[row["k"]] += 1
        feas_k[row["k"]] += bool(row.get("feasible"))
for k in sorted(by_k):
    if cnt_k[k] < 5:
        continue
    v = sorted(by_k[k])
    print("  k=%-3d n=%-3d  median reach %.3f   P(feasible)=%.3f" % (k, cnt_k[k], v[len(v)//2], feas_k[k]/cnt_k[k]))
print()
print("TWO-SIDED FSD MACRO (per-member factors, ONE action)")
fs = [r["fsd"] for r in R]
print("  feasible: %d/%d = %.3f" % (sum(f["feasible"] for f in fs), len(fs), sum(f["feasible"] for f in fs)/len(fs)))
print("  median reach margin: %.3f" % statistics.median([f["margin"] for f in fs]))
mr = [f["mass"]/r["B"] for f, r in zip(fs, R) if r["B"] > 0]
print("  median mass / budget: %.3f" % statistics.median(mr))
print()
print("HEAD TO HEAD at the SAME state: best uniform action of ANY arity vs the two-sided macro")
u_ok = sum(1 for r in R if r["kstar"] is not None)
f_ok = sum(1 for r in R if r["fsd"]["feasible"])
both = sum(1 for r in R if r["kstar"] is not None and r["fsd"]["feasible"])
uonly = sum(1 for r in R if r["kstar"] is not None and not r["fsd"]["feasible"])
fonly = sum(1 for r in R if r["kstar"] is None and r["fsd"]["feasible"])
print("  uniform-any-arity feasible : %d/%d = %.3f" % (u_ok, tot, u_ok/tot))
print("  two-sided macro feasible   : %d/%d = %.3f" % (f_ok, tot, f_ok/tot))
print("  both %d | uniform only %d | macro only %d | neither %d" % (both, uonly, fonly, tot - both - uonly - fonly))
