"""Analysis of the gate matrix: corruption-rate curve, address-vs-value at matched rank
correlation, and the family sweep. Everything paired by problem and clustered."""
import json, statistics, collections, math
from math import comb, sqrt, erf
from pathlib import Path
P = Path("/ocean/projects/mch250030p/wxu7/llm_finetune/results/api_guidance/matrix.jsonl")
rows = [json.loads(l) for l in open(P) if l.strip()]
errs = [r for r in rows if r.get("err")]
print("episodes %d | with turns %d | errors %d" % (len(rows), sum(1 for r in rows if r["turns"]), len(errs)))
if errs:
    print("  sample error:", errs[0]["err"][:120])

def ep(r):
    ht = hr = tot = 0; ch = 0.0; rhos = []
    for t in r["turns"]:
        if t.get("rho") is not None:
            rhos.append(t["rho"])
        for s in t["supports"]:
            if not s:
                continue
            tot += 1
            ht += int(t["j_true"] in s)
            hr += int(t["row_shown"] in s)
            ch += min(1.0, len(s) / max(t["n"], 1))
    if not tot:
        return None
    return {"true": ht/tot, "row": hr/tot, "chance": ch/tot, "n": tot,
            "rho": statistics.mean(rhos) if rhos else None, "feas": int(r["feasible"])}

cell = collections.defaultdict(dict)     # (model,mode,level) -> pid -> ep
for r in rows:
    e = ep(r)
    if e:
        cell[(r["model"], r["mode"], r["level"])][r["problem_id"]] = e

def short(m):
    return m.split("anthropic.")[-1].replace("-v1:0", "")

def norm_p(z):
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

def paired(A, B, keys):
    d = [A[k] - B[k] for k in keys]
    m = sum(d)/len(d); sd = statistics.stdev(d) if len(d) > 1 else 0.0
    se = sd/sqrt(len(d)) if sd else 0.0
    hi = sum(1 for x in d if x > 0); lo = sum(1 for x in d if x < 0); dd = hi+lo
    sp = 1.0 if dd == 0 else min(1.0, 2*sum(comb(dd,i) for i in range(0, min(hi,lo)+1))/2**dd)
    return m, se, (norm_p(m/se) if se else 0.0), hi, lo, sp

S45 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

print("\n" + "="*78)
print("(a) CORRUPTION-RATE SWEEP  [Sonnet 4.5, fraction of rows permuted]")
print("="*78)
print("  rho   n    mean displayed-vs-true rank corr   P(true worst)  P(shown row)  chance  feasible")
base = None
for lv in (0.0, 0.25, 0.5, 0.75, 1.0):
    d = cell.get((S45, "perm", lv), {})
    if not d:
        continue
    ks = sorted(d)
    rc = [d[k]["rho"] for k in ks if d[k]["rho"] is not None]
    print("  %.2f  %-4d %+.3f                            %.3f          %.3f         %.3f   %.3f"
          % (lv, len(ks), statistics.mean(rc) if rc else float('nan'),
             statistics.mean(d[k]["true"] for k in ks),
             statistics.mean(d[k]["row"] for k in ks),
             statistics.mean(d[k]["chance"] for k in ks),
             statistics.mean(d[k]["feas"] for k in ks)))

print("\n  paired contrasts vs rho=0 (same problems):")
d0 = cell.get((S45, "perm", 0.0), {})
for lv in (0.25, 0.5, 0.75, 1.0):
    d = cell.get((S45, "perm", lv), {})
    ks = sorted(set(d0) & set(d))
    if len(ks) < 5:
        continue
    m, se, p, hi, lo, sp = paired({k: d[k]["true"] for k in ks}, {k: d0[k]["true"] for k in ks}, ks)
    mf, _, _, hf, lf, spf = paired({k: d[k]["feas"] for k in ks}, {k: d0[k]["feas"] for k in ks}, ks)
    print("    rho=%.2f (n=%d): P(true worst) %+.3f (sign %d-%d p=%.2e) | feasible %+.3f (sign %d-%d p=%.2e)"
          % (lv, len(ks), m, hi, lo, sp, mf, hf, lf, spf))

print("\n" + "="*78)
print("(b) ADDRESS vs VALUE CORRUPTION, on the common information axis")
print("="*78)
print("  Both corruptions are scored by the SAME quantity: the rank correlation between the")
print("  displayed per-element margin ordering and the true one. If address corruption costs")
print("  more at matched rank correlation, 'address' is doing real work.")
print()
print("  mode    level  n    rank corr   P(true worst)  P(shown row)  feasible")
pts = []
for mode, lvs in (("perm", (0.0, 0.25, 0.5, 0.75, 1.0)), ("noise", (0.3, 0.8, 2.0))):
    for lv in lvs:
        d = cell.get((S45, mode, lv), {})
        if not d:
            continue
        ks = sorted(d)
        rc = [d[k]["rho"] for k in ks if d[k]["rho"] is not None]
        r_ = statistics.mean(rc) if rc else float('nan')
        tw = statistics.mean(d[k]["true"] for k in ks)
        fe = statistics.mean(d[k]["feas"] for k in ks)
        pts.append((mode, lv, r_, tw, fe, len(ks)))
        print("  %-6s  %.2f   %-4d %+.3f      %.3f          %.3f         %.3f"
              % (mode, lv, len(ks), r_, tw, statistics.mean(d[k]["row"] for k in ks), fe))
print()
print("  matched-rank-correlation comparison (nearest neighbours across modes):")
for mode, lv, r_, tw, fe, n_ in pts:
    if mode != "noise":
        continue
    near = min([p for p in pts if p[0] == "perm"], key=lambda p: abs(p[2] - r_))
    print("    noise %.2f (corr %+.3f, true-worst %.3f, feas %.3f)  vs  perm %.2f (corr %+.3f, "
          "true-worst %.3f, feas %.3f)   ->  address penalty %+.3f true-worst, %+.3f feasible"
          % (lv, r_, tw, fe, near[1], near[2], near[3], near[4], near[3] - tw, near[4] - fe))

print("\n" + "="*78)
print("(c) MODEL FAMILIES  [truthful vs fully permuted]")
print("="*78)
print("  model                              n    truthful->true  shuffled->true  shuffled->row  chance   Delta")
for m in sorted({k[0] for k in cell}):
    d0 = cell.get((m, "perm", 0.0), {}); d1 = cell.get((m, "perm", 1.0), {})
    ks = sorted(set(d0) & set(d1))
    if len(ks) < 5:
        continue
    t0 = statistics.mean(d0[k]["true"] for k in ks)
    t1 = statistics.mean(d1[k]["true"] for k in ks)
    r1 = statistics.mean(d1[k]["row"] for k in ks)
    c1 = statistics.mean(d1[k]["chance"] for k in ks)
    mm, se, p, hi, lo, sp = paired({k: d1[k]["true"] for k in ks}, {k: d0[k]["true"] for k in ks}, ks)
    mr, _, _, hr2, lr2, spr = paired({k: d1[k]["row"] for k in ks}, {k: d0[k]["row"] for k in ks}, ks)
    print("  %-34s %-4d %.3f           %.3f           %.3f          %.3f    %+.3f (sign %d-%d p=%.1e)"
          % (short(m), len(ks), t0, t1, r1, c1, mm, hi, lo, sp))
    print("       follows a FALSE address vs a TRUE one: %+.3f (sign %d-%d, p=%.2e)  <- null means "
          "equally reliable" % (mr, hr2, lr2, spr))
    bl, _, _, hb, lb, spb = paired({k: d1[k]["true"] for k in ks}, {k: d1[k]["chance"] for k in ks}, ks)
    print("       shuffled true-worst vs chance: %+.3f (sign %d-%d, p=%.2e)" % (bl, hb, lb, spb))
