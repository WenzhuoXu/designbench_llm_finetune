import json, sys, math
from collections import Counter, defaultdict
from math import comb

def sign_test(a, b):
    up = sum(1 for i in range(len(a)) if b[i] > a[i])
    dn = sum(1 for i in range(len(a)) if b[i] < a[i])
    m = up + dn
    if m == 0: return up, dn, 1.0
    kk = min(up, dn)
    return up, dn, min(1.0, sum(comb(m, i) for i in range(kk+1)) / (2.0**m) * 2)

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print("=" * 70)
    print(path, " rows:", len(rows))
    pids = sorted(set(r["problem_id"] for r in rows))
    print("distinct problems:", len(pids), " first:", pids[0], " last:", pids[-1])
    budgets = sorted(set(r["budget"] for r in rows))
    arms = sorted(set(r["arm"] for r in rows))
    print("budgets:", budgets, " arms:", arms)
    # completeness: every (pid,arm,budget) exactly once?
    c = Counter((r["problem_id"], r["arm"], r["budget"]) for r in rows)
    dup = [k for k, v in c.items() if v != 1]
    missing = [(p, a, b) for p in pids for a in arms for b in budgets if (p, a, b) not in c]
    print("duplicate cells:", len(dup), " missing cells:", len(missing))
    errs = [r for r in rows if r.get("err")]
    print("errored episodes:", len(errs), Counter(r["err"][:40] for r in errs).most_common(3))
    mism = [r for r in rows if r.get("calls",0) >= 0 and r.get("calls") != r.get("fea")]
    print("calls != fea :", len(mism))
    over = [r for r in rows if r.get("calls",0) > r.get("budget",0)]
    print("budget overruns:", len(over))
    under = defaultdict(list)
    for r in rows: under[(r["arm"], r["budget"])].append(r.get("calls",0))
    for B in budgets:
        sub = [r for r in rows if r["budget"] == B]
        by = {a: {r["problem_id"]: (1.0 if r["feasible"] else 0.0) for r in sub if r["arm"]==a} for a in arms}
        common = sorted(set.intersection(*[set(by[a]) for a in arms]))
        print("--- budget %d  paired n=%d" % (B, len(common)))
        for a in arms:
            v = [by[a][p] for p in common]
            cl = [r["calls"] for r in sub if r["arm"]==a and r["calls"]>=0]
            atcap = sum(1 for x in cl if x >= B)
            print("    %-7s feas %.4f  (%d/%d)  mean calls %.1f  at-cap %d/%d"
                  % (a, sum(v)/len(v), int(sum(v)), len(v), sum(cl)/max(len(cl),1), atcap, len(cl)))
        for x, y in (("random","llm"), ("fsd","llm"), ("fsd","random")):
            if x in arms and y in arms:
                u,d,p = sign_test([by[x][q] for q in common], [by[y][q] for q in common])
                print("    %-6s vs %-6s  %+.4f  disc %d-%d  p=%.4g"
                      % (y, x, sum(by[y][q] for q in common)/len(common)-sum(by[x][q] for q in common)/len(common), u, d, p))
        for a in arms:
            pk = Counter()
            for r in sub:
                if r["arm"]==a:
                    for k,v2 in (r.get("picked") or {}).items(): pk[k]+=v2
            if pk: print("    %-7s moves: %s" % (a, dict(pk.most_common())))
        # llm diagnostics
        lm = [r for r in sub if r["arm"]=="llm"]
        if lm and any(r.get("model_calls") for r in lm):
            mc = sum(r.get("model_calls",0) for r in lm); pa = sum(r.get("parsed",0) for r in lm)
            print("    llm model_calls/problem %.2f  parsed/call %.2f  turns/problem %.2f  invalid %d  repeat %d"
                  % (mc/len(lm), pa/max(mc,1), sum(r.get("turns",0) for r in lm)/len(lm),
                     sum(r.get("invalid",0) for r in lm), sum(r.get("repeat",0) for r in lm)))
        rn = [r for r in sub if r["arm"]=="random"]
        if rn:
            print("    rnd turns/problem %.2f  cands %d  invalid %d  repeat %d"
                  % (sum(r.get("turns",0) for r in rn)/len(rn),
                     sum(r.get("cands",0) for r in rn), sum(r.get("invalid",0) for r in rn),
                     sum(r.get("repeat",0) for r in rn)))
