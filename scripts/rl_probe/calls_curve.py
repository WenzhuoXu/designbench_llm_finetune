"""Feasible rate against simulator calls: greedy runs (1 call per turn) vs best-of-K runs (K calls per turn).

An episode counts as feasible within N calls when its design is feasible after a turn t (0-based) with
(t + 1) * cost <= N, where cost = 1 for a greedy file and K (from the turn's best_of record) for a best-of file.
Per variant, per-problem rates averaged over problems (each file's runs pooled within its arm).

usage: python calls_curve.py --a NAME FILE... --b NAME FILE... [--variants v1,v2] [--calls 1,2,4,8,...]
"""
import argparse
import collections
import json


def load(paths, variants):
    per = collections.defaultdict(list)          # (variant, problem) -> [(cost per turn, first feasible turn or None)]
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            ep = r.get("episode") or r
            if r.get("rl_error") or ep.get("crashed") or ep.get("feasible_start"):
                continue
            if variants and ep["variant"] not in variants:
                continue
            ff, cost = None, 1
            for i, t in enumerate(ep.get("turns") or []):
                if t.get("best_of"):
                    cost = int(t["best_of"]["k"])
                if ff is None and t.get("feasible_after"):
                    ff = i
            per[(ep["variant"], ep["problem_id"])].append((cost, ff))
    return per


ap = argparse.ArgumentParser()
ap.add_argument("--a", nargs="+", required=True)
ap.add_argument("--b", nargs="+", required=True)
ap.add_argument("--variants", default="")
ap.add_argument("--calls", default="1,2,4,8,12,16,20,24,40,80,160")
a = ap.parse_args()
vs = set(a.variants.split(",")) if a.variants else None
A, B = load(a.a[1:], vs), load(a.b[1:], vs)
calls = [int(x) for x in a.calls.split(",")]
keys = sorted(set(A) & set(B))
fams = sorted({v for v, _ in keys})


def rate(eps, n):
    return sum(1 for c, ff in eps if ff is not None and (ff + 1) * c <= n) / len(eps)


print("feasible within N simulator calls (per-problem rate, mean over problems); A=%s B=%s" % (a.a[0], a.b[0]))
print("%-10s %4s " % ("variant", "prob") + " ".join("%11s" % ("N=%d" % n) for n in calls))
for f in fams + ["all"]:
    ks = [k for k in keys if f == "all" or k[0] == f]
    ra = ["%.3f" % (sum(rate(A[k], n) for k in ks) / len(ks)) for n in calls]
    rb = ["%.3f" % (sum(rate(B[k], n) for k in ks) / len(ks)) for n in calls]
    print("%-10s %4d " % (f, len(ks)) + " ".join("%5s/%5s" % (x, y) for x, y in zip(ra, rb)))
