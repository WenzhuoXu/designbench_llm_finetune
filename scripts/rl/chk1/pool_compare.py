"""pool_compare.py: dx38 with every new move checked (chk1 collection, v3rc --checks 4 --check-actions) against dx38 greedy
(one call per turn, the open-rules collection) on the same training-pool problems, from compact episode lines
(compact_eps.py). Per family: problems in both, feasible@12 / @20 per-problem rates (runs pooled), simulator calls per
turn in the check arm, calls up to the first feasible design, and problems higher in each arm.

usage: python pool_compare.py --greedy FILE... --checks FILE...
"""
import argparse
import collections
import json
import statistics


def load(paths):
    per = collections.defaultdict(list)
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("crashed") or e.get("feasible_start") or e.get("rl_error") not in (None, "calls_turns_mismatch"):
                continue
            per[(e["variant"], e["problem_id"])].append(e)
    return per


ap = argparse.ArgumentParser()
ap.add_argument("--greedy", nargs="+", required=True)
ap.add_argument("--checks", nargs="+", required=True)
a = ap.parse_args()
G, K = load(a.greedy), load(a.checks)
keys = sorted(set(G) & set(K))
print("problems: greedy %d, checks %d, both %d" % (len(G), len(K), len(keys)))
print("%-10s %5s | %7s %7s | %7s %7s  %7s | %6s %6s | %s" % ("family", "prob", "g@12", "k@12", "g@20", "k@20", "k-g@20",
                                                          "calls/", "calls", "higher g/k"))
print("%-10s %5s | %7s %7s | %7s %7s  %7s | %6s %6s |" % ("", "", "", "", "", "", "", "turn", "to f1"))
for fam in sorted({v for v, _ in keys}) + ["all"]:
    ks = [k for k in keys if fam == "all" or k[0] == fam]
    rate = lambda D, k, f: sum(e[f] for e in D[k]) / len(D[k])          # noqa: E731
    g12 = statistics.mean(rate(G, k, "feasible@12") for k in ks)
    k12 = statistics.mean(rate(K, k, "feasible@12") for k in ks)
    g20 = statistics.mean(rate(G, k, "feasible@20") for k in ks)
    k20 = statistics.mean(rate(K, k, "feasible@20") for k in ks)
    calls = [c for k in ks for e in K[k] for c in e["calls"]]
    to_f = [sum(e["calls"][:e["first_feasible_turn"] + 1]) for k in ks for e in K[k]
            if e.get("first_feasible_turn") is not None]
    hg = sum(1 for k in ks if rate(G, k, "feasible@20") > rate(K, k, "feasible@20"))
    hk = sum(1 for k in ks if rate(K, k, "feasible@20") > rate(G, k, "feasible@20"))
    print("%-10s %5d | %7.3f %7.3f | %7.3f %7.3f  %+7.3f | %6.2f %6.1f | %d / %d" % (
        fam, len(ks), g12, k12, g20, k20, k20 - g20, statistics.mean(calls) if calls else float("nan"),
        statistics.median(to_f) if to_f else float("nan"), hg, hk))
