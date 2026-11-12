"""Turn 0 of every problem: the 8 rollouts of a group start from the same state, so their first
replies are 8 samples of the policy at one state. How often is a single sample legal / improving,
and how often is at least one of k samples (what a simulator check of k proposals would find)?
Also: how many distinct first actions, and the episode outcome by first-move outcome.

usage: python turn0_bestofk.py <rollouts dir>
"""
import collections
import glob
import itertools
import json
import os
import sys

import numpy as np

H, BONUS = 12, 0.2
groups = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(sys.argv[1], "step_*.compact.jsonl"))):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        ep = r["episode"]
        turns = ep.get("turns", [])
        if not turns:
            continue
        t = turns[0]
        phi0, phi1 = ep.get("phi_start"), t.get("phi_after")
        legal = not t.get("fail")
        up = bool(legal and phi0 is not None and phi1 is not None and phi1 > phi0)
        ff = ep.get("first_feasible_turn")
        ok = ff is not None and ff <= H
        act = json.dumps(t.get("action"), sort_keys=True)
        groups[(os.path.basename(f), r["group"])].append((r["variant"], legal, up, ok, act, bool(t.get("feasible_after"))))


def any_of_k(flags, k):
    n = len(flags)
    if n < k:
        return float("nan")
    c = list(itertools.combinations(range(n), k))
    return float(np.mean([any(flags[i] for i in s) for s in c]))


fams = sorted({g[0][0] for g in groups.values()})
print("problem groups %d (8 first replies each)" % len(groups))
for what, col in (("first move is legal", 1), ("first move is legal and raises the potential", 2)):
    print("\n== %s" % what)
    print("%-12s %7s %8s %8s %8s %8s %10s %10s" % ("family", "groups", "1 sample", "any of 2", "any of 4", "any of 8", "none of 8", "all of 8"))
    for fam in fams + ["all"]:
        gs = [g for g in groups.values() if fam == "all" or g[0][0] == fam]
        fl = [[x[col] for x in g] for g in gs]
        print("%-12s %7d %8.3f %8.3f %8.3f %8.3f %10.3f %10.3f" % (
            fam, len(gs), np.mean([np.mean(f) for f in fl]), np.nanmean([any_of_k(f, 2) for f in fl]),
            np.nanmean([any_of_k(f, 4) for f in fl]), np.nanmean([any(f) for f in fl]),
            np.mean([not any(f) for f in fl]), np.mean([all(f) for f in fl])))
print("\n== distinct first actions among the 8, and episode success by first-move outcome")
print("%-12s %16s %22s %22s %22s" % ("family", "distinct actions", "success | illegal first", "success | legal, no gain", "success | legal, gain"))
for fam in fams + ["all"]:
    gs = [g for g in groups.values() if fam == "all" or g[0][0] == fam]
    rows = [x for g in gs for x in g]
    d = np.mean([len({x[4] for x in g}) for g in gs])

    def rate(sel):
        s = [x[3] for x in rows if sel(x)]
        return "%.3f (n %d)" % (np.mean(s), len(s)) if s else "-"
    print("%-12s %16.2f %22s %22s %22s" % (fam, d, rate(lambda x: not x[1]), rate(lambda x: x[1] and not x[2]), rate(lambda x: x[2])))
# within a group: does the first-move outcome separate the rollouts that succeed? (group-centred)
num = den = 0.0
for g in groups.values():
    a = np.array([float(x[2]) for x in g])
    y = np.array([float(x[3]) for x in g])
    if a.std() > 0 and y.std() > 0:
        num += ((a - a.mean()) * (y - y.mean())).sum()
        den += ((a - a.mean()) ** 2).sum()
print("\nwithin a problem: success rate difference between a first move that gains and one that does not: %.3f" % (num / den if den else float("nan")))
