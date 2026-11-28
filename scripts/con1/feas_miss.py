"""Applied moves whose truth is infeasible but whose answer writes FEASIBLE AFTER: yes, per family: which constraints the
simulator's CHECK RESULT lists as VIOLATED after the move, and what the answer's VIOLATED AFTER line names (CPU; 11-28).
    python feas_miss.py QUERIES PREDS FAMILY [FAMILY ...]
"""
import collections
import json
import re
import sys

qs = {json.loads(l)["sid"]: json.loads(l) for l in open(sys.argv[1])}
fams = set(sys.argv[3:])
viol = collections.defaultdict(collections.Counter)
wrote = collections.defaultdict(collections.Counter)
n = collections.Counter()
for l in open(sys.argv[2]):
    r = json.loads(l)
    q = qs.get(r["sid"])
    if not q or q["variant"] not in fams:
        continue
    m = q["moves"][r["qi"]]
    t = r.get("text") or ""
    fe = re.search(r"FEASIBLE AFTER:\s*(\S+)", t)
    if m["verdict"] == "refused" or m["feasible"] or not fe or not fe.group(1).lower().startswith("y"):
        continue
    f = q["variant"]
    n[f] += 1
    vs = tuple(sorted(re.findall(r"^\s+(\S+)\s+.*VIOLATED\s*$", m.get("result") or "", re.M)))
    viol[f][vs] += 1
    va = re.search(r"VIOLATED AFTER:\s*(.*)", t)
    wrote[f][(va.group(1).strip()[:60] if va else "-")] += 1
for f in sorted(n):
    print("%s: %d infeasible-truth answers written feasible" % (f, n[f]))
    print("  violated after (simulator):", viol[f].most_common(6))
    print("  VIOLATED AFTER written:", wrote[f].most_common(4))
print("FEAS_MISS_DONE")
