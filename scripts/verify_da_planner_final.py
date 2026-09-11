import json, sys, zlib
from math import comb
from pathlib import Path
from collections import defaultdict

P = Path("/ocean/projects/mch250030p/wxu7/llm_finetune/results/api_guidance/da_planner_final_smoke.jsonl")
DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench/data/problems_hard")

rows = [json.loads(l) for l in open(P) if l.strip()]
print("episodes on disk:", len(rows))
print("keys sample:", sorted(rows[0].keys()))

by = defaultdict(dict); calls = defaultdict(dict); secs = defaultdict(list)
dupes = defaultdict(int)
for r in rows:
    a, p = r.get("arm"), r.get("problem_id")
    if (a, p) in dupes: pass
    dupes[(a, p)] += 1
    by[a][p] = 1.0 if r.get("feasible") else 0.0
    calls[a][p] = int(r.get("calls", 0) or 0)
    secs[a].append(r.get("secs", 0))

print("duplicate (arm,pid) rows:", sum(1 for k, v in dupes.items() if v > 1))
print("rows with err:", sum(1 for r in rows if r.get("err")))
for r in rows:
    if r.get("err"): print("   ERR:", r.get("arm"), r.get("problem_id"), r.get("err")[:120])

arms = ["planner", "fsd", "plain_llm"]
for a in arms:
    print("arm %-10s episodes %d  distinct pids %d" % (a, sum(1 for r in rows if r.get("arm") == a), len(by[a])))

pids = sorted(set.intersection(*[set(by[a]) for a in arms]))
n = len(pids)
print("\npaired on", n, "problems")

# is the problem set exactly the sorted-glob slice 150:210?
files = sorted(DB.glob("*.json"))
sl = [json.load(open(f)).get("problem_id") for f in files[150:210]]
print("slice[150:210] matches paired pids exactly:", sorted(sl) == pids)
print("first/last of slice:", sl[0], sl[-1])
print("total problem files:", len(files))

def sign_test(a, b):
    up = sum(1 for i in range(len(a)) if b[i] > a[i])
    dn = sum(1 for i in range(len(a)) if b[i] < a[i])
    m = up + dn
    if m == 0: return up, dn, 1.0
    kk = min(up, dn)
    return up, dn, min(1.0, sum(comb(m, i) for i in range(kk + 1)) / (2.0 ** m) * 2)

rate = {a: sum(by[a][p] for p in pids) / n for a in arms}
for a in arms:
    print("  %-10s %.4f  (%d/%d)" % (a, rate[a], int(round(rate[a] * n)), n))

for lo, hi in [("fsd", "planner"), ("plain_llm", "planner"), ("plain_llm", "fsd")]:
    u, d, p = sign_test([by[lo][x] for x in pids], [by[hi][x] for x in pids])
    print("  %s vs %s: %+.4f  discordant %d-%d  p=%.4g" % (hi, lo, rate[hi] - rate[lo], u, d, p))

print("\nmodel calls:")
for a in arms:
    v = [calls[a][p] for p in pids]
    print("  %-10s mean %.2f min %d max %d total %d" % (a, sum(v) / len(v), min(v), max(v), sum(v)))

# how many planner wins came with ZERO model calls (i.e. solved by the opening FSD pass)?
z = [p for p in pids if calls["planner"][p] == 0]
print("\nplanner episodes with 0 model calls:", len(z), " of which feasible:", sum(by["planner"][p] for p in z))
print("  fsd feasible on those same problems:", sum(by["fsd"][p] for p in z))
# planner-wins-over-fsd breakdown by model calls
w = [p for p in pids if by["planner"][p] > by["fsd"][p]]
print("\nplanner-beats-fsd problems:", len(w))
print("  their planner call counts:", sorted(calls["planner"][p] for p in w))
w2 = [p for p in pids if by["planner"][p] > by["plain_llm"][p]]
print("planner-beats-plain_llm problems:", len(w2))
print("  their planner call counts:", sorted(calls["planner"][p] for p in w2))
print("\nwall by arm (sum secs): ", {a: round(sum(secs[a])) for a in arms})
