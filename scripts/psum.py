"""Summarise a planner results jsonl: paired rates and exact sign test against the FSD control."""
import sys, json
from pathlib import Path
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts")
from probe26_presentation import sign_test

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
arms = sorted({r["arm"] for r in rows})
by = {a: {r["problem_id"]: (1.0 if r["feasible"] else 0.0) for r in rows if r["arm"] == a}
      for a in arms}
pids = sorted(set.intersection(*[set(by[a]) for a in arms]))
print("%s  |  paired on %d problems\n" % (Path(sys.argv[1]).name, len(pids)))
rate = {a: sum(by[a][p] for p in pids) / max(len(pids), 1) for a in arms}
for a in arms:
    print("  %-9s %.4f" % (a, rate[a]))
if "fsd" in arms and "planner" in arms:
    u, d, p = sign_test([by["fsd"][x] for x in pids], [by["planner"][x] for x in pids])
    print("\n  planner vs FSD control: %+.4f   discordant %d-%d   exact sign p = %.3g"
          % (rate["planner"] - rate["fsd"], u, d, p))
for k in ("props", "wins", "restarts"):
    v = [r.get(k, 0) for r in rows if r["arm"] == "planner" and k in r]
    if v:
        print("  %-9s mean %.2f" % (k, sum(v) / len(v)))
