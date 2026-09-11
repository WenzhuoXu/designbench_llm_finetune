"""All pairwise paired sign tests between arms in a results jsonl."""
import sys, json, itertools
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts")
from probe26_presentation import sign_test

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
arms = sorted({r["arm"] for r in rows})
by = {a: {r["problem_id"]: (1.0 if r["feasible"] else 0.0) for r in rows if r["arm"] == a}
      for a in arms}
pids = sorted(set.intersection(*[set(by[a]) for a in arms]))
rate = {a: sum(by[a][p] for p in pids) / len(pids) for a in arms}
print("paired on %d problems\n" % len(pids))
for a in sorted(arms, key=lambda x: -rate[x]):
    print("  %-9s %.4f" % (a, rate[a]))
print()
for a, b in itertools.combinations(sorted(arms, key=lambda x: -rate[x]), 2):
    u, d, p = sign_test([by[b][x] for x in pids], [by[a][x] for x in pids])
    print("  %-8s vs %-8s: %+.4f   discordant %d-%d   exact sign p = %.3g"
          % (a, b, rate[a] - rate[b], u, d, p))
