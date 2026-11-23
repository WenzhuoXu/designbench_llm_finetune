"""split_g2e.py <states.jsonl> <samples.jsonl> <k> <n_first> <share2 out>: the states of the g2e job still missing
replies, in the job's own order (engibench, then grid2op, file order inside a family). The first n_first stay with the
job (it is restarted with SAMPLE_LIMIT=n_first); the rest are written to <share2 out> for a second job.
Prints the counts per family of both shares."""
import collections
import json
import sys

states_p, samples_p, k, n_first, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
first = ["engibench", "grid2op"]
rank = {v: i for i, v in enumerate(first)}
lines = [l for l in open(states_p) if l.strip()]
states = [(json.loads(l), l) for l in lines]
states.sort(key=lambda t: rank.get(t[0]["variant"], len(rank)))
have = collections.defaultdict(set)
for raw in open(samples_p, "rb"):
    if not raw.endswith(b"\n"):
        continue
    try:
        o = json.loads(raw)
    except ValueError:
        continue
    if o.get("error"):
        continue
    have[o["sid"]].add(int(o["sample"]))
todo = [(s, l) for s, l in states if len(have[s["sid"]]) < k]
mine, other = todo[:n_first], todo[n_first:]
with open(out, "w") as fo:
    for s, l in other:
        fo.write(l if l.endswith("\n") else l + "\n")
fam = lambda xs: dict(sorted(collections.Counter(s["variant"] for s, _ in xs).items()))
print("states %d, complete %d, to do %d (partial %d)" % (len(states), len(states) - len(todo), len(todo),
                                                     sum(1 for s, _ in todo if have[s["sid"]])))
print("share 1 (this job, SAMPLE_LIMIT=%d): %s" % (len(mine), fam(mine)))
print("share 2 (%s): %d %s" % (out, len(other), fam(other)))
