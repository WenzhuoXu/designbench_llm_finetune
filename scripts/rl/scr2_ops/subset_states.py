"""subset_states.py <states.jsonl> <samples.jsonl> <family> <count> <out.jsonl> [skip=0] [k=8]: the states of the family
that still miss replies in the samples file, in file order: `count` of them after the first `skip`, as full state lines."""
import collections
import json
import sys

states_p, samples_p, family, count, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
skip = int(sys.argv[6]) if len(sys.argv) > 6 else 0
k = int(sys.argv[7]) if len(sys.argv) > 7 else 8
have = collections.defaultdict(set)
for raw in open(samples_p, "rb"):
    try:
        o = json.loads(raw)
    except ValueError:
        continue
    if not o.get("error"):
        have[o["sid"]].add(int(o["sample"]))
todo, n_fam = [], 0
for raw in open(states_p, "rb"):
    if raw.strip():
        o = json.loads(raw)
        if o["variant"] == family:
            n_fam += 1
            if len(have[o["sid"]]) < k:
                todo.append((o["sid"], raw if raw.endswith(b"\n") else raw + b"\n"))
take = todo[skip:skip + count]
with open(out, "wb") as f:
    for _, raw in take:
        f.write(raw)
print("%s: %d states, %d still miss replies; written %d (after skipping %d): first %s, last %s" % (
    family, n_fam, len(todo), len(take), skip, take[0][0] if take else None, take[-1][0] if take else None))
