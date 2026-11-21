"""split_states.py <states.jsonl> <samples.jsonl> <fraction> <out: states for this machine> <out: sids for the other> [k=8]
The states that still miss replies, in the sampler's order (photonics, grid2op, engibench; file order inside a
family): the last <fraction> of them go to the other machine (their sids, last state first), the rest stay here.
The states file written for this machine is the whole list without the other machine's states, so replies already
sampled for any state stay usable and nothing is sampled twice."""
import collections
import json
import sys

states_p, samples_p, frac, out_states, out_sids = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
k = int(sys.argv[6]) if len(sys.argv) > 6 else 8
FIRST = ["photonics", "grid2op", "engibench"]
have = collections.defaultdict(set)
for raw in open(samples_p, "rb"):
    try:
        o = json.loads(raw)
    except ValueError:
        continue
    if not o.get("error"):
        have[o["sid"]].add(int(o["sample"]))
lines, meta = [], []
for raw in open(states_p, "rb"):
    if raw.strip():
        o = json.loads(raw)
        lines.append(raw if raw.endswith(b"\n") else raw + b"\n")
        meta.append((o["sid"], o["variant"]))
rank = {v: i for i, v in enumerate(FIRST)}
order = sorted(range(len(meta)), key=lambda i: rank.get(meta[i][1], len(rank)))      # stable, as the sampler sorts
todo = [i for i in order if len(have[meta[i][0]]) < k]
n_other = int(round(frac * len(todo)))
other = todo[len(todo) - n_other:] if n_other else []
other_set = set(other)
with open(out_states, "wb") as f:
    for i, raw in enumerate(lines):
        if i not in other_set:
            f.write(raw)
with open(out_sids, "w", encoding="utf-8", newline="\n") as f:
    for i in reversed(other):
        f.write(meta[i][0] + "\n")
cnt = lambda idx: dict(sorted(collections.Counter(meta[i][1] for i in idx).items()))
print("states %d, with all %d replies %d, still to do %d %s" % (len(meta), k, len(meta) - len(todo), len(todo), cnt(todo)))
print("other machine: %d states %s; missing replies there %d" % (len(other), cnt(other), sum(k - len(have[meta[i][0]]) for i in other)))
keep = [i for i in todo if i not in other_set]
print("this machine: %d states to do %s; missing replies here %d; states file %d lines" % (
    len(keep), cnt(keep), sum(k - len(have[meta[i][0]]) for i in keep), len(lines) - len(other)))
