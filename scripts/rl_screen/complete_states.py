"""complete_states.py <samples.jsonl> <scored.jsonl> <snapshot out> [k=8]: the replies of the states that have all k
replies in the samples file and no state record in the scored file yet, copied line for line into the snapshot.
Last line printed: `snapshot states N`."""
import collections
import json
import os
import sys

samples, scored, out = sys.argv[1], sys.argv[2], sys.argv[3]
k = int(sys.argv[4]) if len(sys.argv) > 4 else 8
done = set()
try:
    for line in open(scored, encoding="utf-8"):
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("state"):
            done.add(o["sid"])
except FileNotFoundError:
    pass
per = collections.defaultdict(dict)
n_lines = n_bad = 0
with open(samples, "rb") as f:
    for raw in f:
        n_lines += 1
        if not raw.endswith(b"\n"):
            n_bad += 1
            continue
        try:
            o = json.loads(raw)
        except ValueError:
            n_bad += 1
            continue
        if o.get("error") or o["sid"] in done:
            continue
        per[o["sid"]][int(o["sample"])] = raw
full = [sid for sid, d in per.items() if sorted(d) == list(range(k))]
with open(out + ".tmp", "wb") as fo:
    for sid in full:
        for i in range(k):
            fo.write(per[sid][i])
os.replace(out + ".tmp", out)
fam = collections.Counter(s.split("|")[0] for s in full)
print("samples lines %d (unreadable %d), states scored already %d, states with replies and not scored %d, of which complete %s" % (
    n_lines, n_bad, len(done), len(per), dict(sorted(fam.items()))))
print("snapshot states %d" % len(full))
