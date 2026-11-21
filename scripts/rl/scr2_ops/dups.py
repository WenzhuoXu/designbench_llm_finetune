"""dups.py <samples.jsonl>: replies recorded more than once (same sid + sample): how many, whether their text differs,
and where in the file they are."""
import collections
import json
import sys

seen = {}
dup = []
n = bad = err = 0
for i, raw in enumerate(open(sys.argv[1], "rb"), 1):
    n += 1
    try:
        o = json.loads(raw)
    except ValueError:
        bad += 1
        continue
    if o.get("error"):
        err += 1
        continue
    key = (o["sid"], int(o["sample"]))
    if key in seen:
        dup.append((key, seen[key][0], i, seen[key][1] == o.get("sha1")))
    else:
        seen[key] = (i, o.get("sha1"))
print("lines %d, unreadable %d, error records %d, distinct replies %d, recorded again %d (same text %d, other text %d)" % (
    n, bad, err, len(seen), len(dup), sum(1 for d in dup if d[3]), sum(1 for d in dup if not d[3])))
fam = collections.Counter(d[0][0].split("|")[0] for d in dup)
print("by family", dict(fam), "states involved", len({d[0][0] for d in dup}))
for d in dup[:12]:
    print("  %s sample %d: lines %d and %d, same text %s" % (d[0][0], d[0][1], d[1], d[2], d[3]))
