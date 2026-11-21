"""clean_samples.py <samples.jsonl> <scored.jsonl> <out samples> <out scored>: a fresh copy of the samples file with one
reply per (sid, sample): where a reply was recorded twice, the later one is kept (the one the compute node could
read). The scored file is copied without the records of the states involved, so the last scoring pass scores those
states again from the clean file. Lines keep their order."""
import collections
import json
import sys

src, scored, out, out_scored = sys.argv[1:5]
last = {}
n = bad = 0
with open(src, "rb") as f:
    for i, raw in enumerate(f):
        n += 1
        try:
            o = json.loads(raw)
        except ValueError:
            bad += 1
            continue
        if o.get("error") or not raw.endswith(b"\n"):
            bad += 1
            continue
        last[(o["sid"], int(o["sample"]))] = i
keep = set(last.values())
dropped_sids = set()
kept = 0
with open(src, "rb") as f, open(out, "wb") as fo:
    for i, raw in enumerate(f):
        if i in keep:
            fo.write(raw)
            kept += 1
        else:
            try:
                dropped_sids.add(json.loads(raw)["sid"])
            except ValueError:
                pass
per = collections.Counter(k[0] for k in last)
print("samples: %d lines, %d not kept as they are (unreadable, error or cut), %d distinct replies written, states %d, replies per state %s" % (
    n, bad, kept, len(per), dict(collections.Counter(per.values()))))
print("states with a reply recorded twice:", sorted(dropped_sids))
n_s = n_drop = 0
with open(scored, "rb") as f, open(out_scored, "wb") as fo:
    for raw in f:
        n_s += 1
        try:
            o = json.loads(raw)
        except ValueError:
            n_drop += 1
            continue
        if o["sid"] in dropped_sids or not raw.endswith(b"\n"):
            n_drop += 1
            continue
        fo.write(raw)
print("scored: %d lines, %d left out (the states above, or unreadable)" % (n_s, n_drop))
