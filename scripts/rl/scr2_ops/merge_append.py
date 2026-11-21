"""merge_append.py <samples.jsonl> <incoming.jsonl[.gz]> [--dry]: append to the samples file the replies of the incoming
file that it does not hold yet (by sid + sample). Only complete, error-free replies sampled under the same prompt tag
and instruction as the samples file are taken. Run it only while no sampler writes to the samples file."""
import collections
import gzip
import json
import os
import sys

samples, incoming = sys.argv[1], sys.argv[2]
dry = "--dry" in sys.argv[3:]
have, tags = set(), collections.Counter()
with open(samples, "rb") as f:
    for raw in f:
        try:
            o = json.loads(raw)
        except ValueError:
            continue
        if not o.get("error"):
            have.add((o["sid"], int(o["sample"])))
            tags[(o.get("prompt"), o.get("prompt_sha1"), o.get("mode"))] += 1
assert len(tags) == 1, "the samples file holds replies under several prompts: %s" % dict(tags)
tag = next(iter(tags))
opener = gzip.open if incoming.endswith(".gz") else open
add, n_in = [], 0
skip = collections.Counter()
with opener(incoming, "rb") as f:
    for raw in f:
        n_in += 1
        if not raw.endswith(b"\n"):
            skip["cut line"] += 1
            continue
        try:
            o = json.loads(raw)
        except ValueError:
            skip["unreadable"] += 1
            continue
        if o.get("error") or "answer" not in o:
            skip["error record"] += 1
            continue
        if (o.get("prompt"), o.get("prompt_sha1"), o.get("mode")) != tag:
            skip["another prompt"] += 1
            continue
        key = (o["sid"], int(o["sample"]))
        if key in have:
            skip["already there"] += 1
            continue
        have.add(key)
        add.append(raw)
fam = collections.Counter(json.loads(r)["sid"].split("|")[0] for r in add)
print("samples file: %d replies under %s" % (sum(tags.values()), tag))
print("incoming: %d lines; skipped %s; to append %d %s" % (n_in, dict(skip), len(add), dict(fam)))
if dry or not add:
    print("nothing written")
    sys.exit(0)
size = os.path.getsize(samples)
with open(samples, "rb") as f:
    f.seek(-1, os.SEEK_END)
    cut = f.read(1) != b"\n"
with open(samples, "ab") as f:
    if cut:
        f.write(b"\n")
    f.write(b"".join(add))
print("MERGE_DONE appended %d replies (%d bytes -> %d bytes)" % (len(add), size, os.path.getsize(samples)))
