"""exclude_states.py <states.jsonl> <out.jsonl> <other states file or sid list> [...]: the states list without the
states given to other machines (files of state lines, or text files with one sid per line)."""
import json
import sys

src, out, others = sys.argv[1], sys.argv[2], sys.argv[3:]
drop = set()
for p in others:
    n0 = len(drop)
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            drop.add(json.loads(line)["sid"] if line.startswith("{") else line)
    print("%s: %d states" % (p, len(drop) - n0))
kept = dropped = 0
with open(src, "rb") as f, open(out + ".tmp", "wb") as fo:
    for raw in f:
        if not raw.strip():
            continue
        if json.loads(raw)["sid"] in drop:
            dropped += 1
        else:
            fo.write(raw if raw.endswith(b"\n") else raw + b"\n")
            kept += 1
import os
os.replace(out + ".tmp", out)
assert dropped == len(drop), "%d of the %d states to leave out are not in %s" % (len(drop) - dropped, len(drop), src)
print("kept %d states, left out %d -> %s" % (kept, dropped, out))
