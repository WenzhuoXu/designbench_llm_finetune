"""why_missed.py <samples.jsonl> <line> [<line> ...]: how the sampler's own reader (text mode, utf-8) sees the given
lines of the file compared with a byte-wise read."""
import json
import sys

path = sys.argv[1]
want = {int(x) for x in sys.argv[2:]}
raws = {}
with open(path, "rb") as f:
    for i, raw in enumerate(f, 1):
        if i in want or i + 1 in want:
            raws[i] = raw
for i in sorted(raws):
    raw = raws[i]
    print("line %d: %d bytes, starts %r, ends %r, carriage returns %d, U+2028/2029 %d, U+0085 %d, form feeds %d" % (
        i, len(raw), raw[:60], raw[-30:], raw.count(b"\r"), raw.count(b"\xe2\x80\xa8") + raw.count(b"\xe2\x80\xa9"),
        raw.count(b"\xc2\x85"), raw.count(b"\x0c") + raw.count(b"\x0b") + raw.count(b"\x1c") + raw.count(b"\x1d") + raw.count(b"\x1e")))
    try:
        o = json.loads(raw)
        print("   bytes -> json ok: sid %s sample %s" % (o["sid"], o["sample"]))
    except ValueError as e:
        print("   bytes -> json fails:", str(e)[:100])
# the sampler's reader
keys_text, bad_text = set(), 0
n_text = 0
for line in open(path, encoding="utf-8"):
    n_text += 1
    try:
        o = json.loads(line)
    except ValueError:
        bad_text += 1
        continue
    if not o.get("error"):
        keys_text.add((o["sid"], int(o["sample"])))
keys_bin = set()
n_bin = 0
for raw in open(path, "rb"):
    n_bin += 1
    try:
        o = json.loads(raw)
    except ValueError:
        continue
    if not o.get("error"):
        keys_bin.add((o["sid"], int(o["sample"])))
print("text-mode read: %d lines, %d unreadable, %d distinct replies; byte read: %d lines, %d distinct replies; missing in text mode: %s" % (
    n_text, bad_text, len(keys_text), n_bin, len(keys_bin), sorted(keys_bin - keys_text)[:5]))
