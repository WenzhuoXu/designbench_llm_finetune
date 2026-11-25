"""Rewrite an append-written gzip JSONL whose last member may be cut (a killed writer): keep every complete line, and
keep only the .done keys whose records survived. usage: python clean_gz.py IN.jsonl.gz OUT.jsonl.gz"""
import gzip
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
keys, n = set(), 0
with gzip.open(dst, "wt", encoding="utf-8") as fo:
    try:
        for line in gzip.open(src, "rt", encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                break
            fo.write(line if line.endswith("\n") else line + "\n")
            keys.add(r.get("key") or r.get("sid"))
            n += 1
    except (EOFError, OSError):
        pass
done = []
try:
    done = [l.strip() for l in open(src + ".done") if l.strip()]
except FileNotFoundError:
    pass
kept = [k for k in done if k in keys]
open(dst + ".done", "w").write("".join(k + "\n" for k in kept))
print("clean_gz: %d lines, %d keys, .done %d of %d kept" % (n, len(keys), len(kept), len(done)))
