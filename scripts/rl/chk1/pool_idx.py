"""pool_idx.py: which lines of a pool (by index) have a finished episode in the given files, as compact ranges.
Run on each machine with the same pool file (pool_fwd.jsonl); union the outputs to see what neither machine has done.

usage: python pool_idx.py POOL EPISODES... [--rev]   (--rev: POOL is pool_rev; print pool_fwd indices, 1319 - i)
"""
import json
import sys

args = [a for a in sys.argv[1:] if a != "--rev"]
rev = "--rev" in sys.argv[1:]
pool, eps = args[0], args[1:]
keys = []
for l in open(pool, encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        keys.append((r["variant"], r["spec"]["problem_id"]))
n = len(keys)
pos = {k: (n - 1 - i if rev else i) for i, k in enumerate(keys)}
done, bad = set(), 0
for p in eps:
    for l in open(p, encoding="utf-8"):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except ValueError:
            bad += 1
            continue
        ep = r.get("episode") or r
        k = (ep.get("variant"), ep.get("problem_id"))
        if k in pos:
            done.add(pos[k])
rng, s = [], None
for i in sorted(done) + [None]:
    if s is not None and (i is None or i != e + 1):
        rng.append("%d-%d" % (s, e) if e > s else "%d" % s)
        s = None
    if i is not None:
        if s is None:
            s = i
        e = i
print("pool %d, done %d (unreadable %d): %s" % (n, len(done), bad, ",".join(rng)))
