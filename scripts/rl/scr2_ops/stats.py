"""stats.py <samples.jsonl> [skip lines]: per family, the replies' prompt and completion tokens and cues."""
import collections
import json
import sys

skip = int(sys.argv[2]) if len(sys.argv) > 2 else 0
by = collections.defaultdict(list)
keys = None
for i, line in enumerate(open(sys.argv[1], encoding="utf-8")):
    if i < skip:
        continue
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if r.get("error"):
        continue
    keys = keys or sorted(r)
    c = r.get("cues")
    by[str(r["sid"]).split("|")[0]].append((r.get("prompt_tokens") or 0, r.get("tokens") or 0, len(c) if isinstance(c, list) else (c or 0)))
print("keys", keys)
for fam, v in sorted(by.items()):
    n = len(v)
    p = sorted(x[0] for x in v)
    t = sorted(x[1] for x in v)
    print("%-10s replies %6d  prompt tokens mean %7.0f median %6d p90 %6d  completion tokens mean %6.0f median %5d p90 %5d  cues mean %.2f" % (
        fam, n, sum(p) / n, p[n // 2], p[int(n * 0.9)], sum(t) / n, t[n // 2], t[int(n * 0.9)], sum(x[2] for x in v) / n))
