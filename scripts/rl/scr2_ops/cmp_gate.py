"""cmp_gate.py <gate_scored.jsonl> <scored.jsonl>: the records the scoring loop wrote for the gate states against the
records the launcher's gate wrote for the same replies (timings left out). Numbers are compared with a tolerance;
anything else must be equal."""
import collections
import json
import sys

SKIP = {"t", "seconds", "elapsed"}
TOL = 1e-9


def load(path, only=None):
    recs = {}
    for line in open(path, encoding="utf-8"):
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if only is not None and o["sid"] not in only:
            continue
        key = (o["sid"], "state" if o.get("state") else int(o["sample"]))
        recs[key] = {k: v for k, v in o.items() if k not in SKIP}
    return recs


worst = [0.0]


def same(x, y, path, out):
    if isinstance(x, bool) or isinstance(y, bool) or x is None or y is None or isinstance(x, str) or isinstance(y, str):
        if x != y:
            out.append((path, x, y))
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
        d = abs(x - y) / max(1.0, abs(x), abs(y))
        worst[0] = max(worst[0], d)
        if d > TOL:
            out.append((path, x, y))
    elif isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                out.append((path + "." + k, x.get(k), y.get(k)))
            else:
                same(x[k], y[k], path + "." + k, out)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            out.append((path + ".len", len(x), len(y)))
        else:
            for i, (p, q) in enumerate(zip(x, y)):
                same(p, q, "%s[%d]" % (path, i), out)
    elif x != y:
        out.append((path, x, y))


a = load(sys.argv[1])
b = load(sys.argv[2], {k[0] for k in a})
both = sorted(set(a) & set(b), key=str)
exact = sum(1 for k in both if a[k] == b[k])
real = []
fam = collections.Counter()
for k in both:
    if a[k] != b[k]:
        fam[k[0].split("|")[0]] += 1
        out = []
        same(a[k], b[k], "", out)
        if out:
            real.append((k, out))
print("gate records %d, found in scored.jsonl %d (missing %d); identical %d; differing only in the last digits of numbers %d %s; "
      "differing beyond %.0e or in a label %d; largest relative difference of a number %.2e" % (
          len(a), len(both), len(set(a) - set(b)), exact, len(both) - exact - len(real), dict(fam), TOL, len(real), worst[0]))
for k, out in real[:8]:
    print("  ", k, out[:3])
