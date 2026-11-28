"""Arm delta's CHANGES lines (CPU; 11-28): how the signed change is written, against the truth. Per family, on true
up/down lines of truth-applied unflagged questions where the answer is applied and writes the line: the share written
as exactly zero (every digit 0: read as 'same'), the share whose sign is right, and the share of true changes below the
smallest nonzero value the line's own decimals can show (|change| < 0.5e-3 when written with 3 decimals).
    python delta_layout.py QUERIES QFLAGS PREDS
"""
import collections
import json
import re
import sys

qs = {json.loads(l)["sid"]: json.loads(l) for l in open(sys.argv[1], encoding="utf-8")}
qf = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open(sys.argv[2]) if l.strip()}
NUM = re.compile(r"^\s*[-*]?\s*([A-Za-z][A-Za-z0-9_]*)\s*[:=]?\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|same|from 0)")
c = collections.defaultdict(collections.Counter)
for l in open(sys.argv[3], encoding="utf-8"):
    r = json.loads(l)
    k = (r["sid"], r["qi"])
    q = qs.get(r["sid"])
    if q is None or k in qf:
        continue
    m = q["moves"][r["qi"]]
    t = (r.get("text") or "").replace("**", "")
    if m["verdict"] == "refused" or "CHANGES:" not in t:
        continue
    f = q["variant"]
    for ln in t.split("CHANGES:", 1)[1].splitlines():
        g = NUM.match(ln)
        if not g or g.group(1) not in q["metrics0"]:
            continue
        b, a = q["metrics0"][g.group(1)], m["metrics_after"].get(g.group(1))
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        if abs(a - b) <= max(1e-5 * abs(b), 1e-12):
            continue
        x = g.group(2)
        cc = c[f]
        cc["lines"] += 1
        if x in ("same", "from 0"):
            cc["word " + x] += 1
            continue
        v = float(x)
        dec = len(x.split(".")[1]) if "." in x else 0
        cc["written zero"] += v == 0
        cc["sign right"] += v != 0 and (v > 0) == (a > b)
        cc["true change below the written precision"] += abs(a - b) < 0.5 * 10 ** (-dec)
        cc["3 decimals"] += dec == 3
for f in sorted(c) + ["ALL"]:
    cc = sum(c.values(), collections.Counter()) if f == "ALL" else c[f]
    n = cc["lines"] or 1
    print("  %-11s true up/down lines written %5d | exactly zero %.3f | sign right %.3f | true change below written "
          "precision %.3f | 3 decimals %.3f | word same %d" % (f, cc["lines"], cc["written zero"] / n, cc["sign right"] / n,
                                                                cc["true change below the written precision"] / n,
                                                                cc["3 decimals"] / n, cc["word same"]))
print("DELTA_LAYOUT_DONE")
