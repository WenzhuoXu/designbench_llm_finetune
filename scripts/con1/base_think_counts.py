"""Counts of a thinking run (CPU; con1-2m-rows, 11-27), per group and family: answers, errors, how the think block
ended (closed by the model / cut at the budget / end of turn), think tokens (median), answers cut at the answer cap,
unreadable answers (facets.pred_facets), answers the layout-only normalisation changed (raw vs think_norm.py file),
applied answers that leave out metric lines (metrics of the state's metrics0 not read from CHANGES), and the omitted
lines. Counts only; the think text is not read.

    python base_think_counts.py RAW.jsonl NORM.jsonl [QUERIES]
"""
import collections
import json
import statistics
import sys

import os  # noqa: E402
sys.path.insert(0, os.environ.get("CON1_LIB", "/storage/wxu/con1/repo/scripts/con1"))   # CON1_LIB: another facets.py
import facets as F  # noqa: E402
FMT = os.environ.get("FMT", "facetm")                     # the answer form the file was asked in

raw_p, norm_p = sys.argv[1:3]
QF = sys.argv[3] if len(sys.argv) > 3 else "/storage/wxu/con1/arm/main/queries_v3.jsonl"
qs = {json.loads(l)["sid"]: json.loads(l) for l in open(QF)}


def load(p):
    out = {}
    for l in open(p):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        out.setdefault((r["sid"], r["qi"]), r)
    return out


raw, norm = load(raw_p), load(norm_p)
C = collections.defaultdict(collections.Counter)
T = collections.defaultdict(list)
for key, r in norm.items():
    q = qs[r["sid"]]
    for g in ((q["group"], q["variant"]), (q["group"], "all")):
        c = C[g]
        c["answers"] += 1
        if str(r.get("finish", "")).startswith("error"):
            c["errors"] += 1
            continue
        c["think " + str(r.get("think_end"))] += 1
        T[g].append(r.get("think_tokens") or 0)
        c["cut at the answer cap"] += r.get("finish") == "length"
        pf = F.pred_facets(r.get("text"), FMT, q)
        c["unreadable"] += pf["regime"] == "unreadable"
        c["normalised"] += (raw.get(key) or {}).get("text") != r.get("text")
        if F.is_applied(pf["regime"]):
            miss = [k for k in q["metrics0"] if k not in pf.get("dir", {})]
            c["applied answers"] += 1
            c["applied answers leaving out lines"] += bool(miss)
            c["lines left out"] += len(miss)
            c["lines expected"] += len(q["metrics0"])
print("%s (normalised: %s)" % (raw_p, norm_p))
cols = ["answers", "errors", "think closed", "think budget", "think im_end", "unreadable", "normalised",
        "cut at the answer cap", "applied answers", "applied answers leaving out lines", "lines left out", "lines expected"]
print("  %-28s %s %s" % ("group / family", " ".join("%9s" % x.replace("think ", "th_")[:9] for x in cols), "think med"))
for g in sorted(C):
    print("  %-28s %s %9d" % ("/".join(g), " ".join("%9d" % C[g][x] for x in cols), statistics.median(T[g]) if T[g] else -1))
print("BASE_THINK_COUNTS_DONE")
