"""Per-question agreement of the two readers (facets.pred_facets via dir_split.read, indep_score.read_answer) on one
answer file, and what the answers write on FEASIBLE AFTER per family against the truth (CPU; 11-28).
    python reader_agree.py QUERIES PREDS FORM
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, "/storage/wxu/con1/full_eval")
import dir_split as DS  # noqa: E402

qs = {json.loads(l)["sid"]: json.loads(l) for l in open(sys.argv[1])}
recs = DS.load(sys.argv[2])
fmt = sys.argv[3]
c = collections.Counter()
fe = collections.defaultdict(collections.Counter)
ex = []
for (sid, qi), r in sorted(recs.items()):
    q = qs[sid]
    a = DS.read(r.get("text"), fmt, q, "facets")
    b = DS.read(r.get("text"), fmt, q, "indep")
    c["n"] += 1
    c["verdict same"] += a[0] == b[0]
    c["feasible same"] += a[1] == b[1]
    c["dirs same"] += a[2] == b[2]
    c["all same"] += a == b
    if a != b and len(ex) < 5:
        ex.append((sid, qi, a, b))
    m = q["moves"][qi]
    t = "refused" if m["verdict"] == "refused" else ("feasible" if m["feasible"] else "infeasible")
    w = re.search(r"FEASIBLE AFTER:\s*(\S+)", r.get("text") or "")
    fe[q["variant"]]["truth %s / wrote %s" % (t, (w.group(1).lower() if w else "-"))] += 1
print("reader agreement per question:", dict(c))
for e in ex:
    print("  differ:", e)
for f in sorted(fe):
    print(" ", f, dict(sorted(fe[f].items())))
print("READER_AGREE_DONE")
