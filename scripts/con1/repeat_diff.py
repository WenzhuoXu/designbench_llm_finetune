"""Run-to-run change between two runs of the same arm on the same questions (CPU; 11-28): per family and in all,
share of questions whose answer text differs, whose think text differs, whose think_end differs; per facet (facets
reader, dir_split.read): verdict differs, FEASIBLE AFTER differs (both applied), direction differs on a named line
(both applied, both write it), and the count of questions right in one run and wrong in the other, per facet.
    python repeat_diff.py QUERIES A.jsonl B.jsonl FORM [SUBSET.tsv]
"""
import collections
import json
import sys

sys.path.insert(0, "/storage/wxu/con1/full_eval")
import dir_split as DS  # noqa: E402

qs = {json.loads(l)["sid"]: json.loads(l) for l in open(sys.argv[1])}
A, B, fmt = DS.load(sys.argv[2]), DS.load(sys.argv[3]), sys.argv[4]
sub = None
if len(sys.argv) > 5:
    sub = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open(sys.argv[5]) if l.strip()}
keys = sorted(k for k in A if k in B and k[0] in qs and (sub is None or k in sub))
c = collections.defaultdict(collections.Counter)
for k in keys:
    q = qs[k[0]]
    m = q["moves"][k[1]]
    a, b = A[k], B[k]
    va, fa, da = DS.read(a.get("text"), fmt, q, "facets")
    vb, fb, db = DS.read(b.get("text"), fmt, q, "facets")
    tv = "refused" if m["verdict"] == "refused" else "applied"
    for s in (q["variant"], "all"):
        x = c[s]
        x["n"] += 1
        x["text differs"] += a.get("text") != b.get("text")
        x["think differs"] += a.get("think") != b.get("think")
        x["think_end differs"] += a.get("think_end") != b.get("think_end")
        x["both closed"] += a.get("think_end") == "closed" == b.get("think_end")
        x["verdict differs"] += va != vb
        x["verdict right in one only"] += (va == tv) != (vb == tv)
        if va == vb == "applied" and tv == "applied":
            x["feas both"] += 1
            x["feas differs"] += fa != fb
            x["feas right in one only"] += (fa == bool(m["feasible"])) != (fb == bool(m["feasible"]))
            for met in set(da) & set(db):
                if met in DS.F.DIRECT.get(q["variant"], set()) or met not in m["metrics_after"]:
                    continue
                td = DS.tdir(float(q["metrics0"][met]), float(m["metrics_after"][met]))
                x["lines both"] += 1
                x["dir differs"] += da[met] != db[met]
                x["dir right in one only"] += (da[met] == td) != (db[met] == td)
print("# %s vs %s on %d questions (form %s)" % (sys.argv[2], sys.argv[3], len(keys), fmt))
for s in sorted(c, key=lambda s: (s == "all", s)):
    x = c[s]
    n = x["n"]
    print("  %-11s n %3d | text differs %.2f  think differs %.2f  think_end differs %.2f (both closed %d) | verdict "
          "differs %d (right in one only %d) | feasible differs %d of %d (right in one only %d) | direction differs %d of "
          "%d lines (right in one only %d)" % (
              s, n, x["text differs"] / n, x["think differs"] / n, x["think_end differs"] / n, x["both closed"],
              x["verdict differs"], x["verdict right in one only"], x["feas differs"], x["feas both"],
              x["feas right in one only"], x["dir differs"], x["lines both"], x["dir right in one only"]))
print("REPEAT_DIFF_DONE")
