"""Does an answer's FEASIBLE AFTER agree with its own VIOLATED AFTER line? Per family, applied truth and applied answer
(VERDICT: applied): counts of (FEASIBLE yes|no) x (VIOLATED AFTER none | names constraints | missing), and feasibility
accuracy under two readings: the FEASIBLE AFTER line (the scored reading) and VIOLATED AFTER (none = feasible), on the
questions where both lines are present. Flagged questions (no-op, off-range) out. (CPU; 11-28)
    python feas_consist.py QUERIES FLAGS PREDS [PREDS ...]
"""
import collections
import json
import re
import sys

qs = {json.loads(l)["sid"]: json.loads(l) for l in open(sys.argv[1])}
fl = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open(sys.argv[2]) if l.strip()}
for path in sys.argv[3:]:
    c = collections.defaultdict(collections.Counter)
    seen = set()
    for l in open(path):
        r = json.loads(l)
        k = (r["sid"], r["qi"])
        if k in seen or k in fl or r.get("pass", "zero") != "zero" or r["sid"] not in qs:
            continue
        seen.add(k)
        q = qs[r["sid"]]
        m = q["moves"][r["qi"]]
        t = (r.get("text") or "").replace("**", "").replace("`", "")
        v = re.search(r"VERDICT:\s*(\w+)", t)
        if m["verdict"] == "refused" or not v or not v.group(1).lower().startswith("appl"):
            continue
        fe = re.search(r"FEASIBLE AFTER:\s*(\w+)", t)
        va = re.search(r"VIOLATED AFTER:\s*(.*)", t)
        f = q["variant"]
        g = q["group"]
        fw = "-" if not fe else ("yes" if fe.group(1).lower().startswith("y") else "no")
        vw = "-" if not va else ("none" if va.group(1).strip().lower() in ("none", "", "none.") else "names")
        for scope in (f, "ALL " + g):
            c[scope]["n"] += 1
            c[scope]["feas %s / viol %s" % (fw, vw)] += 1
            if fw != "-" and vw != "-":
                tf = bool(m["feasible"])
                c[scope]["both"] += 1
                c[scope]["hit_feasline"] += (fw == "yes") == tf
                c[scope]["hit_violline"] += (vw == "none") == tf
    print("##", path)
    for s in sorted(c):
        x = c[s]
        cells = ", ".join("%s %d" % (k, x[k]) for k in sorted(x) if k.startswith("feas "))
        b = x["both"] or 1
        print("  %-26s n %4d | %s | acc FEASIBLE line %.3f, acc VIOLATED line %.3f (n %d)" % (
            s, x["n"], cells, x["hit_feasline"] / b, x["hit_violline"] / b, x["both"]))
print("FEAS_CONSIST_DONE")
