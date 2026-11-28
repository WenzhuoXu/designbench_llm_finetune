"""Where facets.pred_facets and the attack reader (lib.parse) read an answer file differently (CPU; 11-28): per
family, applied answers whose verdict or direction set differs; the first examples with the answer text.
    python reader_vs_attack.py QUERIES PREDS FORM [N_EXAMPLES]
"""
import collections
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "/storage/wxu/con1/full_eval")
import dir_split as DS  # noqa: E402
sys.path.insert(0, "/storage/wxu/con1/review_tmp/v1128D/attack")
import lib  # noqa: E402

qs = {json.loads(l)["sid"]: json.loads(l) for l in open(sys.argv[1], encoding="utf-8")}
fmt = sys.argv[3]
nex = int(sys.argv[4]) if len(sys.argv) > 4 else 3
c = collections.defaultdict(collections.Counter)
ex = []
for l in open(sys.argv[2], encoding="utf-8"):
    r = json.loads(l)
    q = qs[r["sid"]]
    v, _, d = DS.read(r.get("text"), fmt, q, "facets")
    p = lib.parse(r.get("text"), list(q["metrics0"]))
    f = q["variant"]
    c[f]["answers"] += 1
    c[f]["verdict differs"] += (v == "applied") != (p["v"] == "applied")
    if v == "applied" and p["v"] == "applied":
        da, dp = set(d), set(p["d"])
        c[f]["lines facets"] += len(da)
        c[f]["lines attack"] += len(dp)
        c[f]["only attack"] += len(dp - da)
        c[f]["only facets"] += len(da - dp)
        c[f]["word differs"] += sum(d[k] != p["d"][k] for k in da & dp)
        if (dp - da or v != p["v"]) and len(ex) < nex:
            ex.append((r["sid"], r["qi"], f, sorted(dp - da)[:5], (r.get("text") or "")[:700]))
    elif (v == "applied") != (p["v"] == "applied") and len(ex) < nex:
        ex.append((r["sid"], r["qi"], f, "verdict facets %s attack %s" % (v, p["v"]), (r.get("text") or "")[:700]))
for f in sorted(c):
    print("  %-11s %s" % (f, dict(c[f])))
for e in ex:
    print("--- %s q%d %s %s\n%s" % e)
print("READER_VS_ATTACK_DONE")
