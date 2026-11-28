"""How often answers leave the answer form (CPU; 11-28): per answer file on subset_300, the answers with a VERDICT
line at a line start, with a markdown table, with more than 300 answer tokens, and the share answered refused among
truth-applied unflagged questions.
    python answer_form_counts.py PREDS [PREDS ...]
"""
import collections
import json
import re
import sys

B = "/storage/wxu/con1/full_eval/base_think2/"
qs = {json.loads(l)["sid"]: json.loads(l) for l in open("/storage/wxu/con1/arm/main/queries_v3_inv2cs.jsonl")}
fl = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open("/storage/wxu/con1/full_eval/inv2cs/flagged_any.tsv") if l.strip()}
sub = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open(B + "subset_300.tsv") if l.strip()}
for name in sys.argv[1:]:
    c = collections.Counter()
    for l in open(name):
        r = json.loads(l)
        k = (r["sid"], r["qi"])
        if k not in sub:
            continue
        t = r.get("text") or ""
        c["answers"] += 1
        c["VERDICT line at a line start"] += bool(re.search(r"^\**VERDICT\**\s*:", t, re.M))
        c["markdown table"] += ("|---" in t or "| ---" in t or "|--" in t)
        c["answer tokens > 300"] += (r.get("answer_tokens") or 0) > 300
        m = qs[r["sid"]]["moves"][r["qi"]]
        if k not in fl and m["verdict"] != "refused":
            c["truth-applied unflagged"] += 1
            c["  answered refused"] += bool(re.search(r"VERDICT\**\s*:\s*\**\s*refused", t, re.I))
    print(name.split("/")[-1], dict(c))
