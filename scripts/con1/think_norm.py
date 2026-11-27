"""Thinking diagnostic: layout-only normalisation of answers for facets.py (CPU; con1-2m-rows, 11-27).

    python think_norm.py preds_L.jsonl preds_L_norm.jsonl

The base model writes the facetm lines with punctuation facets.pred_facets does not read ('max_temp_rise: up, 10-20%',
'max_temp_rise down, 10-20%'; facets expects 'max_temp_rise down 10-20%'), so its directions and bins are read as
missing. Per line: a ':' right after a leading word, and a ',' right after up / down / same / satisfied / violated /
'at limit', are replaced by a space. Nothing else changes (words, numbers, order); the count of changed answers is
printed.
"""
import json
import re
import sys

src, dst = sys.argv[1:3]
COLON = re.compile(r"^(\s*[A-Za-z_][\w.]*)\s*:\s+(?=\S)", re.M)
COMMA = re.compile(r"\b(up|down|same|satisfied|violated|at limit)\s*,\s*", re.I)
KEEP = ("VERDICT", "REASON", "MARGINS AFTER", "FEASIBLE AFTER", "VIOLATED AFTER", "CHANGES")
n = c = 0
with open(dst, "w", encoding="utf-8") as fo:
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        t = r.get("text")
        if t:
            out = []
            for ln in t.split("\n"):
                if not ln.lstrip().upper().startswith(KEEP):
                    ln = COMMA.sub(lambda m: m.group(1) + " ", COLON.sub(r"\1 ", ln))
                out.append(ln)
            t2 = "\n".join(out)
            c += t2 != t
            r["text"] = t2
        n += 1
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print("%d answers, %d changed by the layout normalisation -> %s" % (n, c, dst))
