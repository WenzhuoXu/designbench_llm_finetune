"""Per-question flags for the con1 test questions (CPU; con1-2m-rows, 11-28, parent's order 1c). Nothing is removed
from the question file; tables report the flagged questions apart from the headline.

    python make_flags.py QUERIES.jsonl FLAGS.tsv

  no-op      the simulator applied the move and no metric changed (every metrics_after value equal to metrics0 within
             the 'same' tolerance, |a - b| <= max(1e-5 |b|, 1e-12))
  off-range  thermal2 PLACE_BLOCK / MOVE_SINK whose slot / site code lies outside the code range the prompt prints
             ('slot is a code q0-qN', 'site is a code s0-sM')
Output: one line per flagged question, 'sid<TAB>qi<TAB>flag'; a question with both flags gets two lines. Counts per
group and family are printed.
"""
import collections
import json
import re
import sys

src, dst = sys.argv[1:3]
c = collections.Counter()
out = []
for line in open(src):
    q = json.loads(line)
    m0 = q["metrics0"]
    slot = re.search(r"slot is a code q(\d+)-q(\d+)", q["user"])
    site = re.search(r"site is a code s(\d+)-s(\d+)", q["user"])
    for j, m in enumerate(q["moves"]):
        g = (q["group"], q["variant"])
        c[g + ("questions",)] += 1
        if m["verdict"] != "refused":
            after = m["metrics_after"]
            if all(abs(float(after[k]) - float(b)) <= max(1e-5 * abs(float(b)), 1e-12) for k, b in m0.items() if k in after):
                out.append((q["sid"], j, "no-op"))
                c[g + ("no-op",)] += 1
        if q["variant"] == "thermal2":
            mm = re.match(r"(PLACE_BLOCK|MOVE_SINK)\(\s*\d+\s*,\s*([qs])(\d+)\s*\)", m["move"])
            if mm:
                rng = slot if mm.group(2) == "q" else site
                if rng and not (int(rng.group(1)) <= int(mm.group(3)) <= int(rng.group(2))):
                    out.append((q["sid"], j, "off-range"))
                    c[g + ("off-range",)] += 1
                    c[g + ("off-range applied",)] += m["verdict"] != "refused"
with open(dst, "w") as fo:
    for sid, j, f in out:
        fo.write("%s\t%d\t%s\n" % (sid, j, f))
for k in sorted(c):
    print("  %-16s %-11s %-18s %d" % (k + (c[k],)))
print("flags: %d lines -> %s" % (len(out), dst))
