"""K4 question list (11-28, parent's order): every unflagged question (no no-op / off-range flag on the common-seed
truth) of inventory2, molopt, thermal2, plus the subset_300 questions of photonics, frame3d2, seir2, engibench, grid2op.
    python k4_subset.py QUERIES FLAGS SUBSET_300 OUT.tsv
"""
import collections
import json
import sys

qp, fp, sp, out = sys.argv[1:5]
flag = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open(fp) if l.strip()}
sub = [(l.split("\t")[0], int(l.split("\t")[1])) for l in open(sp) if l.strip()]
fam = {}
keys = []
for l in open(qp, encoding="utf-8"):
    q = json.loads(l)
    fam[q["sid"]] = q["variant"]
    if q["variant"] in ("inventory2", "molopt", "thermal2"):
        keys += [(q["sid"], j) for j in range(len(q["moves"])) if (q["sid"], j) not in flag]
extra = [k for k in sub if fam[k[0]] in ("photonics", "frame3d2", "seir2", "engibench", "grid2op")]
keys += extra
assert len(set(keys)) == len(keys)
c = collections.Counter(fam[k[0]] for k in keys)
cf = collections.Counter(fam[k[0]] for k in extra if k in flag)
open(out, "w").write("".join("%s\t%d\n" % k for k in keys))
print("K4 questions %d; per family %s; flagged among the subset_300 part %s" % (len(keys), dict(sorted(c.items())), dict(cf)))
