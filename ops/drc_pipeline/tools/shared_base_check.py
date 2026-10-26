"""How many engibench training problems (b001 idx < 70, b004 idx 130-259) share their base design
(family, source_row) with an evaluation problem (b001 idx 70-129), and how many rows come from them."""
import collections
import json
import os

P = "/storage/wxu/corpus"
D = "/storage/wxu/datav4"


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def idx(s):
    return int(s["problem_id"].rsplit("_", 1)[-1])


def key(s):
    for k in ("source_row", "row", "base_row", "dataset_row"):
        if k in s:
            return (s.get("family"), s[k])
    return (s.get("family"), None)


specs = load(P + "/out/engibench/engibench/accepted/problems.jsonl")
print("spec keys:", sorted(specs[0].keys()))
b4 = os.path.join(P, "out_b004/engibench/engibench/accepted/problems.jsonl")
specs4 = load(b4) if os.path.exists(b4) else []
ev = {key(s) for s in specs if idx(s) >= 70}
tr1 = {s["problem_id"] for s in specs if idx(s) < 70 and key(s) in ev}
tr4 = {s["problem_id"] for s in specs4 if key(s) in ev}
print("eval base keys: %d; families %s" % (len(ev), dict(collections.Counter(k[0] for k in ev))))
print("b001 training problems sharing an eval base: %d of %d" % (len(tr1), sum(idx(s) < 70 for s in specs)))
print("b004 problems sharing an eval base: %d of %d" % (len(tr4), len(specs4)))
for sp in ("multitool", "set_parameter"):
    for name, shared, rel in (("b001", tr1, "rows/engibench"), ("b004", tr4, "batches/b004/rows/engibench")):
        f = os.path.join(D, "datav4_" + sp, rel, "train_own_act.jsonl")
        if not os.path.exists(f):
            print(sp, name, "no file", f); continue
        rows = load(f)
        print("%-13s %s engibench rows: %d, from shared-base problems: %d" % (sp, name, len(rows), sum(r["instance"] in shared for r in rows)))
