"""Write datav4_corpus/engibench/shared_base_with_eval.json: every engibench TRAINING problem (any
batch) whose base design (family, source_row) is also the base of an evaluation problem (b001
indices 70-129), with the evaluation problems it shares it with and the rows it contributes per
batch and spelling. Owner decision 2026-10-24: keep these rows, flagged."""
import collections
import glob
import json
import os
import sys

D = "/storage/wxu/datav4"
C = os.path.join(D, "datav4_corpus", "engibench")


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def idx(pid):
    return int(pid.rsplit("_", 1)[-1])


batches = {}
for d in sorted(glob.glob(os.path.join(C, "accepted*"))):
    name = os.path.basename(d)
    batch = "b001" if name == "accepted" else name.replace("accepted_", "")
    batches[batch] = load(os.path.join(d, "problems.jsonl"))
main = batches.get("b001", [])
eval_by_key = collections.defaultdict(list)
for s in main:
    if idx(s["problem_id"]) >= 70:
        eval_by_key[(s["family"], s["source_row"])].append(s["problem_id"])
flagged = []
for batch, specs in batches.items():
    for s in specs:
        if batch == "b001" and idx(s["problem_id"]) >= 70:
            continue
        k = (s["family"], s["source_row"])
        if k in eval_by_key:
            flagged.append({"problem_id": s["problem_id"], "corpus_batch": batch, "family": s["family"],
                            "source_row": s["source_row"], "eval_problems": eval_by_key[k]})
ids = {f["problem_id"] for f in flagged}
rows = {}
for sp in ("multitool", "set_parameter"):
    for f in glob.glob(os.path.join(D, "datav4_" + sp, "**", "engibench", "*.jsonl"), recursive=True):
        rel = os.path.relpath(f, os.path.join(D, "datav4_" + sp))
        if "_groups" in rel:
            continue
        n = sum(r.get("instance") in ids for r in load(f))
        if n:
            rows["%s/%s" % (sp, rel)] = n
out = {"what": "engibench training problems whose base design (family, source_row: load case, boundary "
               "conditions, reference optimum) is also the base of an evaluation problem (b001 indices 70-129); "
               "their specs differ (start design, targets). Owner decision 2026-10-24: kept as training rows, "
               "flagged; drop rows whose instance is listed here if a strictly base-disjoint engibench evaluation "
               "is wanted.",
       "n_problems": len(flagged), "by_batch": dict(collections.Counter(f["corpus_batch"] for f in flagged)),
       "rows_from_flagged_problems": rows, "problems": flagged}
dst = os.path.join(C, "shared_base_with_eval.json")
json.dump(out, open(dst, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("n_problems", "by_batch", "rows_from_flagged_problems")}, indent=1))
print("wrote", dst)
