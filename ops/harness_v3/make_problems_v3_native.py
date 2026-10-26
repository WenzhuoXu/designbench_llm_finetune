"""problems_v3_native.jsonl: the evaluation problems of the native-verb variants, in the harness's
--problems-file format ({"variant": v, "spec": spec} per line, the accepted spec verbatim).

Rule (recorded in <out>.meta.json): per variant, the LAST --n problems BY INDEX (the integer suffix
of problem_id, e.g. molopt_0129 -> 129) of <corpus>/<v>/<v>/accepted/problems.jsonl. With 130
accepted problems that is indices 70..129; indices 0..69 are left for own-row training, so the two
sets are disjoint by construction (own-row selection must take its problems from the complement of
the ids listed in the meta file). A variant whose accepted/ does not exist yet (generation still
running, or the gates failed) is skipped and listed under "todo" in the meta file; rerun this script
when it lands and the file grows in place (one line per problem, variants in --variants order).

    python make_problems_v3_native.py --corpus /storage/wxu/corpus/out \
        --variants molopt grid2op engibench photonics --n 60 --out problems_v3_native.jsonl
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="run_generation --out dir (<corpus>/<v>/<v>/accepted)")
    ap.add_argument("--variants", nargs="+", default=["molopt", "grid2op", "engibench", "photonics"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = {"rule": "per variant, the LAST %d problems by index (integer suffix of problem_id) of "
                    "<corpus>/<v>/<v>/accepted/problems.jsonl; the lower indices are reserved for "
                    "own-row training, so evaluation and training problems are disjoint" % a.n,
            "corpus": a.corpus, "n_per_variant": a.n, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "variants": {}, "todo": []}
    n_lines = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for v in a.variants:
            path = os.path.join(a.corpus, v, v, "accepted", "problems.jsonl")
            if not os.path.isfile(path):
                meta["todo"].append({"variant": v, "missing": path,
                                     "note": "accepted/ not written yet (generation running or gates failed)"})
                print("%-10s TODO: %s missing" % (v, path), flush=True)
                continue
            rows = []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            idx = [(int(str(r["problem_id"]).rsplit("_", 1)[1]), r) for r in rows]
            idx.sort(key=lambda t: t[0])
            take = idx[-a.n:]
            train_ids = [r["problem_id"] for _, r in idx[:-a.n]]
            for _, r in take:
                fo.write(json.dumps({"variant": v, "spec": r}) + "\n")
                n_lines += 1
            meta["variants"][v] = {
                "accepted_total": len(rows), "index_min": idx[0][0], "index_max": idx[-1][0],
                "eval_index_from": take[0][0], "eval_index_to": take[-1][0], "n_eval": len(take),
                "eval_problem_ids": [r["problem_id"] for _, r in take],
                "reserved_for_training_ids": train_ids,
            }
            print("%-10s accepted %d, eval = indices %d..%d (%d problems), %d left for training"
                  % (v, len(rows), take[0][0], take[-1][0], len(take), len(train_ids)), flush=True)
    with open(a.out + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print("wrote %d problems -> %s (+ .meta.json)" % (n_lines, a.out))


if __name__ == "__main__":
    main()
