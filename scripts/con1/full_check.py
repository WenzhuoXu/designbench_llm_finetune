"""con1 full row set: independent checks of a build (docs/con1_full_rows.md).

    python full_check.py --rows train.jsonl dev.jsonl --stats stats.json --stage1 train_facet.jsonl dev_facet.jsonl \\
        --q 12 --like train_facet.jsonl

- no (state, move) pair twice in the set, none of stage 1's pairs in it;
- every row: tier con1_conseq, supervise every_assistant, the stats' answer format, 1..q questions, messages
  alternating as conseq_format expects, the same keys as a stage-1 row (--like);
- per family and split: simulated outcomes = stage-1 outcomes + outcomes in the set + dropped (the stats' counts
  against the rows actually written);
- dev rows only from dev states, train rows only from train states (the state's split, from the stats' sids).
Prints one line per check and CHECK_OK or CHECK_FAILED.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--stage1", nargs="+", required=True)
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--like", required=True)
    a = ap.parse_args(argv)
    st = json.load(open(a.stats))
    s1 = collections.defaultdict(set)
    for p in a.stage1:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            s1[r["sid"]].update(r["con1"]["moves"])
    like = json.loads(open(a.like, encoding="utf-8").readline())
    keys = sorted(like)
    ok = True
    seen = set()
    dup = overlap = bad_rows = 0
    n_q = collections.Counter()
    for p, split in zip(a.rows, ("train", "dev")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            ms = r["con1"]["moves"]
            roles = [m["role"] for m in r["messages"]]
            if (sorted(r) != keys or r["tier"] != "con1_conseq" or r["supervise"] != "every_assistant"
                    or r["format"] != st["target"] or not 1 <= len(ms) <= a.q
                    or roles != ["system", "user", "assistant"] + ["user", "assistant"] * (len(ms) - 1)):
                bad_rows += 1
            for mv in ms:
                k = (r["sid"], mv)
                if k in seen:
                    dup += 1
                seen.add(k)
                if mv in s1.get(r["sid"], ()):
                    overlap += 1
            n_q["%s/%s" % (split, r["domain"])] += len(ms)
    print("pairs in the set %d; duplicated pairs %d; pairs shared with stage 1 %d; malformed rows %d" % (
        len(seen), dup, overlap, bad_rows))
    ok &= dup == 0 and overlap == 0 and bad_rows == 0
    for k, f in sorted(st["families"].items()):
        lhs = f["outcomes_simulated"]
        rhs = f["excluded_stage1"] + n_q[k] + f["dropped_too_long"]
        good = lhs == rhs and n_q[k] == f["questions"]
        ok &= good
        print("%-16s simulated %8d = stage-1 %7d + set %8d + dropped %d  %s" % (
            k, lhs, f["excluded_stage1"], n_q[k], f["dropped_too_long"], "ok" if good else "MISMATCH"))
    s1_pairs = sum(len(v) for v in s1.values())
    ex = sum(f["excluded_stage1"] for f in st["families"].values())
    print("stage-1 pairs %d, of which found in the states and excluded %d" % (s1_pairs, ex))
    ok &= ex == s1_pairs
    print("CHECK_OK" if ok else "CHECK_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
