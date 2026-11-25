"""con1 skill test, step 1: the fixed question set (docs/consequence_round_con1.md §3).

    python test_set.py --states test_states.jsonl.gz --out queries.jsonl [--q 8] [--per-family 60] [--seed 0]

Per test state 8 moves by the training recipe (build_rows.pick_moves: own moves, the 2 best legal moves, off-grid
moves on verb-surface families, uniform legal moves), in a seeded random order. One line per state:
{sid, variant, group, problem_id, t, system, user, phi0, bad0, metrics0, moves: [{move, src, verdict, phi, viol,
feasible, refused, result, metrics_after, args}]} with group = heldout_problem (the 8 training families on benchmark
problems) or heldout_family (inventory2, molopt, thermal2).
"""
import argparse
import collections
import gzip
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_rows import HELD_OUT, pick_moves  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--per-family", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random("con1-test-%d" % a.seed)
    by = collections.defaultdict(list)
    seen = set()                                           # first copy of a state wins (e.g. its off-menu version)
    for p in a.states:
        f = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")
        for line in f:
            st = json.loads(line)
            if st["split"] != "test":
                raise SystemExit("not a test state: %s" % st["sid"])
            if "/r1/" not in st["key"] or st["sid"] in seen:  # pair a's first run: one state per problem
                continue
            seen.add(st["sid"])
            by[st["variant"]].append(st)
    c = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as fo:
        for v in sorted(by):
            sts = sorted(by[v], key=lambda s: s["sid"])
            rng.shuffle(sts)
            for st in sts[:a.per_family]:
                moves = pick_moves(st, a.q, rng)
                if len(moves) < 2:
                    c["skipped (fewer than 2 moves) %s" % v] += 1
                    continue
                keep = {k: st[k] for k in ("sid", "variant", "problem_id", "t", "k", "system", "user", "phi0", "bad0",
                                           "metrics0", "binding", "n_legal", "turn_log")}
                keep["group"] = "heldout_family" if v in HELD_OUT else "heldout_problem"
                keep["moves"] = moves
                fo.write(json.dumps(keep, ensure_ascii=False) + "\n")
                c["states %s" % v] += 1
                for m in moves:
                    c["moves %s %s" % (v, m["verdict"])] += 1
                    for s in m["src"]:
                        c["src %s %s" % (v, s)] += 1
    for k in sorted(c):
        print("%-40s %d" % (k, c[k]))
    print("TEST_SET_DONE")


if __name__ == "__main__":
    main()
