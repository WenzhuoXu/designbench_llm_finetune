"""Row-by-row check of a rebuilt full row set against the running set (CPU; con1-2m-rows, 11-27): the only change is the
move request cut from each row's prompt.

    python full_pair_check.py --old train_running.jsonl --new train_nc.jsonl

Both files list the states in the same order (build_full_rows.py: input order of the states, a state's rows in
order). Per state (sid): if both sets cut it into the same rows (the same move lists in the same order), every row
must be byte-identical in every field and message except the first user message, and that message must equal
conseq_lib.consequence_user(old state text) + the old first QUESTION unchanged. States cut differently (the token cap
binds on the old, longer prompt and not on the new one) are counted apart, and their (move -> answer message) pairs
must still be identical. Prints PAIR_CHECK_PASS or PAIR_CHECK_FAIL.
"""
import argparse
import collections
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conseq_lib import consequence_user  # noqa: E402

QMARK = "\n\nQUESTION (no move is made now)"


def by_state(path):
    rows = (json.loads(l) for l in open(path, encoding="utf-8"))
    return itertools.groupby(rows, key=lambda r: r["sid"])


def answers(rows):
    out = {}
    for r in rows:
        for mv, k in zip(r["con1"]["moves"], range(2, len(r["messages"]), 2)):
            out[mv] = json.dumps(r["messages"][k], sort_keys=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    a = ap.parse_args()
    c = collections.Counter()
    bad = []
    for (s_old, g_old), (s_new, g_new) in itertools.zip_longest(by_state(a.old), by_state(a.new), fillvalue=(None, None)):
        if s_old != s_new:
            bad.append("state order differs: %s vs %s" % (s_old, s_new))
            break
        old, new = list(g_old), list(g_new)
        c["states"] += 1
        if [r["con1"]["moves"] for r in old] != [r["con1"]["moves"] for r in new]:
            c["states cut into rows differently"] += 1
            c["rows old (those states)"] += len(old)
            c["rows new (those states)"] += len(new)
            ao, an = answers(old), answers(new)
            if ao != an:
                bad.append("%s: answers differ" % s_old)
            continue
        for ro, rn in zip(old, new):
            c["rows compared"] += 1
            ko = next(i for i, m in enumerate(ro["messages"]) if m["role"] == "user")
            u = ro["messages"][ko]["content"]
            cut = u.rfind(QMARK)
            want = consequence_user(u[:cut]) + u[cut:]
            rest_o = dict(ro, messages=[m for i, m in enumerate(ro["messages"]) if i != ko])
            rest_n = dict(rn, messages=[m for i, m in enumerate(rn["messages"]) if i != ko])
            if rn["messages"][ko]["content"] != want or {k: v for k, v in ro["messages"][ko].items() if k != "content"} \
                    != {k: v for k, v in rn["messages"][ko].items() if k != "content"}:
                bad.append("%s: first user message is not the old one with the move request cut" % s_old)
            elif json.dumps(rest_o, sort_keys=True) != json.dumps(rest_n, sort_keys=True):
                bad.append("%s: a field or message other than the prompt differs" % s_old)
            else:
                c["rows identical except the cut"] += 1
                c["prompt chars removed"] += len(u) - len(want)
    for k in sorted(c):
        print("  %-40s %d" % (k, c[k]))
    for b in bad[:10]:
        print("  !!! " + b)
    print("  failures: %d" % len(bad))
    print("PAIR_CHECK_%s" % ("FAIL" if bad else "PASS"))


if __name__ == "__main__":
    main()
