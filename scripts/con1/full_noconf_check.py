"""Move-request check of a rebuilt full row set (CPU; con1-2m-rows, 11-27; owner order: the move request out of every
con1 prompt, conseq_lib.consequence_user, con1 eff023a).

    python full_noconf_check.py --rows train.jsonl dev.jsonl [dev_clean.jsonl] --stats stats_new.json \\
        --ref-stats stats_running.json

1. Marks: no row of any --rows file (any message, any role) contains one of conseq_lib.MOVE_REQUEST_MARKS ("YOUR
   MOVE", "<action>", "end your reply", "60-150 words", "Make exactly one change", "Propose a different move"); the
   first user message carries the OPERATIONS heading exactly once. Counted per file and per mark; a hit fails.
2. Counts per split and family, read from the row files themselves (rows, questions = con1.moves per row) and from the
   two stats files (rows, questions, states, tokens, prompt tokens): the new set against the running set. The same
   outcomes must be there (questions and states equal per split and family); rows may differ only where the token cap
   splits a state differently; tokens are expected lower by the cut text.
Prints NOCONF_CHECK_PASS or NOCONF_CHECK_FAIL.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conseq_lib import MOVE_REQUEST_MARKS, OPS_HEADING  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--ref-stats", required=True)
    a = ap.parse_args()
    ok = True
    print("marks checked: %s; heading %r" % (", ".join(MOVE_REQUEST_MARKS), OPS_HEADING))
    files = {}
    for path in a.rows:
        hits, c = collections.Counter(), collections.Counter()
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            msgs = r["messages"]
            c["rows"] += 1
            c["questions"] += len(r["con1"]["moves"])
            c["rows/" + r["domain"]] += 1
            c["questions/" + r["domain"]] += len(r["con1"]["moves"])
            for m in msgs:
                for mk in MOVE_REQUEST_MARKS:
                    if mk in (m.get("content") or "") or mk in (m.get("reasoning_content") or ""):
                        hits[mk] += 1
            first_user = next(m["content"] for m in msgs if m["role"] == "user")
            c["first user with the OPERATIONS heading once"] += first_user.count("\n" + OPS_HEADING) == 1
        files[path] = c
        bad = sum(hits.values()) or c["first user with the OPERATIONS heading once"] != c["rows"]
        ok &= not bad
        print("%s: rows %d, questions %d, mark hits %s, OPERATIONS heading once in %d of %d rows -> %s" % (
            path, c["rows"], c["questions"], dict(hits) or "none", c["first user with the OPERATIONS heading once"],
            c["rows"], "FAIL" if bad else "ok"))
    new, ref = json.load(open(a.stats)), json.load(open(a.ref_stats))
    nf, rf = new["families"], ref["families"]
    print("\nper split / family: new set vs running set (%s, sha1 %s)" % (a.ref_stats, ref.get("train_sha1")))
    print("  %-20s %8s %8s | %9s %9s | %7s %7s | %14s %14s | %6s" % (
        "split/family", "rows", "(run)", "questions", "(run)", "states", "(run)", "tokens", "(run)", "tok %"))
    tot = collections.Counter()
    for k in sorted(set(nf) | set(rf)):
        n, r = nf.get(k, {}), rf.get(k, {})
        same = n.get("questions") == r.get("questions") and n.get("states") == r.get("states")
        ok &= same
        for f in ("rows", "questions", "states", "tokens", "prompt_tokens"):
            tot[(k.split("/")[0], f, "new")] += n.get(f, 0)
            tot[(k.split("/")[0], f, "run")] += r.get(f, 0)
        print("  %-20s %8d %8d | %9d %9d | %7d %7d | %14d %14d | %+5.1f%s" % (
            k, n.get("rows", 0), r.get("rows", 0), n.get("questions", 0), r.get("questions", 0), n.get("states", 0),
            r.get("states", 0), n.get("tokens", 0), r.get("tokens", 0),
            100.0 * (n.get("tokens", 0) / max(1, r.get("tokens", 0)) - 1), "" if same else "  <- outcomes differ"))
    for sp in ("train", "dev"):
        print("  %-20s %8d %8d | %9d %9d | %7d %7d | %14d %14d | %+5.1f   prompt tokens %d (run %d)" % (
            sp + " total", tot[(sp, "rows", "new")], tot[(sp, "rows", "run")], tot[(sp, "questions", "new")],
            tot[(sp, "questions", "run")], tot[(sp, "states", "new")], tot[(sp, "states", "run")],
            tot[(sp, "tokens", "new")], tot[(sp, "tokens", "run")],
            100.0 * (tot[(sp, "tokens", "new")] / max(1, tot[(sp, "tokens", "run")]) - 1),
            tot[(sp, "prompt_tokens", "new")], tot[(sp, "prompt_tokens", "run")]))
    print("  train rows in stats %d (running set %d); longest row %s (running %s)" % (
        new.get("train_rows", -1), ref.get("train_rows", -1), new.get("longest"), ref.get("longest")))
    print("NOCONF_CHECK_%s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
