"""Deterministic check of the consequence prompt without the move request (owner order 11-27, conseq_lib.consequence_user).

    python noconf_check.py --pairs queries_v2.jsonl queries_v3.jsonl     (a rebuilt question file against its source)
    python noconf_check.py --states S.jsonl.gz [S2 ...]                  (every state prompt a row builder would cut)
    [--fmt facetm] [--out report.json]

Per prompt, the old text is the served user text and the new one the cut text (the paired file's, or consequence_user's).
Checks, per family:
  marks      the full new prompt (system + user + the first QUESTION in --fmt) contains none of "YOUR MOVE", "<action>",
             "end your reply", "60-150 words" (and, reported apart, "Make exactly one change", "Propose a different move")
  ops        every operation signature line of the old prompt (a line starting VERB( in the old operations section)
             and every line of that section is still in the new prompt, in the same order
  identical  a line-level diff (difflib, no autojunk) of old vs new: the only removed lines are the move-request
             sentences and the refusal note's last line, the only inserted line is the OPERATIONS heading; every other
             line is byte-identical. The removed and inserted strings are counted per family.
  same       (--pairs) every field but 'user' and 'prompt' is identical between the two files, line for line
Exit status 1 when any prompt fails a check.
"""
import argparse
import collections
import difflib
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conseq_lib import consequence_user  # noqa: E402
from targets import question  # noqa: E402

MARKS = ("YOUR MOVE", "<action>", "end your reply", "60-150 words")
MARKS_EXTRA = ("Make exactly one change", "Propose a different move")
OPS_LINE = "The operations in this problem are exactly:\n"
SIG = re.compile(r"^[A-Z][A-Z0-9_]*\(")


def opener(p):
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")


def ops_section(old: str) -> list:
    k = old.find(OPS_LINE)
    if k < 0:
        return []
    return [ln for ln in old[k + len(OPS_LINE):].splitlines(keepends=True) if ln != "Propose a different move.\n"]


def in_order(needles: list, hay: list) -> bool:
    it = iter(hay)
    return all(any(n == h for h in it) for n in needles)


class Tally:
    def __init__(self):
        self.n = collections.Counter()
        self.fail = collections.Counter()
        self.marks = collections.defaultdict(collections.Counter)
        self.removed = collections.defaultdict(collections.Counter)
        self.inserted = collections.defaultdict(collections.Counter)
        self.sigs = collections.Counter()
        self.first_fail = {}

    def add(self, v, sid, system, old, new, fmt, move):
        self.n[v] += 1
        full = system + "\n" + new + question(fmt, move or "X()", True)
        bad = []
        for m in MARKS + MARKS_EXTRA:
            if m in full:
                self.marks[v][m] += 1
                bad.append("mark:" + m)
        ops = ops_section(old)
        sigs = [ln for ln in ops if SIG.match(ln)]
        self.sigs[v] += len(sigs)
        nl = new.splitlines(keepends=True)
        if not sigs or not in_order(ops, nl):
            bad.append("ops")
        ol = old.splitlines(keepends=True)
        sm = difflib.SequenceMatcher(None, ol, nl, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            for ln in ol[i1:i2]:
                self.removed[v][ln] += 1
            for ln in nl[j1:j2]:
                self.inserted[v][ln] += 1
        allowed_rm = set(REMOVABLE)
        rm = [ln for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal" for ln in ol[i1:i2]]
        ins = [ln for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal" for ln in nl[j1:j2]]
        if any(ln not in allowed_rm for ln in rm) or any(ln != "OPERATIONS\n" for ln in ins) \
                or ins.count("OPERATIONS\n") != 1 or "YOUR MOVE\n" not in rm:
            bad.append("identical")
        if bad:
            self.fail[v] += 1
            self.first_fail.setdefault(v, {"sid": sid, "why": bad})
        return not bad

    def report(self):
        out = {"families": {}}
        for v in sorted(self.n):
            out["families"][v] = {
                "prompts": self.n[v], "failed": self.fail[v], "signature_lines": self.sigs[v],
                "marks_left": dict(self.marks[v]),
                "removed": {k: c for k, c in self.removed[v].most_common()},
                "inserted": {k: c for k, c in self.inserted[v].most_common()},
                "first_fail": self.first_fail.get(v)}
        out["prompts"] = sum(self.n.values())
        out["failed"] = sum(self.fail.values())
        return out


REMOVABLE = None


def main(argv=None):
    global REMOVABLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs=2, default=None)
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--fmt", default="facetm")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from conseq_lib import MOVE_REQUEST
    REMOVABLE = MOVE_REQUEST.splitlines(keepends=True) + ["Propose a different move.\n"]
    t = Tally()
    same = collections.Counter()
    if a.pairs:
        with opener(a.pairs[0]) as f0, opener(a.pairs[1]) as f1:
            for l0, l1 in zip(f0, f1):
                q0, q1 = json.loads(l0), json.loads(l1)
                r0 = {k: x for k, x in q0.items() if k not in ("user", "prompt")}
                r1 = {k: x for k, x in q1.items() if k not in ("user", "prompt")}
                same["identical other fields" if r0 == r1 else "DIFFERENT other fields"] += 1
                if q1["user"] != consequence_user(q0["user"]):
                    same["DIFFERENT from consequence_user"] += 1
                t.add(q0["variant"], q0["sid"], q1["system"], q0["user"], q1["user"], a.fmt,
                      (q1.get("moves") or [{}])[0].get("move"))
    for p in a.states or []:
        with opener(p) as f:
            for line in f:
                st = json.loads(line)
                try:
                    new = consequence_user(st["user"])
                except ValueError as e:
                    t.n[st["variant"]] += 1
                    t.fail[st["variant"]] += 1
                    t.first_fail.setdefault(st["variant"], {"sid": st.get("sid"), "why": [str(e)]})
                    continue
                mv = (st.get("moves") or [{}])[0].get("move")
                t.add(st["variant"], st.get("sid"), st["system"], st["user"], new, a.fmt, mv)
    rep = t.report()
    rep["pairs"] = dict(same)
    rep["inputs"] = {"pairs": a.pairs, "states": a.states}
    for v, r in rep["families"].items():
        print("%-11s prompts %6d failed %d signature lines %7d marks left %s" % (
            v, r["prompts"], r["failed"], r["signature_lines"], r["marks_left"] or "none"))
        for k, c in r["removed"].items():
            print("    removed  x%-6d %r" % (c, k[:110] + ("..." if len(k) > 110 else "")))
        for k, c in r["inserted"].items():
            print("    inserted x%-6d %r" % (c, k))
        if r["first_fail"]:
            print("    first fail:", r["first_fail"])
    print("pairs:", dict(same))
    print("TOTAL prompts %d failed %d" % (rep["prompts"], rep["failed"]))
    if a.out:
        json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("NOCONF_CHECK_DONE")
    return 1 if rep["failed"] or same.get("DIFFERENT other fields") or same.get("DIFFERENT from consequence_user") else 0


if __name__ == "__main__":
    sys.exit(main())
