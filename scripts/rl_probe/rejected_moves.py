"""What rejected moves cost in the stored rollouts. The harness spends the turn on a move the state
refuses and serves the same prompt again (the refusal and its reason are recorded, not shown to the
model). Per family: the share of turns spent on rejected moves, how often the very next move is
the same rejected move again, how often a rejected move had already been rejected in that episode,
legal moves per episode, and episode success by whether any move was rejected.

usage: python rejected_moves.py <rollouts dir>
"""
import collections
import glob
import json
import os
import sys

import numpy as np

H = 12
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "step_*.compact.jsonl")))
have = {os.path.basename(f)[:10] for f in files}
files += [f for f in sorted(glob.glob(os.path.join(d, "step_*.jsonl")))
          if not f.endswith(".compact.jsonl") and os.path.basename(f)[:10] not in have]
S = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(files):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        ep = r["episode"]
        turns = [t for t in ep.get("turns", []) if "episode has ended" not in str(t.get("fail"))]
        if not turns:
            continue
        s = S[r["variant"]]
        ff = ep.get("first_feasible_turn")
        ok = ff is not None and ff <= H
        ill = [bool(t.get("fail")) and str(t.get("fail")).startswith("illegal") for t in turns]
        act = [json.dumps(t.get("action") or t.get("move"), sort_keys=True) for t in turns]
        s["turns"].append(len(turns))
        s["illegal_turns"].append(sum(ill))
        s["legal_moves"].append(len(turns) - sum(ill))
        s["ok"].append(ok)
        s["ok_clean" if not any(ill) else "ok_rejected"].append(ok)
        s["first_illegal"].append(ill[0])
        s["ok_first_illegal" if ill[0] else "ok_first_legal"].append(ok)
        seen = set()
        for k, (i, a) in enumerate(zip(ill, act)):
            if i:
                s["seen_before"].append(a in seen)
                if k + 1 < len(turns):
                    s["next_same"].append(act[k + 1] == a)
                    s["next_illegal"].append(ill[k + 1])
                seen.add(a)
        if not ok:
            s["failed_no_legal"].append(sum(ill) == len(turns))
            s["failed_illegal_share"].append(sum(ill) / len(turns))


def m(x):
    return "%.3f" % np.mean(x) if len(x) else "    -"


print("%-10s %8s %10s %12s %12s %12s %12s %10s %12s %12s %12s %12s" % (
    "family", "episodes", "turns/ep", "rejected", "next = same", "next also", "rejected", "success", "success,", "success,",
    "failed eps:", "failed eps:"))
print("%-10s %8s %10s %12s %12s %12s %12s %10s %12s %12s %12s %12s" % (
    "", "", "", "share of", "move", "rejected", "earlier in", "", "no move", ">= 1 move", "no legal", "share of"))
print("%-10s %8s %10s %12s %12s %12s %12s %10s %12s %12s %12s %12s" % (
    "", "", "", "turns", "", "", "the episode", "", "rejected", "rejected", "move at all", "turns rejected"))
for fam in sorted(S):
    s = S[fam]
    print("%-10s %8d %10.2f %12.3f %12s %12s %12s %10s %12s %12s %12s %12s" % (
        fam, len(s["ok"]), np.mean(s["turns"]), np.sum(s["illegal_turns"]) / np.sum(s["turns"]), m(s["next_same"]), m(s["next_illegal"]),
        m(s["seen_before"]), m(s["ok"]), "%s (n %d)" % (m(s["ok_clean"]), len(s["ok_clean"])),
        "%s (n %d)" % (m(s["ok_rejected"]), len(s["ok_rejected"])), m(s["failed_no_legal"]), m(s["failed_illegal_share"])))
print("\nfirst move rejected -> episode success, against first move accepted")
for fam in sorted(S):
    s = S[fam]
    if any(s["first_illegal"]):
        print("%-10s first move rejected in %.3f of episodes; success %s (n %d) against %s (n %d)" % (
            fam, np.mean(s["first_illegal"]), m(s["ok_first_illegal"]), len(s["ok_first_illegal"]),
            m(s["ok_first_legal"]), len(s["ok_first_legal"])))
