"""Per-variant turn statistics of a --best-of K benchmark output (harness v3rk).

Per variant: episodes (crashed / feasible-start left out), feasible@12 / @20, turns played, and over turns:
candidates applied / refused / unparsed, distinct moves among the K, turns where at least one candidate makes the
design feasible, turns where every candidate is refused or unparsed (reply 0 kept), and turns where reply 0
(one T=0.8 sample, the closest stand-in for a single draw) would have made the design feasible while the kept one did.

usage: python bestk_stats.py FILE
"""
import collections
import json
import sys

S = collections.defaultdict(collections.Counter)
for line in open(sys.argv[1], encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    ep = r.get("episode") or {}
    if r.get("rl_error") not in (None, "calls_turns_mismatch") or ep.get("crashed") or ep.get("feasible_start"):
        continue
    v = ep["variant"]
    s = S[v]
    s["episodes"] += 1
    s["f12"] += bool(ep.get("feasible@12"))
    s["f20"] += bool(ep.get("feasible@20"))
    for t in ep.get("turns") or []:
        b = t.get("best_of")
        if not b:
            s["turns_without_best_of"] += 1
            continue
        s["turns"] += 1
        c = b["cands"]
        s["cands"] += len(c)
        s["applied"] += sum(1 for x in c if "phi" in x)
        s["refused"] += sum(1 for x in c if "refused" in x)
        s["unparsed"] += sum(1 for x in c if x.get("parse"))
        s["distinct"] += len({x.get("move") for x in c if x.get("move")})
        s["any_feasible"] += any(x.get("feasible") for x in c)
        s["none_applied"] += b["applied"] == 0
        s["pick_nonzero"] += b["pick"] != 0
        s["r0_feasible"] += bool(c and c[0].get("feasible"))
        s["kept_feasible"] += bool(t.get("feasible_after"))

print("%-10s %4s %6s %6s %6s | %6s %6s %6s %6s %6s %6s %6s %6s" % (
    "variant", "n", "f@12", "f@20", "turns", "appl", "refus", "unpars", "distin", "anyF", "noneA", "r0F", "keptF"))
for v in sorted(S):
    s = S[v]
    n, T, C = s["episodes"], max(s["turns"], 1), max(s["cands"], 1)
    print("%-10s %4d %6.3f %6.3f %6.1f | %6.3f %6.3f %6.3f %6.2f %6.3f %6.3f %6.3f %6.3f" % (
        v, n, s["f12"] / max(n, 1), s["f20"] / max(n, 1), s["turns"] / max(n, 1),
        s["applied"] / C, s["refused"] / C, s["unparsed"] / C, s["distinct"] / T, s["any_feasible"] / T,
        s["none_applied"] / T, s["r0_feasible"] / T, s["kept_feasible"] / T))
    if s["turns_without_best_of"]:
        print("   %d turns without a best_of record" % s["turns_without_best_of"])
