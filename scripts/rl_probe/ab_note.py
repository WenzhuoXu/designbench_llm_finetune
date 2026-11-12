"""Two evaluation runs of the same policy on the same problems, the second with the harness's
--reject-note (dataxp_policy_eval_v3n.py): per family, feasibility at 12 and 20 turns, the share of
turns spent on refused moves, how often the move after a refusal is the same move, episodes frozen
on one refused move to the last turn, and the paired outcome per problem (feasible in one arm only).

usage: python ab_note.py <plain episodes.jsonl> <note episodes.jsonl> [--run 1]
"""
import argparse
import collections
import json
import math

ap = argparse.ArgumentParser()
ap.add_argument("plain")
ap.add_argument("note")
ap.add_argument("--run", type=int, default=1)
a = ap.parse_args()


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        ep = r.get("episode") or r
        # an episode with no turns (the harness ends it before the first move) stays in: it is in the
        # launcher's own denominators
        if r.get("member", a.run) != a.run or ep.get("crashed"):
            continue
        out[(ep["variant"], ep["problem_id"])] = ep
    return out


def stats(ep):
    turns = ep.get("turns") or []
    ill = [str(t.get("fail") or "").startswith("illegal") for t in turns]
    act = [json.dumps(t.get("action") or t.get("move"), sort_keys=True) for t in turns]
    nxt = [(act[k + 1] == act[k], ill[k + 1]) for k in range(len(turns) - 1) if ill[k]]
    k = len(turns) - 1
    while k > 0 and ill[k] and ill[k - 1] and act[k] == act[k - 1]:
        k -= 1
    return {"f12": bool(ep.get("feasible@12")), "f20": bool(ep.get("feasible@20")), "turns": len(turns), "refused": sum(ill),
            "next_same": sum(x[0] for x in nxt), "next_refused": sum(x[1] for x in nxt), "n_next": len(nxt),
            "frozen": bool(turns and ill[-1] and len(turns) - k >= 3), "noted": sum(1 for t in turns if t.get("reject_note")),
            "tokens": sum(t.get("completion_tokens") or 0 for t in turns)}


P, N = load(a.plain), load(a.note)
both = sorted(set(P) & set(N))
print("episodes: plain %d, note %d, problems in both %d (run %d)" % (len(P), len(N), len(both), a.run))
fams = sorted({k[0] for k in both})
hdr = "%-10s %4s | %-7s %7s %7s %9s %10s %8s %7s | %s"
print(hdr % ("family", "n", "arm", "feas@12", "feas@20", "refused", "next same", "next", "frozen", "feasible@20 in one arm only"))
print(hdr % ("", "", "", "", "", "turns", "move", "refused", "eps", "(plain only / note only), sign test p"))
for fam in fams + ["all"]:
    keys = [k for k in both if fam == "all" or k[0] == fam]
    sp, sn = [stats(P[k]) for k in keys], [stats(N[k]) for k in keys]
    po = sum(x["f20"] and not y["f20"] for x, y in zip(sp, sn))
    no = sum(y["f20"] and not x["f20"] for x, y in zip(sp, sn))
    m = po + no
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(0, min(po, no) + 1)) / 2 ** m) if m else 1.0
    for name, s in (("plain", sp), ("note", sn)):
        tot = lambda c: sum(x[c] for x in s)                                    # noqa: E731
        print("%-10s %4d | %-7s %7.3f %7.3f %4d/%-5d %5d/%-5d %4d/%-4d %7d | %s" % (
            fam if name == "plain" else "", len(keys), name, tot("f12") / len(keys), tot("f20") / len(keys), tot("refused"), tot("turns"),
            tot("next_same"), tot("n_next"), tot("next_refused"), tot("n_next"), tot("frozen"),
            ("%d / %d, p = %.3f" % (po, no, p)) if name == "plain" else
            "prompts carrying a note %d; completion tokens %.2fM (plain %.2fM)" % (tot("noted"), tot("tokens") / 1e6, sum(x["tokens"] for x in sp) / 1e6)))
