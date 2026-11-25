"""con1: does a consequence-trained adapter still do the real task? (owner's question, 11-25)

    python bench_format.py --arm NAME FILE [FILE ...] --arm pref1 FILE [FILE ...] [--variants ...]

Per arm and family, over every benchmark turn (rollout episode records, v3r --open-rules, T = 0):
  rationale  share of turns whose reply has a non-empty reasoning before the answer
  parsed     share of turns whose answer yields a move the harness reads (fail not in no_action / unparsed /
             think_unclosed / error)
  leak       share of turns whose reasoning or answer contains text of a result message or answer format
             (CHECK RESULT, CONSTRAINT STATUS (after this move), EFFECT OF THIS MOVE, Feasible after this move,
             VERDICT:, FEASIBLE AFTER:, VIOLATED AFTER:, EXPECTED RESULT, CHANGES:, METRICS (before -> after))
  tokens     median completion tokens per turn
  f20        feasible@20 per episode (pooled over the arm's runs)
feasible@20 against pref1 is compared with scripts/rl_probe/ab_pooled.py (paired by problem).
"""
import argparse
import collections
import json
import statistics

LEAK = ("CHECK RESULT", "CONSTRAINT STATUS (after this move)", "EFFECT OF THIS MOVE", "Feasible after this move",
        "VERDICT:", "FEASIBLE AFTER:", "VIOLATED AFTER:", "EXPECTED RESULT", "CHANGES:", "METRICS (before -> after)")
PARSE_FAILS = ("no_action", "unparsed", "think_unclosed", "error")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", action="append", required=True, help="NAME FILE [FILE ...]")
    ap.add_argument("--variants", nargs="*", default=None)
    a = ap.parse_args(argv)
    print("%-8s %-10s %6s %6s %9s %7s %7s %8s %6s" % ("arm", "family", "eps", "turns", "rationale", "parsed", "leak",
                                                    "tok_med", "f20"))
    for arm in a.arm:
        name, files = arm[0], arm[1:]
        by = collections.defaultdict(collections.Counter)
        toks = collections.defaultdict(list)
        for p in files:
            for line in open(p, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                ep = r.get("episode") or r
                v = ep["variant"]
                if (a.variants and v not in a.variants) or ep.get("crashed") or ep.get("feasible_start"):
                    continue
                c = by[v]
                c["eps"] += 1
                c["f20"] += bool(ep.get("feasible@20"))
                for t in ep.get("turns") or []:
                    c["turns"] += 1
                    c["rationale"] += bool((t.get("reasoning") or "").strip())
                    c["parsed"] += (t.get("fail") or "") not in PARSE_FAILS
                    txt = (t.get("reasoning") or "") + "\n" + (t.get("answer") or "")
                    c["leak"] += any(s in txt for s in LEAK)
                    if t.get("completion_tokens"):
                        toks[v].append(int(t["completion_tokens"]))
        tot = collections.Counter()
        for v in sorted(by):
            c = by[v]
            tot.update(c)
            print("%-8s %-10s %6d %6d %9.3f %7.3f %7.3f %8s %6.3f" % (
                name, v, c["eps"], c["turns"], c["rationale"] / max(1, c["turns"]), c["parsed"] / max(1, c["turns"]),
                c["leak"] / max(1, c["turns"]), int(statistics.median(toks[v])) if toks[v] else "-",
                c["f20"] / max(1, c["eps"])))
        print("%-8s %-10s %6d %6d %9.3f %7.3f %7.3f %8s %6.3f" % (
            name, "ALL", tot["eps"], tot["turns"], tot["rationale"] / max(1, tot["turns"]),
            tot["parsed"] / max(1, tot["turns"]), tot["leak"] / max(1, tot["turns"]), "-", tot["f20"] / max(1, tot["eps"])))


if __name__ == "__main__":
    main()
