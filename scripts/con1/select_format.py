"""con1: apply the pre-registered format rule to round 1's facet scores (docs section 11).

    python select_format.py facets_r1.json ROWS_DIR OUT_FORMAT_FILE
The arm with the highest selection score over all families (primary) wins; arms within 1 se of it (the bootstrap se
of the paired difference) count as tied, and among tied arms the one with the fewest answer characters per question
in its training rows wins. Writes the format name (masked / facet / delta / absolute) to OUT_FORMAT_FILE and prints
the table.
"""
import json
import sys

ARMS = {"M": "masked", "F": "facet", "D": "delta", "S": "absolute"}


def chars_per_answer(path):
    n = c = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        for m in r["messages"]:
            if m["role"] == "assistant":
                n += 1
                c += len(m.get("content") or "")
    return c / max(1, n)


def main():
    res = json.load(open(sys.argv[1]))
    rows_dir, out = sys.argv[2], sys.argv[3]
    key = next(k for k in res if k.startswith("group/all"))
    prefix = key.split("/")[1]
    sel = {a: res.get("group/%s/%s/select" % (prefix, a)) for a in ARMS}
    sel = {a: v for a, v in sel.items() if v}
    if not sel:
        raise SystemExit("no arm scores in %s" % sys.argv[1])
    best = max(sel, key=lambda a: sel[a]["value"])
    tied = [best]
    for a in sel:
        if a == best:
            continue
        d = res.get("diff/%s/%s-%s/select" % (prefix, best, a)) or res.get("diff/%s/%s-%s/select" % (prefix, a, best))
        if d and abs(d["value"]) <= d["se"]:
            tied.append(a)
    cost = {a: chars_per_answer("%s/rows_%s.jsonl" % (rows_dir, ARMS[a])) for a in tied}
    win = min(tied, key=lambda a: cost[a])
    for a in sorted(sel, key=lambda a: -sel[a]["value"]):
        print("%s %-9s select %.3f +- %.3f%s" % (a, ARMS[a], sel[a]["value"], sel[a]["se"],
                                                  "  (tied, %.0f chars/answer)" % cost[a] if a in cost else ""))
    print("WINNER %s = %s" % (win, ARMS[win]))
    open(out, "w").write(ARMS[win] + "\n")


if __name__ == "__main__":
    main()
