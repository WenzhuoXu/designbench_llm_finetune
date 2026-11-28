"""Line-level flags of the con1 question set (CPU; con1-2m-rows, 11-28, parent's order A).

    python make_line_flags.py QUERIES OUT.tsv [QUESTION_FLAGS.tsv]

One line per flagged metric line of a truth-applied move: sid<TAB>qi<TAB>metric<TAB>flag. A line may carry several flags.
  exogenous      the state has at least two applied moves, the metric's after-value is the same number for every
                 applied move of the state (relative difference <= 1e-9), and it differs from the before-value
                 (direction up/down under the harness threshold): the outcome does not depend on the move
  categorical    thermal2 hotspot_node (a node label scored as up/down)
  subthreshold   true direction up or down under the harness threshold (relative change > 1e-5) but the relative
                 change is below 1e-3 (0.1%)
Directions as everywhere in con1: 'same' when |after - before| <= max(1e-5 |before|, 1e-12).
Prints counts per family and flag: all lines, and true up/down lines on questions without a question flag (no-op,
off-range; QUESTION_FLAGS, when given), the line set the direction tables use.
"""
import collections
import json
import sys

CATEGORICAL = {"thermal2": {"hotspot_node"}}


def tdir(b, a):
    return "same" if abs(a - b) <= max(1e-5 * abs(b), 1e-12) else ("up" if a > b else "down")


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def main():
    qp, out = sys.argv[1], sys.argv[2]
    qflag = set()
    if len(sys.argv) > 3:
        qflag = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open(sys.argv[3]) if l.strip()}
    rows = []
    cnt = collections.defaultdict(collections.Counter)
    met = collections.defaultdict(collections.Counter)
    for l in open(qp, encoding="utf-8"):
        q = json.loads(l)
        f = q["variant"]
        m0 = q["metrics0"]
        app = [(j, m) for j, m in enumerate(q["moves"]) if m["verdict"] != "refused" and m.get("metrics_after")]
        exo = set()
        if len(app) >= 2:
            for k, b in m0.items():
                vals = [m["metrics_after"].get(k) for _, m in app]
                if not all(num(v) for v in vals) or not num(b):
                    continue
                v0 = float(vals[0])
                if all(abs(float(v) - v0) <= 1e-9 * max(abs(v0), 1e-12) for v in vals) and tdir(float(b), v0) != "same":
                    exo.add(k)
        for j, m in app:
            for k, b in m0.items():
                a = m["metrics_after"].get(k)
                if not (num(a) and num(b)):
                    continue
                td = tdir(float(b), float(a))
                fl = []
                if k in exo:
                    fl.append("exogenous")
                if k in CATEGORICAL.get(f, ()):
                    fl.append("categorical")
                if td != "same" and abs(float(a) - float(b)) < 1e-3 * abs(float(b)):
                    fl.append("subthreshold")
                for x in fl:
                    rows.append((q["sid"], j, k, x))
                    cnt[(f, q["group"])][x] += 1
                    if td != "same" and (q["sid"], j) not in qflag:
                        cnt[(f, q["group"])][x + " ud"] += 1
                        met[(f, x)][k] += 1
    with open(out, "w") as fo:
        for r in rows:
            fo.write("%s\t%d\t%s\t%s\n" % r)
    print("line flags: %d lines -> %s" % (len(rows), out))
    tot = collections.defaultdict(collections.Counter)
    for (f, g), c in sorted(cnt.items()):
        print("  %-11s %-15s %s" % (f, g, ", ".join("%s %d" % (k, v) for k, v in sorted(c.items()))))
        tot[g].update(c)
    for g, c in sorted(tot.items()):
        print("  ALL %-15s %s" % (g, ", ".join("%s %d" % (k, v) for k, v in sorted(c.items()))))
    print("metrics carrying the flag (true up/down lines on unflagged questions):")
    for (f, x), c in sorted(met.items()):
        print("  %-11s %-13s %s" % (f, x, ", ".join("%s %d" % kv for kv in c.most_common(8))))
    print("MAKE_LINE_FLAGS_DONE")


if __name__ == "__main__":
    main()
