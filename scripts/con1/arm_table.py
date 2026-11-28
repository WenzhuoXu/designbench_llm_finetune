"""Thinking arms side by side on the same questions, split by how each think block ended (CPU; con1-2m-rows, 11-28).

    python arm_table.py --queries queries_v3_inv2cs.jsonl --flags flags_v3_inv2cs.tsv [--subset subset_300.tsv] \\
        --arm LABEL=preds.jsonl:FORM [--arm ...] --out REPORT.txt

Questions: those every arm answered (and in --subset). Truth from --queries (inventory2 on the common-seed truth when
queries_v3_inv2cs.jsonl is given). Per arm:
  think_end counts per family; think-token quartiles of closed blocks per family
  then, flagged questions out (no-op, off-range), for all / closed / cut blocks, per group (trained families =
  heldout_problem, unseen = heldout_family), pooled over the group's questions and as the mean over its families:
    verdict    share with the right verdict (applied vs refused; unreadable wrong)
    feas       applied truth and applied answer: share with the right FEASIBLE AFTER, and the balanced accuracy
               (mean of the feasible and the infeasible recall)
    dir        coupled lines (metrics not in facets.DIRECT) of applied-applied questions on which the answer writes a
               direction: accuracy (plain) and balanced over the true classes; 'absent' = share of those lines with no
               direction written
  each number from facets.pred_facets (fixed reader) and, in brackets, from indep_score.read_answer (second scorer)
  the flagged questions: counts and verdict accuracy per flag
"""
import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dir_split as DS  # noqa: E402  (read, load, tdir; facets from CON1_LIB, indep_score)

F = DS.F


def stats(qs, keys, rec, fmt, reader):
    c = collections.Counter()
    for (sid, qi) in keys:
        q = qs[sid]
        m = q["moves"][qi]
        v, f, d = DS.read(rec[(sid, qi)].get("text"), fmt, q, reader)
        tv = "refused" if m["verdict"] == "refused" else "applied"
        c["n"] += 1
        c["verdict"] += v == tv
        if tv != "applied" or v != "applied":
            continue
        tf = bool(m["feasible"])
        c["feas_n"] += 1
        c["feas_hit"] += f == tf
        c["feas_n_%s" % tf] += 1
        c["feas_hit_%s" % tf] += f == tf
        direct = F.DIRECT.get(q["variant"], set())
        for k, b0 in q["metrics0"].items():
            if k not in m["metrics_after"] or k in direct:
                continue
            td = DS.tdir(float(b0), float(m["metrics_after"][k]))
            c["lines"] += 1
            if k not in d:
                c["absent"] += 1
                continue
            c["dir_n"] += 1
            c["dir_hit"] += d[k] == td
            c["dn_" + td] += 1
            c["dh_" + td] += d[k] == td
    return c


def numbers(c):
    nan = float("nan")
    r = lambda a, b: c[a] / c[b] if c[b] else nan  # noqa: E731
    rec = [c["feas_hit_%s" % t] / c["feas_n_%s" % t] for t in (True, False) if c["feas_n_%s" % t]]
    drec = [c["dh_" + x] / c["dn_" + x] for x in ("up", "down", "same") if c["dn_" + x]]
    return {"verdict": r("verdict", "n"), "feas": r("feas_hit", "feas_n"),
            "feas_BA": sum(rec) / len(rec) if rec else nan, "dir": r("dir_hit", "dir_n"),
            "dir_BA": sum(drec) / len(drec) if drec else nan, "absent": r("absent", "lines")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--flags", required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    qs = {json.loads(l)["sid"]: json.loads(l) for l in open(a.queries)}
    flags = collections.defaultdict(set)
    for l in open(a.flags):
        if l.strip():
            s, j, f = l.rstrip("\n").split("\t")
            flags[(s, int(j))].add(f)
    sub = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open(a.subset) if l.strip()} if a.subset else None
    arms = []
    for spec in a.arm:
        lab, rest = spec.split("=", 1)
        path, fmt = rest.rsplit(":", 1)
        arms.append((lab, DS.load(path), fmt))
    keys = set.intersection(*[set(r) for _, r, _ in arms])
    keys = sorted(k for k in keys if k[0] in qs and (sub is None or k in sub))
    out = []
    w = out.append
    w("# arms on the same %d questions; truth %s; flags %s" % (len(keys), a.queries, a.flags))
    for lab, recs, fmt in arms:
        w("# %-34s %s (form %s)" % (lab, [p for p in a.arm if p.startswith(lab + "=")][0].split("=", 1)[1], fmt))
    fams = sorted({qs[k[0]]["variant"] for k in keys})
    w("\n## think_end per family (closed / budget / other) and think tokens of closed blocks (q25 / median / q75)")
    for lab, recs, fmt in arms:
        w("  %s" % lab)
        for f in fams + ["all"]:
            ks = [k for k in keys if f == "all" or qs[k[0]]["variant"] == f]
            te = collections.Counter(str(recs[k].get("think_end")) for k in ks)
            ls = sorted(recs[k].get("think_tokens") or 0 for k in ks if recs[k].get("think_end") == "closed")
            qt = "%d / %d / %d" % (ls[len(ls) // 4], ls[len(ls) // 2], ls[3 * len(ls) // 4]) if ls else "-"
            w("    %-11s n %4d  closed %4d (%.2f)  budget %4d  other %d  closed tokens %s" % (
                f, len(ks), te["closed"], te["closed"] / max(1, len(ks)), te["budget"],
                len(ks) - te["closed"] - te["budget"], qt))
    head = [k for k in keys if not flags.get(k)]
    for g in ("heldout_problem", "heldout_family"):
        gk = [k for k in head if qs[k[0]]["group"] == g]
        gf = sorted({qs[k[0]]["variant"] for k in gk})
        w("\n## %s, flagged questions out (%d questions); pooled over questions | mean over families; facets [indep]" % (g, len(gk)))
        w("  %-26s %-7s %5s | %-15s %-15s %-15s %-15s %-15s %-15s" % ("arm", "blocks", "n", "verdict", "feas", "feas_BA",
                                                                       "dir", "dir_BA", "absent"))
        for lab, recs, fmt in arms:
            for part in ("all", "closed", "cut"):
                ks = [k for k in gk if part == "all" or (recs[k].get("think_end") == "closed") == (part == "closed")]
                if not ks:
                    continue
                vals = {}
                for reader in ("facets", "indep"):
                    pooled = numbers(stats(qs, ks, recs, fmt, reader))
                    per = [numbers(stats(qs, [k for k in ks if qs[k[0]]["variant"] == f], recs, fmt, reader)) for f in gf
                           if any(qs[k[0]]["variant"] == f for k in ks)]
                    mean = {s: (sum(p[s] for p in per if not math.isnan(p[s])) /
                                max(1, sum(1 for p in per if not math.isnan(p[s])))) for s in pooled}
                    vals[reader] = (pooled, mean)
                cells = []
                for s in ("verdict", "feas", "feas_BA", "dir", "dir_BA", "absent"):
                    p, m = vals["facets"][0][s], vals["facets"][1][s]
                    pi, mi = vals["indep"][0][s], vals["indep"][1][s]
                    cells.append("%.2f|%.2f [%.2f|%.2f]" % (p, m, pi, mi))
                w("  %-26s %-7s %5d | %s" % (lab, part, len(ks), "  ".join(cells)))
    for flag in ("no-op", "off-range"):
        fk = [k for k in keys if flag in flags.get(k, ())]
        if not fk:
            continue
        w("\n## flagged: %s (%d questions; truth: applied%s)" % (flag, len(fk), ", every metric unchanged" if flag == "no-op" else ""))
        for lab, recs, fmt in arms:
            c = stats(qs, fk, recs, fmt, "facets")
            ci = stats(qs, fk, recs, fmt, "indep")
            te = collections.Counter(str(recs[k].get("think_end")) for k in fk)
            w("  %-26s verdict right %d of %d [%d]; think closed %d" % (lab, c["verdict"], c["n"], ci["verdict"], te["closed"]))
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("ARM_TABLE_DONE")


if __name__ == "__main__":
    main()
