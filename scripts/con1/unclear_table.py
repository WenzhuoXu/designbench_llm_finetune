"""Arm D-unclear table (CPU; con1-2m-rows, 11-28, parent's order): can the base say in one run which directions it
cannot determine?

    CON1_LIB=lib_v9 python unclear_table.py --queries queries_v3_inv2cs.jsonl --qflags flags_v3_inv2cs.tsv \\
        --lflags line_flags_v3_inv2cs.tsv --comp comp_preds.pkl --comp-best comp_best.pkl --ref S1A,S1H \\
        --unclear preds_base_v3_dirunc_8k_sub300.jsonl --armd D8K --armd-rep REP_ORCH --armd-rep REP_DRC \\
        [--reader facets|indep] --out REPORT.txt

Questions: subset questions answered by D-unclear, arm D and both arm-D repeats; truth-applied, no question flag.
Lines: coupled metrics (not facets.DIRECT) without a line flag (the headline set); 'moving' = true up/down,
'unchanged' = true same. D-unclear's word per line: up / down / same / unclear (its own class, read for the 'dirunc'
form), or absent (no line, or the answer is not applied).
Tables (per family and pooled per group):
  unclear share among the lines D-unclear writes, on moving and on unchanged lines (+ absent share)
  moving lines, split committed / unclear: D-unclear's accuracy on committed lines; arm D, stage 1 and TABLE on the
    same committed lines; arm D, stage 1 and TABLE on the unclear lines; also restricted to TABLE-wrong lines
  moving lines by the three arm-D runs (D, Orchard repeat, drc repeat): all three right / all three wrong / they disagree
    -> unclear share of D-unclear on each
Defects block: think_end; answers without a VERDICT line at a line start; markdown tables; answers over 300 answer
tokens; answers cut at the answer cap; applied answers whose every line is 'unclear'.
"""
import argparse
import collections
import json
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/storage/wxu/con1/full_eval")
import dir_split as DS  # noqa: E402

F = DS.F


def main():
    ap = argparse.ArgumentParser()
    for x in ("queries", "qflags", "lflags", "comp", "comp_best", "ref", "unclear", "armd", "out"):
        ap.add_argument("--" + x.replace("_", "-"), required=True)
    ap.add_argument("--armd-rep", action="append", required=True)
    ap.add_argument("--reader", choices=("facets", "indep"), default="facets")
    a = ap.parse_args()
    qs = {}
    for l in open(a.queries, encoding="utf-8"):
        q = json.loads(l)
        qs[q["sid"]] = q
    qflag = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open(a.qflags) if l.strip()}
    lflag = {tuple([l.split("\t")[0], int(l.split("\t")[1]), l.split("\t")[2]]) for l in open(a.lflags) if l.strip()}
    comp = pickle.load(open(a.comp, "rb"))
    best = pickle.load(open(a.comp_best, "rb"))
    U = DS.load(a.unclear)
    D = DS.load(a.armd)
    R = [DS.load(p) for p in a.armd_rep]
    S1 = {}
    for p in a.ref.split(","):
        S1.update(DS.load(p))
    keys = sorted(k for k in U if k in D and all(k in r for r in R) and k in S1 and k[0] in qs)
    out = []
    w = out.append
    w("# unclear_table, reader %s; D-unclear %s; arm D %s; repeats %s; %d questions" % (
        a.reader, a.unclear, a.armd, a.armd_rep, len(keys)))
    # defects
    fams = sorted({qs[k[0]]["variant"] for k in keys})
    w("\n## defects (all %d questions)" % len(keys))
    te = collections.Counter(str(U[k].get("think_end")) for k in keys)
    ted = collections.Counter(str(D[k].get("think_end")) for k in keys)
    w("  think_end D-unclear %s; arm D %s" % (dict(te), dict(ted)))
    c = collections.Counter()
    for k in keys:
        t = U[k].get("text") or ""
        c["no VERDICT line at a line start"] += not re.search(r"^\**VERDICT\**\s*:", t, re.M)
        c["markdown table"] += ("|--" in t or "| --" in t)
        c["answer tokens > 300"] += (U[k].get("answer_tokens") or 0) > 300
        c["cut at the answer cap"] += U[k].get("finish") == "length"
        v, _, d = DS.read(t, "dirunc", qs[k[0]], a.reader)
        if v == "applied":
            c["applied answers"] += 1
            c["applied answers with every written line 'unclear'"] += bool(d) and all(x == "unclear" for x in d.values())
            c["applied answers with at least one 'unclear'"] += any(x == "unclear" for x in d.values())
    w("  %s" % dict(c))
    # lines
    C = collections.defaultdict(collections.Counter)
    for k in keys:
        q = qs[k[0]]
        m = q["moves"][k[1]]
        if k in qflag or m["verdict"] == "refused":
            continue
        fam = q["variant"]
        vu, _, du = DS.read(U[k].get("text"), "dirunc", q, a.reader)
        rd = [DS.read(x[k].get("text"), "dir", q, a.reader) for x in [D] + R]
        vs, _, ds = DS.read(S1[k].get("text"), S1[k].get("target", "facetm"), q, a.reader)
        for met, b0 in q["metrics0"].items():
            av = m["metrics_after"].get(met)
            if not (isinstance(av, (int, float)) and isinstance(b0, (int, float))) or met in F.DIRECT.get(fam, set()) \
                    or (k[0], k[1], met) in lflag:
                continue
            td = DS.tdir(float(b0), float(av))
            wu = du.get(met) if vu == "applied" and met in du else None
            x = C[fam]
            kind = "moving" if td != "same" else "unchanged"
            x[kind + " lines"] += 1
            if wu is None:
                x[kind + " absent"] += 1
                continue
            x[kind + " written"] += 1
            x[kind + " unclear"] += wu == "unclear"
            if td == "same":
                continue
            dok = [r[0] == "applied" and r[2].get(met) == td for r in rd]
            sok = vs == "applied" and ds.get(met) == td
            tok = comp.get((k, met), {}).get(best.get(fam)) == td
            for part in ("all", "tablewrong"):
                if part == "tablewrong" and tok:
                    continue
                side = "unc" if wu == "unclear" else "com"
                x["%s %s n" % (part, side)] += 1
                x["%s %s U" % (part, side)] += wu == td
                x["%s %s D" % (part, side)] += dok[0]
                x["%s %s S1" % (part, side)] += sok
                x["%s %s TAB" % (part, side)] += tok
            grp = "3 right" if all(dok) else "3 wrong" if not any(dok) else "disagree"
            x["runs %s n" % grp] += 1
            x["runs %s unclear" % grp] += wu == "unclear"
    r_ = lambda x, a_, b_: x[a_] / x[b_] if x[b_] else float("nan")  # noqa: E731
    for g, glab in (("heldout_problem", "trained"), ("heldout_family", "unseen")):
        gf = [f for f in fams if qs[[k for k in keys if qs[k[0]]["variant"] == f][0][0]]["group"] == g]
        if not gf:
            continue
        rows = [(f, C[f]) for f in gf] + [("POOLED " + glab, sum((C[f] for f in gf), collections.Counter()))]
        w("\n## %s: unclear share among written lines (moving | unchanged); absent share" % glab)
        for f, x in rows:
            w("  %-18s moving %4d: unclear %.3f absent %.3f | unchanged %4d: unclear %.3f absent %.3f" % (
                f, x["moving lines"], r_(x, "moving unclear", "moving written"), r_(x, "moving absent", "moving lines"),
                x["unchanged lines"], r_(x, "unchanged unclear", "unchanged written"),
                r_(x, "unchanged absent", "unchanged lines")))
        for part, title in (("all", "moving lines"), ("tablewrong", "moving lines the TABLE gets wrong")):
            w("\n## %s, %s: committed (D-unclear wrote up/down/same) vs unclear" % (glab, title))
            w("  %-18s | committed n: D-unclear  arm D  S1   TABLE | unclear n: arm D  S1   TABLE" % "scope")
            for f, x in rows:
                p = part
                w("  %-18s | %5d: %.3f  %.3f  %.3f  %.3f | %5d: %.3f  %.3f  %.3f" % (
                    f, x["%s com n" % p], r_(x, "%s com U" % p, "%s com n" % p), r_(x, "%s com D" % p, "%s com n" % p),
                    r_(x, "%s com S1" % p, "%s com n" % p), r_(x, "%s com TAB" % p, "%s com n" % p),
                    x["%s unc n" % p], r_(x, "%s unc D" % p, "%s unc n" % p), r_(x, "%s unc S1" % p, "%s unc n" % p),
                    r_(x, "%s unc TAB" % p, "%s unc n" % p)))
        w("\n## %s: moving lines by the three arm-D runs (D, Orchard repeat, drc repeat) -> D-unclear's unclear share" % glab)
        for f, x in rows:
            w("  %-18s " % f + " | ".join("%s %4d: unclear %.3f" % (gname, x["runs %s n" % gname],
                                                                    r_(x, "runs %s unclear" % gname, "runs %s n" % gname))
                                          for gname in ("3 right", "3 wrong", "disagree")))
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("UNCLEAR_TABLE_DONE")


if __name__ == "__main__":
    main()
