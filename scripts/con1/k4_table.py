"""K4 table (CPU; con1-2m-rows, 11-28, parent's order): is a right answer among the base model's own samples when its
temperature-0 answer (arm D) is wrong?

    python k4_table.py --queries queries_v3_inv2cs.jsonl --qflags flags_v3_inv2cs.tsv --lflags line_flags_v3_inv2cs.tsv \\
        --comp comp_preds.pkl --comp-best comp_best.pkl --armd preds_base_v3_dir_8k.jsonl \\
        --k4 preds_base_v3_dir_8k_k4_t08.jsonl [--samples 4] [--reader facets|indep] [--boot 1000] --out REPORT.txt

Questions: those of the K4 file with every sample answered (no error record) and an arm-D answer, without a question
flag, truth applied. Lines: true up/down lines of coupled metrics (not facets.DIRECT) without a line flag (the
'headline' line set of arm_split.py); also the same restricted to the lines the no-physics TABLE gets wrong. A sample
or arm D 'gives' a line when it answers applied and writes a direction for the metric; an absent line or a refused
answer is wrong and counts as the word 'none'.
Per family and pooled per group (trained = heldout_problem, unseen = heldout_family):
  accuracy of each sample, of their mean, of the majority of the samples (the word with strictly more votes than any
  other word, 'none' included; a tie is no majority and is wrong), arm D and TABLE; majority - D and mean - D (se:
  bootstrap over states, seed 0)
  on lines arm D got wrong: share with >= 1 sample right, >= 2 right, all samples wrong
  on lines arm D got right: share with >= 1 sample wrong
  agreement: share of lines where all samples give the same word ('none' included)
Defects block: think_end per sample and family (closed / budget / other) against arm D on the same questions; the share
of refused answers on truth-applied questions per sample against arm D; error records.
"""
import argparse
import collections
import json
import math
import os
import pickle
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/storage/wxu/con1/full_eval")
import dir_split as DS  # noqa: E402  (tdir; facets from CON1_LIB, indep_score)

F, IS = DS.F, DS.IS


def read(text, q, reader):
    v, _, d = DS.read(text, "dir", q, reader)
    return v, d


def main():
    ap = argparse.ArgumentParser()
    for x in ("queries", "qflags", "lflags", "comp", "comp_best", "armd", "k4", "out"):
        ap.add_argument("--" + x.replace("_", "-"), required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--reader", choices=("facets", "indep"), default="facets")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    S = a.samples
    qs = {}
    for l in open(a.queries, encoding="utf-8"):
        q = json.loads(l)
        qs[q["sid"]] = q
    qflag = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open(a.qflags) if l.strip()}
    lflag = set()
    for l in open(a.lflags):
        if l.strip():
            s, j, k, _ = l.rstrip("\n").split("\t")
            lflag.add((s, int(j), k))
    comp = pickle.load(open(a.comp, "rb"))
    best = pickle.load(open(a.comp_best, "rb"))
    armd = DS.load(a.armd)
    k4 = collections.defaultdict(dict)
    errors = collections.Counter()
    for l in open(a.k4, encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            errors["torn line"] += 1
            continue
        if str(r.get("finish", "")).startswith("error"):
            errors["error record"] += 1
            continue
        k4[(r["sid"], r["qi"])].setdefault(int(r["sample"]), r)
    keys = sorted(k for k, v in k4.items() if len(v) == S and k in armd and k[0] in qs)
    out = []
    w = out.append
    w("# k4_table, reader %s; K4 file %s; arm D %s; %d questions with all %d samples (%d with fewer); %s" % (
        a.reader, a.k4, a.armd, len(keys), S, sum(1 for v in k4.values() if len(v) < S), dict(errors) or "no errors"))
    w("# truth %s; question flags %s; line flags %s; TABLE %s (best per family)" % (a.queries, a.qflags, a.lflags, a.comp))
    fams = sorted({qs[k[0]]["variant"] for k in keys})
    # defects: think_end and refusals
    w("\n## defects: think_end (closed share) per sample vs arm D on the same questions; refused answers on truth-applied "
      "questions (question flags out)")
    for f in fams + ["ALL"]:
        ks = [k for k in keys if f == "ALL" or qs[k[0]]["variant"] == f]
        cl = ["%.3f" % (sum(k4[k][s].get("think_end") == "closed" for k in ks) / len(ks)) for s in range(1, S + 1)]
        bud = ["%.3f" % (sum(k4[k][s].get("think_end") == "budget" for k in ks) / len(ks)) for s in range(1, S + 1)]
        cd = sum(armd[k].get("think_end") == "closed" for k in ks) / len(ks)
        ka = [k for k in ks if k not in qflag and qs[k[0]]["moves"][k[1]]["verdict"] != "refused"]
        ref = ["%.3f" % (sum(read(k4[k][s].get("text"), qs[k[0]], a.reader)[0] != "applied" for k in ka) / max(1, len(ka)))
               for s in range(1, S + 1)]
        rd = sum(read(armd[k].get("text"), qs[k[0]], a.reader)[0] != "applied" for k in ka) / max(1, len(ka))
        w("  %-11s n %4d | closed: samples %s, arm D %.3f | budget: samples %s | not-applied answers on %d applied "
          "questions: samples %s, arm D %.3f" % (f, len(ks), " ".join(cl), cd, " ".join(bud), len(ka), " ".join(ref), rd))
    # line statistics per state
    C = collections.defaultdict(collections.Counter)
    for k in keys:
        q = qs[k[0]]
        m = q["moves"][k[1]]
        if k in qflag or m["verdict"] == "refused":
            continue
        fam = q["variant"]
        rd = read(armd[k].get("text"), q, a.reader)
        rs = [read(k4[k][s].get("text"), q, a.reader) for s in range(1, S + 1)]
        direct = F.DIRECT.get(fam, set())
        for met, b0 in q["metrics0"].items():
            av = m["metrics_after"].get(met)
            if not (isinstance(av, (int, float)) and isinstance(b0, (int, float))) or met in direct \
                    or (k[0], k[1], met) in lflag:
                continue
            td = DS.tdir(float(b0), float(av))
            if td == "same":
                continue
            word = lambda r: (r[1].get(met) if r[0] == "applied" and met in r[1] else "none")  # noqa: E731
            wd, ws = word(rd), [word(r) for r in rs]
            votes = collections.Counter(ws)
            top = votes.most_common()
            maj = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else None
            tab_ok = comp.get((k, met), {}).get(best.get(fam)) == td
            nright = sum(x == td for x in ws)
            for part in ("all", "tablewrong"):
                if part == "tablewrong" and tab_ok:
                    continue
                c = C[(part, fam, k[0])]
                c["n"] += 1
                c["d"] += wd == td
                c["tab"] += tab_ok
                c["maj"] += maj == td
                c["tie"] += maj is None
                c["agree"] += len(votes) == 1
                c["samples_right"] += nright
                for s in range(S):
                    c["s%d" % (s + 1)] += ws[s] == td
                if wd == td:
                    c["dr_n"] += 1
                    c["dr_any_wrong"] += nright < S
                else:
                    c["dw_n"] += 1
                    c["dw_ge1"] += nright >= 1
                    c["dw_ge2"] += nright >= 2
                    c["dw_all_wrong"] += nright == 0
    r_ = lambda c, x, y: c[x] / c[y] if c[y] else float("nan")  # noqa: E731

    def boot(cs, fn, needed):
        tot = sum(cs, collections.Counter())
        pt = fn(tot)
        if len(cs) < 2:
            return pt, float("nan")
        V = np.array([[c[x] for x in needed] for c in cs], dtype=float)
        idx = np.random.default_rng(0).integers(0, len(cs), size=(a.boot, len(cs)))
        sums = V[idx].sum(axis=1)
        dr = [fn(collections.Counter(dict(zip(needed, row)))) for row in sums]
        dr = [x for x in dr if not math.isnan(x)]
        return pt, (statistics.pstdev(dr) if len(dr) > 1 else float("nan"))

    for part, title in (("all", "true up/down lines (headline set)"), ("tablewrong", "lines the TABLE gets wrong")):
        for g, glab in (("heldout_problem", "trained (subset_300 questions)"), ("heldout_family", "unseen (every unflagged question)")):
            gf = [f for f in fams if any(qs[k[0]]["group"] == g and qs[k[0]]["variant"] == f for k in keys)]
            if not gf:
                continue
            w("\n## %s, %s" % (glab, title))
            w("  %-11s %5s | %s  mean   maj  (tie) | D     TABLE | maj-D (se)     mean-D (se)    | D wrong: n, >=1 right, "
              ">=2 right, all wrong | D right: n, >=1 wrong | all %d agree" % (
                  "family", "lines", " ".join("s%d   " % (s + 1) for s in range(S)), S))
            for f in gf + ["POOLED"]:
                cs = [C[x] for x in sorted(C) if x[0] == part and (x[1] == f or (f == "POOLED" and x[1] in gf))]
                t = sum(cs, collections.Counter())
                if not t["n"]:
                    continue
                nd = ["n", "d", "maj", "samples_right"]
                dm = boot(cs, lambda c: r_(c, "maj", "n") - r_(c, "d", "n"), nd)
                dn = boot(cs, lambda c: c["samples_right"] / (S * c["n"]) - r_(c, "d", "n") if c["n"] else float("nan"), nd)
                w("  %-11s %5d | %s  %.3f  %.3f (%.3f) | %.3f %.3f | %+.3f (%.3f) %+.3f (%.3f) | %d, %.3f, %.3f, %.3f | "
                  "%d, %.3f | %.3f" % (
                      f, t["n"], " ".join("%.3f" % r_(t, "s%d" % (s + 1), "n") for s in range(S)),
                      t["samples_right"] / (S * t["n"]), r_(t, "maj", "n"), r_(t, "tie", "n"), r_(t, "d", "n"),
                      r_(t, "tab", "n"), dm[0], dm[1], dn[0], dn[1], t["dw_n"], r_(t, "dw_ge1", "dw_n"),
                      r_(t, "dw_ge2", "dw_n"), r_(t, "dw_all_wrong", "dw_n"), t["dr_n"], r_(t, "dr_any_wrong", "dr_n"),
                      r_(t, "agree", "n")))
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("K4_TABLE_DONE")


if __name__ == "__main__":
    main()
