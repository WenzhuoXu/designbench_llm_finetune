"""Thinking arms against stage 1 and a no-physics table, the 11-28 scoring (CPU; con1-2m-rows, parent's order A).

    python arm_split.py --queries queries_v3_inv2cs.jsonl --qflags flags_v3_inv2cs.tsv --lflags line_flags_v3_inv2cs.tsv \\
        --ref s1_trained.jsonl,s1_unseen.jsonl --comp comp_preds.pkl --comp-best comp_best.pkl \\
        --arm LABEL=preds.jsonl:FORM [--arm ...] [--subset S.tsv] [--select LABEL:closed|budget ...] \\
        [--reader facets|indep] [--boot 1000] --out REPORT.txt

Questions: those every arm and stage 1 answered (and in --subset, and meeting every --select LABEL:END, END = the
arm's think_end). Truth from --queries. Question flags (no-op, off-range) apart; the rest is the headline set.
Per arm, stage 1 ('S1') and, for directions, the no-physics table ('TABLE'):
  think_end per family; think tokens of closed blocks (q25 / median / q75)
  verdict      share with the right verdict (applied vs refused; unreadable wrong)
  feasibility  truth-applied headline questions, every answer counted (refused, unreadable or a missing line = wrong):
               primary  'VIOLATED AFTER is none' (none = feasible); accuracy, balanced accuracy, both recalls
               second   the FEASIBLE AFTER line (labelled row); always-'no' accuracy on the same questions
  direction    true up/down lines of truth-applied headline questions, an absent line or a refused answer = wrong:
               accuracy per model; TABLE = the leave-one-problem-out no-physics table of the attack check
               (review_tmp/v1128D/attack/a2_comp.py: verb / verb+sign / verb+element+sign, the best per family, read
               from its comp_preds.pkl / comp_best.pkl); arm - S1, arm - TABLE, arm - first arm (se: bootstrap over
               states, 1,000 draws, seed 0); accuracy on the lines TABLE gets wrong; absent share
               line sets: 'headline' = coupled metrics (not facets.DIRECT) without a line flag; 'all' = every metric
               without a line flag; then each line flag apart (exogenous, categorical, subthreshold)
  3-class      headline lines (true up / down / same) on which the model writes a direction: balanced and plain
Readers: 'facets' = facets.pred_facets (CON1_LIB), 'indep' = indep_score.read_answer; presence of the FEASIBLE AFTER and
VIOLATED AFTER lines is read from the text (markdown '**' and backticks dropped).
"""
import argparse
import collections
import json
import math
import os
import pickle
import re
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/storage/wxu/con1/full_eval")
import dir_split as DS  # noqa: E402  (load, tdir; facets from CON1_LIB, indep_score)

F, IS = DS.F, DS.IS
SETS = ("headline", "all", "exogenous", "categorical", "subthreshold")
FEAS_LINE = re.compile(r"FEASIBLE AFTER\s*:\s*(\w+)", re.I)
VIOL_LINE = re.compile(r"VIOLATED AFTER\s*:", re.I)


def read(text, fmt, q, reader):
    """(verdict, feasible by the FEASIBLE line or None, feasible by 'VIOLATED AFTER is none' or None, {metric: dir})"""
    t = (text or "").replace("**", "").replace("`", "")
    if reader == "facets":
        pf = F.pred_facets(text, fmt, q)
        reg, viol, dirs = pf["regime"], pf.get("violated"), dict(pf.get("dir") or {})
    else:
        IS.FMT = "facetm" if fmt in ("facetm", "facet") else fmt
        p = IS.read_answer(text, q["metrics0"])
        reg, viol, dirs = p["regime"], p.get("viol"), dict(p.get("dir") or {})
    if reg == "unreadable":
        return "unreadable", None, None, {}
    if not reg.startswith("applied"):
        return "refused", None, None, {}
    fl = FEAS_LINE.search(t)
    fline = (fl.group(1).lower().startswith("y")) if fl else None
    vnone = (not viol) if VIOL_LINE.search(t) else None
    return "applied", fline, vnone, dirs


def se_of(draws):
    d = [x for x in draws if not math.isnan(x)]
    return statistics.pstdev(d) if len(d) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qflags", required=True)
    ap.add_argument("--lflags", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--comp", required=True)
    ap.add_argument("--comp-best", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--select", action="append", default=[])
    ap.add_argument("--reader", choices=("facets", "indep"), default="facets")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--keys-out", default=None, help="write the questions used (sid<TAB>qi) to this file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    qs = {}
    for l in open(a.queries, encoding="utf-8"):
        q = json.loads(l)
        qs[q["sid"]] = q
    qflag = collections.defaultdict(set)
    for l in open(a.qflags):
        if l.strip():
            s, j, f = l.rstrip("\n").split("\t")
            qflag[(s, int(j))].add(f)
    lflag = collections.defaultdict(set)
    for l in open(a.lflags):
        if l.strip():
            s, j, k, f = l.rstrip("\n").split("\t")
            lflag[(s, int(j), k)].add(f)
    comp = pickle.load(open(a.comp, "rb"))
    best = pickle.load(open(a.comp_best, "rb"))
    arms = []
    for spec in a.arm:
        lab, rest = spec.split("=", 1)
        path, fmt = rest.rsplit(":", 1)
        arms.append((lab, DS.load(path), fmt, path))
    ref = {}
    for p in a.ref.split(","):
        ref.update(DS.load(p))
    sub = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open(a.subset) if l.strip()} if a.subset else None
    sel = [s.split(":") for s in a.select]
    recs = {lab: rr for lab, rr, _, _ in arms}
    keys = set(ref)
    for lab in recs:
        keys &= set(recs[lab])
    keys = sorted(k for k in keys if k[0] in qs and (sub is None or k in sub)
                  and all(recs[lab][k].get("think_end") == end for lab, end in sel))
    if a.keys_out:
        with open(a.keys_out, "w") as fo:
            for k in keys:
                fo.write("%s\t%d\n" % k)
    models = [lab for lab, _, _, _ in arms] + ["S1"]
    recs["S1"] = ref
    fmts = {lab: fmt for lab, _, fmt, _ in arms}
    out = []
    w = out.append
    w("# arm_split, reader %s; %d questions; truth %s; question flags %s; line flags %s" % (
        a.reader, len(keys), a.queries, a.qflags, a.lflags))
    for lab, _, fmt, path in arms:
        w("#   %s = %s (form %s)" % (lab, path, fmt))
    w("#   S1 = %s; TABLE = %s (best per family: %s)" % (a.ref, a.comp, ", ".join(
        "%s %s" % (f, best[f]) for f in sorted(best) if f not in ("TRAINED", "UNSEEN"))))
    if sub is not None or sel:
        w("#   subset %s; select %s" % (a.subset, a.select))
    fams = sorted({qs[k[0]]["variant"] for k in keys})
    w("\n## think_end per family (closed / budget / other) and think tokens of closed blocks (q25 / median / q75)")
    for lab, rr, _, _ in arms:
        w("  %s" % lab)
        for f in fams + ["all"]:
            ks = [k for k in keys if f == "all" or qs[k[0]]["variant"] == f]
            te = collections.Counter(str(rr[k].get("think_end")) for k in ks)
            ls = sorted(rr[k].get("think_tokens") or 0 for k in ks if rr[k].get("think_end") == "closed")
            qt = "%d / %d / %d" % (ls[len(ls) // 4], ls[len(ls) // 2], ls[3 * len(ls) // 4]) if ls else "-"
            w("    %-11s n %4d  closed %4d (%.3f)  budget %4d  other %d  closed tokens %s" % (
                f, len(ks), te["closed"], te["closed"] / max(1, len(ks)), te["budget"],
                len(ks) - te["closed"] - te["budget"], qt))
    # per-state sufficient statistics
    S = collections.defaultdict(collections.Counter)            # (fam, sid) -> Counter
    for k in keys:
        q = qs[k[0]]
        m = q["moves"][k[1]]
        fam = q["variant"]
        c = S[(fam, k[0])]
        tv = "refused" if m["verdict"] == "refused" else "applied"
        rd = {M: read(recs[M][k].get("text"), fmts.get(M, recs[M][k].get("target", "facetm")), q, a.reader) for M in models}
        if qflag.get(k):
            for fl in qflag[k]:
                c["qf_%s_n" % fl] += 1
                for M in models:
                    c["qf_%s_v_%s" % (fl, M)] += rd[M][0] == tv
            continue
        c["n"] += 1
        for M in models:
            c["v_" + M] += rd[M][0] == tv
        if tv != "applied":
            continue
        tf = bool(m["feasible"])
        c["fq_n"] += 1
        c["fq_n_%s" % tf] += 1
        for M in models:
            v, fline, vnone, _ = rd[M]
            for how, val in (("viol", vnone), ("line", fline)):
                ok = v == "applied" and val is not None and val == tf
                c["f_%s_%s_%s" % (how, M, tf)] += ok
                c["f_%s_%s_missing" % (how, M)] += v == "applied" and val is None
        direct = F.DIRECT.get(fam, set())
        after = m["metrics_after"]
        for met, b0 in q["metrics0"].items():
            av = after.get(met)
            if not (isinstance(av, (int, float)) and isinstance(b0, (int, float))):
                continue
            td = DS.tdir(float(b0), float(av))
            fl = lflag.get((k[0], k[1], met), set())
            sets = list(fl)
            if not fl:
                sets.append("all")
                if met not in direct:
                    sets.append("headline")
            tab = comp.get((k, met), {}).get(best.get(fam))
            for s in sets:
                if td == "same":
                    if s == "headline":
                        for M in models:
                            v, _, _, d = rd[M]
                            if v == "applied" and met in d:
                                c["g3_%s_n_same" % M] += 1
                                c["g3_%s_hit_same" % M] += d[met] == "same"
                    continue
                c["d_%s_n" % s] += 1
                c["d_%s_cov" % s] += tab is not None
                tok = tab == td
                c["d_%s_hit_TABLE" % s] += tok
                if not tok:
                    c["d_%s_tw_n" % s] += 1
                for M in models:
                    v, _, _, d = rd[M]
                    hit = v == "applied" and d.get(met) == td
                    c["d_%s_hit_%s" % (s, M)] += hit
                    c["d_%s_abs_%s" % (s, M)] += not (v == "applied" and met in d)
                    if not tok:
                        c["d_%s_tw_hit_%s" % (s, M)] += hit
                if s == "headline":
                    for M in models:
                        v, _, _, d = rd[M]
                        if v == "applied" and met in d:
                            c["g3_%s_n_%s" % (M, td)] += 1
                            c["g3_%s_hit_%s" % (M, td)] += d[met] == td
    groups = [("heldout_problem", "trained"), ("heldout_family", "unseen")]

    def boot(cs, fn, needed):
        """fn(Counter) -> float; (point, se) with the states resampled (a.boot draws, seed 0: the same draws for
        every statistic of a scope, so differences are paired). needed: the counter keys fn reads."""
        tot = sum(cs, collections.Counter())
        pt = fn(tot)
        if len(cs) < 2:
            return pt, float("nan")
        V = np.array([[c[k] for k in needed] for c in cs], dtype=float)
        idx = np.random.default_rng(0).integers(0, len(cs), size=(a.boot, len(cs)))
        sums = V[idx].sum(axis=1)
        return pt, se_of([fn(collections.Counter(dict(zip(needed, row)))) for row in sums])

    def r(c, x, y):
        return c[x] / c[y] if c[y] else float("nan")

    def feas_stats(c, how, M):
        n1, n0 = c["fq_n_True"], c["fq_n_False"]
        h1, h0 = c["f_%s_%s_True" % (how, M)], c["f_%s_%s_False" % (how, M)]
        rec = [x for x in ((h1 / n1) if n1 else None, (h0 / n0) if n0 else None) if x is not None]
        return ((h1 + h0) / (n1 + n0) if n1 + n0 else float("nan"), sum(rec) / len(rec) if rec else float("nan"),
                h1 / n1 if n1 else float("nan"), h0 / n0 if n0 else float("nan"))

    def cells_of(scope_fams, gname):
        return [S[k] for k in sorted(S) if k[0] in scope_fams and qs[k[1]]["group"] == gname]

    for gname, glab in groups:
        gf = [f for f in fams if any(qs[k[0]]["variant"] == f and qs[k[0]]["group"] == gname for k in keys)]
        if not gf:
            continue
        scopes = [(f, [f]) for f in gf] + [("POOLED " + glab, gf)]
        w("\n" + "=" * 100)
        w("== %s families (%s): %s" % (glab, gname, ", ".join(gf)))
        # verdict
        w("\n## verdict accuracy, headline questions (question flags out)")
        w("  %-22s %5s  %s" % ("scope", "n", "  ".join("%-8s" % M for M in models)))
        for sc, fs in scopes:
            cs = cells_of(fs, gname)
            t = sum(cs, collections.Counter())
            w("  %-22s %5d  %s" % (sc, t["n"], "  ".join("%-8.3f" % r(t, "v_" + M, "n") for M in models)))
        # feasibility
        w("\n## feasibility, truth-applied headline questions; refused / unreadable / missing line = wrong")
        w("   primary: 'VIOLATED AFTER is none'; second (labelled): the FEASIBLE AFTER line. acc | BA | recall feasible"
          " | recall infeasible; always-'no' = share infeasible")
        for sc, fs in scopes:
            cs = cells_of(fs, gname)
            t = sum(cs, collections.Counter())
            if not t["fq_n"]:
                continue
            w("  %-22s n %4d (feasible %d)  always-no %.3f" % (sc, t["fq_n"], t["fq_n_True"], r(t, "fq_n_False", "fq_n")))
            for how, lab in (("viol", "VIOLATED none"), ("line", "FEASIBLE line (2nd)")):
                parts = []
                for M in models:
                    acc, ba, r1, r0 = feas_stats(t, how, M)
                    parts.append("%s %.3f|%.3f|%.3f|%.3f (miss %d)" % (M, acc, ba, r1, r0, t["f_%s_%s_missing" % (how, M)]))
                w("    %-20s %s" % (lab, "   ".join(parts)))
            if sc.startswith("POOLED"):
                for M in models[:-1]:
                    for how in ("viol", "line"):
                        nd = ["fq_n", "fq_n_True", "fq_n_False", "f_%s_%s_True" % (how, M), "f_%s_%s_False" % (how, M),
                              "f_viol_S1_True", "f_viol_S1_False"]
                        d_acc = boot(cs, lambda c, M=M, how=how: feas_stats(c, how, M)[0] - feas_stats(c, "viol", "S1")[0], nd)
                        d_ba = boot(cs, lambda c, M=M, how=how: feas_stats(c, how, M)[1] - feas_stats(c, "viol", "S1")[1], nd)
                        d_no = boot(cs, lambda c, M=M, how=how: feas_stats(c, how, M)[0] - r(c, "fq_n_False", "fq_n"), nd)
                        w("    %s (%s) - S1 (VIOLATED): acc %+.3f (%.3f), BA %+.3f (%.3f); acc - always-no %+.3f (%.3f)" % (
                            M, how, d_acc[0], d_acc[1], d_ba[0], d_ba[1], d_no[0], d_no[1]))
        # directions
        for s in SETS:
            w("\n## direction, true up/down lines, line set '%s'; absent or refused = wrong; TABLE = no-physics table" % s)
            w("  %-22s %6s | %s | %-6s | %s | on lines TABLE gets wrong: n, %s" % (
                "scope", "lines", "  ".join("%-6s" % M for M in models), "TABLE",
                "  ".join("%s-S1 (se)  %s-TAB (se)" % (M, M) for M in models[:-1]) +
                ("  " + "  ".join("%s-%s (se)" % (M, models[0]) for M in models[1:-1]) if len(models) > 2 else ""),
                " ".join(models)))
            for sc, fs in scopes:
                cs = cells_of(fs, gname)
                t = sum(cs, collections.Counter())
                n = t["d_%s_n" % s]
                if not n:
                    continue
                acc = {M: r(t, "d_%s_hit_%s" % (s, M), "d_%s_n" % s) for M in models + ["TABLE"]}
                diffs = []
                nd = ["d_%s_n" % s, "d_%s_hit_S1" % s, "d_%s_hit_TABLE" % s] + ["d_%s_hit_%s" % (s, M) for M in models[:-1]]
                for M in models[:-1]:
                    x = boot(cs, lambda c, M=M: r(c, "d_%s_hit_%s" % (s, M), "d_%s_n" % s) - r(c, "d_%s_hit_S1" % s, "d_%s_n" % s), nd)
                    y = boot(cs, lambda c, M=M: r(c, "d_%s_hit_%s" % (s, M), "d_%s_n" % s) - r(c, "d_%s_hit_TABLE" % s, "d_%s_n" % s), nd)
                    diffs.append("%+.3f (%.3f) %+.3f (%.3f)" % (x[0], x[1], y[0], y[1]))
                for M in models[1:-1]:
                    x = boot(cs, lambda c, M=M: r(c, "d_%s_hit_%s" % (s, M), "d_%s_n" % s) -
                             r(c, "d_%s_hit_%s" % (s, models[0]), "d_%s_n" % s), nd)
                    diffs.append("%+.3f (%.3f)" % x)
                tw = t["d_%s_tw_n" % s]
                w("  %-22s %6d | %s | %.3f  | %s | %d, %s | absent %s | table coverage %d" % (
                    sc, n, "  ".join("%.3f " % acc[M] for M in models), acc["TABLE"], "  ".join(diffs), tw,
                    " ".join("%.3f" % r(t, "d_%s_tw_hit_%s" % (s, M), "d_%s_tw_n" % s) for M in models),
                    " ".join("%s %.3f" % (M, r(t, "d_%s_abs_%s" % (s, M), "d_%s_n" % s)) for M in models),
                    t["d_%s_cov" % s]))
        # 3-class on lines given
        w("\n## 3-class direction on headline lines the model writes (true up / down / same): balanced | plain (n)")
        for sc, fs in scopes:
            t = sum(cells_of(fs, gname), collections.Counter())
            parts = []
            for M in models:
                rec = [t["g3_%s_hit_%s" % (M, x)] / t["g3_%s_n_%s" % (M, x)] for x in ("up", "down", "same")
                       if t["g3_%s_n_%s" % (M, x)]]
                nn = sum(t["g3_%s_n_%s" % (M, x)] for x in ("up", "down", "same"))
                hh = sum(t["g3_%s_hit_%s" % (M, x)] for x in ("up", "down", "same"))
                parts.append("%s %.3f|%.3f (%d)" % (M, sum(rec) / len(rec) if rec else float("nan"), hh / nn if nn else float("nan"), nn))
            w("  %-22s %s" % (sc, "   ".join(parts)))
        # flagged questions
        for fl in ("no-op", "off-range"):
            t = sum(cells_of(gf, gname), collections.Counter())
            if t["qf_%s_n" % fl]:
                w("\n## flagged questions '%s': %d; verdict right: %s" % (fl, t["qf_%s_n" % fl], "  ".join(
                    "%s %d" % (M, t["qf_%s_v_%s" % (fl, M)]) for M in models)))
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("ARM_SPLIT_DONE")


if __name__ == "__main__":
    main()
