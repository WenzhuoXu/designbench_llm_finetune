"""The 11-28 base-model read, scored the parent's way (CPU; con1-2m-rows, 11-28): one thinking arm against the stage-1
final's v3 answers, paired by question, bootstrap over states (1,000 draws, seed 0, states resampled within family;
group = mean over families).

    python dir_split.py --queries queries_v3.jsonl --base preds_ARM.jsonl --base-fmt dir|delta|facetm \\
        --ref preds_s1_trained.jsonl,preds_s1_unseen.jsonl --flags flags_v3.tsv --reader facets|indep --out REPORT.txt

Readers: 'facets' = facets.pred_facets (CON1_LIB, the fixed reader), 'indep' = indep_score.read_answer (written apart);
the same statistics from each are the two values of every number.
Per question: truth from the record (applied / refused; feasible; per-metric direction, 'same' within 1e-5 relative).
Statistics (per family, then the mean over families; base, stage 1, base - stage 1 with se):
  verdict_acc   share of questions whose answer has the right verdict (applied vs refused; unreadable = wrong)
  feas_acc      applied truth and applied answer: share with the right FEASIBLE AFTER
  feas_BA       the same, mean of the feasible and the infeasible recall (both recalls reported)
  dir_given     coupled lines (metrics not in facets.DIRECT) of applied truth and applied base answer on which the
                base answer writes a direction: direction accuracy, balanced over the true classes up / down / same,
                and plain; stage 1 on the same lines
  absent        those lines on which the base answer writes no direction (share of the coupled lines of applied-applied
                questions); stage 1's absent share on the same questions
Splits: headline (no-op and off-range questions out), no-op, off-range; and the headline split by the base answer's
think_end (closed / budget). Also: think_end counts and the think-token quartiles of closed blocks per family, and the
per family x verb x metric table of wrong directions among closed blocks (base).
"""
import argparse
import collections
import json
import math
import os
import random
import re
import statistics
import sys

sys.path.insert(0, os.environ.get("CON1_LIB", "/storage/wxu/con1/full_eval/lib_v5"))
sys.path.insert(0, "/storage/wxu/con1/full_eval")
import facets as F  # noqa: E402
import dirlayer as DL  # noqa: E402
import indep_score as IS  # noqa: E402


def tdir(b, a):
    return "same" if abs(a - b) <= max(1e-5 * abs(b), 1e-12) else ("up" if a > b else "down")


def read(text, fmt, q, reader):
    """(verdict 'applied'|'refused'|'unreadable', feasible bool|None, {metric: dir})"""
    if reader == "facets":
        pf = F.pred_facets(text, fmt, q)
        reg = pf["regime"]
        if reg == "unreadable":
            return "unreadable", None, {}
        if not reg.startswith("applied"):
            return "refused", None, {}
        return "applied", reg == "applied-feasible", dict(pf.get("dir") or {})
    IS.FMT = "facetm" if fmt in ("facetm", "facet") else fmt
    p = IS.read_answer(text, q["metrics0"])
    reg = p["regime"]
    if reg == "unreadable":
        return "unreadable", None, {}
    if not reg.startswith("applied"):
        return "refused", None, {}
    return "applied", reg == "applied-feasible", dict(p.get("dir") or {})


def load(path):
    out = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get("pass", "zero") == "zero" and not str(r.get("finish", "")).startswith("error"):
            out.setdefault((r["sid"], r["qi"]), r)
    return out


def counters(qs, keys, base, ref, bfmt, reader):
    """per (family, sid) Counter of sufficient statistics, plus the per-line wrong-direction table"""
    C = collections.defaultdict(collections.Counter)
    wrong = collections.Counter()
    for (sid, qi) in keys:
        q = qs[sid]
        m = q["moves"][qi]
        fam = q["variant"]
        c = C[(fam, sid)]
        rb, rr = base[(sid, qi)], ref[(sid, qi)]
        vb, fb, db = read(rb.get("text"), bfmt, q, reader)
        vr, fr, dr = read(rr.get("text"), rr.get("target", "facetm"), q, reader)
        tv = "refused" if m["verdict"] == "refused" else "applied"
        c["n"] += 1
        c["vb"] += vb == tv
        c["vr"] += vr == tv
        if tv != "applied":
            continue
        tf = bool(m["feasible"])
        for who, v, f in (("b", vb, fb), ("r", vr, fr)):
            if v == "applied":
                c["f%s_n_%s" % (who, tf)] += 1
                c["f%s_hit_%s" % (who, tf)] += f == tf
        if vb != "applied" or vr != "applied":
            continue
        after, m0 = m["metrics_after"], q["metrics0"]
        direct = F.DIRECT.get(fam, set())
        for k, b0 in m0.items():
            if k not in after or k in direct:
                continue
            td = tdir(float(b0), float(after[k]))
            c["lines"] += 1
            if k not in db:
                c["absent_b"] += 1
                c["absent_r_on_absent"] += k not in dr
                continue
            c["absent_r"] += k not in dr
            pr = dr.get(k, "same")
            c["g_n_" + td] += 1
            c["gb_hit_" + td] += db[k] == td
            c["gr_hit_" + td] += pr == td
            if rb.get("think_end") == "closed" and db[k] != td:
                wrong[(fam, DL.verb_of(m), k, td, db[k])] += 1
            if rb.get("think_end") == "closed":
                wrong[(fam, DL.verb_of(m), k, "n")] += 1
    return C, wrong


def fam_stats(c):
    def ba(pfx):
        r = [c[pfx + "_hit_" + x] / c["g_n_" + x] for x in ("up", "down", "same") if c["g_n_" + x]]
        return sum(r) / len(r) if r else float("nan")
    def plain(pfx):
        n = sum(c["g_n_" + x] for x in ("up", "down", "same"))
        return sum(c[pfx + "_hit_" + x] for x in ("up", "down", "same")) / n if n else float("nan")
    def feas(who):
        rec = {t: (c["f%s_hit_%s" % (who, t)] / c["f%s_n_%s" % (who, t)]) if c["f%s_n_%s" % (who, t)] else float("nan")
               for t in (True, False)}
        n = c["f%s_n_True" % who] + c["f%s_n_False" % who]
        acc = (c["f%s_hit_True" % who] + c["f%s_hit_False" % who]) / n if n else float("nan")
        xs = [x for x in rec.values() if not math.isnan(x)]
        return acc, (sum(xs) / len(xs) if xs else float("nan")), rec[True], rec[False]
    fb, fr = feas("b"), feas("r")
    n = c["n"] or float("nan")
    lines = c["lines"] or float("nan")
    return {"verdict_acc": (c["vb"] / n, c["vr"] / n), "feas_acc": (fb[0], fr[0]), "feas_BA": (fb[1], fr[1]),
            "feas_rec_feasible": (fb[2], fr[2]), "feas_rec_infeasible": (fb[3], fr[3]),
            "dir_given_BA": (ba("gb"), ba("gr")), "dir_given_plain": (plain("gb"), plain("gr")),
            "absent": (c["absent_b"] / lines, (c["absent_r"] + c["absent_r_on_absent"]) / lines)}


STATS = ("verdict_acc", "feas_acc", "feas_BA", "feas_rec_feasible", "feas_rec_infeasible", "dir_given_BA",
         "dir_given_plain", "absent")


def table(C, title, w, boot=1000):
    fams = sorted({f for f, _ in C})
    by = {f: [C[k] for k in sorted(C) if k[0] == f] for f in fams}
    tot = lambda cs: sum(cs, collections.Counter())  # noqa: E731
    point = {f: fam_stats(tot(by[f])) for f in fams}
    rng = random.Random(0)
    draws = []
    for _ in range(boot):
        d = {}
        for f in fams:
            cs = by[f]
            d[f] = fam_stats(tot([cs[rng.randrange(len(cs))] for _ in cs]))
        draws.append(d)
    gm = lambda xs: (sum(x for x in xs if not math.isnan(x)) / sum(1 for x in xs if not math.isnan(x)))  \
        if any(not math.isnan(x) for x in xs) else float("nan")  # noqa: E731
    w("## %s (questions %d)" % (title, sum(C[k]["n"] for k in C)))
    w("  %-12s %-20s %8s %8s %16s" % ("family", "statistic", "base", "stage 1", "base - s1 (se)"))
    for scope in fams + ["mean"]:
        vs = fams if scope == "mean" else [scope]
        nq = sum(C[k]["n"] for k in C if k[0] in vs)
        for s in STATS:
            b = gm([point[f][s][0] for f in vs])
            r = gm([point[f][s][1] for f in vs])
            diffs = [gm([d[f][s][0] for f in vs]) - gm([d[f][s][1] for f in vs]) for d in draws]
            diffs = [x for x in diffs if not math.isnan(x)]
            se = statistics.pstdev(diffs) if len(diffs) > 1 else float("nan")
            w("  %-12s %-20s %8.3f %8.3f %+8.3f (%.3f)%s" % (scope if s == STATS[0] else "", s, b, r, b - r, se,
                                                            ("   n=%d" % nq) if s == STATS[0] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-fmt", required=True)
    ap.add_argument("--ref", required=True, help="comma-separated stage-1 answer files")
    ap.add_argument("--flags", required=True)
    ap.add_argument("--reader", choices=("facets", "indep"), default="facets")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    qs = {json.loads(l)["sid"]: json.loads(l) for l in open(a.queries)}
    base = load(a.base)
    ref = {}
    for p in a.ref.split(","):
        ref.update(load(p))
    flags = collections.defaultdict(set)
    for l in open(a.flags):
        if l.strip():
            s, j, f = l.rstrip("\n").split("\t")
            flags[(s, int(j))].add(f)
    keys = sorted(k for k in base if k in ref and k[0] in qs)
    out = []
    w = out.append
    w("# %s vs stage-1 final v3; base form %s; reader %s; bootstrap over states %d" % (a.base, a.base_fmt, a.reader, a.boot))
    w("# questions with both answers: %d (base %d, stage 1 %d)" % (len(keys), len(base), len(ref)))
    # think_end and closed-block lengths
    te = collections.defaultdict(collections.Counter)
    tl = collections.defaultdict(list)
    for k in keys:
        r = base[k]
        f = qs[k[0]]["variant"]
        te[f][str(r.get("think_end"))] += 1
        if r.get("think_end") == "closed":
            tl[f].append(r.get("think_tokens") or 0)
    w("## think_end per family; think tokens of closed blocks (q25 / median / q75 / max)")
    for f in sorted(te) + ["all"]:
        cnt = sum(te.values(), collections.Counter()) if f == "all" else te[f]
        ls = sorted(sum(tl.values(), [])) if f == "all" else sorted(tl[f])
        qt = (ls[len(ls) // 4], ls[len(ls) // 2], ls[3 * len(ls) // 4], ls[-1]) if ls else ()
        n = sum(cnt.values())
        w("  %-12s n %5d  %s  closed %.3f  closed-block tokens %s" % (f, n, dict(cnt), cnt["closed"] / max(1, n), qt))
    groups = {"headline (no-op and off-range out)": [k for k in keys if not flags.get(k)],
              "no-op": [k for k in keys if "no-op" in flags.get(k, ())],
              "off-range": [k for k in keys if "off-range" in flags.get(k, ())]}
    for gname in ("heldout_problem", "heldout_family"):
        for title, ks in groups.items():
            ks = [k for k in ks if qs[k[0]]["group"] == gname]
            if not ks:
                continue
            C, wrong = counters(qs, ks, base, ref, a.base_fmt, a.reader)
            table(C, "%s, %s" % (gname, title), w, a.boot)
            if title.startswith("headline"):
                for tend in ("closed", "budget"):
                    kt = [k for k in ks if base[k].get("think_end") == tend]
                    if kt:
                        C2, _ = counters(qs, kt, base, ref, a.base_fmt, a.reader)
                        table(C2, "%s, %s, base think block %s" % (gname, title, tend), w, a.boot)
                w("## %s: wrong directions among CLOSED blocks, per family x verb x metric (lines n, wrong; true -> written)" % gname)
                cells = sorted({k[:3] for k in wrong})
                for cell in cells:
                    n = wrong[cell + ("n",)]
                    errs = {(k[3], k[4]): v for k, v in wrong.items() if k[:3] == cell and len(k) == 5}
                    ne = sum(errs.values())
                    if n:
                        w("  %-11s %-22s %-22s n %4d wrong %4d (%.2f)  %s" % (cell + (n, ne, ne / n, ", ".join(
                            "%s->%s %d" % (t, p, v) for (t, p), v in sorted(errs.items(), key=lambda x: -x[1])))))
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("DIR_SPLIT_DONE")


if __name__ == "__main__":
    main()
