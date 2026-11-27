"""Several models' answers to the same questions, side by side, with paired differences against a reference model
(CPU only; con1-2m-rows, 11-26).

    python compare_models.py --pred LABEL=PREDS.jsonl [--pred ...] --ref LABEL --group heldout_family --out REPORT.txt

Scored with facets.py's own truth_facets / pred_facets / move_counts / family_scores (imported, unchanged), as
copy_type.py: per predictor and state a Counter of move statistics; family scores from the summed Counters; paired se
from one bootstrap over states shared by all predictors (1,000 draws, seed 0); the group = mean over families.
Passes: zero (every question) and seq (positions >= 1, an earlier outcome of the state shown). A question a model
did not answer, or answered with an error, is left out for every model (so the comparison stays paired); the readable
share is the share of answers facets.py can read (regime not 'unreadable') among the answers given.
"""
import argparse
import collections
import json
import math
import sys

import numpy as np

sys.path.insert(0, "/storage/wxu/con1/repo/scripts/con1")
import facets as F  # noqa: E402

METRICS = ("regime_phys", "cpl_dir_BA", "cpl_size")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="/storage/wxu/con1/arm/main/queries_v2.jsonl")
    ap.add_argument("--pred", action="append", required=True, help="LABEL=preds.jsonl (LABEL = the predictor field)")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models = [p.split("=", 1) for p in a.pred]
    names = [m for m, _ in models]
    qs = {}
    for line in open(a.queries):
        q = json.loads(line)
        if q["group"] == a.group:
            qs[q["sid"]] = q
    fams = sorted({q["variant"] for q in qs.values()})
    sids = {v: sorted(s for s, q in qs.items() if q["variant"] == v) for v in fams}
    T = {s: [F.truth_facets(q, m) for m in q["moves"]] for s, q in qs.items()}
    PF = {(m, ps): {} for m in names for ps in ("zero", "seq")}
    given = collections.Counter()
    for lab, path in models:
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("predictor") != lab or r["sid"] not in qs or r.get("pass") not in ("zero", "seq"):
                continue
            if str(r.get("finish", "")).startswith("error"):
                given[(lab, r["pass"], "error")] += 1
                continue
            key = (r["sid"], r["qi"])
            if key not in PF[(lab, r["pass"])]:
                PF[(lab, r["pass"])][key] = F.pred_facets(r.get("text"), r.get("target", "verbatim"), qs[r["sid"]])
                given[(lab, r["pass"], "answers")] += 1
                given[(lab, r["pass"], "readable")] += PF[(lab, r["pass"])][key]["regime"] != "unreadable"
                if r.get("finish") == "length":
                    given[(lab, r["pass"], "length")] += 1
    rng = np.random.default_rng(0)
    boots = [{v: rng.integers(len(sids[v]), size=len(sids[v])) for v in fams} for _ in range(1000)]
    out = []
    w = out.append
    w("models side by side, group %s, reference %s; se: bootstrap over states shared by all predictors (1000, seed 0)" % (a.group, a.ref))
    for m in names:
        for ps in ("zero", "seq"):
            n = given[(m, ps, "answers")]
            w("  %-22s %-4s answers %5d, readable %5d (%.3f), errors %d, cut at the token limit %d" % (
                m, ps, n, given[(m, ps, "readable")], given[(m, ps, "readable")] / max(1, n), given[(m, ps, "error")],
                given[(m, ps, "length")]))
    for ps in ("zero", "seq"):
        per = collections.defaultdict(dict)
        nq = collections.Counter()
        unread = collections.Counter()
        for s, q in qs.items():
            for j in range(1 if ps == "seq" else 0, len(q["moves"])):
                if any((s, j) not in PF[(m, ps)] for m in names):
                    nq["dropped"] += 1
                    continue
                nq[q["variant"]] += 1
                for m in names:
                    pf = PF[(m, ps)][(s, j)]
                    unread[(m, q["variant"])] += pf["regime"] == "unreadable"
                    per[m].setdefault(s, collections.Counter()).update(F.move_counts(q, T[s][j], pf, None))
                per["nochange"].setdefault(s, collections.Counter()).update(F.move_counts(q, T[s][j], F.nochange_facets(q), None))
        P = names + ["nochange"]
        keys = sorted({k for p in P for c in per[p].values() for k in c})
        kidx = {k: i for i, k in enumerate(keys)}
        MAT = {}
        for p in P:
            for v in fams:
                M = np.zeros((len(sids[v]), len(keys)))
                for i, s in enumerate(sids[v]):
                    for k, n in per[p].get(s, {}).items():
                        M[i, kidx[k]] = n
                MAT[(p, v)] = M
        fs = lambda p, v, idx=None: F.family_scores(collections.Counter(dict(zip(keys, (MAT[(p, v)] if idx is None else MAT[(p, v)][idx]).sum(0)))))  # noqa: E731
        PT = {(p, v): fs(p, v) for p in P for v in fams}
        BS = {(p, v): [fs(p, v, b[v]) for b in boots] for p in P for v in fams}
        gm = lambda xs: float(np.mean([x for x in xs if not math.isnan(x)])) if any(not math.isnan(x) for x in xs) else float("nan")  # noqa: E731
        w("")
        w("== pass %s (%s); questions answered by every model: %s%s" % (
            ps, "all questions" if ps == "zero" else "positions >= 1", {v: nq[v] for v in fams},
            ", left out %d" % nq["dropped"] if nq["dropped"] else ""))
        for scope in fams + ["all"]:
            vs = fams if scope == "all" else [scope]
            nn = sum(nq[v] for v in vs)
            w("  %s (%d questions)" % (scope, nn))
            for p in P:
                vals = []
                for mm in METRICS:
                    pt = gm([PT[(p, v)][mm] for v in vs])
                    se = np.nanstd([gm([BS[(p, v)][j][mm] for v in vs]) for j in range(len(boots))], ddof=1)
                    vals.append("%s %.3f+-%.3f" % (mm, pt, se))
                rd = "" if p == "nochange" else "  readable %.3f" % (1 - sum(unread[(p, v)] for v in vs) / max(1, nn))
                w("    %-22s " % p + "  ".join(vals) + rd)
            for p in names:
                if p == a.ref:
                    continue
                row = []
                for mm in METRICS:
                    pt = gm([PT[(p, v)][mm] for v in vs]) - gm([PT[(a.ref, v)][mm] for v in vs])
                    bs = [gm([BS[(p, v)][j][mm] for v in vs]) - gm([BS[(a.ref, v)][j][mm] for v in vs]) for j in range(len(boots))]
                    row.append("%s %+.3f+-%.3f" % (mm, pt, np.nanstd(bs, ddof=1)))
                w("    %-22s " % ("%s - %s" % (p, a.ref)) + "  ".join(row))
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("COMPARE_DONE")


if __name__ == "__main__":
    main()
