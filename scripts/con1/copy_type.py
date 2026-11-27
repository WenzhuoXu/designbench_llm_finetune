"""Same-type copiers for the sequential pass (CPU only, answers that exist; con1-2m-rows, 11-26).

    python copy_type.py --preds PREDS.jsonl --model LABEL --group heldout_family|heldout_problem --out REPORT.txt
        [--queries queries_v2.jsonl] [--sign review|value]

Scored with facets.py's own truth_facets / pred_facets / move_counts / family_scores / kshot_baselines / _as_binned
(imported, unchanged), the review agents' way (review_tmp/a1_oracle.py): per predictor and state a Counter of move
statistics, family scores from the summed Counters, paired se by one bootstrap over states shared by all predictors
(1,000 draws, seed 0), 'all' = mean over families.

Questions: the sequential pass's positions >= 1 (an earlier outcome of the state was shown). Predictors on them:
  model         the model's sequential answer;      model_zero  its zero-shot answer to the same question
  copy_last / copy_mode                      facets.py's k-shot baselines (all earlier shown outcomes)
  copy_same_type       the most recent earlier shown move of the same type (verb + sign of the argument change); else
                       the most recent of the same verb; else copy_last
  copy_same_type_mode  facets.py's copy_mode over the earlier shown moves of the same type; else of the same verb; else
                       copy_last
Type = (verb_of, sign_of) of review_tmp/a1_oracle.py, the functions taken from that file unchanged; --sign review keeps
its SIGN_VERBS (SET_BASE_STOCK, SET_REVIEW_PERIOD, SET_CONDUCTANCE: the held-out families' scalar verbs); --sign value
lets sign_of apply to every move whose arguments carry a value and its previous value (v, v_old: the verb-surface
families' parameters; native verbs have no v_old, so their type is the verb).
Subsets: all positions >= 1; those where a same-type earlier move had been shown; those where none had.
"""
import argparse
import ast
import collections
import json
import math
import sys

import numpy as np

sys.path.insert(0, "/storage/wxu/con1/repo/scripts/con1")
import facets as F  # noqa: E402

REVIEW = "/storage/wxu/con1/review_tmp/a1_oracle.py"
METRICS = ("regime_phys", "cpl_dir_BA", "cpl_size")


def review_type_fns(sign_mode):
    """verb_of, sign_of (and SIGN_VERBS) exactly as review_tmp/a1_oracle.py defines them (its other code not run)."""
    tree = ast.parse(open(REVIEW).read())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ("verb_of", "sign_of"))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "SIGN_VERBS" for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), REVIEW, "exec"), ns)
    if sign_mode == "value":
        class Every:
            def __contains__(self, x):
                return True
        ns["SIGN_VERBS"] = Every()
    return ns["verb_of"], ns["sign_of"], ns.get("SIGN_VERBS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="/storage/wxu/con1/arm/main/queries_v2.jsonl")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--model", required=True, help="the predictor label in the preds file")
    ap.add_argument("--group", required=True)
    ap.add_argument("--sign", choices=("review", "value"), default="review")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    verb_of, sign_of, _ = review_type_fns(a.sign)
    typ = lambda m: (verb_of(m), sign_of(m))                     # noqa: E731
    qs = {}
    for line in open(a.queries):
        q = json.loads(line)
        if q["group"] == a.group:
            qs[q["sid"]] = q
    fams = sorted({q["variant"] for q in qs.values()})
    sids = {v: sorted(s for s, q in qs.items() if q["variant"] == v) for v in fams}
    T = {s: [F.truth_facets(q, m) for m in q["moves"]] for s, q in qs.items()}
    # the model's answers, read as facets.py main() reads them
    PF = {"seq": {}, "zero": {}}
    for line in open(a.preds):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("predictor") != a.model or r["sid"] not in qs or str(r.get("finish", "")).startswith("error"):
            continue
        ps = r.get("pass")
        if ps in PF and (r["sid"], r["qi"]) not in PF[ps]:
            PF[ps][(r["sid"], r["qi"])] = F.pred_facets(r.get("text"), r.get("target", "verbatim"), qs[r["sid"]])
    per = {sub: collections.defaultdict(dict) for sub in ("all", "same_type_shown", "none_shown")}
    share = collections.Counter()
    for s, q in qs.items():
        mv = q["moves"]
        for j in range(1, len(mv)):
            if (s, j) not in PF["seq"] or (s, j) not in PF["zero"]:
                share["missing"] += 1
                continue
            t = T[s][j]
            earlier_t = [i for i in range(j) if typ(mv[i]) == typ(mv[j])]
            earlier_v = [i for i in range(j) if verb_of(mv[i]) == verb_of(mv[j])]
            kb = F.kshot_baselines(q, T[s][:j])
            if earlier_t:
                cst = F._as_binned(q, T[s][earlier_t[-1]])
                cstm = F.kshot_baselines(q, [T[s][i] for i in earlier_t])["copy_mode"]
            elif earlier_v:
                cst = F._as_binned(q, T[s][earlier_v[-1]])
                cstm = F.kshot_baselines(q, [T[s][i] for i in earlier_v])["copy_mode"]
            else:
                cst = cstm = kb["copy_last"]
            preds = {"model": PF["seq"][(s, j)], "model_zero": PF["zero"][(s, j)], "copy_last": kb["copy_last"],
                     "copy_mode": kb["copy_mode"], "copy_same_type": cst, "copy_same_type_mode": cstm}
            subs = ("all", "same_type_shown" if earlier_t else "none_shown")
            share[(q["variant"], "n")] += 1
            share[(q["variant"], "same_type")] += bool(earlier_t)
            share[(q["variant"], "same_verb")] += bool(earlier_v)
            for name, pf in preds.items():
                c = F.move_counts(q, t, pf, None)
                for sub in subs:
                    per[sub][name].setdefault(s, collections.Counter()).update(c)
    names = ["model", "model_zero", "copy_last", "copy_mode", "copy_same_type", "copy_same_type_mode"]
    rng = np.random.default_rng(0)
    boots = [{v: rng.integers(len(sids[v]), size=len(sids[v])) for v in fams} for _ in range(1000)]
    out = []
    w = out.append
    w("same-type copiers, sequential pass (positions >= 1): %s, model %s, group %s, sign %s" % (a.preds, a.model, a.group, a.sign))
    w("type = (verb, sign of the argument change) from %s; se: bootstrap over states shared by all predictors (1000, seed 0)" % REVIEW)
    if share["missing"]:
        w("questions without both a sequential and a zero-shot answer (left out): %d" % share["missing"])
    w("share of questions with an earlier shown move of the same type (same verb):")
    for v in fams:
        n = share[(v, "n")]
        w("  %-10s %4d questions: same type %.3f (%d), same verb %.3f" % (
            v, n, share[(v, "same_type")] / max(1, n), share[(v, "same_type")], share[(v, "same_verb")] / max(1, n)))
    N = sum(share[(v, "n")] for v in fams)
    w("  %-10s %4d questions: same type %.3f" % ("all", N, sum(share[(v, "same_type")] for v in fams) / max(1, N)))
    for sub in ("all", "same_type_shown", "none_shown"):
        P = per[sub]
        keys = sorted({k for p in P for c in P[p].values() for k in c})
        kidx = {k: i for i, k in enumerate(keys)}
        MAT = {}
        for p in names:
            for v in fams:
                M = np.zeros((len(sids[v]), len(keys)))
                for i, s in enumerate(sids[v]):
                    for k, n in P[p].get(s, {}).items():
                        M[i, kidx[k]] = n
                MAT[(p, v)] = M
        fs = lambda p, v, idx=None: F.family_scores(collections.Counter(dict(zip(keys, (MAT[(p, v)] if idx is None else MAT[(p, v)][idx]).sum(0)))))  # noqa: E731
        PT = {(p, v): fs(p, v) for p in names for v in fams}
        BS = {(p, v): [fs(p, v, b[v]) for b in boots] for p in names for v in fams}
        gm = lambda xs: float(np.mean([x for x in xs if not math.isnan(x)])) if any(not math.isnan(x) for x in xs) else float("nan")  # noqa: E731
        w("")
        w("== subset: %s" % {"all": "all sequential questions (positions >= 1)",
                              "same_type_shown": "a same-type move had been shown earlier",
                              "none_shown": "NO same-type move had been shown earlier"}[sub])
        for scope in fams + ["all"]:
            nq = sum(int(MAT[("model", v)][:, kidx["moves"]].sum()) for v in (fams if scope == "all" else [scope])) if "moves" in kidx else -1
            w("  %s (%d questions)" % (scope, nq))
            for p in names:
                if scope == "all":
                    vals = [(gm([PT[(p, v)][m] for v in fams]), np.nanstd([gm([BS[(p, v)][j][m] for v in fams]) for j in range(len(boots))], ddof=1)) for m in METRICS]
                else:
                    vals = [(PT[(p, scope)][m], np.nanstd([BS[(p, scope)][j][m] for j in range(len(boots))], ddof=1)) for m in METRICS]
                w("    %-20s " % p + "  ".join("%s %.3f+-%.3f" % (m, x, e) for m, (x, e) in zip(METRICS, vals)))
            for p in names[1:]:
                row = []
                for m in METRICS:
                    if scope == "all":
                        pt = gm([PT[("model", v)][m] for v in fams]) - gm([PT[(p, v)][m] for v in fams])
                        bs = [gm([BS[("model", v)][j][m] for v in fams]) - gm([BS[(p, v)][j][m] for v in fams]) for j in range(len(boots))]
                    else:
                        pt = PT[("model", scope)][m] - PT[(p, scope)][m]
                        bs = [BS[("model", scope)][j][m] - BS[(p, scope)][j][m] for j in range(len(boots))]
                    row.append("%s %+.3f+-%.3f" % (m, pt, np.nanstd(bs, ddof=1)))
                w("    model - %-14s " % p + "  ".join(row))
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("COPY_TYPE_DONE")


if __name__ == "__main__":
    main()
