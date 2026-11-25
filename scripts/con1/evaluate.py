"""con1 skill test, step 3: score every predictor on the fixed question set (docs/consequence_round_con1.md §3-5).

    python evaluate.py --queries queries.jsonl --problems bench_problems.jsonl --code C --harness V3RC \\
        --preds preds_con1.jsonl preds_pref1.jsonl preds_trees.jsonl --majority-from native.jsonl.gz scripted.jsonl.gz \\
        --out report.txt [--json report.json] [--boot 2000]

Predictors: every label found in --preds (model messages are read with score_pred.py; tree lines carry a verdict and
a score), plus 'majority' (the family's most common verdict in the training transitions; on held-out families the
test set's own majority, labelled oracle) and, on the sequential pass, 'nearest_probe' (the true verdict of the most
similar earlier move of the conversation: same verb, then the closest arguments; majority when none).

Metrics (zero-shot pass unless stated): V 3-way verdict accuracy; D direction accuracy on moves applied and
predicted applied; PO within-state pairwise ordering of applied moves; N metric-sign accuracy and share closer than
'no change'; refusal precision / recall; k-shot gain G(k) = sequential minus zero-shot correctness on the same move.
Group values are means over the group's families; se by bootstrap over states (resampled within each family);
paired differences between predictors use the same resamples.
"""
import argparse
import collections
import gzip
import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pred import PhiContext, metric_scores, rank_value, read_message, verdict_of  # noqa: E402

NATIVE = ["engibench", "grid2op", "photonics"]
REPLAY = ["circuit2", "frame3d2", "grid2", "seir2", "traffic2"]
HELD = ["inventory2", "molopt", "thermal2"]
GROUPS = {"heldout_problem(8)": NATIVE + REPLAY, "native(3)": NATIVE, "replay(5)": REPLAY, "heldout_family(3)": HELD}


def opener(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def args_vec(m):
    a = (m.get("args") or {}).get("args") or {}
    out = {}
    for k, v in a.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = str(v)
    return (m.get("args") or {}).get("verb") or m["move"].split("(", 1)[0], out


def nearest_probe(moves, j, majority):
    verb, aj = args_vec(moves[j])
    best, bd = None, None
    for i in range(j):
        vi, ai = args_vec(moves[i])
        if vi != verb:
            continue
        d = 0.0
        for k in set(aj) | set(ai):
            x, y = aj.get(k), ai.get(k)
            if isinstance(x, float) and isinstance(y, float):
                d += abs(x - y) / (abs(x) + abs(y) + 1e-9)
            else:
                d += 0.0 if x == y else 1.0
        if bd is None or d < bd:
            best, bd = moves[i], d
    return best["verdict"] if best else majority


class Scored:
    """Per predictor, pass, state: lists of per-move outcomes."""

    def __init__(self):
        self.d = collections.defaultdict(dict)        # (predictor, pass) -> sid -> {qi: rec}

    def add(self, pred, ps, sid, qi, rec):
        self.d[(pred, ps)].setdefault(sid, {})[qi] = rec


def state_metrics(q, recs):
    """Per-state sums for one predictor/pass: dict of (numerator, denominator) pairs."""
    out = collections.defaultdict(lambda: [0.0, 0.0])
    moves = q["moves"]
    for qi, m in enumerate(moves):
        r = recs.get(qi)
        if r is None:
            continue
        tv = m["verdict"]
        pv = r["verdict"]
        out["V"][0] += pv == tv
        out["V"][1] += 1
        if tv != "refused" and pv in ("better", "worse"):
            out["D"][0] += pv == tv
            out["D"][1] += 1
        out["refused_tp"][0] += (pv == "refused" and tv == "refused")
        out["refused_pred"][0] += pv == "refused"
        out["refused_true"][0] += tv == "refused"
        out["unreadable"][0] += pv == "unreadable"
        out["unreadable"][1] += 1
        if r.get("N"):
            out["N_sign"][0] += r["N"]["sign_ok"]
            out["N_sign"][1] += r["N"]["n"]
            out["N_closer"][0] += r["N"]["closer"]
            out["N_closer"][1] += r["N"]["n"]
    # PO over truly applied moves with different true rank
    app = [qi for qi, m in enumerate(moves) if m["verdict"] != "refused" and qi in recs]
    for x in range(len(app)):
        for y in range(x + 1, len(app)):
            i, j = app[x], app[y]
            ti = rank_value(moves[i]["verdict"], moves[i].get("phi"), moves[i].get("viol"), q["variant"])
            tj = rank_value(moves[j]["verdict"], moves[j].get("phi"), moves[j].get("viol"), q["variant"])
            if ti == tj:
                continue
            pi, pj = recs[i]["rank"], recs[j]["rank"]
            out["PO"][1] += 1
            if pi == pj:
                out["PO"][0] += 0.5
            else:
                out["PO"][0] += (pi > pj) == (ti > tj)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--preds", nargs="*", default=[])
    ap.add_argument("--majority-from", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from conseq_lib import Env
    env = Env(a.code, a.harness)
    specs = {}
    for p in a.problems:
        for line in opener(p):
            r = json.loads(line)
            specs[r["spec"]["problem_id"]] = (r["variant"], r["spec"])
    qs = {}
    for line in open(a.queries, encoding="utf-8"):
        q = json.loads(line)
        qs[q["sid"]] = q
    # majority verdict per family from the training transitions
    maj = {}
    cnt = collections.defaultdict(collections.Counter)
    for p in a.majority_from:
        for line in opener(p):
            st = json.loads(line)
            if st["split"] != "train":
                continue
            for m in st["moves"]:
                cnt[st["variant"]][m["verdict"]] += 1
    test_cnt = collections.defaultdict(collections.Counter)
    for q in qs.values():
        for m in q["moves"]:
            test_cnt[q["variant"]][m["verdict"]] += 1
    for v in test_cnt:
        maj[v] = (cnt[v] or test_cnt[v]).most_common(1)[0][0]
    S = Scored()
    ctxs = {}

    def ctx_of(q):
        if q["sid"] not in ctxs:
            v, spec = specs[q["problem_id"]]
            ev = env.D.PhiEvaluator(env.domain(v), spec)
            ctxs[q["sid"]] = PhiContext(ev, v, q["phi0"], q.get("bad0") or 0)
        return ctxs[q["sid"]]

    n_bad = collections.Counter()
    for p in a.preds:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            q = qs.get(r["sid"])
            if q is None:
                n_bad["pred for an unknown state"] += 1
                continue
            if r.get("predictor") == "trees":
                for pr in r["preds"]:
                    rk = (-1, 0.0, 0.0) if pr["verdict"] == "refused" else (0, 0.0, pr["score"])
                    S.add("trees", r["pass"], r["sid"], pr["qi"], {"verdict": pr["verdict"], "rank": rk})
                continue
            m = q["moves"][r["qi"]]
            ctx = ctx_of(q)
            pred = read_message(r.get("text") or "", q["metrics0"])
            verdict, phi = verdict_of(pred, ctx)
            viol = (pred["metrics"] or {}).get("min_feature_violations") if pred.get("metrics") else None
            rec = {"verdict": verdict, "rank": rank_value(verdict, phi, viol, q["variant"])}
            if pred["kind"] == "applied" and m["verdict"] != "refused":
                rec["N"] = metric_scores(pred["metrics"], m["metrics_after"], q["metrics0"])
            rec["exact"] = (r.get("text") or "").strip() == m["result"].strip()
            S.add(r["predictor"], r["pass"], r["sid"], r["qi"], rec)
    # baselines without a model
    for q in qs.values():
        for qi, m in enumerate(q["moves"]):
            S.add("majority", "zero", q["sid"], qi, {"verdict": maj[q["variant"]], "rank": (0, 0.0, 0.0)})
            if qi > 0:
                nv = nearest_probe(q["moves"], qi, maj[q["variant"]])
                S.add("nearest_probe", "seq", q["sid"], qi, {"verdict": nv, "rank": (0, 0.0, 0.0)})
    # per-state sums
    fam_states = collections.defaultdict(list)
    for sid, q in qs.items():
        fam_states[q["variant"]].append(sid)
    sums = {}
    for key, per in S.d.items():
        sums[key] = {sid: state_metrics(qs[sid], recs) for sid, recs in per.items()}
    # k-shot gains (sequential vs zero-shot on the same move), per predictor
    gains = {}
    for (pred, ps), per in S.d.items():
        if ps != "seq" or pred == "nearest_probe":
            continue
        z = S.d.get((pred, "zero"), {})
        g = {}
        for sid, recs in per.items():
            for qi, r in recs.items():
                zr = z.get(sid, {}).get(qi)
                if zr is None:
                    continue
                tv = qs[sid]["moves"][qi]["verdict"]
                g.setdefault(sid, {})[qi] = (r["verdict"] == tv) - (zr["verdict"] == tv)
        gains[pred] = g
    rng = np.random.default_rng(a.seed)
    fams = sorted(fam_states)
    boots = [{v: rng.choice(len(fam_states[v]), size=len(fam_states[v]), replace=True) for v in fams}
             for _ in range(a.boot)]

    def fam_value(table, v, metric, idx=None):
        sids = fam_states[v] if idx is None else [fam_states[v][i] for i in idx]
        num = den = 0.0
        for sid in sids:
            t = table.get(sid)
            if t is None or metric not in t:
                continue
            num += t[metric][0]
            den += t[metric][1]
        return num / den if den else float("nan")

    def group_value(table, fl, metric, b=None):
        vals = [fam_value(table, v, metric, None if b is None else b[v]) for v in fl if v in fam_states]
        vals = [x for x in vals if not math.isnan(x)]
        return sum(vals) / len(vals) if vals else float("nan")

    def gain_table(pred, ks):
        g = gains.get(pred, {})
        return {sid: {"G": [sum(x for qi, x in d.items() if qi in ks), sum(1 for qi in d if qi in ks)]}
                for sid, d in g.items()}

    lines = []
    res = {"families": {}, "groups": {}, "diffs": {}, "gains": {}}
    keys = sorted(sums)
    metrics = ["V", "D", "PO", "N_sign", "N_closer", "unreadable"]
    lines.append("predictors/passes: %s" % ", ".join("%s/%s" % k for k in keys))
    lines.append("majority verdict per family: %s" % maj)
    lines.append("test states per family: %s" % {v: len(fam_states[v]) for v in fams})
    lines.append("")
    lines.append("== per family (zero-shot pass; seq = sequential pass with earlier true results in context)")
    hdr = "%-14s %-10s" % ("predictor", "family") + "".join("%9s" % m for m in metrics) + "%9s%9s" % ("ref_P", "ref_R")
    lines.append(hdr)
    for key in keys:
        pred, ps = key
        for v in fams:
            row = [fam_value(sums[key], v, m) for m in metrics]
            tp = sum(sums[key][s]["refused_tp"][0] for s in fam_states[v] if s in sums[key])
            pp = sum(sums[key][s]["refused_pred"][0] for s in fam_states[v] if s in sums[key])
            tt = sum(sums[key][s]["refused_true"][0] for s in fam_states[v] if s in sums[key])
            if all(math.isnan(x) for x in row):
                continue
            lines.append("%-14s %-10s" % ("%s/%s" % (pred, ps), v) + "".join("%9.3f" % x for x in row)
                         + "%9s%9s" % ("%.3f" % (tp / pp) if pp else "-", "%.3f" % (tp / tt) if tt else "-"))
            res["families"]["%s/%s/%s" % (pred, ps, v)] = dict(zip(metrics, row))
    lines.append("")
    lines.append("== groups (mean over families; +- se by bootstrap over states)")
    for gname, fl in GROUPS.items():
        for key in keys:
            vals = []
            for m in ["V", "D", "PO"]:
                point = group_value(sums[key], fl, m)
                bs = [group_value(sums[key], fl, m, b) for b in boots]
                bs = [x for x in bs if not math.isnan(x)]
                se = float(np.std(bs, ddof=1)) if len(bs) > 1 else float("nan")
                vals.append((point, se))
                res["groups"]["%s/%s/%s/%s" % (gname, key[0], key[1], m)] = {"value": point, "se": se}
            if all(math.isnan(p) for p, _ in vals):
                continue
            lines.append("%-20s %-22s " % (gname, "%s/%s" % key) + "  ".join("%s %.3f +- %.3f" % (m, p, s)
                                                                          for m, (p, s) in zip(["V", "D", "PO"], vals)))
    lines.append("")
    lines.append("== paired differences (zero-shot), group mean, bootstrap se")
    preds_zero = sorted({k[0] for k in keys if k[1] == "zero"})
    pairs = [(x, y) for x in preds_zero for y in preds_zero if x != y and x not in ("majority",) and
             (x.startswith("con1") or (x == "trees" and y == "majority"))]
    for gname, fl in GROUPS.items():
        for x, y in pairs:
            for m in ["V", "D", "PO"]:
                tx, ty = sums.get((x, "zero")), sums.get((y, "zero"))
                if tx is None or ty is None:
                    continue
                point = group_value(tx, fl, m) - group_value(ty, fl, m)
                bs = [group_value(tx, fl, m, b) - group_value(ty, fl, m, b) for b in boots]
                bs = [z for z in bs if not math.isnan(z)]
                if math.isnan(point) or len(bs) < 2:
                    continue
                se = float(np.std(bs, ddof=1))
                res["diffs"]["%s/%s-%s/%s" % (gname, x, y, m)] = {"value": point, "se": se}
                lines.append("%-20s %-28s %-3s %+.3f +- %.3f  (%.1f se)" % (gname, "%s - %s" % (x, y), m, point, se,
                                                                         point / se if se > 0 else float("nan")))
    lines.append("")
    lines.append("== k-shot gain G = correct(sequential) - correct(zero-shot), same move; and V of the sequential pass")
    for pred in sorted(gains):
        for gname, fl in GROUPS.items():
            for ks, lab in ((set(range(1, 8)), "k1-7"), ({4, 5, 6, 7}, "k4-7"), ({1}, "k1")):
                t = gain_table(pred, ks)
                point = group_value(t, fl, "G")
                bs = [group_value(t, fl, "G", b) for b in boots]
                bs = [z for z in bs if not math.isnan(z)]
                if math.isnan(point) or len(bs) < 2:
                    continue
                se = float(np.std(bs, ddof=1))
                res["gains"]["%s/%s/%s" % (pred, gname, lab)] = {"value": point, "se": se}
                lines.append("%-14s %-20s %-5s G %+.3f +- %.3f" % (pred, gname, lab, point, se))
    # sequential V at k = 4..7 vs nearest probe (I1)
    lines.append("")
    lines.append("== sequential pass, V at positions 4-7: predictor vs nearest_probe (I1 control)")
    for (pred, ps), per in S.d.items():
        if ps != "seq" or pred == "nearest_probe":
            continue
        npb = S.d.get(("nearest_probe", "seq"), {})
        for gname, fl in GROUPS.items():
            def tab(dd):
                out = {}
                for sid, recs in dd.items():
                    q = qs[sid]
                    c = sum(recs[qi]["verdict"] == q["moves"][qi]["verdict"] for qi in recs if qi >= 4)
                    n = sum(1 for qi in recs if qi >= 4)
                    out[sid] = {"V": [c, n]}
                return out
            ta, tb = tab(per), tab(npb)
            point = group_value(ta, fl, "V") - group_value(tb, fl, "V")
            bs = [group_value(ta, fl, "V", b) - group_value(tb, fl, "V", b) for b in boots]
            bs = [z for z in bs if not math.isnan(z)]
            if math.isnan(point) or len(bs) < 2:
                continue
            se = float(np.std(bs, ddof=1))
            res["gains"]["%s/%s/seqV47-nearest" % (pred, gname)] = {"value": point, "se": se,
                                                                    "pred": group_value(ta, fl, "V"),
                                                                    "nearest": group_value(tb, fl, "V")}
            lines.append("%-14s %-20s V(k4-7) %.3f vs nearest %.3f: %+.3f +- %.3f" % (
                pred, gname, group_value(ta, fl, "V"), group_value(tb, fl, "V"), point, se))
    # exact-match share (model messages word for word)
    lines.append("")
    for (pred, ps), per in sorted(S.d.items()):
        ex = [r.get("exact") for recs in per.values() for r in recs.values() if "exact" in r]
        if ex:
            lines.append("exact message match %s/%s: %.3f of %d" % (pred, ps, sum(ex) / len(ex), len(ex)))
    if n_bad:
        lines.append("!!! %s" % dict(n_bad))
    txt = "\n".join(lines)
    open(a.out, "w", encoding="utf-8").write(txt + "\n")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
