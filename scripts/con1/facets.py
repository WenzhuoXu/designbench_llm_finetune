"""con1 facet scorer (docs/consequence_round_con1.md §8): every answer format is read back into the same physical
facets and scored per move against the truth and against a no-change baseline.

    python facets.py --queries queries.jsonl --preds preds_B.jsonl preds_F.jsonl ... [--groups heldout_problem]
        [--variants ...] [--pass zero] [--boot 1000] [--out report.txt] [--json report.json]

Facets of one move (truth from the simulated record; prediction from the answer text in its format):
  regime     refused / applied-infeasible / applied-feasible
  violated   the set of constraints violated after the move (applied moves)
  crossings  constraints whose violated status flips relative to the current design
  direction  per metric: up / down / same (same: change within 1e-5 of the value, targets.is_same)
  size       per metric with a true change from a nonzero value: right direction and relative change within the same
             or an adjacent half-decade bin (targets.BINS)
Scores per family (pooled over test moves): regime balanced accuracy and per-class recall; refusal precision /
recall; violated exact-set accuracy; crossing recall and false-crossing rate; direction balanced accuracy over
up/down/same; size share; feasible-after (yes) recall / precision. The no-change baseline ('nochange') is scored the
same way. Selection score = mean over families of mean(regime BA, direction BA, size share). Diagnostic only:
within-state discrimination = P(true Phi up | predicted up) - P(true Phi up | predicted down) over applied moves,
with the predicted Phi from reconstructed after values (bin geometric midpoints for the facet format).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from targets import BIN_NAMES, BINS, direction, size_bin, violated_after  # noqa: E402

NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
STATUS_CUR = re.compile(r"^\s+([A-Za-z_][\w.\-]*)\s+\S+\s+limit .*?(VIOLATED)?", re.M)
REGIMES = ("refused", "applied-infeasible", "applied-feasible")


def current_violated(user: str) -> tuple:
    """(constraints of the problem, those violated now) from the prompt's CONSTRAINT STATUS (current design)."""
    i = user.find("CONSTRAINT STATUS")
    if i < 0:
        return [], set()
    j = user.find("\n\n", i)
    block = user[i:j if j > 0 else len(user)]
    cons, viol = [], set()
    for line in block.splitlines()[1:]:
        m = re.match(r"^\s+([A-Za-z_][\w.\-]*)\s+\S+\s+limit", line)
        if m:
            cons.append(m.group(1))
            if "VIOLATED" in line:
                viol.add(m.group(1))
    return cons, viol


def truth_facets(q: dict, m: dict) -> dict:
    if m["verdict"] == "refused":
        return {"regime": "refused"}
    after = m["metrics_after"]
    out = {"regime": "applied-feasible" if m["feasible"] else "applied-infeasible",
           "violated": set(violated_after(m["result"])), "dir": {}, "bin": {}, "after": dict(after)}
    for k, b in q["metrics0"].items():
        if k in after:
            out["dir"][k] = direction(float(b), float(after[k]))
            out["bin"][k] = size_bin(float(b), float(after[k]))
    return out


def _bins_from_values(m0: dict, after: dict) -> tuple:
    d, b = {}, {}
    for k, v in after.items():
        if k in m0:
            d[k] = direction(float(m0[k]), float(v))
            b[k] = size_bin(float(m0[k]), float(v))
    return d, b


def _mid_value(before: float, dr: str, bn: str):
    """A value inside the bin (geometric midpoint of the relative change) for Phi reconstruction."""
    if dr == "same" or bn == "same":
        return before
    if bn == "from 0" or before == 0:
        return None
    i = BIN_NAMES.index(bn) if bn in BIN_NAMES else None
    if i is None:
        return None
    lo = BINS[i - 1][0] if i > 0 else BINS[0][0] / 3.0
    hi = BINS[i][0] if math.isfinite(BINS[i][0]) else lo * 3.0
    r = math.sqrt(lo * hi)
    return before + (r * abs(before) if dr == "up" else -r * abs(before))


def pred_facets(text: str, fmt: str, q: dict) -> dict:
    """Facets of a predicted answer; regime 'unreadable' when nothing can be read."""
    text = text or ""
    m0 = q["metrics0"]
    if fmt in ("verbatim", "masked"):
        from score_pred import read_message
        r = read_message(text, m0)
        if r["kind"] != "applied":
            return {"regime": r["kind"]}
        fe = r.get("feasible_said")
        out = {"regime": "applied-feasible" if fe else "applied-infeasible", "violated": set(violated_after(text))}
        d, b = _bins_from_values(m0, r["metrics"])
        out.update(dir=d, bin=b, after=r["metrics"])
        return out
    low = text.lower()
    mv = re.search(r"VERDICT:\s*(\w+)", text)
    if not mv:
        if "refused" in low or "rejected" in low:
            return {"regime": "refused"}
        return {"regime": "unreadable"}
    if mv.group(1).lower().startswith("refus"):
        return {"regime": "refused"}
    fe = re.search(r"FEASIBLE AFTER:\s*(\w+)", text)
    feasible = bool(fe and fe.group(1).lower().startswith("y"))
    vl = re.search(r"VIOLATED AFTER:\s*(.*)", text)
    viol = set()
    if vl and vl.group(1).strip().lower() not in ("none", ""):
        viol = {x.strip() for x in vl.group(1).split(",") if x.strip()}
    out = {"regime": "applied-feasible" if feasible else "applied-infeasible", "violated": viol, "dir": {},
           "bin": {}, "after": dict(m0)}
    body = text.split("CHANGES:", 1)[1] if "CHANGES:" in text else ""
    for line in body.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] not in m0:
            continue
        k, rest = parts[0], " ".join(parts[1:])
        b0 = float(m0[k])
        if rest.startswith("same"):
            out["dir"][k], out["bin"][k] = "same", "same"
            continue
        if fmt == "facet":
            dr = parts[1] if parts[1] in ("up", "down") else None
            bn = " ".join(parts[2:]) if dr else None
            if dr is None:
                continue
            out["dir"][k] = dr
            out["bin"][k] = bn if bn in BIN_NAMES or bn == "from 0" else "?"
            v = _mid_value(b0, dr, out["bin"][k])
            if v is not None:
                out["after"][k] = v
        else:                                              # delta: '+x (+y%)' or '+x from 0'
            mm = re.match(r"(%s)" % NUM, parts[1])
            if not mm:
                continue
            try:
                a = b0 + float(mm.group(1))
            except ValueError:
                continue
            out["after"][k] = a
            out["dir"][k] = direction(b0, a)
            out["bin"][k] = size_bin(b0, a)
    return out


def nochange_facets(q: dict) -> dict:
    m0 = q["metrics0"]
    _, viol = current_violated(q["user"])
    return {"regime": "applied-infeasible", "violated": viol, "dir": {k: "same" for k in m0},
            "bin": {k: "same" for k in m0}, "after": dict(m0)}


def move_counts(q, t, p, ctx=None):
    """Per-move sufficient statistics."""
    c = collections.Counter()
    c["regime_true_" + t["regime"]] += 1
    c["regime_hit_" + t["regime"]] += p["regime"] == t["regime"]
    c["pred_refused"] += p["regime"] == "refused"
    c["unreadable"] += p["regime"] == "unreadable"
    c["moves"] += 1
    if t["regime"] == "applied-feasible":
        c["feas_true"] += 1
        c["feas_hit"] += p["regime"] == "applied-feasible"
    c["feas_pred"] += p["regime"] == "applied-feasible"
    c["feas_pred_ok"] += p["regime"] == "applied-feasible" and t["regime"] == "applied-feasible"
    if t["regime"] != "refused" and p["regime"] in ("applied-infeasible", "applied-feasible"):
        c["viol_n"] += 1
        c["viol_exact"] += p.get("violated", set()) == t["violated"]
        cons, now = current_violated(q["user"])
        for k in cons:
            tc = (k in now) != (k in t["violated"])
            pc = (k in now) != (k in p.get("violated", set()))
            c["cross_true"] += tc
            c["cross_hit"] += tc and pc
            c["nocross_true"] += not tc
            c["false_cross"] += (not tc) and pc
    if t["regime"] != "refused":
        for k, td in t["dir"].items():
            pd = (p.get("dir") or {}).get(k, "same" if p["regime"] in ("refused", "unreadable") else "same")
            if p["regime"] in ("refused", "unreadable"):
                pd = None                                   # no metric prediction: wrong on every line
            c["dir_true_" + td] += 1
            c["dir_hit_" + td] += pd == td
            tb = t["bin"][k]
            if tb not in ("same", "from 0"):
                c["size_n"] += 1
                pb = (p.get("bin") or {}).get(k)
                if pd == td and pb in BIN_NAMES and abs(BIN_NAMES.index(pb) - BIN_NAMES.index(tb)) <= 1:
                    c["size_hit"] += 1
    return c


def family_scores(cs: collections.Counter) -> dict:
    def rate(a, b):
        return cs[a] / cs[b] if cs[b] else float("nan")
    rec = [cs["regime_hit_" + r] / cs["regime_true_" + r] for r in REGIMES if cs["regime_true_" + r]]
    drec = [cs["dir_hit_" + d] / cs["dir_true_" + d] for d in ("up", "down", "same") if cs["dir_true_" + d]]
    out = {"regime_BA": float(np.mean(rec)) if rec else float("nan"),
           "dir_BA": float(np.mean(drec)) if drec else float("nan"),
           "size": rate("size_hit", "size_n"),
           "viol_exact": rate("viol_exact", "viol_n"),
           "cross_recall": rate("cross_hit", "cross_true"),
           "false_cross": rate("false_cross", "nocross_true"),
           "refused_recall": rate("regime_hit_refused", "regime_true_refused"),
           "refused_precision": cs["regime_hit_refused"] / cs["pred_refused"] if cs["pred_refused"] else float("nan"),
           "feas_recall": rate("feas_hit", "feas_true"),
           "feas_precision": cs["feas_pred_ok"] / cs["feas_pred"] if cs["feas_pred"] else float("nan"),
           "unreadable": rate("unreadable", "moves")}
    for r in REGIMES:
        out["recall_" + r] = rate("regime_hit_" + r, "regime_true_" + r)
    sel = [out["regime_BA"], out["dir_BA"], out["size"]]
    sel = [x for x in sel if not math.isnan(x)]
    out["select"] = float(np.mean(sel)) if sel else float("nan")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--preds", nargs="*", default=[])
    ap.add_argument("--groups", nargs="*", default=["heldout_problem"])
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--pass", dest="pass_", default="zero")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    qs = {}
    for line in open(a.queries, encoding="utf-8"):
        q = json.loads(line)
        if a.groups and q["group"] not in a.groups:
            continue
        if a.variants and q["variant"] not in a.variants:
            continue
        qs[q["sid"]] = q
    # per predictor -> sid -> Counter
    per = collections.defaultdict(dict)
    for q in qs.values():
        cs = collections.Counter()
        for m in q["moves"]:
            cs.update(move_counts(q, truth_facets(q, m), nochange_facets(q)))
        per["nochange"][q["sid"]] = cs
    formats = {}
    for p in a.preds:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("pass") != a.pass_ or r["sid"] not in qs:
                continue
            q = qs[r["sid"]]
            fmt = r.get("target", "verbatim")
            formats[r["predictor"]] = fmt
            t = truth_facets(q, q["moves"][r["qi"]])
            if r.get("facets") is not None:                # a predictor that outputs facets directly (trees)
                pf = dict(r["facets"])
                pf["violated"] = set(pf.get("violated") or [])
            else:
                pf = pred_facets(r.get("text"), fmt, q)
            per[r["predictor"]].setdefault(r["sid"], collections.Counter()).update(move_counts(q, t, pf))
    fams = sorted({q["variant"] for q in qs.values()})
    sids = {v: sorted(s for s, q in qs.items() if q["variant"] == v) for v in fams}
    rng = np.random.default_rng(a.seed)
    boots = [{v: rng.integers(len(sids[v]), size=len(sids[v])) for v in fams} for _ in range(a.boot)]

    def fam_counter(pred, v, idx=None):
        c = collections.Counter()
        ss = sids[v] if idx is None else [sids[v][i] for i in idx]
        for s in ss:
            c.update(per[pred].get(s, collections.Counter()))
        return c

    def group(pred, key, b=None):
        vals = [family_scores(fam_counter(pred, v, None if b is None else b[v]))[key] for v in fams]
        vals = [x for x in vals if not math.isnan(x)]
        return float(np.mean(vals)) if vals else float("nan")

    lines = ["facet profile, pass %s, groups %s, families %s, states %s" % (
        a.pass_, a.groups, fams, {v: len(sids[v]) for v in fams})]
    keys = ["select", "regime_BA", "dir_BA", "size", "viol_exact", "cross_recall", "false_cross", "refused_recall",
            "refused_precision", "feas_recall", "feas_precision", "unreadable"]
    preds = ["nochange"] + sorted(k for k in per if k != "nochange")
    lines.append("%-10s %-11s" % ("predictor", "family") + "".join("%10s" % k[:10] for k in keys))
    res = {}
    for pr in preds:
        for v in fams:
            f = family_scores(fam_counter(pr, v))
            res["%s/%s" % (pr, v)] = f
            lines.append("%-10s %-11s" % (pr, v) + "".join("%10.3f" % f[k] for k in keys))
    lines.append("")
    lines.append("group means (+- se, bootstrap over states)")
    for pr in preds:
        row = []
        for k in ("select", "regime_BA", "dir_BA", "size"):
            pt = group(pr, k)
            bs = [group(pr, k, b) for b in boots]
            se = float(np.nanstd(bs, ddof=1))
            res["group/%s/%s" % (pr, k)] = {"value": pt, "se": se}
            row.append("%s %.3f +- %.3f" % (k, pt, se))
        lines.append("%-10s %s  [%s]" % (pr, "  ".join(row), formats.get(pr, "-")))
    lines.append("")
    lines.append("paired differences of the selection score (+- se)")
    arms = [p for p in preds if p != "nochange"]
    for i, x in enumerate(arms):
        for y in arms[i + 1:] + ["nochange"]:
            pt = group(x, "select") - group(y, "select")
            bs = [group(x, "select", b) - group(y, "select", b) for b in boots]
            se = float(np.nanstd(bs, ddof=1))
            lines.append("  %s - %s: %+.3f +- %.3f" % (x, y, pt, se))
            res["diff/%s-%s" % (x, y)] = {"value": pt, "se": se}
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt + "\n")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
