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
REGIMES = ("refused-rule", "refused-physical", "applied-infeasible", "applied-feasible")
PHYS_REGIMES = ("refused-physical", "applied-infeasible", "applied-feasible")
# A refusal whose reason is a simulation outcome (the power flow diverges / no AC solution / blackout: the move would
# end the episode) is physics; one whose reason is a rule (bad arguments, cooldown, forced outage, ambiguous redispatch,
# a move that changes nothing, the fabrication rule, an unknown element) is not (parent review 11-25, doc section 11).
PHYSICAL_REFUSAL = re.compile(r"would end the episode|diverge|no AC solution|blackout|isolated load|non-connected grid",
                              re.I)


def refusal_kind(reason: str) -> str:
    return "refused-physical" if PHYSICAL_REFUSAL.search(reason or "") else "refused-rule"


def is_applied(regime: str) -> bool:
    return regime.startswith("applied")
VERDICTS = ("refused", "worse", "better")
# 'Directly moved' metrics: bookkeeping of the move itself (amounts, sizes, counts, costs of what was set), readable
# without solving any physics. Every other metric is a response the simulator computes; the coupling facet is the
# direction / size facets on those response metrics only. Fixed 11-25 before any result (doc section 11).
DIRECT = {
    "circuit2": {"area", "power_mw"}, "engibench": {"volume", "n_components"}, "frame3d2": {"mass"},
    "grid2": {"commitment_cost", "n_committed"},
    "grid2op": {"step", "redispatch_mw", "redispatch_cost", "curtailed_mw", "storage_cost", "lines_out",
                "lines_unavailable"},
    "photonics": {"fill_fraction", "n_islands", "min_feature_violations"}, "seir2": {"doses_per_capita", "cost"},
    "traffic2": {"lanes_used"}, "inventory2": set(),
    "molopt": {"heavy_atoms", "mol_weight", "n_rings", "max_ring_size", "rot_bonds"},
    "thermal2": {"cost", "link_cost", "sink_cost"},
}


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
        return {"regime": refusal_kind(m.get("refused") or m["result"]), "verdict": "refused"}
    after = m["metrics_after"]
    out = {"regime": "applied-feasible" if m["feasible"] else "applied-infeasible", "verdict": m["verdict"],
           "violated": set(violated_after(m["result"])), "dir": {}, "bin": {}, "change": {}, "after": dict(after)}
    for k, b in q["metrics0"].items():
        if k in after:
            out["dir"][k] = direction(float(b), float(after[k]))
            out["bin"][k] = size_bin(float(b), float(after[k]))
            out["change"][k] = float(after[k]) - float(b)
    return out


def _bins_from_values(m0: dict, after: dict) -> tuple:
    d, b, ch = {}, {}, {}
    for k, v in after.items():
        if k in m0:
            d[k] = direction(float(m0[k]), float(v))
            b[k] = size_bin(float(m0[k]), float(v))
            ch[k] = float(v) - float(m0[k])
    return d, b, ch


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
        if r["kind"] == "refused":
            reason = text.split("refused by the simulator:", 1)[1] if "refused by the simulator:" in text else text
            return {"regime": refusal_kind(reason)}
        if r["kind"] != "applied":
            return {"regime": r["kind"]}
        fe = r.get("feasible_said")
        out = {"regime": "applied-feasible" if fe else "applied-infeasible", "violated": set(violated_after(text))}
        d, b, ch = _bins_from_values(m0, r["metrics"])
        out.update(dir=d, bin=b, change=ch, after=r["metrics"])
        return out
    low = text.lower()
    mv = re.search(r"VERDICT:\s*(\w+)", text)
    if not mv:
        if "refused" in low or "rejected" in low:
            return {"regime": refusal_kind(text)}
        return {"regime": "unreadable"}
    if mv.group(1).lower().startswith("refus"):
        rs = re.search(r"REASON:\s*(.*)", text)
        return {"regime": refusal_kind(rs.group(1) if rs else text)}
    fe = re.search(r"FEASIBLE AFTER:\s*(\w+)", text)
    feasible = bool(fe and fe.group(1).lower().startswith("y"))
    vl = re.search(r"VIOLATED AFTER:\s*(.*)", text)
    viol = set()
    if vl and vl.group(1).strip().lower() not in ("none", ""):
        viol = {x.strip() for x in vl.group(1).split(",") if x.strip()}
    out = {"regime": "applied-feasible" if feasible else "applied-infeasible", "violated": viol, "dir": {},
           "bin": {}, "change": {}, "after": dict(m0)}
    if "CHANGES:" in text:
        body = text.split("CHANGES:", 1)[1]
    elif "METRICS (before -> after):" in text:
        body = text.split("METRICS (before -> after):", 1)[1]
    else:
        body = ""
    for line in body.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] not in m0:
            continue
        k, rest = parts[0], " ".join(parts[1:])
        b0 = float(m0[k])
        if rest.startswith("same"):
            out["dir"][k], out["bin"][k] = "same", "same"
            continue
        if fmt in ("facet", "facetm", "facetc"):          # facetm's / facetc's blocks sit before CHANGES: not read here
            dr = parts[1] if parts[1] in ("up", "down") else None
            bn = " ".join(parts[2:]) if dr else None
            if dr is None:
                continue
            out["dir"][k] = dr
            out["bin"][k] = bn if bn in BIN_NAMES or bn == "from 0" else "?"
            v = _mid_value(b0, dr, out["bin"][k])
            if v is not None:
                out["after"][k] = v
                out["change"][k] = v - b0
        elif fmt == "absolute":                            # 'b -> a' (4 significant digits each)
            mm = re.match(r"(%s)\s*->\s*(%s)" % (NUM, NUM), rest)
            if not mm:
                continue
            try:
                bw, aw = float(mm.group(1)), float(mm.group(2))
            except ValueError:
                continue
            dch = aw - bw                                  # the change as written (both values rounded alike)
            out["after"][k] = b0 + dch
            out["change"][k] = dch
            out["dir"][k] = "same" if dch == 0 else ("up" if dch > 0 else "down")
            out["bin"][k] = size_bin(b0, b0 + dch) if dch != 0 else "same"
        else:                                              # delta: '+x (+y%)' or '+x from 0'
            mm = re.match(r"(%s)" % NUM, parts[1])
            if not mm:
                continue
            try:
                a = b0 + float(mm.group(1))
            except ValueError:
                continue
            out["after"][k] = a
            out["change"][k] = a - b0
            out["dir"][k] = direction(b0, a)
            out["bin"][k] = size_bin(b0, a)
    return out


def nochange_facets(q: dict) -> dict:
    m0 = q["metrics0"]
    _, viol = current_violated(q["user"])
    return {"regime": "applied-infeasible", "violated": viol, "dir": {k: "same" for k in m0},
            "bin": {k: "same" for k in m0}, "after": dict(m0)}


def _as_binned(q: dict, f: dict) -> dict:
    """A truth record as a facet-format answer would carry it: the size from the bin midpoint, not the exact change."""
    if not is_applied(f["regime"]):
        return {"regime": f["regime"]}
    out = {"regime": f["regime"], "violated": set(f["violated"]), "dir": dict(f["dir"]), "bin": dict(f["bin"]),
           "change": {}, "after": dict(q["metrics0"])}
    for k, dr in f["dir"].items():
        b0 = float(q["metrics0"][k])
        v = _mid_value(b0, dr, f["bin"].get(k, "same"))
        if v is not None:
            out["change"][k], out["after"][k] = v - b0, v
    return out


def kshot_baselines(q: dict, seen: list) -> dict:
    """Predictions for the next move from the outcomes already shown at this state (the sequential pass's context),
    with no physics: copy_last = the last shown outcome; copy_mode = per facet the most common shown value (ties:
    the most recent), the violated set of the most recent shown move with that regime."""
    last = _as_binned(q, seen[-1])
    regs = [f["regime"] for f in seen]
    reg = max(reversed(regs), key=regs.count)               # reversed: ties go to the most recent
    mode = {"regime": reg}
    if is_applied(reg):
        src = next(f for f in reversed(seen) if f["regime"] == reg)
        app = [f for f in seen if is_applied(f["regime"])]
        mode.update(violated=set(src["violated"]), dir={}, bin={}, change={}, after=dict(q["metrics0"]))
        for k in q["metrics0"]:
            ds = [f["dir"][k] for f in app if k in f["dir"]]
            if not ds:
                continue
            dr = max(reversed(ds), key=ds.count)
            bs = [f["bin"][k] for f in app if f["dir"].get(k) == dr]
            bn = max(reversed(bs), key=bs.count)
            mode["dir"][k], mode["bin"][k] = dr, bn
            v = _mid_value(float(q["metrics0"][k]), dr, bn)
            if v is not None:
                mode["change"][k], mode["after"][k] = v - float(q["metrics0"][k]), v
    return {"copy_last": last, "copy_mode": mode}


def move_counts(q, t, p, ctx=None):
    """Per-move sufficient statistics."""
    c = collections.Counter()
    c["regime_true_" + t["regime"]] += 1
    c["regime_hit_" + t["regime"]] += p["regime"] == t["regime"]
    c["pred_refused"] += p["regime"].startswith("refused")
    c["refused_any_true"] += t["regime"].startswith("refused")
    c["refused_any_hit"] += t["regime"].startswith("refused") and p["regime"].startswith("refused")
    c["unreadable"] += p["regime"] == "unreadable"
    c["moves"] += 1
    if t["regime"] in PHYS_REGIMES:                        # physics regime: applied moves and physical refusals
        c["phys_true_" + t["regime"]] += 1
        c["phys_hit_" + t["regime"]] += p["regime"] == t["regime"]
    if t["regime"] == "applied-feasible":
        c["feas_true"] += 1
        c["feas_hit"] += p["regime"] == "applied-feasible"
    c["feas_pred"] += p["regime"] == "applied-feasible"
    c["feas_pred_ok"] += p["regime"] == "applied-feasible" and t["regime"] == "applied-feasible"
    if is_applied(t["regime"]) and is_applied(p["regime"]):
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
    direct = DIRECT.get(q["variant"], set())
    if is_applied(t["regime"]):
        for k, td in t["dir"].items():
            pd = (p.get("dir") or {}).get(k, "same")
            if not is_applied(p["regime"]):
                pd = None                                   # no metric prediction: wrong on every line
            pres = ("", "bk_") if k in direct else ("", "cpl_")
            if q["variant"] == "photonics" and k == "target_loss":
                pres = pres + ("adj_",)                     # the simulator interface includes an adjoint map of it
            for pre in pres:
                c[pre + "dir_true_" + td] += 1
                c[pre + "dir_hit_" + td] += pd == td
            tch = t["change"][k]
            if td != "same" and float(q["metrics0"][k]) != 0.0:
                pch = (p.get("change") or {}).get(k)
                ok = bool(pd == td and pch and tch and abs(math.log2(abs(pch / tch))) <= 1.0)
                for pre in pres:
                    c[pre + "size_n"] += 1
                    c[pre + "size_hit"] += ok
                if pd == td and pch and tch:
                    c["logratio_n"] += 1
                    c["logratio_abs_sum"] += abs(math.log2(abs(pch / tch)))
    # verdict (refused / worse / better): truth from the record; the prediction from the predictor (trees) or from the
    # problem's potential of the predicted metric values (ctx)
    pv = p.get("verdict")
    if pv is None and ctx is not None:
        if not is_applied(p["regime"]):
            pv = "refused" if p["regime"].startswith("refused") else "unreadable"
        else:
            try:
                after = p.get("after") or {}
                phi = ctx.phi(after)
                viol = float(after.get("min_feature_violations") or 0.0) if q["variant"] == "photonics" else 0.0
                pv = ctx.verdict(phi, viol)
            except Exception:                              # noqa: BLE001
                pv = "unreadable"
    if pv is not None:
        tv = t["verdict"]
        c["v_true_" + tv] += 1
        c["v_hit_" + tv] += pv == tv
        if tv != "refused":                                # the verdict on truly applied moves
            c["va_true_" + tv] += 1
            c["va_hit_" + tv] += pv == tv
        if t["regime"] != "refused-rule":                  # the physics verdict: diverged (refused), worse, better
            c["vp_true_" + tv] += 1
            c["vp_hit_" + tv] += pv == tv
        if tv != "refused" and pv in ("better", "worse"):
            c["disc_pred_" + pv] += 1
            c["disc_true_better_given_" + pv] += tv == "better"
    return c


def family_scores(cs: collections.Counter) -> dict:
    def rate(a, b):
        return cs[a] / cs[b] if cs[b] else float("nan")
    rec = [cs["regime_hit_" + r] / cs["regime_true_" + r] for r in REGIMES if cs["regime_true_" + r]]
    drec = [cs["dir_hit_" + d] / cs["dir_true_" + d] for d in ("up", "down", "same") if cs["dir_true_" + d]]
    def ba(prefix, classes):
        r = [cs[prefix + "hit_" + x] / cs[prefix + "true_" + x] for x in classes if cs[prefix + "true_" + x]]
        return float(np.mean(r)) if r else float("nan")
    crec = [cs["cpl_dir_hit_" + d] / cs["cpl_dir_true_" + d] for d in ("up", "down", "same") if cs["cpl_dir_true_" + d]]
    vrec = [cs["v_hit_" + v] / cs["v_true_" + v] for v in VERDICTS if cs["v_true_" + v]]
    cross_r, false_c = rate("cross_hit", "cross_true"), rate("false_cross", "nocross_true")
    cross_ba = float(np.nanmean([cross_r, 1.0 - false_c])) if not (math.isnan(cross_r) and math.isnan(false_c)) else float("nan")
    feas_ba = ba("phys_", PHYS_REGIMES)
    phys = [x for x in (feas_ba, cross_ba) if not math.isnan(x)]
    out = {"regime_BA": float(np.mean(rec)) if rec else float("nan"),
           "dir_BA": float(np.mean(drec)) if drec else float("nan"),
           "size": rate("size_hit", "size_n"),
           "cpl_dir_BA": float(np.mean(crec)) if crec else float("nan"),
           "cpl_size": rate("cpl_size_hit", "cpl_size_n"),
           "V_bal": float(np.mean(vrec)) if vrec else float("nan"),
           "V_bal_applied": ba("va_", ("worse", "better")),
           "V_bal_phys": ba("vp_", VERDICTS),
           "refused_physical_recall": rate("regime_hit_refused-physical", "regime_true_refused-physical"),
           "refused_rule_recall": rate("regime_hit_refused-rule", "regime_true_refused-rule"),
           "feas_BA": feas_ba,
           "cross_BA": cross_ba,
           "regime_phys": float(np.mean(phys)) if phys else float("nan"),
           "bk_dir_BA": ba("bk_dir_", ("up", "down", "same")),
           "bk_size": rate("bk_size_hit", "bk_size_n"),
           "adj_dir_BA": ba("adj_dir_", ("up", "down", "same")),
           "adj_size": rate("adj_size_hit", "adj_size_n"),
           "logratio_mean_abs": rate("logratio_abs_sum", "logratio_n"),
           "disc": (rate("disc_true_better_given_better", "disc_pred_better")
                    - rate("disc_true_better_given_worse", "disc_pred_worse")),
           "viol_exact": rate("viol_exact", "viol_n"),
           "cross_recall": rate("cross_hit", "cross_true"),
           "false_cross": rate("false_cross", "nocross_true"),
           "refused_recall": rate("refused_any_hit", "refused_any_true"),
           "refused_precision": cs["refused_any_hit"] / cs["pred_refused"] if cs["pred_refused"] else float("nan"),
           "feas_recall": rate("feas_hit", "feas_true"),
           "feas_precision": cs["feas_pred_ok"] / cs["feas_pred"] if cs["feas_pred"] else float("nan"),
           "unreadable": rate("unreadable", "moves")}
    for r in REGIMES:
        out["recall_" + r] = rate("regime_hit_" + r, "regime_true_" + r)
    for v in VERDICTS:
        out["recall_v_" + v] = rate("v_hit_" + v, "v_true_" + v)
    sel = [out["regime_phys"], out["cpl_dir_BA"], out["cpl_size"]]   # parent review 11-25: physics terms only
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
    ap.add_argument("--kshot-baselines", action="store_true", help="add copy_last / copy_mode: the next move's "
                    "outcome copied from the outcomes shown before it at the same state (positions >= 1)")
    ap.add_argument("--min-qi", type=int, default=0, help="only questions at this position or later (1: the questions "
                    "the sequential pass asks, for a same-question zero vs sequential comparison)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--problems", nargs="*", default=[], help="problem specs (bench_problems.jsonl): with --code and "
                    "--harness, the predicted Phi verdict (V_bal, disc) is computed for formats that give values")
    ap.add_argument("--code", default=None)
    ap.add_argument("--harness", default=None)
    a = ap.parse_args(argv)
    ctxs = {}
    if a.problems and a.code and a.harness:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from conseq_lib import Env
        from score_pred import PhiContext
        env = Env(a.code, a.harness)
        specs = {}
        for path in a.problems:
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                specs[r["spec"]["problem_id"]] = (r["variant"], r["spec"])
        evs = {}

        def ctx_of(q):
            if q["sid"] not in ctxs:
                v, spec = specs[q["problem_id"]]
                if q["problem_id"] not in evs:
                    evs[q["problem_id"]] = env.D.PhiEvaluator(env.domain(v), spec)
                ctxs[q["sid"]] = PhiContext(evs[q["problem_id"]], v, q["phi0"], q.get("bad0") or 0)
            return ctxs[q["sid"]]
    else:
        def ctx_of(q):
            return None
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
        for i, m in enumerate(q["moves"]):
            if i < a.min_qi:
                continue
            cs.update(move_counts(q, truth_facets(q, m), nochange_facets(q), ctx_of(q)))
        per["nochange"][q["sid"]] = cs
        if a.kshot_baselines:                              # the sequential pass's context, copied without physics
            T = [truth_facets(q, m) for m in q["moves"]]
            for j in range(max(1, a.min_qi), len(T)):
                for name, pf in kshot_baselines(q, T[:j]).items():
                    per[name].setdefault(q["sid"], collections.Counter()).update(move_counts(q, T[j], pf, ctx_of(q)))
    formats = {}
    seen = set()
    for p in a.preds:
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:                             # a line torn by a failed write (the client rewrites it)
                continue
            if (r.get("pass") != a.pass_ or r["sid"] not in qs or str(r.get("finish", "")).startswith("error")
                    or r["qi"] < a.min_qi):
                continue
            key = (r["predictor"], r["sid"], r["pass"], r["qi"])
            if key in seen:                                # a resumed client can repeat a question: first answer counts
                continue
            seen.add(key)
            q = qs[r["sid"]]
            fmt = r.get("target", "verbatim")
            formats[r["predictor"]] = fmt
            t = truth_facets(q, q["moves"][r["qi"]])
            if r.get("facets") is not None:                # a predictor that outputs facets directly (trees)
                pf = dict(r["facets"])
                pf["violated"] = set(pf.get("violated") or [])
                if "change" not in pf:                     # size from the predicted bin's midpoint, as for the facet format
                    pf["change"], pf["after"] = {}, dict(q["metrics0"])
                    for k, dr in (pf.get("dir") or {}).items():
                        if k in q["metrics0"]:
                            b0 = float(q["metrics0"][k])
                            v = _mid_value(b0, dr, (pf.get("bin") or {}).get(k, "same"))
                            if v is not None:
                                pf["change"][k], pf["after"][k] = v - b0, v
            else:
                pf = pred_facets(r.get("text"), fmt, q)
            per[r["predictor"]].setdefault(r["sid"], collections.Counter()).update(move_counts(q, t, pf, ctx_of(q)))
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

    def group(pred, key, b=None, fl=None):
        vals = [family_scores(fam_counter(pred, v, None if b is None else b[v]))[key] for v in (fl or fams)]
        vals = [x for x in vals if not math.isnan(x)]
        return float(np.mean(vals)) if vals else float("nan")

    # primary: all families; sensitivity check: without photonics (its simulator interface includes an adjoint map;
    # owner 11-25: the map stays, it is part of that simulator's interface)
    fsets = [("all (primary)", fams)]
    if "photonics" in fams and len(fams) > 1:
        fsets.append(("no_photonics (sensitivity check)", [v for v in fams if v != "photonics"]))

    lines = ["facet profile, pass %s, groups %s, families %s, states %s" % (
        a.pass_, a.groups, fams, {v: len(sids[v]) for v in fams})]
    keys = ["select", "regime_phys", "feas_BA", "cross_BA", "V_bal_phys", "V_bal_applied", "cpl_dir_BA", "cpl_size",
            "V_bal", "refused_physical_recall", "refused_rule_recall",
            "regime_BA", "dir_BA", "size", "bk_dir_BA", "bk_size", "adj_dir_BA", "adj_size", "logratio_mean_abs",
            "viol_exact", "refused_recall", "refused_precision", "feas_recall", "feas_precision", "disc", "unreadable"]
    preds = ["nochange"] + sorted(k for k in per if k != "nochange")
    lines.append("%-10s %-11s" % ("predictor", "family") + "".join("%10s" % k[:10] for k in keys))
    res = {}
    for pr in preds:
        for v in fams:
            f = family_scores(fam_counter(pr, v))
            res["%s/%s" % (pr, v)] = f
            lines.append("%-10s %-11s" % (pr, v) + "".join("%10.3f" % f[k] for k in keys))
    gkeys = ("select", "regime_phys", "V_bal_phys", "cpl_dir_BA", "cpl_size", "bk_dir_BA", "V_bal")
    rkeys = ("select", "V_bal_phys", "cpl_dir_BA", "regime_phys")
    arms = [p for p in preds if p != "nochange"]
    for fname, fl in fsets:
        lines.append("")
        lines.append("group means over %s %s (+- se, bootstrap over states)" % (fname, fl))
        for pr in preds:
            row = []
            for k in gkeys:
                pt = group(pr, k, None, fl)
                bs = [group(pr, k, b, fl) for b in boots]
                se = float(np.nanstd(bs, ddof=1))
                res["group/%s/%s/%s" % (fname, pr, k)] = {"value": pt, "se": se}
                row.append("%s %.3f+-%.3f" % (k, pt, se))
            lines.append("%-10s %s  [%s]" % (pr, " ".join(row), formats.get(pr, "-")))
        lines.append("paired differences over %s (+- se): selection score and the rule metrics" % fname)
        for i, x in enumerate(arms):
            for y in arms[i + 1:] + ["nochange"]:
                row = []
                for k in rkeys:
                    pt = group(x, k, None, fl) - group(y, k, None, fl)
                    bs = [group(x, k, b, fl) - group(y, k, b, fl) for b in boots]
                    se = float(np.nanstd(bs, ddof=1))
                    res["diff/%s/%s-%s/%s" % (fname, x, y, k)] = {"value": pt, "se": se}
                    row.append("%s %+.3f+-%.3f" % (k, pt, se))
                lines.append("  %-22s %s" % ("%s - %s" % (x, y), " ".join(row)))
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt + "\n")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
