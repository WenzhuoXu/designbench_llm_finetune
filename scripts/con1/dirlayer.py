"""con1 direction control layer (docs/consequence_round_con1.md section 12): model minus the best no-physics
comparator on the same cell-balanced items, paired bootstrap over problems. CPU only; re-run per answer file.

    python dirlayer.py --queries queries_v2.jsonl --states test_states.jsonl.gz --pred s1=preds.jsonl [--pred ...]
        --cmp rule_vs=preds_rule_vs.jsonl --cmp rule_elem=... --cmp rule_cat=... [--cmp trees=...]
        [--thermal-reader d3_reader.py] [--obs obs_test.jsonl] [--pass zero|seq] [--both a,b] [--out out.json]

Reused from the 11-26 test review (review_tmp): c1_pilot.py's verb/sign/element keys and cell balance, f5_rulepred.py's
rule answer files (--cmp), d3_reader.py's thermal2 reader (executed from its own source), d5_regime.py's margin
predictor. rule_vs is required: every comparator backs off to it where it has no value.
"""
import argparse
import collections
import gzip
import json
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import facets as F  # noqa: E402

SIGN_VERBS = {                                             # c1_pilot.py
    "SET_BASE_STOCK", "SET_REVIEW_PERIOD", "SET_CONDUCTANCE",
    "SET_WIDTH", "SET_LENGTH", "SET_BIAS_CURRENT", "SET_COMP_CAP", "SET_MIRROR_RATIO", "SET_NULLING_RESISTOR",
    "SET_RADIUS", "MOVE_JOINT", "DISPATCH", "SET_PHASE_SHIFT",
    "SET_INTERVENTION_DURATION", "SET_INTERVENTION_START", "SET_VACCINE_SHARE", "SET_CONTACT_REDUCTION",
    "SET_TESTING_RATE", "SET_SCHOOL_POLICY", "SET_CYCLE", "SET_GREEN_SPLIT", "SET_LANES", "SET_OFFSET",
    "REDISPATCH", "SET_STORAGE",
}
EXTRA_EXCL = {"inventory2": {"order_cost", "expedite_cost"}}
DROP = {"thermal2": {"hotspot_node"}, "grid2op": {"served_load_mw", "energy_cost", "operating_cost"}}
UNSEEN = ("inventory2", "molopt", "thermal2")
MOLOPT = {"logp": "molopt/logp+tpsa", "tpsa": "molopt/logp+tpsa"}
FLOOR = 0.01


def opener(p):
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")


def verb_of(m):
    return m["move"].split("(")[0].strip()


def _args(m):
    a = m.get("args") or {}
    return a, (a.get("args") if isinstance(a.get("args"), dict) else {})


def sign_of(m):
    a, aa = _args(m)
    if verb_of(m) not in SIGN_VERBS:
        return "na"
    if "v_old" in a and "v" in aa:
        d = float(aa["v"]) - float(a["v_old"])
    elif "mw" in aa:
        d = float(aa["mw"])
    else:
        return "na"
    return "+" if d > 0 else ("-" if d < 0 else "0")


def group_of(v, k):
    if v == "molopt":
        return MOLOPT.get(k, "molopt/oracle+qed+sa")
    return v


def load_preds(paths, qs):
    d = {"zero": {}, "seq": {}}
    for line in (ln for p in paths.split(",") for ln in opener(p)):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        ps = r.get("pass", "zero")
        if ps not in d or r["sid"] not in qs or str(r.get("finish", "")).startswith("error"):
            continue
        key = (r["sid"], r["qi"])
        if key in d[ps]:
            continue
        q = qs[r["sid"]]
        d[ps][key] = dict(r["facets"]) if r.get("facets") is not None else F.pred_facets(
            r.get("text"), r.get("target", "verbatim"), q)
    return d


def pdir(pf, k):
    if pf is None or not F.is_applied(pf["regime"]):
        return None
    return (pf.get("dir") or {}).get(k, "same")


def thermal_reader(path, qs):
    """d3_reader.py's reader(q), executed from its own source (definitions only)."""
    src = open(path, encoding="utf-8").read()
    i, j = src.index("# ---------------- thermal2 reader"), src.index("RD = {")
    ns = {"re": re, "verb_of": verb_of}
    exec(src[i:j], ns)                                     # noqa: S102 (the review's own code)
    out = {}
    for s, q in qs.items():
        if q["variant"] == "thermal2":
            for i2, d in ns["reader"](q).items():
                out[(s, i2)] = {k: ("up" if x > 0 else "down") for k, x in d.items() if "#" not in k and x}
    return out


def photonics_reader(path, qs):
    feats = {}
    for line in opener(path):
        r = json.loads(line)
        if r.get("variant") == "photonics":
            feats[r["sid"]] = r.get("feats") or {}
    out = {}
    for s, q in qs.items():
        if q["variant"] != "photonics" or s not in feats:
            continue
        for i, m in enumerate(q["moves"]):
            g = (feats[s].get(m["move"]) or {}).get("adj_gain")
            if g:
                out[(s, i)] = {"target_loss": "down" if g > 0 else "up"}
    return out


def dn_copier(states_path, qs):
    """grid2op: every move predicted with DO_NOTHING's directions at the same state (test_states)."""
    want = {s for s, q in qs.items() if q["variant"] == "grid2op"}
    dn = {}
    for line in opener(states_path):
        st = json.loads(line)
        if st["sid"] not in want:
            continue
        for m in st["moves"]:
            if m["move"].startswith("DO_NOTHING"):
                t = F.truth_facets(qs[st["sid"]], m)
                if F.is_applied(t["regime"]):
                    dn[st["sid"]] = dict(t["dir"])
    out = {}
    for s in dn:
        for i in range(len(qs[s]["moves"])):
            out[(s, i)] = dn[s]
    return out, len(dn), len(want)


def inv_noise(states_path, live):
    """inventory2: per metric, the 90th percentile of |relative change| over no-op moves (new value = current)."""
    ch = collections.defaultdict(list)
    for line in opener(states_path):
        st = json.loads(line)
        if st["variant"] != "inventory2":
            continue
        m0 = st["metrics0"]
        for m in st["moves"]:
            a, aa = _args(m)
            if "v_old" not in a or "v" not in aa or float(aa["v"]) != float(a["v_old"]) or m["verdict"] == "refused":
                continue
            for k in live:
                b, x = float(m0.get(k, 0) or 0), float((m.get("metrics_after") or {}).get(k, 0) or 0)
                if b:
                    ch[k].append(abs(x - b) / abs(b))
    out = {}
    for k, xs in ch.items():
        xs.sort()
        out[k] = xs[min(len(xs) - 1, int(0.9 * len(xs)))]
    return out, {k: len(v) for k, v in ch.items()}


SFL = re.compile(r"^\s+([A-Za-z_][\w.\-]*)\s+([-+]?[\d.eE+-]+)\s+limit\s*(<=|>=)\s*([-+]?[\d.eE+-]+)(.*)$")


def shortfall(u):                                          # d5_regime.py's state-only predictor
    i = u.find("CONSTRAINT STATUS")
    if i < 0:
        return 0.0
    j = u.find("\n\n", i)
    tot = 0.0
    for line in u[i:j if j > 0 else len(u)].splitlines()[1:]:
        mm = SFL.match(line)
        if mm and "VIOLATED" in mm.group(5):
            try:
                v, lim = float(mm.group(2)), float(mm.group(4))
            except ValueError:
                continue
            tot += abs(v - lim) / (abs(lim) if lim != 0 else 1.0)
    return -tot


def balanced(lines, pred, cond=False):
    """(score, W, signed share): lines = [(cell, k, td, line_id)], pred(line_id, k) -> dir or None."""
    N, A, C = collections.Counter(), collections.Counter(), collections.Counter()
    for cell, k, td, lid in lines:
        p = pred(lid, k)
        key = (cell, k, td)
        N[key] += 1
        A[key] += p in ("up", "down")
        C[key] += p == td
    W = num = sh = 0.0
    for (cell, k) in {(a, b) for (a, b, c) in N}:
        base = A if cond else N
        nu, nd = base.get((cell, k, "up"), 0), base.get((cell, k, "down"), 0)
        w = min(nu, nd)
        if not w:
            continue
        W += w
        num += w * (C[(cell, k, "up")] / nu + C[(cell, k, "down")] / nd) / 2
        if not cond:
            sh += w * (A[(cell, k, "up")] / nu + A[(cell, k, "down")] / nd) / 2
    return (num / W if W else float("nan")), W, (sh / W if W and not cond else float("nan"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--pred", action="append", default=[])
    ap.add_argument("--cmp", action="append", default=[])
    ap.add_argument("--thermal-reader", default=None)
    ap.add_argument("--obs", default=None)
    ap.add_argument("--pass", dest="pass_", default="zero")
    ap.add_argument("--both", default=None, help="name1,name2: also score only lines both give a direction")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    qs = {}
    for line in opener(a.queries):
        q = json.loads(line)
        if not a.families or q["variant"] in a.families:
            qs[q["sid"]] = q
    fams = sorted({q["variant"] for q in qs.values()})
    T = {s: [F.truth_facets(q, m) for m in q["moves"]] for s, q in qs.items()}
    LIVE = {}
    for v in fams:
        excl = set(F.DIRECT.get(v, set())) | EXTRA_EXCL.get(v, set()) | DROP.get(v, set())
        cm = collections.defaultdict(collections.Counter)
        for s, q in qs.items():
            if q["variant"] == v:
                for t in T[s]:
                    if F.is_applied(t["regime"]):
                        for k, d in t["dir"].items():
                            cm[k][d] += 1
        LIVE[v] = sorted(k for k in cm if k not in excl and cm[k]["up"] + cm[k]["down"] > 0)
    noise, nn = inv_noise(a.states, LIVE.get("inventory2", [])) if "inventory2" in fams else ({}, {})
    P = {}
    for spec in a.pred:
        n, p = spec.split("=", 1)
        P[n] = load_preds(p, qs)
    CMP = {}
    for spec in a.cmp:
        n, p = spec.split("=", 1)
        CMP[n] = load_preds(p, qs)["zero"]
    if "rule_vs" not in CMP:
        raise SystemExit("--cmp rule_vs=... is required (the back-off of every comparator)")
    built = {}
    if a.thermal_reader:
        built["reader_thermal2"] = thermal_reader(a.thermal_reader, qs)
    if a.obs:
        built["reader_photonics"] = photonics_reader(a.obs, qs)
    dn, n_dn, n_g = dn_copier(a.states, qs) if "grid2op" in fams else ({}, 0, 0)
    if dn:
        built["donothing"] = dn

    # items, per pass: zero = every applied truth move with a zero answer of the first model; seq = positions >= 1
    base_model = next(iter(P)) if P else None
    items = collections.defaultdict(lambda: {"main": [], "sub": [], "dropped_noise": 0})
    for s, q in qs.items():
        v = q["variant"]
        for i, m in enumerate(q["moves"]):
            if a.pass_ == "seq" and i == 0:
                continue
            t = T[s][i]
            if not F.is_applied(t["regime"]):
                continue
            for k in LIVE[v]:
                td = t["dir"].get(k)
                if td not in ("up", "down"):
                    continue
                b0 = float(q["metrics0"].get(k, 0) or 0)
                rel = abs(t["change"].get(k, 0.0)) / abs(b0) if b0 else math.inf
                g = group_of(v, k)
                if v == "inventory2" and rel <= noise.get(k, 0.0):
                    items[g]["dropped_noise"] += 1
                    continue
                cell = (verb_of(m), sign_of(m))
                items[g]["main" if rel >= FLOOR else "sub"].append((cell, k, td, (s, i)))
    # predictors as functions of (line id, metric)
    def model_fn(name, ps):
        d = P[name][ps]
        return lambda lid, k: pdir(d.get(lid), k)

    def cmp_fn(name):
        d = CMP.get(name)
        vs = CMP["rule_vs"]
        bt = built.get(name)

        def f(lid, k):
            if d is not None:
                x = pdir(d.get(lid), k)
            else:
                x = (bt.get(lid) or {}).get(k)
            return x if x in ("up", "down") else pdir(vs.get(lid), k)
        return f

    def sametype_fn(lid, k):
        s, j = lid
        mv = qs[s]["moves"]
        ty = (verb_of(mv[j]), sign_of(mv[j]))
        for i in range(j - 1, -1, -1):
            if (verb_of(mv[i]), sign_of(mv[i])) == ty and F.is_applied(T[s][i]["regime"]):
                x = T[s][i]["dir"].get(k)
                if x in ("up", "down"):
                    return x
                break
        return pdir(CMP["rule_vs"].get(lid), k)

    comps = {n: cmp_fn(n) for n in list(CMP) + list(built)}
    if a.pass_ == "seq":
        comps["sametype_copier"] = sametype_fn
    models = {n: model_fn(n, a.pass_) for n in P}
    if a.pass_ == "seq":
        models.update({n + "@zero": model_fn(n, "zero") for n in P})
    pid = {s: q["problem_id"] for s, q in qs.items()}
    rng = random.Random(0)
    res, lines_out = {}, []
    hdr = "pass %s | per row: items main (pairs W), sub-floor lines; per model: (a) non-answers wrong, (b) share given a direction, (c) among directed; best comparator; model - best +- se (paired bootstrap over problems)" % a.pass_
    lines_out.append(hdr)
    for g in sorted(items):
        L = items[g]["main"]
        if not L:
            continue
        sc = {n: balanced(L, f) for n, f in comps.items()}
        best = max(sc, key=lambda n: sc[n][0] if sc[n][0] == sc[n][0] else -1)
        row = {"lines_main": len(L), "lines_sub": len(items[g]["sub"]), "dropped_noise": items[g]["dropped_noise"],
               "comparators": {n: sc[n][0] for n in sc}, "best": best, "W": sc[best][1]}
        byp = collections.defaultdict(list)
        for x in L:
            byp[pid[x[3][0]]].append(x)
        probs = sorted(byp)
        draws = []
        for _ in range(a.boot):
            dr = []
            for _ in probs:
                dr.extend(byp[probs[rng.randrange(len(probs))]])
            draws.append(dr)
        mrow = {}
        for n, f in models.items():
            ans = [lid for lid in {x[3] for x in L} if (P[n.split("@")[0]]["zero" if n.endswith("@zero") else a.pass_]).get(lid) is not None]
            if not ans:
                continue
            s_a, W, sh = balanced(L, f)
            s_c, Wc, _ = balanced(L, f, cond=True)
            unread = sum(1 for lid in {x[3] for x in L}
                         if ((P[n.split("@")[0]]["zero" if n.endswith("@zero") else a.pass_]).get(lid) or {}).get("regime") == "unreadable")
            diffs = []
            for dr in draws:
                x, y = balanced(dr, f)[0], balanced(dr, comps[best])[0]
                if x == x and y == y:
                    diffs.append(x - y)
            mu = sum(diffs) / len(diffs) if diffs else float("nan")
            se = (sum((d - mu) ** 2 for d in diffs) / max(1, len(diffs) - 1)) ** 0.5 if diffs else float("nan")
            mrow[n] = {"a": s_a, "b": sh, "c": s_c, "Wc": Wc, "unreadable_lines": unread, "minus_best": s_a - sc[best][0], "se": se}
            sub = items[g]["sub"]
            if sub:
                mrow[n]["a_subfloor"] = balanced(sub, f)[0]
        row["models"] = mrow
        res[g] = row
        txt = "%-22s main %5d lines (W %4d), sub %4d%s | best %s %.3f | " % (
            g + ("*" if g.split("/")[0] in UNSEEN else ""), len(L), sc[best][1], len(items[g]["sub"]),
            (", noise-dropped %d" % items[g]["dropped_noise"]) if g == "inventory2" else "", best, sc[best][0])
        txt += " ; ".join("%s a %.3f b %.2f c %.3f | -best %+.3f +- %.3f" % (n, r["a"], r["b"], r["c"], r["minus_best"], r["se"])
                          for n, r in mrow.items())
        lines_out.append(txt)
        lines_out.append("    comparators: " + "  ".join("%s %.3f" % (n, sc[n][0]) for n in sorted(sc)))
    if a.both:
        n1, n2 = a.both.split(",")
        lines_out.append("lines both %s and %s give a direction (pass %s):" % (n1, n2, a.pass_))
        for g in sorted(items):
            L = [x for x in items[g]["main"] if models[n1](x[3], x[1]) in ("up", "down") and models[n2](x[3], x[1]) in ("up", "down")]
            if L:
                lines_out.append("  %-22s lines %d | %s %.3f | %s %.3f | best comparator on them %.3f" % (
                    g, len(L), n1, balanced(L, models[n1])[0], n2, balanced(L, models[n2])[0],
                    max(balanced(L, f)[0] for f in comps.values())))
    # regime: model minus the margin reader (LOPO threshold), cell-balanced inside verb+sign cells holding both classes
    lines_out.append("regime (applied moves, feasible vs infeasible after; model minus the margin reader that ignores the move):")
    for v in fams:
        rows = []
        for s, q in qs.items():
            if q["variant"] != v:
                continue
            sf = shortfall(q["user"])
            for i, m in enumerate(q["moves"]):
                if a.pass_ == "seq" and i == 0:
                    continue
                t = T[s][i]
                if F.is_applied(t["regime"]):
                    rows.append({"pid": q["problem_id"], "cell": (verb_of(m), sign_of(m)), "y": t["regime"] == "applied-feasible",
                                 "sc": sf, "lid": (s, i)})
        cells = collections.defaultdict(list)
        for r in rows:
            cells[r["cell"]].append(r)

        def rbal(rr_all, pred):
            cc = collections.defaultdict(list)
            for r in rr_all:
                cc[r["cell"]].append(r)
            W = num = 0.0
            for rr in cc.values():
                fe, inf = [r for r in rr if r["y"]], [r for r in rr if not r["y"]]
                w = min(len(fe), len(inf))
                if w:
                    W += w
                    num += w * (sum(pred(r) for r in fe) / len(fe) + sum(not pred(r) for r in inf) / len(inf)) / 2
            return (num / W if W else float("nan")), W
        cand = sorted({r["sc"] for r in rows})[::max(1, len(rows) // 40 or 1)]
        thr = {}
        for p_ in sorted({r["pid"] for r in rows}):
            rest = [r for r in rows if r["pid"] != p_]
            best_t = max(cand, key=lambda tq: rbal(rest, lambda r: r["sc"] >= tq)[0] if rest else 0) if cand else 0.0
            thr[p_] = best_t
        reader = lambda r: r["sc"] >= thr.get(r["pid"], 0.0)  # noqa: E731
        rr_score, W = rbal(rows, reader)
        if not W:
            lines_out.append("  %-11s no verb+sign cell with both classes" % v)
            continue
        txt = "  %-11s applied %d feasible %d W %d | margin reader %.3f" % (v, len(rows), sum(r["y"] for r in rows), W, rr_score)
        byp = collections.defaultdict(list)
        for r in rows:
            byp[r["pid"]].append(r)
        probs = sorted(byp)
        for n in P:
            d = P[n][a.pass_]
            mp = lambda r, d=d: (d.get(r["lid"]) or {}).get("regime") == "applied-feasible"  # noqa: E731
            ms, _ = rbal(rows, mp)
            diffs = []
            for _ in range(a.boot):
                dr = []
                for _ in probs:
                    dr.extend(byp[probs[rng.randrange(len(probs))]])
                x, y = rbal(dr, mp)[0], rbal(dr, reader)[0]
                if x == x and y == y:
                    diffs.append(x - y)
            mu = sum(diffs) / len(diffs) if diffs else float("nan")
            se = (sum((d_ - mu) ** 2 for d_ in diffs) / max(1, len(diffs) - 1)) ** 0.5 if diffs else float("nan")
            txt += " | %s %.3f, minus reader %+.3f +- %.3f" % (n, ms, ms - rr_score, se)
            res.setdefault("regime/" + v, {})[n] = {"model": ms, "reader": rr_score, "diff": ms - rr_score, "se": se, "W": W}
        lines_out.append(txt)
    # can / cannot (section 12 thresholds)
    lines_out.append("can / cannot show state-dependent prediction (main items >= 50 pairs; best comparator <= 0.80; grid2op < 80% lines with DO_NOTHING's direction):")
    for g in sorted(res):
        if g.startswith("regime/"):
            continue
        r = res[g]
        why = []
        if r["W"] < 50:
            why.append("only %d pairs" % r["W"])
        if r["comparators"][r["best"]] > 0.80:
            why.append("best comparator %s %.3f" % (r["best"], r["comparators"][r["best"]]))
        if g == "grid2op" and dn:
            L = items[g]["main"]
            agree = sum(1 for (cell, k, td, lid) in L if (dn.get(lid) or {}).get(k) == td)
            share = agree / max(1, len(L))
            r["donothing_share"] = share
            if share >= 0.8:
                why.append("%.0f%% of lines share DO_NOTHING's direction" % (100 * share))
        lines_out.append("  %-22s %s" % (g, "CAN" if not why else "CANNOT: " + "; ".join(why)))
    if "inventory2" in fams:
        lines_out.append("inventory2 no-op noise (90th pct |rel change|, n no-op lines): " +
                         ", ".join("%s %.4f (%d)" % (k, noise[k], nn[k]) for k in sorted(noise)))
    if dn:
        lines_out.append("grid2op DO_NOTHING outcomes found for %d of %d states" % (n_dn, n_g))
    print("\n".join(lines_out))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
