"""Second, independently written scorer for con1 answer files (CPU; con1-2m-rows, 11-27): recomputes the headline
facets numbers without importing facets.py or targets.py, from the raw question records and the answer text, so a
reported number can be stated twice.

    python indep_score.py --queries queries_v2.jsonl --preds P.jsonl --predictor LAB [--pass zero] [--group heldout_problem]

Definitions (taken from facets.py's documentation, re-implemented here):
  truth regime   refused (physical when the refusal text names a simulation outcome: 'would end the episode',
                 'diverge', 'no AC solution', 'blackout', 'isolated load', 'non-connected grid'; else rule) /
                 applied-feasible / applied-infeasible (the record's 'feasible')
  truth dir      per metric, metrics_after vs metrics0: same when |after - before| <= max(1e-5 |before|, 1e-12)
  answer         'VERDICT: <word>'; none -> refused if the text says refused / rejected, else unreadable; refused ->
                 kind from the REASON line; applied -> feasible when 'FEASIBLE AFTER: y...'; 'VIOLATED AFTER:' list;
                 CHANGES block lines '<metric> <up|down|same> [<bin>]' (metric names as in metrics0, no punctuation)
  unreadable     share of answers read as unreadable
  regime_phys    mean of (balanced accuracy over the physical regimes refused-physical / applied-infeasible /
                 applied-feasible) and (crossing BA: mean of the recall of constraint-status flips and 1 - the false
                 flip rate, constraints and their current status from the prompt's CONSTRAINT STATUS block, statuses
                 after from the record's result message and from the answer's VIOLATED AFTER line; applied answers to
                 applied moves only)
  cpl_dir_BA     applied moves, metrics not 'directly moved' (the per-family DIRECT list): mean over the true classes
                 up / down / same of the recall; an answer that is not 'applied' is wrong on every line, a metric the
                 answer leaves out counts as 'same'
  cpl_size       those lines with a true change from a nonzero value: right direction and |log2(predicted change /
                 true change)| <= 1, the predicted change = the bin's geometric midpoint
Per family pooled over its questions; 'mean' = mean over families. Prints INDEP_SCORE_DONE.
"""
import argparse
import collections
import json
import math
import re

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
PHYS = re.compile(r"would end the episode|diverge|no AC solution|blackout|isolated load|non-connected grid", re.I)
EDGES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
NAMES = ["<0.1%"] + ["%g-%g%%" % (100 * EDGES[i], 100 * EDGES[i + 1]) for i in range(len(EDGES) - 1)] + [">1000%"]
PHYS_REGIMES = ("refused-physical", "applied-infeasible", "applied-feasible")


def same(b, a):
    return abs(a - b) <= max(1e-5 * abs(b), 1e-12)


def tdir(b, a):
    return "same" if same(b, a) else ("up" if a > b else "down")


def mid_change(before, d, name):
    if name not in NAMES or before == 0:
        return None
    i = NAMES.index(name)
    lo = EDGES[i - 1] if i > 0 else EDGES[0] / 3.0
    hi = EDGES[i] if i < len(EDGES) else lo * 3.0
    r = math.sqrt(lo * hi) * abs(before)
    return r if d == "up" else -r


def status_block(text, start, stop=None):
    i = text.find(start)
    if i < 0:
        return None
    j = text.find(stop, i) if stop else text.find("\n\n", i)
    return text[i:j if j > i else len(text)]


def prompt_status(user):
    blk = status_block(user, "CONSTRAINT STATUS")
    cons, now = [], set()
    for line in (blk or "").splitlines()[1:]:
        m = re.match(r"^\s+([A-Za-z_][\w.\-]*)\s+\S+\s+limit", line)
        if m:
            cons.append(m.group(1))
            if "VIOLATED" in line:
                now.add(m.group(1))
    return cons, now


def result_violated(result):
    blk = status_block(result, "CONSTRAINT STATUS", "EFFECT OF THIS MOVE") or ""
    return {m.group(1) for m in re.finditer(r"^\s+([A-Za-z_][\w.\-]*)\s+[^\n]*VIOLATED", blk, re.M)}


def read_answer(text, m0):
    text = text or ""
    v = re.search(r"VERDICT:\s*(\w+)", text)
    if not v:
        if "refused" in text.lower() or "rejected" in text.lower():
            return {"regime": "refused-physical" if PHYS.search(text) else "refused-rule"}
        return {"regime": "unreadable"}
    if v.group(1).lower().startswith("refus"):
        r = re.search(r"REASON:\s*(.*)", text)
        return {"regime": "refused-physical" if PHYS.search(r.group(1) if r else text) else "refused-rule"}
    fe = re.search(r"FEASIBLE AFTER:\s*(\w+)", text)
    out = {"regime": "applied-feasible" if fe and fe.group(1).lower().startswith("y") else "applied-infeasible",
           "viol": set(), "dir": {}, "chg": {}}
    vl = re.search(r"VIOLATED AFTER:\s*(.*)", text)
    if vl and vl.group(1).strip().lower() not in ("none", ""):
        out["viol"] = {x.strip() for x in vl.group(1).split(",") if x.strip()}
    if "CHANGES:" in text:
        for line in text.split("CHANGES:", 1)[1].splitlines():
            w = line.split()
            if len(w) < 2 or w[0] not in m0:
                continue
            if w[1].startswith("same"):
                out["dir"][w[0]] = "same"
            elif w[1] in ("up", "down"):
                out["dir"][w[0]] = w[1]
                out["chg"][w[0]] = mid_change(float(m0[w[0]]), w[1], " ".join(w[2:]))
    return out


def scores(c):
    def rec(pfx, classes):
        r = [c[pfx + "hit_" + x] / c[pfx + "n_" + x] for x in classes if c[pfx + "n_" + x]]
        return sum(r) / len(r) if r else float("nan")
    feas = rec("reg_", PHYS_REGIMES)
    cr = c["cross_hit"] / c["cross_n"] if c["cross_n"] else float("nan")
    fc = c["false_cross"] / c["nocross_n"] if c["nocross_n"] else float("nan")
    xs = [x for x in (cr, 1 - fc) if not math.isnan(x)]
    cross = sum(xs) / len(xs) if xs else float("nan")
    ph = [x for x in (feas, cross) if not math.isnan(x)]
    return {"unreadable": c["unreadable"] / c["answers"] if c["answers"] else float("nan"),
            "regime_phys": sum(ph) / len(ph) if ph else float("nan"),
            "cpl_dir_BA": rec("cpl_", ("up", "down", "same")),
            "cpl_size": c["size_hit"] / c["size_n"] if c["size_n"] else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--pass", dest="pass_", default="zero")
    ap.add_argument("--group", default="heldout_problem")
    a = ap.parse_args()
    qs = {}
    for line in open(a.queries):
        q = json.loads(line)
        if q["group"] == a.group:
            qs[q["sid"]] = q
    ans = {}
    for line in open(a.preds):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("predictor") != a.predictor or r.get("pass") != a.pass_ or r["sid"] not in qs:
            continue
        if str(r.get("finish", "")).startswith("error"):
            continue
        ans.setdefault((r["sid"], r["qi"]), r.get("text"))
    C = collections.defaultdict(collections.Counter)
    for (sid, qi), text in ans.items():
        q = qs[sid]
        m, m0, fam = q["moves"][qi], q["metrics0"], q["variant"]
        c = C[fam]
        p = read_answer(text, m0)
        c["answers"] += 1
        c["unreadable"] += p["regime"] == "unreadable"
        if m["verdict"] == "refused":
            treg = "refused-physical" if PHYS.search(m.get("refused") or m["result"]) else "refused-rule"
        else:
            treg = "applied-feasible" if m["feasible"] else "applied-infeasible"
        if treg in PHYS_REGIMES:
            c["reg_n_" + treg] += 1
            c["reg_hit_" + treg] += p["regime"] == treg
        if not treg.startswith("applied"):
            continue
        applied = p["regime"].startswith("applied")
        if applied:
            cons, now = prompt_status(q["user"])
            tv = result_violated(m["result"])
            for k in cons:
                tc, pc = (k in now) != (k in tv), (k in now) != (k in p["viol"])
                c["cross_n"] += tc
                c["cross_hit"] += tc and pc
                c["nocross_n"] += not tc
                c["false_cross"] += (not tc) and pc
        after = m["metrics_after"]
        for k, b in m0.items():
            if k not in after or k in DIRECT.get(fam, set()):
                continue
            td = tdir(float(b), float(after[k]))
            pd = p["dir"].get(k, "same") if applied else None
            c["cpl_n_" + td] += 1
            c["cpl_hit_" + td] += pd == td
            if td != "same" and float(b) != 0.0:
                tch = float(after[k]) - float(b)
                pch = p["chg"].get(k) if applied else None
                c["size_n"] += 1
                c["size_hit"] += bool(pd == td and pch and tch and abs(math.log2(abs(pch / tch))) <= 1.0)
    keys = ("unreadable", "regime_phys", "cpl_dir_BA", "cpl_size")
    print("independent scorer: %s, predictor %s, pass %s, group %s, answers %d" % (
        a.preds, a.predictor, a.pass_, a.group, len(ans)))
    print("  %-12s %7s " % ("family", "answers") + " ".join("%11s" % k for k in keys))
    S = {}
    for fam in sorted(C):
        S[fam] = scores(C[fam])
        print("  %-12s %7d " % (fam, C[fam]["answers"]) + " ".join("%11.3f" % S[fam][k] for k in keys))
    mean = {k: [S[f][k] for f in S if not math.isnan(S[f][k])] for k in keys}
    print("  %-12s %7d " % ("mean", sum(C[f]["answers"] for f in C)) +
          " ".join("%11.3f" % (sum(v) / len(v) if v else float("nan")) for v in mean.values()))
    print("INDEP_SCORE_DONE")


if __name__ == "__main__":
    main()
