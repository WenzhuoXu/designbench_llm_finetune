"""Second scorer for unclear_table.py's pooled numbers, written apart (CPU; 11-28): the attack check's lib.parse reader
(imported read-only) for verdicts and direction words, its own regex for 'unclear' lines, own truth and counting.

    PYTHONDONTWRITEBYTECODE=1 python unclear_recount.py QUERIES QFLAGS LFLAGS UNCLEAR.jsonl D.jsonl REP1.jsonl REP2.jsonl

Per group (trained / unseen), coupled metrics without a line flag on truth-applied unflagged questions: unclear share
among written lines (moving, unchanged); on moving lines: D-unclear's accuracy on its committed lines and arm D's on
the same lines, arm D's accuracy on the unclear lines; unclear share where the three arm-D runs are all right / all
wrong / disagree.
"""
import collections
import json
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "/storage/wxu/con1/review_tmp/v1128D/attack")
import lib  # noqa: E402

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
UNC = re.compile(r"^[\s>\-*•\d.)|]*`?([A-Za-z][A-Za-z0-9_]*)`?\s*[:=|]?\s*\**\s*unclear\b", re.I)


def way(b, a):
    if abs(a - b) <= max(1e-5 * abs(b), 1e-12):
        return "same"
    return "up" if a > b else "down"


def first(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out.setdefault((r["sid"], r["qi"]), r)
    return out


def words_unclear(text, names):
    p = lib.parse(text, names)
    t = (text or "").replace("**", "")
    unc = set()
    if "CHANGES:" in t:
        for ln in t.split("CHANGES:", 1)[1].splitlines():
            g = UNC.match(ln)
            if g and g.group(1) in names:
                unc.add(g.group(1))
    d = dict(p["d"])
    for k in unc:
        d.setdefault(k, "unclear")
    return p["v"], d


def main():
    qp, fp, lp, up, dp, r1, r2 = sys.argv[1:8]
    Q = {json.loads(l)["sid"]: json.loads(l) for l in open(qp, encoding="utf-8")}
    qf = {(x.split("\t")[0], int(x.split("\t")[1])) for x in open(fp) if x.strip()}
    lf = {(x.split("\t")[0], int(x.split("\t")[1]), x.split("\t")[2]) for x in open(lp) if x.strip()}
    U, D, A, B = first(up), first(dp), first(r1), first(r2)
    T = collections.defaultdict(collections.Counter)
    for key in U:
        if key not in D or key not in A or key not in B or key in qf or key[0] not in Q:
            continue
        q = Q[key[0]]
        mv = q["moves"][key[1]]
        if mv["verdict"] == "refused":
            continue
        names = list(q["metrics0"])
        vu, du = words_unclear(U[key].get("text"), names)
        runs = [lib.parse(x[key].get("text"), names) for x in (D, A, B)]
        g = "trained" if q["group"] == "heldout_problem" else "unseen"
        t = T[g]
        for k, b in q["metrics0"].items():
            a = mv["metrics_after"].get(k)
            if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                continue
            if k in DIRECT.get(q["variant"], set()) or (key[0], key[1], k) in lf:
                continue
            td = way(float(b), float(a))
            wu = du.get(k) if vu == "applied" else None
            kind = "moving" if td != "same" else "unchanged"
            if wu is None:
                continue
            t[kind + " written"] += 1
            t[kind + " unclear"] += wu == "unclear"
            if td == "same":
                continue
            ok = [p["v"] == "applied" and p["d"].get(k) == td for p in runs]
            if wu == "unclear":
                t["unc n"] += 1
                t["unc D"] += ok[0]
            else:
                t["com n"] += 1
                t["com U"] += wu == td
                t["com D"] += ok[0]
            grp = "right3" if all(ok) else "wrong3" if not any(ok) else "disagree"
            t[grp + " n"] += 1
            t[grp + " unclear"] += wu == "unclear"
    print("# unclear_recount (attack reader + own unclear regex): %s" % up)
    f = lambda t, a, b: t[a] / t[b] if t[b] else float("nan")  # noqa: E731
    for g in ("trained", "unseen"):
        t = T[g]
        print("  %-7s unclear share moving %.3f (%d) unchanged %.3f (%d) | committed %d: D-unclear %.3f arm D %.3f | "
              "unclear %d: arm D %.3f | unclear share: 3 right %.3f (%d), 3 wrong %.3f (%d), disagree %.3f (%d)" % (
                  g, f(t, "moving unclear", "moving written"), t["moving written"],
                  f(t, "unchanged unclear", "unchanged written"), t["unchanged written"], t["com n"],
                  f(t, "com U", "com n"), f(t, "com D", "com n"), t["unc n"], f(t, "unc D", "unc n"),
                  f(t, "right3 unclear", "right3 n"), t["right3 n"], f(t, "wrong3 unclear", "wrong3 n"), t["wrong3 n"],
                  f(t, "disagree unclear", "disagree n"), t["disagree n"]))
    print("UNCLEAR_RECOUNT_DONE")


if __name__ == "__main__":
    main()
