"""Second scorer for k4_table.py's pooled numbers, written apart (CPU; 11-28): the attack check's lib.parse reader
(imported read-only), own truth, own counting; no import of facets.py, indep_score.py or dir_split.py.

    PYTHONDONTWRITEBYTECODE=1 python k4_recount.py QUERIES QFLAGS LFLAGS ARMD.jsonl K4.jsonl [SAMPLES]

Per group, pooled over the true up/down lines of coupled metrics without a line flag on truth-applied unflagged
questions with all samples and an arm-D answer: arm D accuracy, mean sample accuracy, majority accuracy (strict
plurality, absent = 'none', tie = wrong), on D-wrong lines the share with >= 1 sample right and with all wrong, on
D-right lines the share with >= 1 sample wrong, and the share where every sample gives the same word.
"""
import collections
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "/storage/wxu/con1/review_tmp/v1128D/attack")
import lib  # noqa: E402  (parse only)

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


def way(b, a):
    if abs(a - b) <= max(1e-5 * abs(b), 1e-12):
        return "same"
    return "up" if a > b else "down"


def main():
    qp, fp, lp, dp, kp = sys.argv[1:6]
    S = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    Q = {}
    for line in open(qp, encoding="utf-8"):
        r = json.loads(line)
        Q[r["sid"]] = r
    qf = {(x.split("\t")[0], int(x.split("\t")[1])) for x in open(fp) if x.strip()}
    lf = {(x.split("\t")[0], int(x.split("\t")[1]), x.split("\t")[2]) for x in open(lp) if x.strip()}
    D = {}
    for line in open(dp, encoding="utf-8"):
        r = json.loads(line)
        D.setdefault((r["sid"], r["qi"]), r)
    K = collections.defaultdict(dict)
    for line in open(kp, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not str(r.get("finish", "")).startswith("error"):
            K[(r["sid"], r["qi"])].setdefault(int(r["sample"]), r)
    T = collections.defaultdict(collections.Counter)
    for key, smp in K.items():
        if len(smp) != S or key not in D or key in qf or key[0] not in Q:
            continue
        q = Q[key[0]]
        mv = q["moves"][key[1]]
        if mv["verdict"] == "refused":
            continue
        names = list(q["metrics0"])
        pd = lib.parse(D[key].get("text"), names)
        ps = [lib.parse(smp[s].get("text"), names) for s in range(1, S + 1)]
        g = "trained" if q["group"] == "heldout_problem" else "unseen"
        for k, b in q["metrics0"].items():
            a = mv["metrics_after"].get(k)
            if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                continue
            if k in DIRECT.get(q["variant"], set()) or (key[0], key[1], k) in lf:
                continue
            td = way(float(b), float(a))
            if td == "same":
                continue
            say = lambda p: p["d"].get(k, "none") if p["v"] == "applied" else "none"  # noqa: E731
            wd = say(pd)
            ws = [say(p) for p in ps]
            cnt = collections.Counter(ws).most_common()
            maj = cnt[0][0] if len(cnt) == 1 or cnt[0][1] > cnt[1][1] else None
            right = sum(x == td for x in ws)
            t = T[g]
            t["lines"] += 1
            t["d"] += wd == td
            t["samples"] += right
            t["maj"] += maj == td
            t["agree"] += len(cnt) == 1
            if wd == td:
                t["dr"] += 1
                t["dr_wrong1"] += right < S
            else:
                t["dw"] += 1
                t["dw_right1"] += right >= 1
                t["dw_all"] += right == 0
    print("# k4_recount (attack lib.parse reader): %s vs %s" % (kp, dp))
    for g in ("trained", "unseen"):
        t = T[g]
        if not t["lines"]:
            continue
        n = t["lines"]
        print("  %-7s lines %d | D %.3f mean sample %.3f majority %.3f | D wrong %d: >=1 right %.3f, all wrong %.3f | "
              "D right %d: >=1 wrong %.3f | all agree %.3f" % (
                  g, n, t["d"] / n, t["samples"] / (S * n), t["maj"] / n, t["dw"], t["dw_right1"] / max(1, t["dw"]),
                  t["dw_all"] / max(1, t["dw"]), t["dr"], t["dr_wrong1"] / max(1, t["dr"]), t["agree"] / n))
    print("K4_RECOUNT_DONE")


if __name__ == "__main__":
    main()
