"""Second scorer for arm_split.py's headline numbers, written apart (CPU; con1-2m-rows, 11-28).

    PYTHONDONTWRITEBYTECODE=1 python recount_split.py QUERIES QFLAGS LFLAGS COMP_PREDS COMP_BEST S1A,S1H ARM.jsonl \\
        [SUBSET.tsv|-] [LABEL:END ...]

Reader: the attack check's own lib.parse (review_tmp/v1128D/attack/lib.py, imported read-only, no bytecode written);
no import of facets.py, indep_score.py or dir_split.py. Truth recomputed here from the question file. Per group
(trained = heldout_problem, unseen = heldout_family), pooled over questions / lines, on truth-applied questions
without a question flag (and in SUBSET, and meeting every LABEL:END on the ARM file's think_end; LABEL is ignored,
the ARM file is the one selected on):
  feasibility by 'VIOLATED AFTER is none' (refused, no verdict or no VIOLATED line = wrong): accuracy, balanced
  direction on true up/down lines of coupled metrics (facets.DIRECT list copied below) without a line flag: arm,
  stage 1, table accuracy; and arm / stage 1 on the lines the table gets wrong
"""
import collections
import json
import pickle
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "/storage/wxu/con1/review_tmp/v1128D/attack")
import lib  # noqa: E402  (parse only)

DIRECT = {  # copy of facets.DIRECT (con1 scripts), kept here so this file imports none of them
    "circuit2": {"area", "power_mw"}, "engibench": {"volume", "n_components"}, "frame3d2": {"mass"},
    "grid2": {"commitment_cost", "n_committed"},
    "grid2op": {"step", "redispatch_mw", "redispatch_cost", "curtailed_mw", "storage_cost", "lines_out",
                "lines_unavailable"},
    "photonics": {"fill_fraction", "n_islands", "min_feature_violations"}, "seir2": {"doses_per_capita", "cost"},
    "traffic2": {"lanes_used"}, "inventory2": set(),
    "molopt": {"heavy_atoms", "mol_weight", "n_rings", "max_ring_size", "rot_bonds"},
    "thermal2": {"cost", "link_cost", "sink_cost"},
}


def direction(b, a):
    if abs(a - b) <= max(1e-5 * abs(b), 1e-12):
        return "same"
    return "up" if a > b else "down"


def first_records(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("pass", "zero") != "zero" or str(r.get("finish", "")).startswith("error"):
            continue
        out.setdefault((r["sid"], r["qi"]), r)
    return out


def main():
    qp, qfp, lfp, cpp, cbp, s1p, armp = sys.argv[1:8]
    subp = sys.argv[8] if len(sys.argv) > 8 and sys.argv[8] != "-" else None
    ends = [x.split(":")[1] for x in sys.argv[9:]]
    Q = {}
    for line in open(qp, encoding="utf-8"):
        r = json.loads(line)
        Q[r["sid"]] = r
    qf = {tuple([p.split("\t")[0], int(p.split("\t")[1])]) for p in open(qfp) if p.strip()}
    lf = set()
    for p in open(lfp):
        if p.strip():
            s, j, k, _ = p.rstrip("\n").split("\t")
            lf.add((s, int(j), k))
    comp = pickle.load(open(cpp, "rb"))
    best = pickle.load(open(cbp, "rb"))
    s1 = {}
    for p in s1p.split(","):
        s1.update(first_records(p))
    arm = first_records(armp)
    sub = None
    if subp:
        sub = {(p.split("\t")[0], int(p.split("\t")[1])) for p in open(subp) if p.strip()}
    T = collections.defaultdict(collections.Counter)
    for key in sorted(set(arm) & set(s1)):
        sid, qi = key
        if sid not in Q or key in qf or (sub is not None and key not in sub):
            continue
        if any(arm[key].get("think_end") != e for e in ends):
            continue
        q = Q[sid]
        mv = q["moves"][qi]
        if mv["verdict"] == "refused":
            continue
        g = "trained" if q["group"] == "heldout_problem" else "unseen"
        fam = q["variant"]
        names = list(q["metrics0"])
        pa = lib.parse(arm[key].get("text"), names)
        ps = lib.parse(s1[key].get("text"), names)
        feas = bool(mv["feasible"])
        t = T[g]
        t["q"] += 1
        t["q_feas_%s" % feas] += 1
        for who, p in (("arm", pa), ("s1", ps)):
            said = None if (p["v"] != "applied" or p["viol"] is None) else (len(p["viol"]) == 0)
            t["fhit_%s_%s" % (who, feas)] += said is not None and said == feas
        for k, b in q["metrics0"].items():
            a = mv["metrics_after"].get(k)
            if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                continue
            if k in DIRECT.get(fam, set()) or (sid, qi, k) in lf:
                continue
            td = direction(float(b), float(a))
            if td == "same":
                continue
            t["lines"] += 1
            tab = comp.get((key, k), {}).get(best.get(fam)) == td
            t["tab"] += tab
            for who, p in (("arm", pa), ("s1", ps)):
                ok = p["v"] == "applied" and p["d"].get(k) == td
                t["d_" + who] += ok
                if not tab:
                    t["tw_" + who] += ok
            t["tw_n"] += not tab
    print("# recount_split (attack lib.parse reader): %s; subset %s; think_end %s" % (armp, subp, ends or "any"))
    for g in ("trained", "unseen"):
        t = T[g]
        if not t["q"]:
            continue
        out = []
        for who in ("arm", "s1"):
            n1, n0 = t["q_feas_True"], t["q_feas_False"]
            h1, h0 = t["fhit_%s_True" % who], t["fhit_%s_False" % who]
            acc = (h1 + h0) / (n1 + n0)
            ba = ((h1 / n1 if n1 else 0) + (h0 / n0 if n0 else 0)) / ((n1 > 0) + (n0 > 0))
            out.append("%s feas %.3f BA %.3f" % (who, acc, ba))
        L = t["lines"] or 1
        TW = t["tw_n"] or 1
        print("  %-7s questions %d, lines %d | %s | direction arm %.3f s1 %.3f table %.3f | table-wrong %d: arm %.3f s1 %.3f"
              % (g, t["q"], t["lines"], "; ".join(out), t["d_arm"] / L, t["d_s1"] / L, t["tab"] / L, t["tw_n"],
                 t["tw_arm"] / TW, t["tw_s1"] / TW))
    print("RECOUNT_SPLIT_DONE")


if __name__ == "__main__":
    main()
