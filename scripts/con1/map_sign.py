"""con1: how much the photonics adjoint map alone predicts (owner, 11-25). For each photonics question that was applied,
the sign of the map's first-order change of target_loss (-adj_gain, obs_features.py) against the sign of the true
change; also the share within factor 2 of the true change.

    python map_sign.py --queries queries.jsonl --feats obs_feats_test.jsonl [--states states_native*.jsonl.gz]
"""
import argparse
import collections
import gzip
import json
import math


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", nargs="*", default=[])
    ap.add_argument("--feats", nargs="+", required=True)
    a = ap.parse_args(argv)
    feats = {}
    for p in a.feats:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["variant"] == "photonics":
                feats[r["sid"]] = r.get("feats") or {}
    c = collections.Counter()
    for p in a.queries:
        f = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")
        for line in f:
            q = json.loads(line)
            if q["variant"] != "photonics" or q["sid"] not in feats:
                continue
            b = float(q["metrics0"]["target_loss"])
            for m in q["moves"]:
                ff = feats[q["sid"]].get(m["move"]) or {}
                if m["verdict"] == "refused" or "adj_gain" not in ff:
                    continue
                d_true = float(m["metrics_after"]["target_loss"]) - b
                d_map = -float(ff["adj_gain"])
                if abs(d_true) <= 1e-5 * max(abs(b), 1e-12):
                    c["true change ~0"] += 1
                    continue
                c["moves"] += 1
                c["sign agrees"] += (d_map > 0) == (d_true > 0) and d_map != 0
                c["within factor 2"] += d_map != 0 and (d_map > 0) == (d_true > 0) and abs(math.log2(abs(d_map / d_true))) <= 1
                c["map says ~0"] += d_map == 0
                for src in m["src"]:
                    c["moves %s" % src] += 1
                    c["sign agrees %s" % src] += (d_map > 0) == (d_true > 0) and d_map != 0
    for k in sorted(c):
        print("%-28s %d" % (k, c[k]))
    if c["moves"]:
        print("sign agreement %.3f, within factor 2 %.3f (n = %d)" % (c["sign agrees"] / c["moves"],
                                                                  c["within factor 2"] / c["moves"], c["moves"]))


if __name__ == "__main__":
    main()
