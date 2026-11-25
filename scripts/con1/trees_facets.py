"""con1 stop-rule baseline on the facets (docs/consequence_round_con1.md §8): boosted trees per family predicting each
move's facets from move + state features (trees.Featurizer), trained on every simulated transition of the given
training states, scored by facets.py exactly like a model's answers.

    python trees_facets.py --train states_*.jsonl.gz [--rows rows.jsonl] --queries queries.jsonl --out preds_trees.jsonl
        [--families ...] [--max-states-per-family N]

Models per family (LightGBM): regime (refused / applied-infeasible / applied-feasible); verdict (refused / worse /
better, the Phi rule); per constraint, violated after
(yes / no); per metric, the joint label 'direction + size bin' (targets.size_bin, e.g. 'up 1-3%' or 'same'), from which
direction and size are read. Output lines: {sid, predictor: 'trees', pass: 'zero', qi, target: 'facets', facets}.
"""
import argparse
import collections
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facets import current_violated, truth_facets  # noqa: E402
from trees import Featurizer  # noqa: E402

PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=10, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=int(os.environ.get("TREES_THREADS", "2")))


def opener(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def read_states(paths, keep=None, fams=None, cap=0):
    out = collections.defaultdict(list)
    for p in paths:
        try:
            for line in opener(p):
                st = json.loads(line)
                if st["split"] != "train" or not st["moves"]:
                    continue
                if keep is not None and st["sid"] not in keep:
                    continue
                if fams and st["variant"] not in fams:
                    continue
                if cap and len(out[st["variant"]]) >= cap:
                    continue
                out[st["variant"]].append(st)
        except EOFError:
            pass
    return out


def label_of(tf, k):
    d, b = tf["dir"].get(k), tf["bin"].get(k)
    if d is None:
        return None
    return "same" if b == "same" else "%s %s" % (d, b)


class Multi:
    """A classifier that also copes with a single class."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y, cats):
        import lightgbm as lgb
        self.classes = sorted(set(y))
        if len(self.classes) == 1:
            self.m = None
            return self
        idx = {c: i for i, c in enumerate(self.classes)}
        self.m = lgb.LGBMClassifier(random_state=self.seed, **PARAMS)
        self.m.fit(X, np.array([idx[c] for c in y]), categorical_feature=cats or "auto")
        return self

    def predict(self, X):
        if self.m is None:
            return [self.classes[0]] * len(X)
        return [self.classes[int(i)] for i in np.argmax(self.m.predict_proba(X), axis=1)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--groups", nargs="*", default=["heldout_problem"])
    ap.add_argument("--max-states-per-family", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    keep = None
    if a.rows:
        keep = set()
        for line in opener(a.rows):
            r = json.loads(line)
            if r.get("tier") == "con1_conseq":
                keep.add(r["sid"])
    data = read_states(a.train, keep, a.families, a.max_states_per_family)
    qs = [json.loads(l) for l in open(a.queries, encoding="utf-8")]
    qs = [q for q in qs if (not a.groups or q["group"] in a.groups) and (not a.families or q["variant"] in a.families)]
    with open(a.out, "w", encoding="utf-8") as fo:
        for v in sorted({q["variant"] for q in qs}):
            if v not in data:
                print("%s: no training states, no tree" % v)
                continue
            fz = Featurizer()
            X_rows, regime, verdict = [], [], []
            viol, lab = collections.defaultdict(list), collections.defaultdict(list)
            cons_all, metrics = set(), set()
            for st in data[v]:
                cons, _ = current_violated(st["user"])
                cons_all |= set(cons)
                metrics |= set(st["metrics0"])
            cons_all, metrics = sorted(cons_all), sorted(metrics)
            for st in data[v]:
                for m in st["moves"]:
                    tf = truth_facets(st, m)
                    X_rows.append(fz.row(st, m, grow=True))
                    regime.append(tf["regime"])
                    verdict.append(m["verdict"])
                    for c in cons_all:
                        viol[c].append(None if tf["regime"] == "refused" else (c in tf["violated"]))
                    for k in metrics:
                        lab[k].append(None if tf["regime"] == "refused" else label_of(tf, k))
            X = fz.matrix(X_rows)
            cats = fz.cat_idx()
            reg = Multi(a.seed).fit(X, regime, cats)
            ver = Multi(a.seed).fit(X, verdict, cats)
            vm, mm = {}, {}
            for c in cons_all:
                ok = [i for i, y in enumerate(viol[c]) if y is not None]
                if ok:
                    vm[c] = Multi(a.seed).fit(X[ok], [viol[c][i] for i in ok], cats)
            for k in metrics:
                ok = [i for i, y in enumerate(lab[k]) if y is not None]
                if ok:
                    mm[k] = Multi(a.seed).fit(X[ok], [lab[k][i] for i in ok], cats)
            print("%s: %d states, %d transitions, %d features, %d constraints, %d metrics" % (
                v, len(data[v]), len(regime), X.shape[1], len(vm), len(mm)), flush=True)
            for q in [q for q in qs if q["variant"] == v]:
                Xq = fz.matrix([fz.row(q, m, grow=False) for m in q["moves"]])
                rg = reg.predict(Xq)
                vg = ver.predict(Xq)
                vp = {c: mdl.predict(Xq) for c, mdl in vm.items()}
                lp = {k: mdl.predict(Xq) for k, mdl in mm.items()}
                for qi in range(len(q["moves"])):
                    f = {"regime": rg[qi], "verdict": vg[qi]}
                    if rg[qi] != "refused":
                        f["violated"] = sorted(c for c in vm if vp[c][qi])
                        f["dir"], f["bin"] = {}, {}
                        for k in lp:
                            L = lp[k][qi]
                            if L == "same":
                                f["dir"][k], f["bin"][k] = "same", "same"
                            else:
                                d, b = L.split(" ", 1)
                                f["dir"][k], f["bin"][k] = d, b
                    fo.write(json.dumps({"sid": q["sid"], "predictor": "trees", "pass": "zero", "qi": qi,
                                         "target": "facets", "facets": f}) + "\n")
    print("TREES_FACETS_DONE")


if __name__ == "__main__":
    main()
