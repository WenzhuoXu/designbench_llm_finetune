"""con1 skill-test baseline (b): boosted trees on move + state features, per family (docs §3).

    python trees.py --train native.jsonl.gz scripted.jsonl.gz --queries queries.jsonl --out preds_trees.jsonl

Trained on every simulated transition (all legal moves, own and off-grid moves) of con1's training states of a
family (--rows: the states its consequence rows use; con1 itself sees 12 of the moves of each): a LightGBM 3-class
classifier for the verdict (refused / worse / better) and a LambdaRank model (one query = one state; label = the
move's quality grade within its state) for ordering. Features: the verb, every argument by name (numbers as numbers,
anything else as a category), on verb-surface families the element's current value and the ratio new / current,
and the state: its metric values, Phi, binding constraint, turn, moves applied, legal-set size. Families never
trained on (held-out families) get no tree prediction. Output: one line per (test state, pass 'zero') with
preds [{qi, move, verdict, score}] (score = ranker output; refused-predicted moves are ranked last by the
evaluator, as for the models).
"""
import argparse
import collections
import gzip
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_rows import HELD_OUT, rank_key  # noqa: E402

VERDICTS = ["refused", "worse", "better"]


def opener(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


class Featurizer:
    def __init__(self):
        self.cols = {}
        self.cats = collections.defaultdict(dict)

    def _cat(self, col, val):
        d = self.cats[col]
        if val not in d:
            d[val] = len(d)
        return d[val]

    def row(self, st, m, grow: bool):
        f = {}
        a = m.get("args") or {}
        f["verb"] = ("cat", a.get("verb") or m["move"].split("(", 1)[0])
        for k, v in (a.get("args") or {}).items():
            x = num(v)
            f["arg:%s" % k] = ("num", x) if x is not None else ("cat", str(v))
        if a.get("v_old") is not None:
            vo = num(a["v_old"])
            vn = num((a.get("args") or {}).get("v"))
            f["v_old"] = ("num", vo)
            if vo not in (None, 0.0) and vn is not None:
                f["ratio"] = ("num", vn / vo)
                f["logratio"] = ("num", math.log(abs(vn / vo)) if vn / vo > 0 else None)
        for k, v in (st.get("metrics0") or {}).items():
            f["m0:%s" % k] = ("num", num(v))
        f["phi0"] = ("num", num(st.get("phi0")))
        f["binding"] = ("cat", str(st.get("binding")))
        f["t"] = ("num", float(st.get("t") or 0))
        f["k"] = ("num", float(st.get("k") or 0))
        f["n_legal"] = ("num", float(st.get("n_legal") or 0))
        f["bad0"] = ("num", float(st.get("bad0") or 0))
        out = {}
        for col, (kind, val) in f.items():
            if col not in self.cols:
                if not grow:
                    continue
                self.cols[col] = (len(self.cols), kind)
            if self.cols[col][1] == "cat":
                if val is None:
                    continue
                if not grow and val not in self.cats[col]:
                    out[col] = -1.0
                    continue
                out[col] = float(self._cat(col, val))
            else:
                out[col] = np.nan if val is None else val
        return out

    def matrix(self, dicts):
        X = np.full((len(dicts), len(self.cols)), np.nan, dtype=np.float64)
        for i, d in enumerate(dicts):
            for col, val in d.items():
                X[i, self.cols[col][0]] = val
        return X

    def cat_idx(self):
        return [i for col, (i, kind) in self.cols.items() if kind == "cat"]


def grade(st, moves):
    """Quality grade 0..4 of each move within its state: refused 0, applied moves by rank quintile 1..4."""
    app = [m for m in moves if m["verdict"] != "refused"]
    order = sorted(app, key=lambda m: rank_key(st["variant"], m))
    pos = {id(m): i for i, m in enumerate(order)}
    g = []
    for m in moves:
        if m["verdict"] == "refused":
            g.append(0)
        else:
            g.append(1 + int(4 * pos[id(m)] / max(1, len(order))) if len(order) > 1 else 2)
    return [min(4, x) for x in g]


def main(argv=None):
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=None, help="con1 training rows: train on exactly the states they use (by sid)")
    ap.add_argument("--max-states", type=int, default=0, help="per family, 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    queries = collections.defaultdict(list)
    for line in open(a.queries, encoding="utf-8"):
        q = json.loads(line)
        queries[q["variant"]].append(q)
    keep = None
    if a.rows:
        keep = set()
        for line in opener(a.rows):
            r = json.loads(line)
            if r.get("tier") == "con1_conseq":
                keep.add(r["sid"])
    data = collections.defaultdict(list)
    for p in a.train:
        for line in opener(p):
            st = json.loads(line)
            if st["split"] != "train" or not st["moves"] or (keep is not None and st["sid"] not in keep):
                continue
            data[st["variant"]].append(st)
    report = []
    with open(a.out, "w", encoding="utf-8") as fo:
        for v in sorted(queries):
            if v in HELD_OUT or v not in data:
                report.append("%s: no tree (family not trained on)" % v)
                continue
            sts = data[v]
            if a.max_states:
                sts = sts[:a.max_states]
            fz = Featurizer()
            dicts, y, grp, rel = [], [], [], []
            for st in sts:
                g = grade(st, st["moves"])
                for m, gg in zip(st["moves"], g):
                    dicts.append(fz.row(st, m, grow=True))
                    y.append(VERDICTS.index(m["verdict"]))
                    rel.append(gg)
                grp.append(len(st["moves"]))
            X = fz.matrix(dicts)
            y = np.array(y)
            classes = sorted(set(y.tolist()))
            params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=a.seed, verbose=-1)
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X, y, categorical_feature=fz.cat_idx() or "auto")
            rk = lgb.LGBMRanker(**params)
            rk.fit(X, np.array(rel), group=grp, categorical_feature=fz.cat_idx() or "auto")
            report.append("%s: trained on %d states, %d transitions, %d features, classes %s" % (
                v, len(sts), len(y), X.shape[1], [VERDICTS[c] for c in classes]))
            for q in queries[v]:
                Xq = fz.matrix([fz.row(q, m, grow=False) for m in q["moves"]])
                proba = clf.predict_proba(Xq)
                score = rk.predict(Xq)
                preds = []
                for qi, (m, pr, sc) in enumerate(zip(q["moves"], proba, score)):
                    preds.append({"qi": qi, "move": m["move"], "verdict": VERDICTS[classes[int(np.argmax(pr))]],
                                  "score": float(sc), "proba": {VERDICTS[c]: float(p) for c, p in zip(classes, pr)}})
                fo.write(json.dumps({"sid": q["sid"], "predictor": "trees", "pass": "zero", "preds": preds}) + "\n")
    print("\n".join(report))
    print("TREES_DONE")


if __name__ == "__main__":
    main()
