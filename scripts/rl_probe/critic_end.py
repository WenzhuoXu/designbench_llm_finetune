"""A fairer test of the value estimate than the one model for all positions of critic_hidden.py:
models fitted for ONE position only (the end of the rationale, where the action is already fixed),
on the hidden state there, on its change over the reply (end - start), and a non-linear model on
principal components. Cross-fitted by problem group; reports R2 within problem group.

usage: python critic_end.py <hidden dir> <rollouts dir>
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
hid, rollouts = sys.argv[1], sys.argv[2]
H_, BONUS = 12, 0.2


def reward(ep):
    t = ep.get("first_feasible_turn")
    return 0.0 if t is None or t > H_ else 1.0 + BONUS * (H_ - t) / H_


def load_step(s):
    f = os.path.join(rollouts, "step_%05d.compact.jsonl" % s)
    if not os.path.exists(f):
        f = os.path.join(rollouts, "step_%05d.jsonl" % s)
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        r.pop("calls", None)
        yield r


metas = sorted(glob.glob(os.path.join(hid, "s*_k*.meta.jsonl")))
episodes = {}
for s in sorted({int(os.path.basename(m)[1:6]) for m in metas}):
    for r in load_step(s):
        episodes[r["key"]] = r
rows, h0, hm, h1 = [], [], [], []
for m in metas:
    h = np.load(m.replace(".meta.jsonl", ".h.npy"), mmap_mode="r")
    for i, line in enumerate(open(m, encoding="utf-8")):
        d = json.loads(line)
        if d.get("error") or d["key"] not in episodes:
            continue
        rows.append(d)
        h0.append(np.asarray(h[i, 0], dtype=np.float32))
        hm.append(np.asarray(h[i, h.shape[1] // 2], dtype=np.float32))
        h1.append(np.asarray(h[i, -1], dtype=np.float32))
h0, hm, h1 = np.stack(h0), np.stack(hm), np.stack(h1)
n = len(rows)
group = np.array([d["group"] for d in rows])
gid = np.unique(group, return_inverse=True)[1]
fam = np.array([d["variant"] for d in rows])
turn = np.array([d["turn"] for d in rows])
legal, up, live = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
R = np.zeros(n)
for i, d in enumerate(rows):
    ep = episodes[d["key"]]["episode"]
    R[i] = reward(ep)
    turns = ep.get("turns", [])
    dead = next((k for k, t in enumerate(turns) if "episode has ended" in str(t.get("fail"))), None)
    if d["turn"] < len(turns) and not (dead is not None and d["turn"] >= dead):
        live[i] = True
        t = turns[d["turn"]]
        legal[i] = 0.0 if t.get("fail") else 1.0
        phi = ep.get("phi_start")
        for k in range(d["turn"]):
            if turns[k].get("phi_after") is not None:
                phi = turns[k]["phi_after"]
        new = t.get("phi_after")
        up[i] = 1.0 if (new is not None and phi is not None and new > phi) else 0.0
print("replies %d, live %d, groups %d" % (n, live.sum(), len(set(group))))


def within(y, v, by):
    yc, vc = y.copy(), v.copy()
    for b in np.unique(by):
        s = by == b
        yc[s] -= y[s].mean()
        vc[s] -= v[s].mean()
    sst = (yc ** 2).sum()
    return 1 - ((yc - vc) ** 2).sum() / sst if sst > 0 else float("nan")


def crossfit(X, y, mask, kind):
    v = np.zeros(n)
    idx = np.where(mask)[0]
    for tr, te in GroupKFold(n_splits=8).split(X[idx], y[idx], gid[idx]):
        tr, te = idx[tr], idx[te]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A, Bt = (X[tr] - mu) / sd, (X[te] - mu) / sd
        if kind == "ridge":
            m = RidgeCV(alphas=np.logspace(1, 6, 11)).fit(A, y[tr])
            v[te] = m.predict(Bt)
        else:
            p = PCA(n_components=64, random_state=0).fit(A)
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
                                              l2_regularization=1.0, random_state=0).fit(p.transform(A), y[tr])
            v[te] = m.predict(p.transform(Bt))
    return v


feats = {"state at the end of the rationale": h1, "state at the start of the reply": h0, "state at the middle": hm,
         "change over the reply (end - start)": h1 - h0}
for lname, y, mask in (("episode reward", R, np.ones(n, dtype=bool)), ("move is legal", legal, live), ("potential goes up", up, live)):
    print("\n== %s (mean %.3f over %d replies)" % (lname, y[mask].mean(), mask.sum()))
    for fname, X in feats.items():
        for kind in ("ridge", "pca64+trees"):
            v = crossfit(X, y, mask, kind)
            sst = ((y[mask] - y[mask].mean()) ** 2).sum()
            line = "   %-38s %-12s R2 overall %6.3f  within problem group %6.3f" % (
                fname, kind, 1 - ((y[mask] - v[mask]) ** 2).sum() / sst, within(y[mask], v[mask], gid[mask]))
            later = mask & (turn > 0)
            line += "  (later turns %6.3f)" % within(y[later], v[later], gid[later])
            if lname == "move is legal":
                line += "  " + " ".join("%s %.2f" % (f[:6], within(y[mask & (fam == f)], v[mask & (fam == f)], gid[mask & (fam == f)]))
                                        for f in ("photonics", "grid2op", "engibench"))
            print(line, flush=True)
