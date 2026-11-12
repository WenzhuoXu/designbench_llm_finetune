"""Can a value estimate read off the model's hidden state place credit inside the rationale?

Input: grad_probe.py --no-grad --hidden-points H output (last hidden state at H positions over the
rationale of every reply) and the run's rollouts. For each label (episode reward; per-turn facts)
a ridge regression hidden state -> label, one model for all positions, cross-fitted by problem
group (a reply is always predicted by a model that never saw its problem). Ridge strength: chosen
inside the training folds (one inner fold held out), never on the predicted fold.

Reported per position: R2 of the held-out predictions, overall and within problem group (label and
prediction both centred per group: what the estimate adds to the group-mean baseline), and how
much of the label a reply's own rationale resolves (R2 at the end minus R2 at the start).

usage: python critic_hidden.py <hidden dir> <rollouts dir> [--out file.npz] [--folds 8]
"""
import argparse
import collections
import glob
import json
import os
import zlib

import numpy as np

H_, BONUS = 12, 0.2
ap = argparse.ArgumentParser()
ap.add_argument("hid")
ap.add_argument("rollouts")
ap.add_argument("--out", default=None)
ap.add_argument("--folds", type=int, default=8)
a = ap.parse_args()
LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


def reward(ep):
    t = ep.get("first_feasible_turn")
    return 0.0 if t is None or t > H_ else 1.0 + BONUS * (H_ - t) / H_


def load_step(s):
    f = os.path.join(a.rollouts, "step_%05d.compact.jsonl" % s)
    if not os.path.exists(f):
        f = os.path.join(a.rollouts, "step_%05d.jsonl" % s)
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        r.pop("calls", None)
        yield r


def turn_facts(r):
    ep = r["episode"]
    turns = ep.get("turns", [])
    phi0 = ep.get("phi_start")
    phi = phi0
    out = []
    first_dead = next((k for k, t in enumerate(turns) if "episode has ended" in str(t.get("fail"))), None)
    for k, t in enumerate(turns):
        dead = first_dead is not None and k >= first_dead
        new = t.get("phi_after")
        if new is None:
            new = phi
        d = (new - phi) if (new is not None and phi is not None) else 0.0
        out.append({"dead": dead, "legal": 0.0 if t.get("fail") else 1.0,
                    "fatal": 1.0 if (first_dead is not None and k == first_dead - 1) else 0.0,
                    "up": 1.0 if d > 0 else 0.0, "feas": 1.0 if t.get("feasible_after") else 0.0})
        phi = new
    return out


# ---------------------------------------------------------------- data
metas = sorted(glob.glob(os.path.join(a.hid, "s*_k*.meta.jsonl")))
steps = sorted({int(os.path.basename(m)[1:6]) for m in metas})
episodes = {}
for s in steps:
    for r in load_step(s):
        episodes[r["key"]] = r
facts = {k: turn_facts(r) for k, r in episodes.items()}
rows, Hs = [], []
for m in metas:
    h = np.load(m.replace(".meta.jsonl", ".h.npy"), mmap_mode="r")
    for i, line in enumerate(open(m, encoding="utf-8")):
        d = json.loads(line)
        if d.get("error") or d["key"] not in episodes:
            continue
        rows.append(d)
        Hs.append(np.asarray(h[i]))
X = np.stack(Hs)                                           # replies x positions x d, float16 (cast where used)
del Hs
n, P, D = X.shape
print("hidden files %d, steps %s, replies %d, positions %d, hidden size %d" % (len(metas), steps, n, P, D))
print("non-finite hidden values:", int((~np.isfinite(X)).sum()), " max |h| %.1f" % float(np.abs(X).max()))
group = np.array([d["group"] for d in rows])
fam = np.array([d["variant"] for d in rows])
turn = np.array([d["turn"] for d in rows])
key = [d["key"] for d in rows]
F = [facts[k][t] if t < len(facts[k]) else None for k, t in zip(key, turn)]
live = np.array([f is not None and not f["dead"] for f in F])
R = np.array([reward(episodes[k]["episode"]) for k in key])
allm = np.ones(n, dtype=bool)


def col(c):
    return np.array([f[c] if f is not None else 0.0 for f in F])


LABELS = collections.OrderedDict()                          # name -> (values, mask of replies it is defined on)
LABELS["episode reward"] = (R, allm)
LABELS["episode succeeds"] = ((R > 0).astype(float), allm)
LABELS["move is legal"] = (col("legal"), live)
LABELS["potential goes up"] = (col("up"), live)
LABELS["turn reaches feasibility"] = (col("feas"), live)
LABELS["move ends the episode"] = (col("fatal"), live)
names = list(LABELS)
Y = np.stack([LABELS[k][0] for k in names], 1)              # replies x labels
M = np.stack([LABELS[k][1] for k in names], 1)
ug = sorted(set(group))
fold_of_group = {g: zlib.crc32(g.encode()) % a.folds for g in ug}
# balance: crc32 buckets can be uneven; reassign by sorted order instead, families interleaved
order = sorted(ug, key=lambda g: (zlib.crc32(g.encode()), g))
fold_of_group = {g: i % a.folds for i, g in enumerate(order)}
fold = np.array([fold_of_group[g] for g in group])
print("groups %d, folds %s, live %d; label means: %s" % (
    len(ug), np.bincount(fold, minlength=a.folds).tolist(), live.sum(),
    ", ".join("%s %.3f" % (k, Y[M[:, j], j].mean()) for j, k in enumerate(names))))

# ---------------------------------------------------------------- sufficient statistics per fold and mask
masks = {"all": allm, "live": live}
mask_of_label = ["all" if LABELS[k][1] is allm else "live" for k in names]
stat = {}
for mk, mm in masks.items():
    for f in range(a.folds):
        sel = mm & (fold == f)
        Xf = X[sel].reshape(-1, D).astype(np.float64)
        Yf = np.repeat(Y[sel], P, axis=0)
        stat[(mk, f)] = (len(Xf), Xf.sum(0), Xf.T @ Xf, Xf.T @ Yf, Yf.sum(0))
        del Xf
print("sufficient statistics done")


def fit(mk, train_folds):
    """Ridge on the standardised hidden state from the given folds: returns predict(X2d, lam index)."""
    nn = sum(stat[(mk, f)][0] for f in train_folds)
    sx = sum(stat[(mk, f)][1] for f in train_folds)
    G = sum(stat[(mk, f)][2] for f in train_folds)
    Xy = sum(stat[(mk, f)][3] for f in train_folds)
    sy = sum(stat[(mk, f)][4] for f in train_folds)
    mu, ybar = sx / nn, sy / nn
    C = G - nn * np.outer(mu, mu)
    sd = np.sqrt(np.maximum(np.diag(C) / nn, 1e-12))
    Cs = C / np.outer(sd, sd) / nn                          # correlation matrix
    b = (Xy - np.outer(mu, sy)) / sd[:, None] / nn
    ev, U = np.linalg.eigh(Cs)
    Ub = U.T @ b
    Ws = [U @ (Ub / (np.maximum(ev, 0)[:, None] + lam)) for lam in LAMS]   # d x labels, per lambda

    def predict(X2, li):
        return ((X2 - mu) / sd) @ Ws[li] + ybar
    return predict


def r2(y, v):
    sst = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - v) ** 2).sum() / sst if sst > 0 else float("nan")


V = np.zeros((n, P, len(names)))                             # held-out predictions
chosen = np.zeros((a.folds, len(names)), dtype=int)
for f in range(a.folds):
    train = [g for g in range(a.folds) if g != f]
    inner = train[f % len(train)]                            # one inner fold held out to choose the ridge strength
    for mk, mm in masks.items():
        lab = [j for j in range(len(names)) if mask_of_label[j] == mk]
        pin = fit(mk, [g for g in train if g != inner])
        sel = mm & (fold == inner)
        Xi = X[sel].reshape(-1, D).astype(np.float64)
        Yi = np.repeat(Y[sel], P, axis=0)
        score = np.array([[r2(Yi[:, j], pin(Xi, li)[:, j]) for j in lab] for li in range(len(LAMS))])
        best = score.argmax(0)
        pout = fit(mk, train)
        te = fold == f
        Xt = X[te].reshape(-1, D).astype(np.float64)
        for jj, j in enumerate(lab):
            chosen[f, j] = best[jj]
            V[te, :, j] = pout(Xt, best[jj])[:, j].reshape(-1, P)
    print("fold %d done; ridge strength chosen per label: %s" % (f, [LAMS[i] for i in chosen[f]]), flush=True)


# ---------------------------------------------------------------- report
def within(y, v, by):
    """R2 after centring label and prediction per ``by``."""
    yc, vc = y.copy(), v.copy()
    for b in np.unique(by):
        s = by == b
        yc[s] -= y[s].mean()
        vc[s] -= v[s].mean()
    sst = (yc ** 2).sum()
    return 1 - ((yc - vc) ** 2).sum() / sst if sst > 0 else float("nan")


gid = np.unique(group, return_inverse=True)[1]
show = [0, P // 8, P // 4, P // 2, 3 * P // 4, P - 1]
for j, nm in enumerate(names):
    mm = M[:, j]
    y = Y[mm, j]
    gm, g_ = np.zeros(len(y)), gid[mm]
    for g in np.unique(g_):
        gm[g_ == g] = y[g_ == g].mean()
    print("\n== %s  (replies %d, mean %.3f, share of variance inside problem groups %.2f)" % (
        nm, mm.sum(), y.mean(), ((y - gm) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)))
    print("   position in rationale:   " + "  ".join("%5.0f%%" % (100 * p / (P - 1)) for p in show))
    print("   R2 overall               " + "  ".join("%6.3f" % r2(y, V[mm, p, j]) for p in show))
    print("   R2 within problem group  " + "  ".join("%6.3f" % within(y, V[mm, p, j], gid[mm]) for p in show))
    t0 = mm & (turn == 0)
    if t0.sum() > 50:
        print("   same, first turn only    " + "  ".join("%6.3f" % within(Y[t0, j], V[t0, p, j], gid[t0]) for p in show))
    later = mm & (turn > 0)
    if later.sum() > 50:
        print("   same, later turns        " + "  ".join("%6.3f" % within(Y[later, j], V[later, p, j], gid[later]) for p in show))
    # how much moves inside one reply, and how noisy the path is
    dv = np.diff(V[mm, :, j], axis=1)
    tot = V[mm, -1, j] - V[mm, 0, j]
    print("   estimate start -> end: sd of the change %.3f; path roughness (sum of squared steps / squared change) %.2f"
          % (tot.std(), (dv ** 2).sum() / max((tot ** 2).sum(), 1e-12)))
    print("   by family, R2 within group at end of rationale: " + "  ".join(
        "%s %.2f" % (f_[:8], within(Y[mm & (fam == f_), j], V[mm & (fam == f_), -1, j], gid[mm & (fam == f_)]))
        for f_ in sorted(set(fam)) if (mm & (fam == f_)).sum() > 30))

# learning curve at the end-of-rationale position: fewer training folds
print("\nlearning curve (R2 within problem group at the end of the rationale, ridge 0.1), training folds used:")
li = LAMS.index(1e-1)
for ktr in (2, 4, a.folds - 1):
    Vk = np.zeros((n, len(names)))
    for f in range(a.folds):
        train = [g for g in range(a.folds) if g != f][:ktr]
        te = fold == f
        for mk in masks:
            lab = [j for j in range(len(names)) if mask_of_label[j] == mk]
            pr = fit(mk, train)(X[te, -1].astype(np.float64), li)
            Vk[np.ix_(np.where(te)[0], lab)] = pr[:, lab]
    print("  %d folds: " % ktr + "  ".join("%s %.3f" % (nm[:18], within(Y[M[:, j], j], Vk[M[:, j], j], gid[M[:, j]]))
                                          for j, nm in enumerate(names)))
if a.out:
    np.savez_compressed(a.out, V=V.astype(np.float32), names=np.array(names), key=np.array(key), turn=turn,
                        group=group, fam=fam, Y=Y, M=M, hpos=np.array([d["hpos"] for d in rows]))
    print("saved", a.out)
