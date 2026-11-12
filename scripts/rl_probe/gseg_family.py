"""Is a training signal shared only inside a task family? gseg_analysis.py averages over all pairs of
problem groups; a direction that only problems of one family have in common is diluted there by the
share of same-family pairs (about 1 in 9). This computes rho separately over same-family pairs and
different-family pairs (same definition: mean pair dot product / mean squared norm, jackknife over
groups), for the label on all segments, with every (reply, segment) gradient clipped to 3x the
median norm.

usage: python gseg_family.py <segment sketch dir> <rollouts dir>
"""
import collections
import glob
import json
import math
import os
import sys

import numpy as np

sk, rollouts = sys.argv[1], sys.argv[2]
H, BONUS = 12, 0.2


def reward(ep):
    t = ep.get("first_feasible_turn")
    return 0.0 if t is None or t > H else 1.0 + BONUS * (H - t) / H


def load_step(s):
    f = os.path.join(rollouts, "step_%05d.compact.jsonl" % s)
    if not os.path.exists(f):
        f = os.path.join(rollouts, "step_%05d.jsonl" % s)
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        r.pop("calls", None)
        yield r


def turn_facts(r):
    ep = r["episode"]
    turns = ep.get("turns", [])
    phi = ep.get("phi_start")
    out = []
    first_dead = next((k for k, t in enumerate(turns) if "episode has ended" in str(t.get("fail"))), None)
    for k, t in enumerate(turns):
        new = t.get("phi_after")
        if new is None:
            new = phi
        d = (new - phi) if (new is not None and phi is not None) else 0.0
        out.append({"dead": first_dead is not None and k >= first_dead, "legal": 0.0 if t.get("fail") else 1.0,
                    "dphi": float(np.sign(d))})
        phi = new
    return out


metas = sorted(glob.glob(os.path.join(sk, "s*_k*.meta.jsonl")))
steps = sorted({int(os.path.basename(m)[1:6]) for m in metas})
episodes = {}
for s in steps:
    for r in load_step(s):
        episodes[r["key"]] = r
facts = {k: turn_facts(r) for k, r in episodes.items()}
units = sorted({r["group"] for r in episodes.values()})
uidx = {g: i for i, g in enumerate(units)}
ufam = np.array([next(r["variant"] for r in episodes.values() if r["group"] == g) for g in units])
U = len(units)
samples = []
for fi, m in enumerate(metas):
    for row, line in enumerate(open(m, encoding="utf-8")):
        d = json.loads(line)
        if d.get("error") or d["key"] not in episodes:
            continue
        samples.append((fi, row, d))
n = len(samples)
B = len(samples[0][2]["g2_seg"])
key = [s[2]["key"] for s in samples]
turn = np.array([s[2]["turn"] for s in samples])
unit = np.array([uidx[s[2]["group"]] for s in samples])
F = [facts[k][t] if t < len(facts[k]) else None for k, t in zip(key, turn)]
live = np.array([f is not None and not f["dead"] for f in F])
g2 = np.array([s[2]["g2_seg"] for s in samples])
nseg = np.array([np.diff(s[2]["seg"]) for s in samples]).astype(float)
lpseg = np.array([s[2]["lp_seg"] for s in samples])


def centred(vals, mask):
    out = np.zeros(n)
    for b in np.unique(unit):
        sel = mask & (unit == b)
        if sel.sum() > 1:
            out[sel] = vals[sel] - vals[sel].mean()
    return out


def col(c):
    return np.array([f[c] if f is not None else 0.0 for f in F])


R = np.array([reward(episodes[k]["episode"]) for k in key])
allm = np.ones(n, dtype=bool)
ones = np.ones(B)
W = collections.OrderedDict()
W["episode"] = np.outer(centred(R, allm), ones)
W["legal move"] = np.outer(centred(col("legal"), live), ones)
W["potential up/down"] = np.outer(centred(col("dphi"), live), ones)
W["control: random sign"] = np.random.default_rng(0).choice([-1.0, 1.0], size=(n, B))
W["positive control: log-prob per token"] = np.stack(
    [centred(lpseg[:, b] / np.maximum(nseg[:, b], 1), nseg[:, b] > 0) for b in range(B)], 1)
names = list(W)
gn = np.sqrt(np.maximum(g2, 1e-30))
scale = np.minimum(1.0, 3 * np.median(gn) / gn)
Wm = (np.stack([W[k] for k in names]) * scale[None]).astype(np.float32)
dim = np.load(metas[0].replace(".meta.jsonl", ".seg.npy"), mmap_mode="r").shape[2]
G = np.zeros((len(names), U, dim), dtype=np.float32)
by_file = collections.defaultdict(list)
for j, s in enumerate(samples):
    by_file[s[0]].append(j)
for fi, m in enumerate(metas):
    js = np.array(by_file.get(fi, []), dtype=int)
    if not len(js):
        continue
    rows = np.array([samples[j][1] for j in js])
    S = np.load(m.replace(".meta.jsonl", ".seg.npy"))[rows].astype(np.float32)
    onehot = np.zeros((U, len(js)), dtype=np.float32)
    onehot[unit[js], np.arange(len(js))] = 1.0
    flat = S.reshape(len(js) * B, dim)
    for k in range(len(names)):
        G[k] += (onehot[:, :, None] * Wm[k, js][None]).reshape(U, len(js) * B) @ flat
    del S, flat
same = (ufam[:, None] == ufam[None, :]) & ~np.eye(U, dtype=bool)
diff = ufam[:, None] != ufam[None, :]


def rho(Kq, pairs):
    return (Kq[pairs].mean() / (np.trace(Kq) / Kq.shape[0])) if pairs.any() else float("nan")


def stat(Kg, pairs):
    r = rho(Kg, pairs)
    jk = []
    for i in range(U):
        keep = np.r_[0:i, i + 1:U]
        jk.append(rho(Kg[np.ix_(keep, keep)], pairs[np.ix_(keep, keep)]))
    jk = np.array(jk)
    return r, math.sqrt((U - 1) / U * ((jk - jk.mean()) ** 2).sum())


fams = sorted(set(ufam))
print("steps %s, groups %d (%s), replies %d; same-family pairs %d, different-family pairs %d" % (
    steps, U, " ".join("%s %d" % (f, (ufam == f).sum()) for f in fams), n, same.sum() // 2, diff.sum() // 2))
print("\n%-40s %22s %22s %s" % ("label on all segments (clipped)", "same family", "different family", "same family, by family (rho)"))
for k, nm in enumerate(names):
    Kg = G[k].astype(np.float64) @ G[k].astype(np.float64).T
    rs, ss = stat(Kg, same)
    rd, sd = stat(Kg, diff)
    per = []
    for f in fams:
        m = ufam == f
        Kf = Kg[np.ix_(m, m)]
        k_ = m.sum()
        per.append("%s %.4f" % (f[:6], (Kf.sum() - np.trace(Kf)) / (k_ * (k_ - 1)) / (np.trace(Kf) / k_)) if k_ > 1 else "%s -" % f[:6])
    up = rs + 2 * ss
    need = "%.0f" % ((1 - rs) / rs) if rs > 2 * ss and rs > 0 else ("at least %.0f" % ((1 - up) / up) if up > 0 else "-")
    print("%-40s %8.4f +- %.4f (%4.1f) %8.4f +- %.4f (%4.1f)  %s" % (nm[:40], rs, ss, rs / ss if ss else 0, rd, sd, rd / sd if sd else 0, " ".join(per)))
    print("%-40s same-family groups for a half-signal gradient: %s" % ("", need))
