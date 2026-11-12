"""Where in the rationale does a training signal have a direction, and does credit placed by a value
estimate have one? Works on grad_probe.py --segments B output (one sketch per equal run of rationale
tokens of every reply).

A weighting is one number per (reply, segment). The gradient of problem group u under it is
G_u = sum_{reply i in u} sum_b c[i, b] * sketch[i, b]; rho = (mean over pairs u != v of <G_u, G_v>)
/ (mean over u of |G_u|^2) is the share of a problem's gradient that the problems have in common
(jackknife se over groups), as in gprobe_analysis.py.

Weightings: a label on all segments (the reference: equals the whole-rationale result), the label
on one segment only, on the last k segments, and credit from a value estimate V (critic_hidden.py
--out): TD inside the reply (V after the segment minus V before it), the label minus V before the
segment (label with a prefix baseline), and a mix.

Reading the table (independent audit, 2026-11-12). The standard error is the one at rho ~ 0; with a
signal present it grows, so "at least N groups" overstates N (most for 'legal move', whose weight sits
in few groups). A direction shared only inside a task family is diluted by the share of same-family
pairs: see gseg_family.py. Weightings that do not sum to zero inside a group (the value-estimate rows)
pick up the constant direction that clipped / unit-scaled gradients have, which biases rho upward.

usage: python gseg_analysis.py <segment sketch dir> <rollouts dir> [--critic V.npz] [--whole <dir of .r.npy sketches>] [--norm clip|unit|raw]
"""
import argparse
import collections
import glob
import json
import math
import os

import numpy as np

H, BONUS = 12, 0.2
ap = argparse.ArgumentParser()
ap.add_argument("sk")
ap.add_argument("rollouts")
ap.add_argument("--critic", default=None)
ap.add_argument("--whole", default=None)
ap.add_argument("--norm", default="clip", choices=("raw", "clip", "unit"),
                help="clip (default): every (reply, segment) gradient clipped to 3x the median norm; unit: scaled to norm 1; "
                     "raw: as stored (one outlier reply can hold most of the energy and hide a signal: do not read raw rows alone)")
a = ap.parse_args()


def reward(ep):
    t = ep.get("first_feasible_turn")
    return 0.0 if t is None or t > H else 1.0 + BONUS * (H - t) / H


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
    phi = ep.get("phi_start")
    out = []
    first_dead = next((k for k, t in enumerate(turns) if "episode has ended" in str(t.get("fail"))), None)
    for k, t in enumerate(turns):
        dead = first_dead is not None and k >= first_dead
        new = t.get("phi_after")
        if new is None:
            new = phi
        d = (new - phi) if (new is not None and phi is not None) else 0.0
        out.append({"dead": dead, "legal": 0.0 if t.get("fail") else 1.0, "up": 1.0 if d > 0 else 0.0,
                    "dphi": float(np.sign(d))})
        phi = new
    return out


metas = sorted(glob.glob(os.path.join(a.sk, "s*_k*.meta.jsonl")))
steps = sorted({int(os.path.basename(m)[1:6]) for m in metas})
episodes = {}
for s in steps:
    for r in load_step(s):
        episodes[r["key"]] = r
facts = {k: turn_facts(r) for k, r in episodes.items()}
groups = collections.defaultdict(list)
for k, r in episodes.items():
    groups[r["group"]].append(k)
units = sorted(groups)
uidx = {g: i for i, g in enumerate(units)}
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
g2 = np.array([s[2]["g2_seg"] for s in samples])                         # replies x segments
nseg = np.array([np.diff(s[2]["seg"]) for s in samples]).astype(float)
lpseg = np.array([s[2]["lp_seg"] for s in samples])
skmax = max(s[2].get("sk_max", 0.0) for s in samples)
print("segment files %d, steps %s, groups %d, replies %d, segments %d, live %d" % (len(metas), steps, U, n, B, live.sum()))
print("largest sketch value %.1f (float16 holds 65504); share of gradient energy per segment: %s" % (
    skmax, " ".join("%.3f" % x for x in g2.sum(0) / g2.sum())))
print("tokens per segment (median) %d; reply log-prob per token by segment: %s" % (
    np.median(nseg), " ".join("%.3f" % x for x in lpseg.sum(0) / np.maximum(nseg.sum(0), 1))))


def centred(vals, mask, by):
    out = np.zeros(n)
    for b in np.unique(by):
        sel = mask & (by == b)
        if sel.sum() > 1:
            out[sel] = vals[sel] - vals[sel].mean()
    return out


def col(c):
    return np.array([f[c] if f is not None else 0.0 for f in F])


R = np.array([reward(episodes[k]["episode"]) for k in key])
allm = np.ones(n, dtype=bool)
lab = collections.OrderedDict()
lab["episode"] = centred(R, allm, unit)
lab["legal move"] = centred(col("legal"), live, unit)
lab["potential up/down"] = centred(col("dphi"), live, unit)
ones = np.ones(B)
W = collections.OrderedDict()                                            # name -> replies x segments
for nm, w in lab.items():
    W[nm + " | all segments"] = np.outer(w, ones)
for nm, w in lab.items():
    for b in range(B):
        e = np.zeros(B)
        e[b] = 1
        W["%s | segment %d only" % (nm, b + 1)] = np.outer(w, e)
    for k in (1, 2, 4):
        e = np.zeros(B)
        e[B - k:] = 1
        W["%s | last %d segments" % (nm, k)] = np.outer(w, e)
        e = np.zeros(B)
        e[:k] = 1
        W["%s | first %d segments" % (nm, k)] = np.outer(w, e)
if a.critic:
    z = np.load(a.critic, allow_pickle=False)
    pos = {(k, int(t)): i for i, (k, t) in enumerate(zip(z["key"], z["turn"]))}
    idx = np.array([pos.get((k, int(t)), -1) for k, t in zip(key, turn)])
    have = idx >= 0
    print("value estimates for %d of %d replies" % (have.sum(), n))
    Pn = z["V"].shape[1]
    bpos = [b * (Pn - 1) // B for b in range(B + 1)]                     # the segment boundaries among the positions
    cn = list(z["names"])
    for cname, y, mask in (("episode reward", R, allm), ("move is legal", col("legal"), live),
                           ("potential goes up", col("up"), live)):
        j = cn.index(cname)
        v = np.zeros((n, B + 1))
        v[have] = z["V"][idx[have]][:, bpos, j]
        m2 = (mask & have)[:, None]
        td = np.where(m2, np.diff(v, axis=1), 0.0)                       # V after the segment - V before it
        mc = np.where(m2, y[:, None] - v[:, :B], 0.0)                    # label - V before the segment
        tail = y - v[:, B]
        lam = 0.5
        gae = np.zeros((n, B))
        for b in range(B):
            gae[:, b] = sum(lam ** l * td[:, b + l] for l in range(B - b)) + lam ** (B - b) * tail
        gae = np.where(m2, gae, 0.0)
        W["value %s | TD inside the reply" % cname] = td
        W["value %s | label - V before the segment" % cname] = mc
        W["value %s | mix (lambda 0.5 per segment)" % cname] = gae
        W["value %s | V change over the reply, on all segments" % cname] = np.where(m2, (v[:, B] - v[:, 0])[:, None] * ones, 0.0)
rng = np.random.default_rng(0)
W["control: random sign per reply and segment"] = rng.choice([-1.0, 1.0], size=(n, B))
W["positive control: log-prob per token, per segment"] = np.stack(
    [centred(lpseg[:, b] / np.maximum(nseg[:, b], 1), nseg[:, b] > 0, unit) for b in range(B)], 1)
names = list(W)
K = len(names)
Wm = np.stack([W[k] for k in names]).astype(np.float32)                  # weightings x replies x segments
gn = np.sqrt(np.maximum(g2, 1e-30))
if a.norm == "clip":
    cmax = 3 * np.median(gn)
    scale = np.minimum(1.0, cmax / gn)
    print("clip: (reply, segment) gradients above %.1f (3x the median norm) are scaled down: %d of %d cells" % (cmax, (scale < 1).sum(), scale.size))
elif a.norm == "unit":
    scale = 1.0 / gn
else:
    scale = np.ones_like(gn)
Wm = Wm * scale[None].astype(np.float32)

dim = np.load(metas[0].replace(".meta.jsonl", ".seg.npy"), mmap_mode="r").shape[2]
G = np.zeros((K, U, dim), dtype=np.float32)
by_file = collections.defaultdict(list)
for j, s in enumerate(samples):
    by_file[s[0]].append(j)
whole_cos, whole_ratio = [], []
for fi, m in enumerate(metas):
    js = np.array(by_file.get(fi, []), dtype=int)
    if not len(js):
        continue
    rows = np.array([samples[j][1] for j in js])
    S = np.load(m.replace(".meta.jsonl", ".seg.npy"))[rows].astype(np.float32)       # replies x B x dim
    onehot = np.zeros((U, len(js)), dtype=np.float32)
    onehot[unit[js], np.arange(len(js))] = 1.0
    flat = S.reshape(len(js) * B, dim)
    for k in range(K):
        c = Wm[k, js]                                                    # replies x B
        G[k] += (onehot[:, :, None] * c[None]).reshape(U, len(js) * B) @ flat
    if a.whole:
        wf = os.path.join(a.whole, os.path.basename(m))
        if os.path.exists(wf):
            wkey = {(json.loads(l)["key"], json.loads(l)["turn"]): i for i, l in enumerate(open(wf, encoding="utf-8"))}
            Wh = np.load(wf.replace(".meta.jsonl", ".r.npy"), mmap_mode="r")
            for jj, j in enumerate(js):
                i = wkey.get((key[j], int(turn[j])))
                if i is not None:
                    x, y = S[jj].sum(0).astype(np.float64), np.asarray(Wh[i], dtype=np.float64)
                    whole_cos.append(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
                    whole_ratio.append(np.linalg.norm(x) / (np.linalg.norm(y) + 1e-30))
    del S, flat
if whole_cos:
    wc, wr = np.array(whole_cos), np.array(whole_ratio)
    print("segments summed against the whole-rationale sketch of the earlier run, %d replies: cosine median %.4f, "
          "5th percentile %.4f; norm ratio median %.4f" % (len(wc), np.median(wc), np.percentile(wc, 5), np.median(wr)))


def rho_stats(Kg):
    m = Kg.shape[0]

    def rho(Kq):
        k = Kq.shape[0]
        T = np.trace(Kq) / k
        S_ = (Kq.sum() - np.trace(Kq)) / (k * (k - 1))
        return S_ / T if T > 0 else 0.0
    r = rho(Kg)
    jk = np.array([rho(Kg[np.ix_(np.r_[0:i, i + 1:m], np.r_[0:i, i + 1:m])]) for i in range(m)])
    return r, math.sqrt((m - 1) / m * ((jk - jk.mean()) ** 2).sum())


print("\n%-64s %9s %8s %7s %s" % ("weighting", "rho", "se", "rho/se", "groups for a half-signal gradient"))
res = {}
for k, nm in enumerate(names):
    Gk = G[k].astype(np.float64)
    r, se = rho_stats(Gk @ Gk.T)
    res[nm] = (r, se)
    need = "%.0f" % ((1 - r) / r) if r > 2 * se and r > 0 else ("at least %.0f" % ((1 - (r + 2 * se)) / (r + 2 * se)) if r + 2 * se > 0 else "-")
    print("%-64s %9.4f %8.4f %7.1f  %s" % (nm[:64], r, se, r / se if se > 0 else 0.0, need))
print("\nrho by segment (label on that segment only)")
for nm in lab:
    print("%-20s %s" % (nm, "  ".join("%7.4f" % res["%s | segment %d only" % (nm, b + 1)][0] for b in range(B))))
    print("%-20s %s" % ("   se", "  ".join("%7.4f" % res["%s | segment %d only" % (nm, b + 1)][1] for b in range(B))))
