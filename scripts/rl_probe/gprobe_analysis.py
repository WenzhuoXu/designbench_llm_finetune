"""Score training signals on the gradient sketches (llm_finetune/rl_dataxp/grad_probe.py output).

For a labelling w (one weight per reply) and a token part (rationale r, action a, both), the gradient
of problem group u is G_u = sum_{i in u} w_i * sketch_i. With G_u = s + n_u (s: the direction the
groups share, n_u: what is particular to a group),
    S = mean over pairs u != v of <G_u, G_v>   estimates |s|^2      (unbiased, also through the sketch)
    T = mean over u of |G_u|^2                 estimates |s|^2 + E|n|^2
    rho = S / T
and a batch of m groups points along s with cos^2 = m*rho / (1 + (m-1)*rho); m_half = (1-rho)/rho
groups are needed for cos^2 = 0.5. Standard errors: delete-one-group jackknife.

usage: python gprobe_analysis.py <sketch dir> <rollouts dir> [--out file.npz]
"""
import argparse
import collections
import glob
import json
import math
import os

import zlib

import numpy as np

H, BONUS = 12, 0.2
ap = argparse.ArgumentParser()
ap.add_argument("sk")
ap.add_argument("rollouts")
ap.add_argument("--out", default=None)
ap.add_argument("--critic-steps", type=int, default=100)
a = ap.parse_args()


def reward(ep):
    t = ep.get("first_feasible_turn")
    return 0.0 if t is None or t > H else 1.0 + BONUS * (H - t) / H


# ---------------------------------------------------------------- per-turn facts from the rollouts
def load_step(s):
    f = os.path.join(a.rollouts, "step_%05d.compact.jsonl" % s)
    if not os.path.exists(f):
        f = os.path.join(a.rollouts, "step_%05d.jsonl" % s)
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        r.pop("calls", None)
        yield r


def turn_facts(r):
    """Per call k of an episode: dict(dead, legal, fatal, dphi, dphi_rel, feas, phi_before, phi_after)."""
    ep = r["episode"]
    turns = ep.get("turns", [])
    phi0 = ep.get("phi_start")
    phi = phi0
    out = []
    first_dead = next((k for k, t in enumerate(turns) if "episode has ended" in str(t.get("fail"))), None)
    for k, t in enumerate(turns):
        fail = t.get("fail")
        dead = first_dead is not None and k >= first_dead
        new = t.get("phi_after")
        if new is None:
            new = phi
        d = (new - phi) if (new is not None and phi is not None) else 0.0
        den = abs(phi0) + 1e-9 if phi0 is not None else 1.0
        out.append({"dead": dead, "legal": 0.0 if fail else 1.0, "fatal": 1.0 if (first_dead is not None and k == first_dead - 1) else 0.0,
                    "dphi": float(np.sign(d)), "dphi_rel": float(np.clip(d / den, -1, 1)),
                    "feas": 1.0 if t.get("feasible_after") else 0.0, "phi_before": phi, "phi_after": new})
        phi = new
    return out


# ---------------------------------------------------------------- the sketches
metas = sorted(glob.glob(os.path.join(a.sk, "s*_k*.meta.jsonl")))
steps = sorted({int(os.path.basename(m)[1:6]) for m in metas})
print("sketch files", len(metas), "steps", steps)
episodes = {}
for s in steps:
    for r in load_step(s):
        episodes[r["key"]] = r
groups = collections.defaultdict(list)
for k, r in episodes.items():
    groups[r["group"]].append(k)
units = sorted(groups)
uidx = {g: i for i, g in enumerate(units)}
fam_of_unit = [episodes[groups[g][0]]["variant"] for g in units]
fams = sorted(set(fam_of_unit))
print("problem groups", len(units), "episodes", len(episodes), "families", fams)

# episode-level labels
ep_ctr, ep_std = {}, {}
for g, keys in groups.items():
    R = np.array([reward(episodes[k]["episode"]) for k in keys])
    m, sd = R.mean(), R.std(ddof=1) if len(R) > 1 else 0.0
    for k, x in zip(keys, R):
        ep_ctr[k] = x - m
        ep_std[k] = (x - m) / (sd + 1e-6) if sd > 1e-6 else 0.0
facts = {k: turn_facts(r) for k, r in episodes.items()}

# ---------------------------------------------------------------- a value estimate for the one-step label
def fit_values(nsteps):
    """V(state) cross-fitted by problem on all logged steps (see critic_fit.py); returns
    {(key, call index): V after the call - V before the call} for the episodes of the probed steps."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    rows, eid_keys, Rs, grp = [], [], [], []
    variants = fams
    for s in range(nsteps):
        try:
            recs = list(load_step(s))
        except FileNotFoundError:
            continue
        for r in recs:
            ep = r["episode"]
            if ep.get("feasible_start") or ep.get("crashed"):
                continue
            R = reward(ep)
            tf = ep.get("first_feasible_turn")
            turns = ep.get("turns", [])
            n = len(turns) if tf is None else min(len(turns), tf)
            phi0 = ep.get("phi_start") or 0.0
            phi, best, fails, dead, nochange = phi0, phi0, 0, 0, 0
            e = len(eid_keys)
            eid_keys.append((r["key"], n))
            Rs.append(R)
            grp.append(zlib.crc32(r["group"].encode("utf-8")))     # not hash(): salted per process
            def row(t, feas):
                den = abs(phi0) + 1e-9
                return [variants.index(r["variant"]), t, H - t, phi0, phi, phi / den, best / den, (phi - phi0) / den,
                        math.copysign(math.log1p(abs(phi)), phi), fails, dead, nochange, feas, e]
            rows.append(row(0, 0))
            for k, t in enumerate(turns[:n]):
                fail = t.get("fail")
                new = t.get("phi_after")
                if fail:
                    fails += 1
                    dead = dead or int("episode has ended" in str(fail))
                if new is None:
                    new = phi
                nochange = nochange + 1 if new == phi else 0
                phi = new
                best = max(best, phi)
                rows.append(row(k + 1, 1 if t.get("feasible_after") else 0))
    X = np.array(rows, dtype=np.float64)
    eid = X[:, -1].astype(int)
    X = X[:, :-1]
    y = np.array(Rs)[eid]
    g = np.array(grp)[eid]
    term = np.array([X[i, 1] == eid_keys[eid[i]][1] for i in range(len(X))])
    V = np.zeros(len(X))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        tr = tr[~term[tr]]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=40,
                                          l2_regularization=1.0, categorical_features=[0], random_state=0)
        m.fit(X[tr], y[tr])
        V[te] = m.predict(X[te])
    V = np.clip(V, 0, 1 + BONUS)
    V[term] = y[term]
    out = {}
    start = {}
    for i in range(len(X)):
        start.setdefault(eid[i], i)
    for e, (key, n) in enumerate(eid_keys):
        if key in episodes:
            i0 = start[e]
            for k in range(n):
                out[(key, k)] = V[i0 + k + 1] - V[i0 + k]
    return out


td = fit_values(a.critic_steps)
print("one-step value labels for", len(td), "turns")

# ---------------------------------------------------------------- weights per labelling
rng = np.random.default_rng(0)
samples = []                                   # (file index, row, key, turn, unit)
for fi, m in enumerate(metas):
    for row, line in enumerate(open(m, encoding="utf-8")):
        d = json.loads(line)
        if d.get("error") or d["key"] not in episodes:
            continue
        samples.append((fi, row, d["key"], d["turn"], uidx[d["group"]], d))
n = len(samples)
print("replies with a sketch", n)
key = [s[2] for s in samples]
turn = [s[3] for s in samples]
unit = np.array([s[4] for s in samples])
F = [facts[k][t] if t < len(facts[k]) else None for k, t in zip(key, turn)]
live = np.array([f is not None and not f["dead"] for f in F])
fam_s = np.array([fams.index(fam_of_unit[u]) for u in unit])


def centred(vals, mask, by):
    """vals minus the mean over the masked replies sharing ``by``; 0 outside the mask."""
    v = np.where(mask, vals, 0.0)
    out = np.zeros(n)
    for b in np.unique(by):
        sel = mask & (by == b)
        if sel.sum() > 1:
            out[sel] = v[sel] - v[sel].mean()
    return out


def col(name):
    return np.array([f[name] if f is not None else 0.0 for f in F])


W = collections.OrderedDict()
W["episode, per-group std (the old run)"] = np.array([ep_std[k] for k in key])
W["episode, centred"] = np.array([ep_ctr[k] for k in key])
W["episode, centred, live turns only"] = np.where(live, W["episode, centred"], 0.0)
for nm, c in (("legal move", "legal"), ("potential went up/down (sign)", "dphi"), ("potential change (relative)", "dphi_rel"),
              ("turn reached feasibility", "feas"), ("move that ended the episode (negated)", "fatal")):
    v = col(c) * (-1.0 if c == "fatal" else 1.0)
    W[nm + " | group baseline"] = centred(v, live, unit)
    W[nm + " | family baseline"] = centred(v, live, fam_s)
W["one-step value change (fitted V)"] = np.array([td.get((k, t), 0.0) for k, t in zip(key, turn)]) * live
meta_of = [s[5] for s in samples]
n_r = np.array([float(d.get("n_r", 0)) for d in meta_of])
lp_r = np.array([float(d.get("lp_r", 0.0)) for d in meta_of])
g2_r = np.array([float(d.get("g2_r", 0.0)) for d in meta_of])
allm = np.ones(n, dtype=bool)
W["episode, centred, each reply / its token count"] = W["episode, centred"] / np.maximum(n_r, 1.0) * n_r.mean()
W["episode, centred, each reply at unit norm"] = W["episode, centred"] / np.sqrt(np.maximum(g2_r, 1e-12)) * math.sqrt(np.median(g2_r))
W["episode success only (positive weights)"] = np.array([1.0 if ep_ctr[k] + 1e-12 > 0 and reward(episodes[k]["episode"]) > 0 else 0.0 for k in key])
W["positive control: rationale length"] = centred(n_r, allm, unit)
W["positive control: reply log-probability"] = centred(lp_r, allm, unit)
W["positive control: log-probability per token"] = centred(lp_r / np.maximum(n_r, 1.0), allm, unit)
W["control: constant 1"] = np.ones(n)
W["control: random sign"] = rng.choice([-1.0, 1.0], size=n)
names = list(W)
Wm = np.stack([W[k] for k in names])                      # labellings x replies
print("live turns %d of %d; legal %.3f of live; fatal moves %d; reached feasibility %d"
      % (live.sum(), n, col("legal")[live].mean(), int(col("fatal").sum()), int(col("feas").sum())))

# ---------------------------------------------------------------- group gradients
U, Lk = len(units), len(names)
dim = np.load(metas[0].replace(".meta.jsonl", ".r.npy"), mmap_mode="r").shape[1]
G = np.zeros((Lk, 2, U, dim), dtype=np.float32)            # labelling, part (r, a), unit
by_file = collections.defaultdict(list)
for j, s in enumerate(samples):
    by_file[s[0]].append(j)
g2 = np.zeros((n, 2))
for fi, m in enumerate(metas):
    js = np.array(by_file.get(fi, []), dtype=int)
    if not len(js):
        continue
    rows = np.array([samples[j][1] for j in js])
    for p, part in enumerate(("r", "a")):
        sk = np.load(m.replace(".meta.jsonl", ".%s.npy" % part))[rows].astype(np.float32)
        g2[js, p] = [samples[j][5].get("g2_" + part, 0.0) for j in js]
        onehot = np.zeros((U, len(js)), dtype=np.float32)
        onehot[unit[js], np.arange(len(js))] = 1.0
        for l in range(Lk):
            G[l, p] += (onehot * Wm[l, js].astype(np.float32)) @ sk
print("per reply: squared gradient norm, rationale part %.3g, action part %.3g (median); action share %.3f"
      % (np.median(g2[:, 0]), np.median(g2[:, 1]), np.median(g2[:, 1] / (g2.sum(1) + 1e-30))))
if a.out:
    np.savez_compressed(a.out, G=G, names=np.array(names), units=np.array(units), fam=np.array(fam_of_unit))


def rho_stats(K, idx=None):
    """(rho, jackknife se, S, T, cos of two random halves) from a Gram matrix of group gradients."""
    if idx is not None:
        K = K[np.ix_(idx, idx)]
    m = K.shape[0]

    def rho(K):
        k = K.shape[0]
        T = np.trace(K) / k
        S = (K.sum() - np.trace(K)) / (k * (k - 1))
        return S / T if T > 0 else 0.0, S, T
    r, S, T = rho(K)
    jk = []
    for i in range(m):
        keep = np.r_[0:i, i + 1:m]
        jk.append(rho(K[np.ix_(keep, keep)])[0])
    jk = np.array(jk)
    se = math.sqrt((m - 1) / m * ((jk - jk.mean()) ** 2).sum())
    return r, se, S, T


def halves_cos(K, reps=400, seed=1):
    r = np.random.default_rng(seed)
    m = K.shape[0]
    cs = []
    for _ in range(reps):
        p = r.permutation(m)
        A, B = p[:m // 2], p[m // 2:]
        ab = K[np.ix_(A, B)].sum()
        aa, bb = K[np.ix_(A, A)].sum(), K[np.ix_(B, B)].sum()
        cs.append(ab / math.sqrt(max(aa, 1e-30) * max(bb, 1e-30)))
    return float(np.mean(cs))


print("\n%-52s %-9s %9s %8s %8s %9s" % ("labelling", "tokens", "rho", "se", "halves", "m_half"))
parts = (("both", lambda g: g[0] + g[1]), ("rationale", lambda g: g[0]), ("action", lambda g: g[1]))
table = {}
for l, nm in enumerate(names):
    for pn, pf in parts:
        Gp = pf(G[l]).astype(np.float64)
        K = Gp @ Gp.T
        r, se, S, T = rho_stats(K)
        hc = halves_cos(K)
        table[(nm, pn)] = (r, se, K)
        mh = (1 - r) / r if r > 0 else float("inf")
        ub = r + 2 * se
        print("%-52s %-9s %9.4f %8.4f %8.3f %9s   upper %.4f -> at least %s groups" % (
            nm[:52], pn, r, se, hc, ("%.0f" % mh) if r > 2 * se and r > 0 else "-", ub, ("%.0f" % ((1 - ub) / ub)) if ub > 0 else "inf"))

print("\nwithin one family (pairs of problems of the same family) and across families, both token parts")
print("%-52s %s" % ("labelling", "  ".join("%9s" % f[:9] for f in fams) + "     across"))
fam_idx = {f: np.array([i for i, x in enumerate(fam_of_unit) if x == f]) for f in fams}
for l, nm in enumerate(names):
    for pn in ("both", "action"):
        K = table[(nm, pn)][2]
        cells = []
        for f in fams:
            r, se, _, _ = rho_stats(K, fam_idx[f])
            cells.append("%5.2f%s" % (r, "*" if r > 2 * se else " "))
        # across families: pairs of groups from different families
        num = den_n = 0.0
        for i in range(U):
            for j in range(U):
                if fam_of_unit[i] != fam_of_unit[j]:
                    num += K[i, j]
                    den_n += 1
        cross = (num / den_n) / (np.trace(K) / U)
        print("%-45s %-6s %s   %7.4f" % (nm[:45], pn, "  ".join("%9s" % c for c in cells), cross))

print("\ncosine between the summed gradients of labellings (both token parts)")
sel = [i for i, nm in enumerate(names) if not nm.startswith("control")]
tot = np.stack([(G[i, 0] + G[i, 1]).sum(0) for i in sel]).astype(np.float64)
nrm = np.linalg.norm(tot, axis=1)
C = (tot @ tot.T) / np.outer(nrm, nrm)
for i, li in enumerate(sel):
    print("%-45s %s" % (names[li][:45], " ".join("%+.2f" % C[i, j] for j in range(len(sel)))))
print("\nrationale part vs action part of the same labelling: cosine of the summed gradients")
for l, nm in enumerate(names):
    r_, a_ = G[l, 0].sum(0).astype(np.float64), G[l, 1].sum(0).astype(np.float64)
    print("%-52s %+.3f   |r| %.3g |a| %.3g" % (nm[:52], r_ @ a_ / (np.linalg.norm(r_) * np.linalg.norm(a_) + 1e-30),
                                           np.linalg.norm(r_), np.linalg.norm(a_)))
