"""The gradient probe's reading on screened first replies (first_move_run.py + grad_probe.py):
does the cross-entropy on a simulator-screened own reply have a direction that problems share,
beyond what any own reply has?

Per problem p: s_p = gradient sketch of the screened reply (the applied reply with the highest
potential after it), r_p = sketch of another, unscreened reply at the same state. With <.,.> over
pairs of different problems and |.|^2 over problems:
    q_sel   = <s_p, s_p'> / |s|^2        what a supervised step on screened replies shares
    q_rnd   = <r_p, r_p'> / |r|^2        what a step on unscreened own replies shares (replies drawn
                                         at T < 1 and scored at T = 1 share a sharpening direction)
    q_cross = <s_p, r_p'> / (|s| |r|)    the part of q_sel that an unscreened reply has too
    rho_d   = <d_p, d_p'> / |d|^2, d = s - r   the part only the screening has
Delete-one-problem jackknife. m rows per optimizer step share m q / (1 + (m - 1) q) of the step.
Reference (gold_analysis.py, the rows that trained dx38, at the dx38 adapter): q = 0.0008 +- 0.0001.

usage: python first_move_analysis.py <probe out dir> <labels.jsonl> [--m 64]
"""
import argparse
import glob
import json
import math
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("probe")
ap.add_argument("labels")
ap.add_argument("--m", type=int, default=64)
a = ap.parse_args()


def off_mean(M, mask):
    """(mean of M over mask, delete-one values): mask is symmetric with a False diagonal."""
    KM = M * mask
    tot, cnt = KM.sum(), mask.sum()
    rs, cs, rc = KM.sum(1), KM.sum(0), mask.sum(1)
    return tot / max(cnt, 1), (tot - rs - cs) / np.maximum(cnt - 2 * rc, 1)


def loo_mean(d):
    n = len(d)
    return d.mean(), (d.sum() - d) / (n - 1)


def jk(full, loo):
    n = len(loo)
    return full, math.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum())


def stats(S, R, mask):
    """q_sel, q_rnd, q_cross, rho_d, q_sel - q_cross: each (value, jackknife se)."""
    S, R = S.astype(np.float64), R.astype(np.float64)
    D = S - R
    out = {}
    ds, dsl = loo_mean((S ** 2).sum(1))
    dr, drl = loo_mean((R ** 2).sum(1))
    dd, ddl = loo_mean((D ** 2).sum(1))
    ss, ssl = off_mean(S @ S.T, mask)
    rr, rrl = off_mean(R @ R.T, mask)
    sr = S @ R.T
    sx, sxl = off_mean(0.5 * (sr + sr.T), mask)
    dm, dml = off_mean(D @ D.T, mask)
    out["q_sel"] = jk(ss / ds, ssl / dsl)
    out["q_rnd"] = jk(rr / dr, rrl / drl)
    out["q_cross"] = jk(sx / math.sqrt(ds * dr), sxl / np.sqrt(dsl * drl))
    out["rho_d"] = jk(dm / dd, dml / ddl)
    out["q_sel - q_cross"] = jk(ss / ds - sx / math.sqrt(ds * dr), ssl / dsl - sxl / np.sqrt(dsl * drl))
    return out


lab = {}
for line in open(a.labels, encoding="utf-8"):
    r = json.loads(line)
    lab[r["key"]] = r
rows, R, A = {}, {}, {}
n_err = 0
for m in sorted(glob.glob(os.path.join(a.probe, "s*_k*.meta.jsonl"))):
    r = np.load(m.replace(".meta.jsonl", ".r.npy"), mmap_mode="r")
    act = np.load(m.replace(".meta.jsonl", ".a.npy"), mmap_mode="r")
    for i, line in enumerate(open(m, encoding="utf-8")):
        x = json.loads(line)
        if x.get("error"):
            n_err += 1
            continue
        rows[x["key"]] = x
        R[x["key"]] = np.asarray(r[i], dtype=np.float32)
        A[x["key"]] = np.asarray(act[i], dtype=np.float32)
groups = sorted({lab[k]["group"] for k in rows if k in lab})
groups = [g for g in groups if g + "/sel" in rows and g + "/rnd" in rows]
fam = np.array([lab[g + "/sel"]["variant"] for g in groups])
n = len(groups)
print("problems with both replies sketched: %d (%s); replies with an error: %d; adapter step %s" % (
    n, " ".join("%s %d" % (f, (fam == f).sum()) for f in np.unique(fam)), n_err, rows[groups[0] + "/sel"].get("adapter")))
for role in ("sel", "rnd"):
    ks = [g + "/" + role for g in groups]
    nt = np.array([rows[k]["n_r"] + rows[k]["n_a"] for k in ks], dtype=float)
    lp = np.array([rows[k]["lp_r"] + rows[k]["lp_a"] for k in ks])
    print("%s: tokens per reply median %d, log-prob per token %.3f, applied %.3f, raised the potential %.3f, episode feasible %.3f" % (
        role, np.median(nt), lp.sum() / nt.sum(), np.mean([lab[k]["legal"] for k in ks]), np.mean([lab[k]["gain"] for k in ks]),
        np.mean([lab[k]["episode_ok"] for k in ks])))
off = ~np.eye(n, dtype=bool)
same = (fam[:, None] == fam[None, :]) & off
diff = fam[:, None] != fam[None, :]
names = ("q_sel", "q_rnd", "q_cross", "rho_d", "q_sel - q_cross")
for pname, get in (("rationale + action", lambda k: R[k] + A[k]), ("rationale", lambda k: R[k]), ("action", lambda k: A[k])):
    S0 = np.stack([get(g + "/sel") for g in groups])
    R0 = np.stack([get(g + "/rnd") for g in groups])
    ns = np.sqrt((S0.astype(np.float64) ** 2).sum(1))
    nr = np.sqrt((R0.astype(np.float64) ** 2).sum(1))
    med = np.median(np.concatenate([ns, nr]))
    ts = np.array([rows[g + "/sel"]["n_r"] + rows[g + "/sel"]["n_a"] for g in groups], dtype=float)
    tr = np.array([rows[g + "/rnd"]["n_r"] + rows[g + "/rnd"]["n_a"] for g in groups], dtype=float)
    print("\n== %s: squared norm median sel %.3g, rnd %.3g" % (pname, np.median(ns ** 2), np.median(nr ** 2)))
    print("%-16s %-18s %s" % ("scaling", "pairs", "  ".join("%-20s" % x for x in names)))
    for sname, cs, cr in (("clip 3x median", np.minimum(1.0, 3 * med / np.maximum(ns, 1e-30)), np.minimum(1.0, 3 * med / np.maximum(nr, 1e-30))),
                          ("unit", 1.0 / np.maximum(ns, 1e-30), 1.0 / np.maximum(nr, 1e-30)),
                          ("mean per token", 1.0 / np.maximum(ts, 1), 1.0 / np.maximum(tr, 1)),
                          ("sum", np.ones(n), np.ones(n))):
        S = S0 * cs[:, None].astype(np.float32)
        Rn = R0 * cr[:, None].astype(np.float32)
        for mname, mask in (("all", off), ("same family", same), ("different family", diff)):
            st = stats(S, Rn, mask)
            print("%-16s %-18s %s" % (sname, mname, "  ".join("%8.5f +- %.5f " % st[x] for x in names)))
        q = stats(S, Rn, off)["q_sel"][0]
        d = stats(S, Rn, off)["rho_d"][0]
        print("%-16s share of a %d-row step: screened replies %.4f, the screening-only part %.4f" % (
            sname, a.m, a.m * q / (1 + (a.m - 1) * q), a.m * d / (1 + (a.m - 1) * max(d, 0))))
