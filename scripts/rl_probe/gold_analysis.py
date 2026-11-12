"""The gradient probe's reading on SFT rows (grad_probe.py output on a gold_rows_to_run.py run
directory): how much of one row's cross-entropy gradient do rows of other problems share?

Every row is one problem instance, so the unit is the row. With g_i the gradient sketch of row i
(rationale + action tokens, or one part),
    q = (mean over pairs i != j of <g_i, g_j>) / (mean over i of |g_i|^2)
over all pairs, pairs of the same family and pairs of different families (delete-one-row
jackknife). A group of m rows of one family then has, against groups of other problems,
    rho_m = m * q_x / (1 + (m - 1) * q_same),   q_x = share * q_same + (1 - share) * q_diff,
with `share` the share of same-family pairs among pairs of groups (--share, 0.106 in the RL probe:
8 families, one problem per family and step). This is the number to set beside the rho of a
labelling of the policy's own replies (gseg_analysis.py), whose groups hold about 64 replies.
Also printed: rho measured directly on groups of 8 rows (random partitions inside a family), and
the rows' log-probability per token.

Scalings: sum (gradient of the summed log-probability, what the probe stores), mean (per token,
what SFT's loss is), clip (sum, every row's gradient clipped to 3x the median norm), unit.

usage: python gold_analysis.py <probe out dir> [<probe out dir> ...] [--share 0.106] [--m 64]
"""
import argparse
import glob
import json
import math
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--share", type=float, default=0.106)
ap.add_argument("--m", type=int, default=64)
a = ap.parse_args()


def q_stat(K, mask):
    n = K.shape[0]
    KM = K * mask
    tot, cnt, tr, d = KM.sum(), mask.sum(), np.trace(K), np.diag(K)
    if cnt == 0:
        return float("nan"), float("nan")
    q = (tot / cnt) / (tr / n)
    rs, rc = KM.sum(1), mask.sum(1)
    qi = ((tot - 2 * rs) / np.maximum(cnt - 2 * rc, 1)) / ((tr - d) / (n - 1))
    return q, math.sqrt((n - 1) / n * ((qi - qi.mean()) ** 2).sum())


def rho_groups(X, fam, m, rng):
    """rho over groups of m rows of one family (one random partition), jackknife over groups."""
    G, gf = [], []
    for f in np.unique(fam):
        idx = rng.permutation(np.where(fam == f)[0])
        for k in range(len(idx) // m):
            G.append(X[idx[k * m:(k + 1) * m]].sum(0))
            gf.append(f)
    G = np.stack(G).astype(np.float64)
    Kg = G @ G.T
    U = len(G)

    def rho(Kq):
        k = Kq.shape[0]
        return ((Kq.sum() - np.trace(Kq)) / (k * (k - 1))) / (np.trace(Kq) / k)
    jk = np.array([rho(Kg[np.ix_(np.r_[0:i, i + 1:U], np.r_[0:i, i + 1:U])]) for i in range(U)])
    return rho(Kg), math.sqrt((U - 1) / U * ((jk - jk.mean()) ** 2).sum()), U


for d in a.dirs:
    metas = sorted(glob.glob(os.path.join(d, "s*_k*.meta.jsonl")))
    rows, R, A = [], [], []
    for m in metas:
        r = np.load(m.replace(".meta.jsonl", ".r.npy"), mmap_mode="r")
        act = np.load(m.replace(".meta.jsonl", ".a.npy"), mmap_mode="r")
        for i, line in enumerate(open(m, encoding="utf-8")):
            x = json.loads(line)
            if x.get("error"):
                continue
            rows.append(x)
            R.append(np.asarray(r[i], dtype=np.float32))
            A.append(np.asarray(act[i], dtype=np.float32))
    R, A = np.stack(R), np.stack(A)
    n = len(rows)
    fam = np.array([x["variant"] for x in rows])
    nr = np.array([x["n_r"] for x in rows], dtype=float)
    na = np.array([x["n_a"] for x in rows], dtype=float)
    lpr = np.array([x["lp_r"] for x in rows])
    lpa = np.array([x["lp_a"] for x in rows])
    off = ~np.eye(n, dtype=bool)
    same = (fam[:, None] == fam[None, :]) & off
    diff = fam[:, None] != fam[None, :]
    print("\n=== %s" % d)
    print("rows %d (%s), adapter step %s, files %d; errors %d" % (
        n, " ".join("%s %d" % (f, (fam == f).sum()) for f in np.unique(fam)), rows[0].get("adapter"), len(metas),
        sum(1 for m in metas for line in open(m, encoding="utf-8") if json.loads(line).get("error"))))
    print("tokens per row: rationale median %d, action median %d; log-prob per token: rationale %.3f, action %.3f, whole %.3f" % (
        np.median(nr), np.median(na), lpr.sum() / nr.sum(), lpa.sum() / na.sum(), (lpr.sum() + lpa.sum()) / (nr.sum() + na.sum())))
    print("log-prob per token by family (whole completion): %s" % " ".join(
        "%s %.3f" % (f[:6], (lpr[fam == f].sum() + lpa[fam == f].sum()) / (nr[fam == f].sum() + na[fam == f].sum())) for f in np.unique(fam)))
    print("squared gradient norm: rationale median %.3g, action median %.3g" % (
        np.median((R.astype(np.float64) ** 2).sum(1)), np.median((A.astype(np.float64) ** 2).sum(1))))
    print("%-34s %20s %20s %20s %10s %18s" % ("part, scaling", "q all pairs", "q same family", "q different family",
                                              "rho_%d" % a.m, "rho, groups of 8"))
    for pname, X0, ntok in (("rationale + action", R + A, nr + na), ("rationale", R, nr), ("action", A, na)):
        norm = np.sqrt((X0.astype(np.float64) ** 2).sum(1))
        for sname, sc in (("sum", np.ones(n)), ("mean per token", 1.0 / np.maximum(ntok, 1)),
                          ("clip 3x median", np.minimum(1.0, 3 * np.median(norm) / np.maximum(norm, 1e-30))),
                          ("unit", 1.0 / np.maximum(norm, 1e-30))):
            X = X0 * sc[:, None].astype(np.float32)
            K = X.astype(np.float64) @ X.astype(np.float64).T
            qa, sa = q_stat(K, off)
            qs, ss = q_stat(K, same)
            qd, sd = q_stat(K, diff)
            qx = a.share * qs + (1 - a.share) * qd
            rho_m = a.m * qx / (1 + (a.m - 1) * qs)
            rng = np.random.default_rng(0)
            rg = [rho_groups(X, fam, 8, rng) for _ in range(10)]
            print("%-34s %8.4f +- %.4f   %8.4f +- %.4f   %8.4f +- %.4f %10.4f   %.4f +- %.4f (%d groups)" % (
                "%s, %s" % (pname, sname), qa, sa, qs, ss, qd, sd, rho_m, np.mean([x[0] for x in rg]), np.mean([x[1] for x in rg]), rg[0][2]))
