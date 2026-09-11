"""
Per-observation refit with honest standard errors.

The cell-level GLM put ordering fidelity at logit +6.22 with residual terms of +0.24
(corruption type) and +0.39 (domain). The whole claim is that those residuals are small
relative to the main effect, and 24 aggregated points cannot establish that -- aggregation
discards within-cell variation in rho and understates the standard errors.

This refits on all individual episodes, using each episode's OWN mean rho rather than its
cell's, and reports Wald standard errors from the inverse Hessian plus a cluster-robust
version grouped by instance, since the same problem appears in every cell.
"""
import sys, json, math, time, collections, statistics
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_collapse import _truss, _synth, CELLS

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")


def irls(X, y, groups=None, iters=80):
    n, k = len(X), len(X[0])
    b = [0.0] * k
    for _ in range(iters):
        g = [0.0] * k
        H = [[0.0] * k for _ in range(k)]
        for i in range(n):
            z = max(-30.0, min(30.0, sum(b[j] * X[i][j] for j in range(k))))
            p = 1.0 / (1.0 + math.exp(-z))
            w = p * (1 - p)
            r = y[i] - p
            for a in range(k):
                g[a] += r * X[i][a]
                for c in range(k):
                    H[a][c] += w * X[i][a] * X[i][c]
        for a in range(k):
            H[a][a] += 1e-8
        M = [row[:] + [g[a]] for a, row in enumerate(H)]
        for c in range(k):
            piv = max(range(c, k), key=lambda r_: abs(M[r_][c]))
            M[c], M[piv] = M[piv], M[c]
            for r_ in range(k):
                if r_ == c or abs(M[c][c]) < 1e-14:
                    continue
                f = M[r_][c] / M[c][c]
                for cc in range(c, k + 1):
                    M[r_][cc] -= f * M[c][cc]
        d = [M[a][k] / M[a][a] if abs(M[a][a]) > 1e-14 else 0.0 for a in range(k)]
        for a in range(k):
            b[a] += d[a]
        if max(abs(x) for x in d) < 1e-10:
            break
    # inverse Hessian for Wald SEs
    H = [[0.0] * k for _ in range(k)]
    scores = []
    for i in range(n):
        z = max(-30.0, min(30.0, sum(b[j] * X[i][j] for j in range(k))))
        p = 1.0 / (1.0 + math.exp(-z))
        w = p * (1 - p)
        r = y[i] - p
        scores.append([r * X[i][a] for a in range(k)])
        for a in range(k):
            for c in range(k):
                H[a][c] += w * X[i][a] * X[i][c]
    A = [row[:] + [1.0 if i == j else 0.0 for j in range(k)] for i, row in enumerate(H)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r_: abs(A[r_][c]))
        A[c], A[piv] = A[piv], A[c]
        d0 = A[c][c]
        if abs(d0) < 1e-14:
            continue
        A[c] = [v / d0 for v in A[c]]
        for r_ in range(k):
            if r_ == c:
                continue
            f = A[r_][c]
            A[r_] = [A[r_][j] - f * A[c][j] for j in range(2 * k)]
    Hinv = [[A[i][k + j] for j in range(k)] for i in range(k)]
    se = [math.sqrt(max(Hinv[a][a], 0.0)) for a in range(k)]
    se_cl = se
    if groups is not None:
        agg = collections.defaultdict(lambda: [0.0] * k)
        for i, gid in enumerate(groups):
            for a in range(k):
                agg[gid][a] += scores[i][a]
        meat = [[0.0] * k for _ in range(k)]
        for v in agg.values():
            for a in range(k):
                for c in range(k):
                    meat[a][c] += v[a] * v[c]
        V = [[sum(Hinv[a][x] * meat[x][yy] for x in range(k)) for yy in range(k)] for a in range(k)]
        V = [[sum(V[a][x] * Hinv[x][c] for x in range(k)) for c in range(k)] for a in range(k)]
        se_cl = [math.sqrt(max(V[a][a], 0.0)) for a in range(k)]
    ll = 0.0
    for i in range(n):
        z = max(-30.0, min(30.0, sum(b[j] * X[i][j] for j in range(k))))
        p = min(1 - 1e-12, max(1e-12, 1.0 / (1.0 + math.exp(-z))))
        ll += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
    return b, se, se_cl, ll


if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    files = [str(f) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    t0 = time.time()
    with Pool(44) as pool:
        rt = [r for r in pool.map(_truss, [(f, m, l) for f in files for (m, l) in CELLS], chunksize=1) if r]
        rs = [r for r in pool.map(_synth, [(s, m, l) for s in range(NS) for (m, l) in CELLS], chunksize=1) if r]
    rows = rt + rs
    ids = ([f for f in files for _ in CELLS][:len(rt)] +
           ["s%d" % s for s in range(NS) for _ in CELLS][:len(rs)])
    print("per-observation refit: %d episodes (%d truss, %d synthetic), %.0fs"
          % (len(rows), len(rt), len(rs), time.time() - t0))
    rho = [r[4] for r in rows]
    y = [float(r[3]) for r in rows]
    perm = [1.0 if r[1] == "perm" else 0.0 for r in rows]
    trs = [1.0 if r[0] == "truss" else 0.0 for r in rows]

    X0 = [[1.0, rho[i]] for i in range(len(rows))]
    X1 = [[1.0, rho[i], perm[i], trs[i]] for i in range(len(rows))]
    b0, s0, c0, ll0 = irls(X0, y, ids)
    b1, s1, c1, ll1 = irls(X1, y, ids)
    print("\n  term                coef     naive SE   cluster SE   z(cluster)")
    print("  rho (fidelity)     %+7.3f   %7.3f    %7.3f     %6.2f" % (b1[1], s1[1], c1[1], b1[1] / max(c1[1], 1e-9)))
    print("  corruption=perm    %+7.3f   %7.3f    %7.3f     %6.2f" % (b1[2], s1[2], c1[2], b1[2] / max(c1[2], 1e-9)))
    print("  domain=truss       %+7.3f   %7.3f    %7.3f     %6.2f" % (b1[3], s1[3], c1[3], b1[3] / max(c1[3], 1e-9)))
    print("\n  LR chi2 for adding both terms: %.2f (2 df)" % (2 * (ll1 - ll0)))
    span = max(rho) - min(rho)
    print("\n  fidelity effect across the observed rho span (%.2f): %+.2f logits"
          % (span, b1[1] * span))
    print("  residuals as a share of that: corruption %.1f%%, domain %.1f%%"
          % (100 * abs(b1[2]) / abs(b1[1] * span), 100 * abs(b1[3]) / abs(b1[1] * span)))
