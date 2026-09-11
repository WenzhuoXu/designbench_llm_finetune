"""Proper binomial GLM on the collapse cells.

The previous fit used fixed-step gradient ascent and reported negative likelihood-ratio
statistics, which is impossible and means it had not converged. This refits by IRLS
(Newton-Raphson with the exact Hessian), which converges in a handful of steps, and reports
the nested-model comparison correctly.

Question: is outcome a function of ordering fidelity alone, or does corruption type / domain
add explanatory power? Cells are (domain, mode, level, mean rho, successes, n).
"""
import math

# domain, mode, level, rho, rate, n   -- from da_collapse.py
CELLS = [
    ("synth", "noise", 0.15, +0.875, 0.287, 300),
    ("synth", "noise", 0.30, +0.780, 0.177, 300),
    ("synth", "noise", 0.50, +0.667, 0.043, 300),
    ("synth", "noise", 0.80, +0.545, 0.007, 300),
    ("synth", "noise", 1.30, +0.362, 0.003, 300),
    ("synth", "noise", 2.50, +0.170, 0.000, 300),
    ("synth", "perm", 0.00, +1.000, 0.377, 300),
    ("synth", "perm", 0.15, +0.928, 0.343, 300),
    ("synth", "perm", 0.30, +0.715, 0.190, 300),
    ("synth", "perm", 0.50, +0.512, 0.087, 300),
    ("synth", "perm", 0.70, +0.303, 0.000, 300),
    ("synth", "perm", 1.00, -0.002, 0.000, 300),
    ("truss", "noise", 0.15, +0.906, 0.427, 150),
    ("truss", "noise", 0.30, +0.853, 0.353, 150),
    ("truss", "noise", 0.50, +0.796, 0.213, 150),
    ("truss", "noise", 0.80, +0.728, 0.093, 150),
    ("truss", "noise", 1.30, +0.637, 0.053, 150),
    ("truss", "noise", 2.50, +0.464, 0.007, 150),
    ("truss", "perm", 0.00, +1.000, 0.507, 150),
    ("truss", "perm", 0.15, +0.893, 0.400, 150),
    ("truss", "perm", 0.30, +0.743, 0.253, 150),
    ("truss", "perm", 0.50, +0.463, 0.093, 150),
    ("truss", "perm", 0.70, +0.317, 0.020, 150),
    ("truss", "perm", 1.00, -0.022, 0.007, 150),
]


def irls(X, y, n, iters=60):
    """Binomial GLM by Newton-Raphson. X rows are feature vectors, y successes, n trials."""
    k = len(X[0])
    b = [0.0] * k
    for _ in range(iters):
        g = [0.0] * k
        H = [[0.0] * k for _ in range(k)]
        for i in range(len(X)):
            z = sum(b[j] * X[i][j] for j in range(k))
            z = max(-30.0, min(30.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
            w = n[i] * p * (1 - p)
            r = y[i] - n[i] * p
            for a in range(k):
                g[a] += r * X[i][a]
                for c in range(k):
                    H[a][c] += w * X[i][a] * X[i][c]
        for a in range(k):
            H[a][a] += 1e-9
        # solve H d = g
        M = [row[:] + [g[a]] for a, row in enumerate(H)]
        for c in range(k):
            piv = max(range(c, k), key=lambda r_: abs(M[r_][c]))
            M[c], M[piv] = M[piv], M[c]
            if abs(M[c][c]) < 1e-14:
                continue
            for r_ in range(k):
                if r_ == c:
                    continue
                f = M[r_][c] / M[c][c]
                for cc in range(c, k + 1):
                    M[r_][cc] -= f * M[c][cc]
        d = [M[a][k] / M[a][a] if abs(M[a][a]) > 1e-14 else 0.0 for a in range(k)]
        step = 1.0
        for a in range(k):
            b[a] += step * d[a]
        if max(abs(x) for x in d) < 1e-10:
            break
    ll = 0.0
    for i in range(len(X)):
        z = max(-30.0, min(30.0, sum(b[j] * X[i][j] for j in range(k))))
        p = min(1 - 1e-12, max(1e-12, 1.0 / (1.0 + math.exp(-z))))
        ll += y[i] * math.log(p) + (n[i] - y[i]) * math.log(1 - p)
    return b, ll


def chi2_p(x, df=1):
    if x <= 0:
        return 1.0
    if df == 1:
        z = math.sqrt(x)
        return math.erfc(z / math.sqrt(2))
    return math.exp(-x / 2)


if __name__ == "__main__":
    rho = [c[3] for c in CELLS]
    n = [c[5] for c in CELLS]
    y = [round(c[4] * c[5]) for c in CELLS]
    isperm = [1.0 if c[1] == "perm" else 0.0 for c in CELLS]
    istruss = [1.0 if c[0] == "truss" else 0.0 for c in CELLS]

    m0 = irls([[1.0, rho[i]] for i in range(len(CELLS))], y, n)
    m1 = irls([[1.0, rho[i], isperm[i]] for i in range(len(CELLS))], y, n)
    m2 = irls([[1.0, rho[i], istruss[i]] for i in range(len(CELLS))], y, n)
    m3 = irls([[1.0, rho[i], isperm[i], istruss[i]] for i in range(len(CELLS))], y, n)

    print("binomial GLM, %d cells, %d trials total\n" % (len(CELLS), sum(n)))
    print("  rho only                      b_rho=%+.3f            logL=%9.2f" % (m0[0][1], m0[1]))
    for name, m, base in (("+ corruption type (perm=1)", m1, m0),
                          ("+ domain (truss=1)", m2, m0)):
        lr = 2 * (m[1] - base[1])
        print("  %-29s coef=%+.3f  logL=%9.2f  LR chi2=%7.2f  p=%.2e"
              % (name, m[0][2], m[1], lr, chi2_p(lr)))
    lr3 = 2 * (m3[1] - m0[1])
    print("  + both                        perm=%+.3f truss=%+.3f  logL=%9.2f  LR chi2=%7.2f (2 df)"
          % (m3[0][2], m3[0][3], m3[1], lr3))
    print("\n  chi2 > 3.84 (1 df) => the term carries real information beyond ordering fidelity.")
    print("  A positive perm coefficient means permutation is LESS harmful than value noise")
    print("  at matched ordering fidelity.")
