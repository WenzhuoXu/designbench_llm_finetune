"""
Firth-penalised logistic regression, for the two model fits that separated.

Sonnet 4 and Haiku returned coefficients of +39.7 and +32.3 with absurd z values: every
success sits above some rho threshold, so the likelihood is monotone in the coefficient and
the MLE diverges. Those are not results, they are the optimiser running away.

Firth's penalty adds (1/2)log|I(beta)| to the log-likelihood, which for logistic regression
modifies the score to

    U_j = sum_i [ y_i - p_i + h_i(1/2 - p_i) ] x_ij

with h_i the hat-matrix diagonal. It gives finite, bias-reduced estimates under separation
and is the standard fix. Profile-penalised-likelihood intervals are used rather than Wald,
since Wald is unreliable exactly where separation bites.
"""
import sys, json, math, statistics, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SRC = Path("/ocean/projects/mch250030p/wxu7/llm_finetune/results/api_guidance/matrix.jsonl")


def _solve(A, b):
    k = len(b)
    M = [A[i][:] + [b[i]] for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-14:
            continue
        for r in range(k):
            if r == c:
                continue
            f = M[r][c] / M[c][c]
            for cc in range(c, k + 1):
                M[r][cc] -= f * M[c][cc]
    return [M[i][k] / M[i][i] if abs(M[i][i]) > 1e-14 else 0.0 for i in range(k)]


def _inv(A):
    k = len(A)
    M = [A[i][:] + [1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        if abs(d) < 1e-14:
            continue
        M[c] = [v / d for v in M[c]]
        for r in range(k):
            if r == c:
                continue
            f = M[r][c]
            M[r] = [M[r][j] - f * M[c][j] for j in range(2 * k)]
    return [[M[i][k + j] for j in range(k)] for i in range(k)]


def firth(X, y, fixed=None, iters=200):
    """Fit by Firth penalty. `fixed` pins coefficient 1 to a value (for profile likelihood)."""
    n, k = len(X), len(X[0])
    b = [0.0] * k
    if fixed is not None:
        b[1] = fixed
    for _ in range(iters):
        W, P = [], []
        for i in range(n):
            z = max(-30.0, min(30.0, sum(b[j] * X[i][j] for j in range(k))))
            p = 1.0 / (1.0 + math.exp(-z))
            P.append(p); W.append(p * (1 - p))
        I = [[sum(W[i] * X[i][a] * X[i][c] for i in range(n)) for c in range(k)] for a in range(k)]
        for a in range(k):
            I[a][a] += 1e-10
        Iinv = _inv(I)
        h = []
        for i in range(n):
            v = sum(X[i][a] * sum(Iinv[a][c] * X[i][c] for c in range(k)) for a in range(k))
            h.append(W[i] * v)
        U = [sum((y[i] - P[i] + h[i] * (0.5 - P[i])) * X[i][a] for i in range(n)) for a in range(k)]
        d = _solve(I, U)
        if fixed is not None:
            d[1] = 0.0
        step = 1.0
        while step > 1e-4 and max(abs(x * step) for x in d) > 5.0:
            step /= 2
        for a in range(k):
            b[a] += step * d[a]
        if max(abs(x * step) for x in d) < 1e-9:
            break
    # penalised log-likelihood
    W = []
    ll = 0.0
    for i in range(n):
        z = max(-30.0, min(30.0, sum(b[j] * X[i][j] for j in range(k))))
        p = min(1 - 1e-12, max(1e-12, 1.0 / (1.0 + math.exp(-z))))
        W.append(p * (1 - p))
        ll += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
    I = [[sum(W[i] * X[i][a] * X[i][c] for i in range(n)) for c in range(k)] for a in range(k)]
    for a in range(k):
        I[a][a] += 1e-10
    # log|I| via LU
    M = [row[:] for row in I]
    logdet = 0.0
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-300:
            continue
        logdet += math.log(abs(M[c][c]))
        for r in range(c + 1, k):
            f = M[r][c] / M[c][c]
            for cc in range(c, k):
                M[r][cc] -= f * M[c][cc]
    return b, ll + 0.5 * logdet


def profile_ci(X, y, b_hat, pll_hat, lo=-5.0, hi=60.0, level=3.841 / 2):
    """Profile penalised-likelihood interval for coefficient 1."""
    def pll(v):
        return firth(X, y, fixed=v)[1]
    out = []
    for direction, start, end in ((-1, b_hat, lo), (+1, b_hat, hi)):
        a, c = start, end
        for _ in range(40):
            m = (a + c) / 2
            if pll_hat - pll(m) < level:
                a = m
            else:
                c = m
        out.append(a)
    return out[0], out[1]


if __name__ == "__main__":
    rows = []
    for line in open(SRC, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rr = [t["rho"] for t in (r.get("turns") or []) if t.get("rho") is not None]
        if rr:
            rows.append({"model": r["model"], "y": float(bool(r["feasible"])),
                         "rho": statistics.mean(rr)})
    print("Firth-penalised refits (the two separated models, plus the others for comparison)\n")
    print("  model                            n   successes   Firth coef   profile 95% CI")
    for m in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == m]
        X = [[1.0, r["rho"]] for r in sub]
        y = [r["y"] for r in sub]
        k = int(sum(y))
        b, pl = firth(X, y)
        try:
            lo_, hi_ = profile_ci(X, y, b[1], pl)
            ci = "[%+.2f, %+.2f]" % (lo_, hi_)
        except Exception:
            ci = "n/a"
        print("  %-30s %4d  %6d     %+8.2f    %s"
              % (m.split("anthropic.")[-1][:28], len(sub), k, b[1], ci))
    allX = [[1.0, r["rho"]] for r in rows]
    ally = [r["y"] for r in rows]
    b, pl = firth(allX, ally)
    lo_, hi_ = profile_ci(allX, ally, b[1], pl)
    print("\n  %-30s %4d  %6d     %+8.2f    [%+.2f, %+.2f]"
          % ("ALL pooled", len(rows), int(sum(ally)), b[1], lo_, hi_))
    print("\n  model-free reference: +7.14")
