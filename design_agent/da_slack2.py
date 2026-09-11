"""
The slack sweep again, on a scale that can actually be compared across slack levels.

The first pass compared logit coefficients across budget-slack levels whose base feasibility
ran from 0.084 to 0.292. Logit coefficients are not comparable across groups with different
outcome variance -- the same underlying effect shows up as a larger coefficient where the
base rate is nearer 0.5 -- so that comparison could not settle anything, and its formal
interaction came out with the sign the budget account did not predict.

This refits and reports AVERAGE MARGINAL EFFECTS, mean_i b * p_i(1 - p_i), which are on the
probability scale and comparable across base rates. Rows are written out so later questions
do not need another sweep.
"""
import sys, json, math, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_fit2 import irls
from da_slack import _truss, _synth, CELLS, SLACK

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
OUT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune/results/da_slack_rows.jsonl")


def fit_ame(rows, crit="ovq"):
    """within-cell logit; returns (ame_rho, z_rho, ame_crit, z_crit, base)."""
    n = len(rows)
    y = [r["y"] for r in rows]
    ids = [r["id"] for r in rows]
    cells = sorted({(r["mode"], r["level"]) for r in rows})
    cix = {c: i for i, c in enumerate(cells)}
    D = [[1.0 if cix[(r["mode"], r["level"])] == j else 0.0 for j in range(1, len(cells))]
         for r in rows]
    X = [[1.0, rows[i]["rho"], rows[i][crit]] + D[i] for i in range(n)]
    b, se, cl, ll = irls(X, y, ids)
    w = 0.0
    for i in range(n):
        z = max(-30.0, min(30.0, sum(b[j] * X[i][j] for j in range(len(b)))))
        p = 1.0 / (1.0 + math.exp(-z))
        w += p * (1 - p)
    w /= n
    return (b[1] * w, b[1] / max(cl[1], 1e-9),
            b[2] * w, b[2] / max(cl[2], 1e-9), sum(y) / n)


if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    files = [str(f) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    t0 = time.time()
    jt = [(f, m, l, s) for f in files for (m, l) in CELLS for s in SLACK]
    js = [(i, m, l, s) for i in range(NS) for (m, l) in CELLS for s in SLACK]
    with Pool(44) as pool:
        rt = [r for r in pool.map(_truss, jt, chunksize=4) if r]
        rs = [r for r in pool.map(_synth, js, chunksize=4) if r]
    rows = rt + rs
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("%d episodes, %.0fs, rows saved to %s" % (len(rows), time.time() - t0, OUT.name))

    print("\nAverage marginal effects on the probability scale (comparable across slack).")
    print("Budget account predicts the rho AME to FALL and the critical-set AME to RISE")
    print("as the cap is relaxed.\n")
    for dm in ("truss", "synth"):
        sub0 = [r for r in rows if r["domain"] == dm]
        print("--- %s ---" % dm)
        print("  slack    n     base     AME rho   (z)      AME crit-quartile   (z)")
        for sl in SLACK:
            sub = [r for r in sub0 if r["slack"] == sl]
            if len(sub) < 100:
                continue
            try:
                ar, zr, ac, zc, base = fit_ame(sub)
                print("  %5.2f  %4d   %.3f    %+7.4f (%5.2f)      %+7.4f (%5.2f)"
                      % (sl, len(sub), base, ar, zr, ac, zc))
            except Exception as e:
                print("  %5.2f  fit failed: %s" % (sl, e))
        print()

    print("--- does the rho AME trend with slack? Spearman over the six levels ---")
    for dm in ("truss", "synth"):
        sub0 = [r for r in rows if r["domain"] == dm]
        xs, ys, cs = [], [], []
        for sl in SLACK:
            sub = [r for r in sub0 if r["slack"] == sl]
            if len(sub) < 100:
                continue
            ar, zr, ac, zc, base = fit_ame(sub)
            xs.append(sl); ys.append(ar); cs.append(ac)
        def sp(a, b_):
            ra = sorted(range(len(a)), key=lambda i: a[i])
            rb = sorted(range(len(b_)), key=lambda i: b_[i])
            Ra = [0] * len(a); Rb = [0] * len(a)
            for i, j in enumerate(ra): Ra[j] = i
            for i, j in enumerate(rb): Rb[j] = i
            m = (len(a) - 1) / 2.0
            num = sum((Ra[i] - m) * (Rb[i] - m) for i in range(len(a)))
            den = (sum((Ra[i] - m) ** 2 for i in range(len(a))) *
                   sum((Rb[i] - m) ** 2 for i in range(len(a)))) ** 0.5
            return num / den if den else 0.0
        print("  %-6s  rho AME vs slack: %+0.3f     crit AME vs slack: %+0.3f"
              % (dm, sp(xs, ys), sp(xs, cs)))
