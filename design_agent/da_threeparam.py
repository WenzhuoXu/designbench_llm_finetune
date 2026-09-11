"""
Third coordinate: magnitude SPREAD. Completing the repair.

The two-coordinate law (ordering fidelity rho, magnitude shape mu) predicts three of four
corner arms from interior cells to within 0.02 -- and misses the wrong-scale arm by 0.25 on the
truss and 0.42 in the synthetic domain. The failure is in my own statistic, not in the idea:
mu is a correlation of sorted log values, and a correlation is invariant to affine rescaling,
so it registers the SHAPE of the magnitude profile and is blind to its SPREAD. The wrong-scale
arm has the right ordering and the right shape stretched over a far wider range. And the
interior grid held the range fixed, so the fit had nothing to learn scale from.

Adding the coordinate that was missing, measurable like the others:

  rho     spearman(true, shown)                            assignment fidelity
  mu      pearson(log sorted shown, log sorted true)       magnitude shape
  s       sd(log shown) / sd(log true)                     magnitude spread, in log units

and varying all three in the grid: assignment, shape, and the width of the ladder. The corner
arms stay out of the fit, and the wrong-scale corner uses a range wider than anything the grid
contains, so predicting it is genuine extrapolation rather than interpolation.
"""
import sys, json, math, random, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
from da_synth import SynthDomain
from da_fit2 import irls
from da_twoparam import spearman, pearson, clean, make_signal
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
CAP, STEPS, BUDGET = 50.0, 6, 200
PERMS = (0.0, 0.4, 0.8)
ALPHAS = (0.3, 0.6, 1.0)
LADDERS = ((0.7, 1.5), (0.5, 3.0), (0.3, 8.0))
STD = (0.5, 3.0)


def stats_of(shown, v):
    ls, lv = [math.log(x) for x in shown], [math.log(x) for x in v]
    sd_s = statistics.pstdev(ls) if len(ls) > 1 else 0.0
    sd_v = statistics.pstdev(lv) if len(lv) > 1 else 0.0
    return (spearman(v, shown),
            pearson(sorted(ls), sorted(lv)),
            (sd_s / sd_v) if sd_v > 1e-9 else 1.0)


def run(dom, st, param, pf, al, ladder, seed):
    rng = random.Random(seed)
    op = T.INSTALLED_OP(3.0)
    acc = [[], [], []]
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1, acc
        if getattr(dom, "calls", 0) >= BUDGET:
            break
        em = dom.element_margins(st)
        if not em:
            break
        shown, v = make_signal(em, pf, al, rng, ladder)
        r, m, sp = stats_of(shown, v)
        if r is None:
            break
        acc[0].append(r); acc[1].append(m); acc[2].append(sp)
        n = dom.n_elements(st)
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(n):
                b = dom.get(st, i, param)
                if not b:
                    continue
                dom.set(nx, i, param, b * min(2.0, max(0.7, op(shown[i], i, tgt))))
            val = T.phi_rho(dom, nx)
            if val > bv:
                best, bv = nx, val
        if best is None:
            break
        st = best
    return (1 if dom.feasible(st) else 0), acc


def _job(t):
    kind, key, pf, al, ladder, tag = t
    try:
        if kind == "truss":
            dom = get_domain("truss"); st = dom.load(json.load(open(key))); pm = "r"
            sd = hash(key) & 0xffff
        else:
            dom = SynthDomain(); st = dom.load({"seed": key, "n": 10}); pm = "x"
            sd = key ^ 0x5EED
        y, acc = run(dom, st, pm, pf, al, ladder, sd)
        if not acc[0]:
            return None
        return (kind, tag, str(key), float(y),
                statistics.mean(acc[0]), statistics.mean(acc[1]), statistics.mean(acc[2]))
    except Exception:
        return None


CORNERS = [("true", 0.0, 0.0, STD), ("rank_only", 0.0, 1.0, STD),
           ("wrong_scale", 0.0, 1.0, (0.05, 40.0)), ("permuted", 1.0, 0.0, STD)]

if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 350
    files = [str(x) for x in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    jobs = []
    for pf in PERMS:
        for al in ALPHAS:
            for ld in LADDERS:
                jobs += [("truss", f, pf, al, ld, "grid") for f in files]
                jobs += [("synth", s, pf, al, ld, "grid") for s in range(NS)]
    for nm, pf, al, ld in CORNERS:
        jobs += [("truss", f, pf, al, ld, nm) for f in files]
        jobs += [("synth", s, pf, al, ld, nm) for s in range(NS)]
    t0 = time.time()
    with Pool(44) as pool:
        got = [r for r in pool.map(_job, jobs, chunksize=4) if r]
    print("%d episodes, %.0fs\n" % (len(got), time.time() - t0))

    for dm in ("truss", "synth"):
        sub = [x for x in got if x[0] == dm]
        gr = [x for x in sub if x[1] == "grid"]
        if len(gr) < 300:
            continue
        y = [x[3] for x in gr]; ids = [x[2] for x in gr]
        print("--- %s: fitted on %d grid episodes (corners excluded) ---" % (dm, len(gr)))
        print("    grid spread range: %.2f to %.2f" % (min(x[6] for x in gr), max(x[6] for x in gr)))
        models = {
            "rho only":            lambda x: [1.0, x[4]],
            "rho + mu":            lambda x: [1.0, x[4], x[5], x[4] * x[5]],
            "rho + mu + spread":   lambda x: [1.0, x[4], x[5], x[6], x[4] * x[5]],
        }
        fits = {}
        for nm, fx in models.items():
            b, se, cl, ll = irls([fx(x) for x in gr], y, ids)
            fits[nm] = (b, fx, ll)
        for nm in models:
            print("    %-20s logL %.1f" % (nm, fits[nm][2]))
        print("\n  extrapolation to corners, by model (error = predicted minus observed):")
        print("    arm            observed    rho only   rho+mu    rho+mu+spread")
        for cn, _, _, _ in CORNERS:
            c = [x for x in sub if x[1] == cn]
            if not c:
                continue
            obs = sum(x[3] for x in c) / len(c)
            mean_x = (None, None, None, None,
                      statistics.mean([x[4] for x in c]),
                      statistics.mean([x[5] for x in c]),
                      statistics.mean([x[6] for x in c]))
            row = []
            for nm in ("rho only", "rho + mu", "rho + mu + spread"):
                b, fx, _ = fits[nm]
                xv = fx(mean_x)
                z = sum(b[j] * xv[j] for j in range(len(b)))
                row.append(1.0 / (1.0 + math.exp(-max(-30, min(30, z)))) - obs)
            print("    %-12s  %.3f      %+.3f     %+.3f     %+.3f"
                  % (cn, obs, row[0], row[1], row[2]))
        print()
