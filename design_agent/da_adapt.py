"""
From law to policy: should the agent's action aggression depend on its signal's fidelity?

Established: outcome is governed by ordering fidelity, and the channel by which a bad ordering
does damage is SHRINKING -- the action rescales every element by a factor read off its shown
margin, and a safe element shown as too healthy gets shrunk into violation. Forbidding shrink
removes that channel.

But shrinking is not simply bad. On a clean signal at the real budget it is worth +22.0pp in
the truss (0.507 against 0.287): freeing material from over-strong elements is how the budget
gets met. The two facts together say aggression should be CONDITIONAL on signal fidelity --
shrink when the signal can be trusted, grow only when it cannot.

Before building an estimator of fidelity from observables, check the oracle version can win at
all. This arm is told the true rho and switches on it. If the oracle policy does not beat the
better of the two fixed policies, no estimator built from noisier information will either, and
the idea is dead cheaply.

Matched evaluation budget throughout: every arm gets the same BUDGET, as in the planner tests.
"""
import sys, json, math, random, statistics, collections, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
from da_synth import SynthDomain
from da_matched_info import corrupt, spearman
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
BUDGET = 200
# a deployment mixture: the agent meets signals of varying quality and does not choose which
MIX = [("perm", 0.0), ("perm", 0.15), ("perm", 0.3), ("perm", 0.5), ("perm", 0.7), ("perm", 1.0),
       ("noise", 0.15), ("noise", 0.3), ("noise", 0.5), ("noise", 0.8), ("noise", 1.3), ("noise", 2.5)]
THETAS = (0.5, 0.65, 0.8, 0.9)


def episode(dom, st, mode, level, seed, policy, theta=0.8, steps=20):
    """policy: 'shrink' | 'grow' | 'oracle'. Oracle sees the turn's true rho."""
    rng = random.Random(seed)
    op = T.INSTALLED_OP(3.0)
    n = dom.n_elements(st)
    for _ in range(steps):
        if dom.feasible(st):
            return 1
        if dom.calls >= BUDGET:
            break
        em = dom.element_margins(st)
        if not em:
            break
        shown = corrupt(em, mode, level, rng)
        if policy == "shrink":
            clip_lo = 0.7
        elif policy == "grow":
            clip_lo = 1.0
        else:
            clip_lo = 0.7 if spearman(em, shown) >= theta else 1.0
        best, bv = None, T.phi_rho(dom, st)
        param = dom.params[0]
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(n):
                b = dom.get(st, i, param)
                if not b:
                    continue
                f = min(2.0, max(clip_lo, op(shown[i], i, tgt)))
                dom.set(nx, i, param, b * f)
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def _truss(t):
    f, mode, lv = t
    try:
        out = {}
        for pol, th in [("shrink", 0), ("grow", 0)] + [("oracle", x) for x in THETAS]:
            dom = get_domain("truss")
            st = dom.load(json.load(open(f)))
            out["%s%s" % (pol, th or "")] = episode(dom, st, mode, lv, hash(f) & 0xffff, pol, th)
        out["id"] = f; out["domain"] = "truss"; out["cell"] = "%s%s" % (mode, lv)
        return out
    except Exception:
        return None


def _synth(t):
    seed, mode, lv = t
    try:
        out = {}
        for pol, th in [("shrink", 0), ("grow", 0)] + [("oracle", x) for x in THETAS]:
            dom = SynthDomain()
            st = dom.load({"seed": seed, "n": 16})
            out["%s%s" % (pol, th or "")] = episode(dom, st, mode, lv, seed ^ 0x5EED, pol, th)
        out["id"] = "s%d" % seed; out["domain"] = "synth"; out["cell"] = "%s%s" % (mode, lv)
        return out
    except Exception:
        return None


def sign_test(a, b):
    """exact two-sided sign test on paired binary outcomes"""
    up = sum(1 for i in range(len(a)) if b[i] > a[i])
    dn = sum(1 for i in range(len(a)) if b[i] < a[i])
    m = up + dn
    if m == 0:
        return up, dn, 1.0
    from math import comb
    k = min(up, dn)
    p = sum(comb(m, i) for i in range(k + 1)) / (2.0 ** m) * 2
    return up, dn, min(1.0, p)


if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    files = [str(f) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    t0 = time.time()
    with Pool(44) as pool:
        rt = [r for r in pool.map(_truss, [(f, m, l) for f in files for (m, l) in MIX], chunksize=2) if r]
        rs = [r for r in pool.map(_synth, [(s, m, l) for s in range(NS) for (m, l) in MIX], chunksize=2) if r]
    rows = rt + rs
    with open("/ocean/projects/mch250030p/wxu7/llm_finetune/results/da_adapt_rows.jsonl",
              "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("%d episodes per arm (%d truss, %d synth), %.0fs, matched budget %d"
          % (len(rows), len(rt), len(rs), time.time() - t0, BUDGET))
    arms = ["shrink", "grow"] + ["oracle%s" % x for x in THETAS]
    for dm in ("truss", "synth", "both"):
        sub = rows if dm == "both" else [r for r in rows if r["domain"] == dm]
        if not sub:
            continue
        print("\n--- %s (n=%d, deployment mixture of signal qualities) ---" % (dm, len(sub)))
        rates = {a: sum(r[a] for r in sub) / len(sub) for a in arms}
        best_fixed = max(("shrink", "grow"), key=lambda a: rates[a])
        for a in arms:
            tag = ""
            if a.startswith("oracle"):
                u, d, p = sign_test([r[best_fixed] for r in sub], [r[a] for r in sub])
                tag = "   vs best fixed (%s): %+.3f, disc %d-%d, p=%.2g" % (
                    best_fixed, rates[a] - rates[best_fixed], u, d, p)
            elif a == best_fixed:
                tag = "   <- best fixed policy"
            print("  %-10s %.3f%s" % (a, rates[a], tag))
