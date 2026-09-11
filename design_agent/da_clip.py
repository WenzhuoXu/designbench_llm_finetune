"""
Does the clip saturate the truss, making per-element exponents irrelevant?

Five accounts have failed. The diagnostic that killed the fifth also suggests the sixth. On the
truss the fixed constant 3.0 is already the median measured exponent (3.094), so induction has
no centring to contribute. And the two arms differ by a factor of 1.7 in step scale yet
disagree on eleven problems out of four hundred, which means the outer target sweep absorbs
scale completely. Neither centring nor scale is available as a channel.

That leaves the clip. The applied factor is clipped to [0.7, 2.0]. Truss margins are wildly
unequal -- most members carry little load and have factors of safety far above requirement, so
their computed factor is far below 0.7 and lands on the bound. An element pinned to the bound
takes the same action whatever exponent produced it, so the two operators coincide wherever the
clip binds. In the synthetic domain margins sit near 1 by construction and the clip should
rarely bind.

  (a) measure what fraction of element-updates are pinned to a clip bound, in both domains
  (b) widen the clip on the truss and re-run induction; if saturation is the mechanism, the
      induction gain should appear as the bound is released

If the gain stays at zero with the clip wide open, saturation is not the answer either.
"""
import sys, json, math, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
from da_synth import SynthDomain
from da_adapt import sign_test
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
CAP, STEPS, SHRINK, BUDGET = 50.0, 6, 0.6, 200
CLIPS = ((0.7, 2.0), (0.5, 3.0), (0.25, 6.0), (0.05, 20.0))


def _sat_truss(f):
    try:
        dom = get_domain("truss")
        st = dom.load(json.load(open(f)))
        em = dom.element_margins(st)
        n = dom.n_elements(st)
        tot = pin = 0
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            for i in range(n):
                try:
                    e0 = min(float(em[i]), CAP)
                except Exception:
                    continue
                if not math.isfinite(e0) or e0 <= 0:
                    continue
                f_ = math.exp(math.log(max(tgt, 1e-9) / e0) / 3.0)
                tot += 1
                if f_ <= 0.7 or f_ >= 2.0:
                    pin += 1
        return (pin, tot)
    except Exception:
        return None


def _sat_synth(seed):
    try:
        dom = SynthDomain()
        st = dom.load({"seed": seed, "n": 10})
        em = dom.element_margins(st)
        tot = pin = 0
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            for i in range(dom.n_elements(st)):
                f_ = math.exp(math.log(max(tgt, 1e-9) / max(em[i], 1e-9)) / 3.0)
                tot += 1
                if f_ <= 0.7 or f_ >= 2.0:
                    pin += 1
        return (pin, tot)
    except Exception:
        return None


def truss_ep(f, mode, clip):
    lo, hi = clip
    dom = get_domain("truss")
    st = dom.load(json.load(open(f)))
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1
        if dom.calls >= BUDGET:
            break
        em = dom.element_margins(st)
        if not em:
            break
        n = dom.n_elements(st)
        ex = None
        if mode == "induced":
            before = dom.calls
            ex = []
            for i in range(n):
                b = dom.get(st, i, "r")
                if not b:
                    ex.append(3.0); continue
                nx = dom.clone(st)
                dom.set(nx, i, "r", b * 1.10)
                m = dom.element_margins(nx)
                try:
                    e = math.log(min(float(m[i]), CAP) / min(float(em[i]), CAP)) / math.log(1.10)
                except Exception:
                    e = 3.0
                ex.append(e if (math.isfinite(e) and e > 0.2) else 3.0)
            dom.calls = before
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(n):
                b = dom.get(st, i, "r")
                if not b:
                    continue
                try:
                    e0 = min(float(em[i]), CAP)
                except Exception:
                    continue
                if not math.isfinite(e0) or e0 <= 0:
                    continue
                d = math.log(max(tgt, 1e-9) / e0)
                lf = d / 3.0 if ex is None else d * SHRINK / max(ex[i], 0.2)
                dom.set(nx, i, "r", b * min(hi, max(lo, math.exp(lf))))
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def _run(t):
    f, mode, clip = t
    try:
        return (clip, mode, f, truss_ep(f, mode, clip))
    except Exception:
        return None


if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    files = [str(x) for x in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    t0 = time.time()
    with Pool(44) as pool:
        st_t = [r for r in pool.map(_sat_truss, files[:250], chunksize=2) if r]
        st_s = [r for r in pool.map(_sat_synth, list(range(250)), chunksize=4) if r]
    for name, g in (("TRUSS", st_t), ("SYNTHETIC", st_s)):
        pin = sum(a for a, _ in g); tot = sum(b for _, b in g)
        print("  %-11s element-updates pinned to a clip bound: %.3f  (%d of %d)"
              % (name, pin / max(tot, 1), pin, tot))
    print()
    with Pool(44) as pool:
        got = [r for r in pool.map(_run, [(f, m, c) for f in files
                                          for c in CLIPS for m in ("fixed", "induced")],
                                    chunksize=2) if r]
    print("  induction on the truss as the clip is widened (%d problems, exponents free)\n" % len(files))
    print("  clip            fixed 3.0    induced    gain      sign test")
    for c in CLIPS:
        fa = {x[2]: x[3] for x in got if x[0] == c and x[1] == "fixed"}
        ia = {x[2]: x[3] for x in got if x[0] == c and x[1] == "induced"}
        pids = sorted(set(fa) & set(ia))
        if not pids:
            continue
        a = [fa[p] for p in pids]; b = [ia[p] for p in pids]
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        u, d, p = sign_test(a, b)
        print("  [%.2f, %.2f]     %.4f       %.4f    %+.4f    %d-%d, p=%.3g"
              % (c[0], c[1], ma, mb, mb - ma, u, d, p))
    print("\n  %.0fs" % (time.time() - t0))
