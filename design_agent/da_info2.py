"""
Does the information ceiling survive a policy with memory, and does a classifier agree?

Two objections to the per-turn ceiling of 17.5%.

FIRST: it is measured one turn at a time, but a real policy sees the whole episode and can
accumulate evidence. If the per-turn observations are conditionally independent given rho,
information adds up and the per-turn figure understates the achievable. So this recomputes at
EPISODE level, aggregating each observable across the episode (mean, spread, last value) and
asking how much that whole record says about the episode's fidelity bit.

SECOND: mutual information is unfamiliar and positively biased, and a reviewer is entitled to
ask what it means operationally. So the same question is asked a second way, with no entropy
in it: fit a classifier on the observables and score it OUT OF SAMPLE. Held-out accuracy
against the base rate, and AUC against 0.5, say directly how well the bit can be predicted.
Two independent routes to the same quantity; if they disagree, the claim is not safe.
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
from da_info import episode, FEATS, CELLS, THETA, qbin, mi_mm
from da_fit2 import irls

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")


def _truss(t):
    f, mode, lv = t
    try:
        dom = get_domain("truss")
        st = dom.load(json.load(open(f)))
        return ("truss", episode(dom, st, mode, lv, hash(f) & 0xffff))
    except Exception:
        return ("truss", [])


def _synth(t):
    seed, mode, lv = t
    try:
        dom = SynthDomain()
        st = dom.load({"seed": seed, "n": 16})
        return ("synth", episode(dom, st, mode, lv, seed ^ 0x5EED))
    except Exception:
        return ("synth", [])


def aggregate(rows):
    """One record per episode: the whole observable history a memory policy would hold."""
    if not rows:
        return None
    out = {}
    for f in FEATS:
        v = [o[f] for o, _ in rows]
        out[f + "_mean"] = statistics.mean(v)
        out[f + "_sd"] = statistics.pstdev(v) if len(v) > 1 else 0.0
        out[f + "_last"] = v[-1]
    rho = statistics.mean([r for _, r in rows])
    return out, rho


def auc(scores, y):
    pairs = sorted(range(len(scores)), key=lambda i: scores[i])
    r = [0.0] * len(scores)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and scores[pairs[j + 1]] == scores[pairs[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[pairs[k]] = avg
        i = j + 1
    n1 = sum(y)
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    s1 = sum(r[i] for i in range(len(y)) if y[i])
    return (s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    files = [str(f) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:NT]]
    t0 = time.time()
    with Pool(44) as pool:
        rt = pool.map(_truss, [(f, m, l) for f in files for (m, l) in CELLS], chunksize=2)
        rs = pool.map(_synth, [(s, m, l) for s in range(NS) for (m, l) in CELLS], chunksize=2)
    eps = []
    for dm, rows in rt + rs:
        a = aggregate(rows)
        if a:
            eps.append((dm, a[0], a[1]))
    print("%d episodes with a usable record, %.0fs" % (len(eps), time.time() - t0))
    AF = [f + s for f in FEATS for s in ("_mean", "_sd", "_last")]

    for dom_sel in ("truss", "synth", "both"):
        sub = eps if dom_sel == "both" else [e for e in eps if e[0] == dom_sel]
        if len(sub) < 400:
            continue
        y = [1 if e[2] >= THETA else 0 for e in sub]
        base = sum(y) / len(y)
        if not (0.02 < base < 0.98):
            continue
        Hy = -(base * math.log2(base) + (1 - base) * math.log2(1 - base))
        print("\n=== %s: %d episodes, P(rho>=%.1f)=%.3f, H(bit)=%.4f ==="
              % (dom_sel, len(sub), THETA, base, Hy))

        # --- route 1: episode-level mutual information, against a permutation null
        cols = {f: qbin([e[1][f] for e in sub], 4) for f in AF}
        rng = random.Random(0)
        best = sorted(AF, key=lambda f: -mi_mm(cols[f], y))[:6]
        jt = [tuple(cols[f][i] for f in best) for i in range(len(sub))]
        i_j = mi_mm(jt, y)
        nulls = []
        for _ in range(12):
            yp = y[:]; rng.shuffle(yp)
            nulls.append(mi_mm(jt, yp))
        exc = max(0.0, i_j - statistics.mean(nulls))
        print("  episode-level MI, 6 strongest aggregates jointly:")
        print("    I = %.5f, null = %.5f, excess = %.5f  ->  %.1f%% of the bit"
              % (i_j, statistics.mean(nulls), exc, 100 * exc / Hy))

        # --- route 2: out-of-sample classifier, no entropy involved
        idx = list(range(len(sub)))
        random.Random(7).shuffle(idx)
        cut = int(0.7 * len(idx))
        tr, te = idx[:cut], idx[cut:]
        mu = {f: statistics.mean([sub[i][1][f] for i in tr]) for f in AF}
        sd = {f: (statistics.pstdev([sub[i][1][f] for i in tr]) or 1.0) for f in AF}
        X = lambda i: [1.0] + [(sub[i][1][f] - mu[f]) / sd[f] for f in AF]
        b, se, cl, ll = irls([X(i) for i in tr], [float(y[i]) for i in tr])
        sc = []
        for i in te:
            z = sum(b[j] * X(i)[j] for j in range(len(b)))
            sc.append(1.0 / (1.0 + math.exp(-max(-30, min(30, z)))))
        yte = [y[i] for i in te]
        bt = sum(yte) / len(yte)
        maj = max(bt, 1 - bt)
        acc = sum(1 for k in range(len(te)) if (sc[k] >= 0.5) == bool(yte[k])) / len(te)
        print("  held-out classifier on all %d aggregates (n_train=%d, n_test=%d):"
              % (len(AF), len(tr), len(te)))
        print("    accuracy %.4f  vs  always-guess-majority %.4f   (lift %+.4f)"
              % (acc, maj, acc - maj))
        print("    AUC %.4f  vs  chance 0.5" % auc(sc, yte))
