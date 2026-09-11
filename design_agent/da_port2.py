"""
Portability test: the same search, both domains, each with its own in-script control.

da_search.py is written against the Domain interface alone -- no members, joints, buckling or
mass anywhere in it. This runs it unchanged on the truss and on the synthetic domain, with each
domain's own fully-stressed analogue as the control in the same run, and reports both.

If the architecture is what it claims to be, the truss result reproduces through the generic
interface and the synthetic domain shows the same shape against its own heuristic. If the truss
number was carried by truss-specific engineering, the generic version will fall short of the
0.9256 measured by the truss-native implementation, and that gap is the honest cost of
portability.
"""
import sys, json, math, random, zlib, argparse, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench",
          "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
from da_synth import SynthDomain
import da_search as S
import da_search2 as S2
from probe26_presentation import sign_test

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
STEPS = 8
NPROP = 16


def seed_of(x):
    return zlib.crc32(str(x).encode()) & 0xffffffff


def _truss(t):
    key, arm = t
    try:
        dom = get_domain("truss")
        st = dom.load(json.load(open(key)))
        rng = random.Random(seed_of(key))
        if arm == "heuristic":
            y = S.heuristic_episode(dom, st, "r", STEPS)
        else:
            y = S2.search_episode(dom, st, "r", STEPS, rng, NPROP)
        return ("truss", arm, str(key), float(y))
    except Exception:
        return None


def _synth(t):
    key, arm = t
    try:
        dom = SynthDomain()
        st = dom.load({"seed": key, "n": 10})
        rng = random.Random(seed_of(key))
        if arm == "heuristic":
            y = S.heuristic_episode(dom, st, "x", STEPS)
        else:
            y = S2.search_episode(dom, st, "x", STEPS, rng, NPROP)
        return ("synth", arm, "s%d" % key, float(y))
    except Exception:
        return None


def report(rows, label, native=None):
    arms = sorted({r[1] for r in rows})
    by = {a: {r[2]: r[3] for r in rows if r[1] == a} for a in arms}
    pids = sorted(set.intersection(*[set(by[a]) for a in arms]))
    if not pids:
        print("  %s: no paired results" % label)
        return
    rate = {a: sum(by[a][p] for p in pids) / len(pids) for a in arms}
    print("\n--- %s (paired on %d instances) ---" % (label, len(pids)))
    for a in sorted(arms, key=lambda x: -rate[x]):
        print("  %-10s %.4f" % (a, rate[a]))
    if "search" in arms and "heuristic" in arms:
        u, d, p = sign_test([by["heuristic"][x] for x in pids], [by["search"][x] for x in pids])
        print("  search vs heuristic: %+.4f   discordant %d-%d   exact sign p = %.3g"
              % (rate["search"] - rate["heuristic"], u, d, p))
    if native is not None:
        print("  truss-native implementation reached %.4f; generic version %.4f"
              % (native, rate.get("search", float("nan"))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", type=int, default=430)
    ap.add_argument("--start", type=int, default=150)
    ap.add_argument("--ns", type=int, default=500)
    ap.add_argument("--procs", type=int, default=44)
    a = ap.parse_args()
    files = [str(f) for f in sorted((DB / "data/problems_hard").glob("*.json"))[a.start:a.start + a.nt]]
    t0 = time.time()
    with Pool(a.procs) as pool:
        tr = [r for r in pool.map(_truss, [(f, arm) for f in files
                                           for arm in ("search", "heuristic")], chunksize=2) if r]
        sy = [r for r in pool.map(_synth, [(i, arm) for i in range(a.ns)
                                           for arm in ("search", "heuristic")], chunksize=8) if r]
    print("%d truss + %d synthetic episodes, %.0fs" % (len(tr), len(sy), time.time() - t0))
    report(tr, "TRUSS, generic search through the Domain interface", native=0.9256)
    report(sy, "SYNTHETIC, same code, one subclass, nothing above it changed")
