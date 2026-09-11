"""Does topology add anything above the sizing ceiling?

Every arm so far is sizing-only, so "the installed constant is the ceiling" is a statement
about sizing, not about the action space. The closed form provably cannot change topology
(Sved & Ginos; Kirsch), so if topology edits help at all, the ceiling argument is incomplete
and the planner direction reopens -- composing a topology edit with a sizing solve is
something no single tool in the library does.

Arms, all procedural, Phi-selected, 20 steps, matched simulator budget:
  sizing        installed constant only                      [the current ceiling]
  topo          single topology edits only
  both          at each step take the best of sizing or a topology edit
"""
import sys, json, math, copy, statistics, collections, time
from math import comb
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
BUDGET = 200


def joints_of(st):
    return list(getattr(st["t"], "joints", []) or [])


def existing_pairs(st):
    out = set()
    for m in st["t"].members:
        try:
            a, b = m.joints[0], m.joints[1]
            ia = st["t"].joints.index(a); ib = st["t"].joints.index(b)
            out.add((min(ia, ib), max(ia, ib)))
        except Exception:
            continue
    return out


def try_add(dom, st, i, j):
    from validation.truss_executor import execute_grammar_action
    from llm_finetune.envs.truss_env import normalize_action
    nx = dom.clone(st)
    try:
        act = "ADD_MEMBER(%d, %d, 6061_T6_Aluminum, Pipe, 0.03, 0.004)" % (i, j)
        if not execute_grammar_action(nx["t"], normalize_action(nx["t"], act)):
            return None
        nx["cache"] = None
        return nx
    except Exception:
        return None


def try_remove(dom, st, i):
    from validation.truss_executor import execute_grammar_action
    from llm_finetune.envs.truss_env import normalize_action
    nx = dom.clone(st)
    try:
        if not execute_grammar_action(nx["t"], normalize_action(nx["t"], "REMOVE_MEMBER(%d)" % i)):
            return None
        nx["cache"] = None
        return nx
    except Exception:
        return None


def topo_step(dom, st):
    """Best single topology edit by Phi."""
    best, bv = None, T.phi_rho(dom, st)
    J = len(joints_of(st))
    have = existing_pairs(st)
    for i in range(J):
        for j in range(i + 1, J):
            if (i, j) in have:
                continue
            if dom.calls >= BUDGET:
                return best
            nx = try_add(dom, st, i, j)
            if nx is None:
                continue
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
    for i in range(dom.n_elements(st)):
        if dom.calls >= BUDGET:
            return best
        nx = try_remove(dom, st, i)
        if nx is None:
            continue
        v = T.phi_rho(dom, nx)
        if v > bv:
            best, bv = nx, v
    return best


def size_step(dom, st, param):
    op = T.INSTALLED_OP(3.0)
    build = lambda target: T.apply_op(dom, st, op, list(range(dom.n_elements(st))), param, target)
    x, v, _ = T.inner_optimise(dom, st, build, {"target": (1.0, 1.8)}, budget=18)
    return x if v > T.phi_rho(dom, st) else None


def run(t):
    spec, arm = t
    try:
        dom = get_domain("truss")
        st = dom.load(spec)
        param = dom.params[0]
        for _ in range(20):
            if dom.feasible(st):
                return spec["problem_id"], arm, 1, dom.calls
            if dom.calls >= BUDGET:
                break
            cands = []
            if arm in ("sizing", "both"):
                c = size_step(dom, st, param)
                if c is not None:
                    cands.append(c)
            if arm in ("topo", "both"):
                c = topo_step(dom, st)
                if c is not None:
                    cands.append(c)
            if not cands:
                break
            nx = max(cands, key=lambda x: T.phi_rho(dom, x))
            if T.phi_rho(dom, nx) <= T.phi_rho(dom, st):
                break
            st = nx
        return spec["problem_id"], arm, (1 if dom.feasible(st) else 0), dom.calls
    except Exception:
        return spec["problem_id"], arm, None, 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    specs = [json.load(open(f)) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:n]]
    ARMS = ["sizing", "topo", "both"]
    t0 = time.time()
    with Pool(44) as pool:
        res = pool.map(run, [(s, a) for s in specs for a in ARMS], chunksize=1)
    solved = collections.defaultdict(dict); calls = collections.defaultdict(list)
    for pid, arm, s, c in res:
        if s is not None:
            solved[arm][pid] = s; calls[arm].append(c)
    keys = sorted(set.intersection(*[set(solved[a]) for a in ARMS]))
    print("topology test, n=%d, budget %d sims/ep, %.0fs" % (len(keys), BUDGET, time.time() - t0))
    for a in ARMS:
        print("  %-7s solved %.3f   sim/ep %5.0f" % (a, statistics.mean(solved[a][k] for k in keys),
                                                     statistics.mean(calls[a])))

    def mc(x, y):
        hi = sum(1 for k in keys if solved[x][k] and not solved[y][k])
        lo = sum(1 for k in keys if solved[y][k] and not solved[x][k])
        d = hi + lo
        p = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        return hi, lo, 100 * (hi - lo) / len(keys), p
    print()
    for x, y in (("both", "sizing"), ("topo", "sizing"), ("both", "topo")):
        hi, lo, d, p = mc(x, y)
        print("  %-6s vs %-6s disc %3d-%-3d  %+5.1fpp  p=%.2e" % (x, y, hi, lo, d, p))
