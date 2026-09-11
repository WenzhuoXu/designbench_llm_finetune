"""
What the margin costs: feasibility against evaluation budget, not against turns.

The best configuration reaches 0.9721 against native fully-stressed design's 0.4930, discordant
206-0. It gets there with 32 candidates per turn, each rolled forward to a horizon of 12, so
every turn spends hundreds of analyses where the heuristic spends one. Reported per turn the
comparison flatters the search enormously; the honest question is where it stands per unit of
compute, and whether a practitioner with a fixed simulation budget should run it at all.

Both sides are instrumented. The Domain already counts its analyses, so the search arms are
measured directly; the native heuristic is counted by wrapping the analysis function it calls.
The search is then given a hard evaluation budget and stopped when it runs out, at budgets
spanning three orders of magnitude:

    100, 250, 500, 1000, 2500, 5000, 10000, 25000 analyses

Each budget is its own arm, on the same problems as the controls, in the same run. The output
is a curve: feasibility versus analyses spent, with the heuristic's own cost and feasibility
marked on it, so the crossing point is visible rather than asserted.
"""
import os, sys, json, math, random, zlib, argparse, time
from pathlib import Path
from multiprocessing import Pool
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts"),
          str(PROJECT / "design_agent")):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
import da_domtools
from da_search import size_pass, potential
from da_search2 import sample_generic, apply_generic
from probe26_presentation import sign_test
import llm_finetune.envs.truss_env as TE
from planner01 import fsd_pass

BUDGETS = (100, 250, 500, 1000, 2500, 5000, 10000, 25000)
POOL, HORIZON = 32, 12
ALLOWED = ("SIZE_PASS",)          # the configuration that won; SCALE and TRIM diluted the pool


def seed_of(x):
    return zlib.crc32(str(x).encode()) & 0xffffffff


def budgeted_rollout(dom, st, param, steps, budget):
    cur = st
    for _ in range(max(0, steps)):
        if dom.calls >= budget:
            return dom.feasible(cur), cur
        if dom.feasible(cur):
            return True, cur
        nxt = size_pass(dom, cur, param)
        if nxt is cur:
            break
        cur = nxt
    return dom.feasible(cur), cur


def propose(dom, st, rng, n):
    try:
        extra = dom.tools() or {}
    except Exception:
        extra = {}
    names = list(ALLOWED) + list(extra)
    out = []
    for _ in range(n):
        kind = rng.choice(names)
        if kind in extra:
            try:
                args = extra[kind][0](dom, st, rng)
            except Exception:
                args = None
            if args is None:
                continue
            out.append((kind, args))
        else:
            out.append((kind, sample_generic(dom, st, rng, kind)))
    return out


def apply_any(dom, st, param, kind, args):
    try:
        extra = dom.tools() or {}
    except Exception:
        extra = {}
    if kind in extra:
        try:
            return extra[kind][1](dom, st, args)
        except Exception:
            return None
    return apply_generic(dom, st, param, kind, args)


def search_budgeted(dom, st, param, rng, budget):
    """Identical to the winning search, stopped hard when the analysis budget runs out."""
    for turn in range(HORIZON):
        if dom.calls >= budget or dom.feasible(st):
            break
        nxt = size_pass(dom, st, param)
        if nxt is not st:
            st = nxt
        if dom.feasible(st):
            break
        remaining = HORIZON - turn - 1
        ok, _ = budgeted_rollout(dom, st, param, remaining, budget)
        if ok:
            return 1, dom.calls
        best_v, best, scored = potential(dom, st), None, []
        for kind, args in propose(dom, st, rng, POOL):
            if dom.calls >= budget:
                break
            cand = apply_any(dom, st, param, kind, args)
            if cand is None:
                continue
            ok, rolled = budgeted_rollout(dom, cand, param, remaining, budget)
            if ok:
                return 1, dom.calls
            v = potential(dom, rolled)
            scored.append((v, cand))
            if v > best_v + 1e-12:
                best_v, best = v, cand
        if scored and dom.calls < budget:
            scored.sort(key=lambda x: -x[0])
            anchor = scored[0][1]
            for kind, args in propose(dom, anchor, rng, min(8, POOL)):
                if dom.calls >= budget:
                    break
                cand = apply_any(dom, anchor, param, kind, args)
                if cand is None:
                    continue
                ok, rolled = budgeted_rollout(dom, cand, param, remaining, budget)
                if ok:
                    return 1, dom.calls
                v = potential(dom, rolled)
                if v > best_v + 1e-12:
                    best_v, best = v, cand
        if best is not None:
            st = best
    return (1 if dom.feasible(st) else 0), dom.calls


def _work(t):
    key, arm = t
    try:
        if arm == "fsd_native":
            n = {"c": 0}
            orig = TE._analyze_truss

            def counted(*a, **k):
                n["c"] += 1
                return orig(*a, **k)
            TE._analyze_truss = counted
            try:
                truss, goals = TE._load_truss_and_goals(json.load(open(key)))
                st = counted(truss, goals)
                for _ in range(8):
                    if st.get("is_feasible"):
                        break
                    nxt = fsd_pass(truss, goals)
                    if nxt is truss:
                        break
                    truss = nxt
                    st = counted(truss, goals)
                return (arm, str(key), 1.0 if st.get("is_feasible") else 0.0, n["c"])
            finally:
                TE._analyze_truss = orig
        if arm == "heur_generic":
            dom = get_domain("truss")
            s = dom.load(json.load(open(key)))
            ok, _ = budgeted_rollout(dom, s, "r", 8, 10 ** 9)
            return (arm, str(key), 1.0 if ok else 0.0, dom.calls)
        budget = int(arm.split("_")[1])
        dom = get_domain("truss")
        s = dom.load(json.load(open(key)))
        y, used = search_budgeted(dom, s, "r", random.Random(seed_of(key)), budget)
        return (arm, str(key), float(y), used)
    except Exception:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=430)
    ap.add_argument("--start", type=int, default=150)
    ap.add_argument("--procs", type=int, default=46)
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/da_cost.jsonl"))
    a = ap.parse_args()
    files = [str(f) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]]
    arms = ["search_%d" % b for b in BUDGETS] + ["heur_generic", "fsd_native"]
    print("problems %d-%d | %d files | %d arms" % (a.start, a.start + a.n, len(files), len(arms)),
          flush=True)
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = [r for r in pool.map(_work, [(f, arm) for f in files for arm in arms],
                                    chunksize=2) if r]
    print("done, %.0fs\n" % (time.time() - t0))
    with open(a.out, "w") as fh:
        for arm, pid, y, c in rows:
            fh.write(json.dumps({"arm": arm, "problem_id": pid,
                                 "feasible": bool(y), "calls": c}) + "\n")

    by = {arm: {r[1]: r[2] for r in rows if r[0] == arm} for arm in arms}
    cost = {arm: [r[3] for r in rows if r[0] == arm] for arm in arms}
    pids = sorted(set.intersection(*[set(by[x]) for x in arms]))
    rate = {arm: sum(by[arm][p] for p in pids) / max(len(pids), 1) for arm in arms}
    mean_cost = {arm: (sum(cost[arm]) / max(len(cost[arm]), 1)) for arm in arms}
    print("paired on %d problems\n" % len(pids))
    print("  arm             feasibility   mean analyses   vs native heuristic")
    for arm in arms:
        tag = ""
        if arm != "fsd_native":
            u, d, p = sign_test([by["fsd_native"][x] for x in pids], [by[arm][x] for x in pids])
            tag = "%+.4f  %d-%d  p=%.2g" % (rate[arm] - rate["fsd_native"], u, d, p)
        print("  %-14s  %.4f        %8.0f      %s" % (arm, rate[arm], mean_cost[arm], tag))
    fb = rate["fsd_native"]
    over = [arm for arm in arms if arm.startswith("search_") and rate[arm] > fb]
    if over:
        print("\n  cheapest search budget that beats the heuristic: %s (%.0f analyses vs %.0f)"
              % (over[0], mean_cost[over[0]], mean_cost["fsd_native"]))
