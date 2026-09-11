"""
Robustness of the headline: how much of it is the search, and how much is the tool library?

The search reaches 0.8814 against native fully-stressed design's 0.4930 on the held-out set,
one loss in 430. Two things could be carrying that. One is the architecture -- rollout scoring,
composition, argmax with a do-nothing branch. The other is simply having good tools, and on the
truss that means the topology moves the domain registers. A framework whose margin evaporates
when a domain has nothing exotic to register is much less useful than one whose search earns
its keep on a thin library.

So the library is cut down, arm by arm, with everything else fixed:

  full        SIZE_PASS, SCALE, TRIM + registered topology      -- the reference
  notopo      generic three only, no registered tools
  notrim      no TRIM
  noscale     no SCALE
  onlysize    SIZE_PASS alone -- the search reduced to choosing a margin each turn

and the two search knobs are swept alongside, since a headline that only holds at one pool size
or one horizon is not a headline:

  pool4 / pool8 / pool32     candidates evaluated per turn (reference 16)
  h4 / h12                   turns (reference 8)

fsd_native is the control and llm is kept live per standing rule. Everything runs on the same
problems in the same invocation.
"""
import os, sys, json, math, random, zlib, argparse, time
from pathlib import Path
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts"),
          str(PROJECT / "design_agent")):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
import da_domtools                       # registers tools() on Domain and TrussDomain
from da_search import size_pass, potential, rollout
from da_search2 import sample_generic, apply_generic
from planner03 import rollout as native_rollout
from llm_finetune.envs.truss_env import _load_truss_and_goals
from probe26_presentation import sign_test

GENERIC = ("SIZE_PASS", "SCALE", "TRIM")

# arm -> (allowed generic tools, use registered tools, pool, horizon)
ARMS = {
    "full":     (GENERIC, True, 16, 8),
    "notopo":   (GENERIC, False, 16, 8),
    "notrim":   (("SIZE_PASS", "SCALE"), True, 16, 8),
    "noscale":  (("SIZE_PASS", "TRIM"), True, 16, 8),
    "onlysize": (("SIZE_PASS",), True, 16, 8),
    "pool4":    (GENERIC, True, 4, 8),
    "pool8":    (GENERIC, True, 8, 8),
    "pool32":   (GENERIC, True, 32, 8),
    "h4":       (GENERIC, True, 16, 4),
    "h12":      (GENERIC, True, 16, 12),
    "best":     (("SIZE_PASS",), True, 32, 12),
    "size_h12": (("SIZE_PASS",), True, 16, 12),
    "size_p32": (("SIZE_PASS",), True, 32, 8),
    "best_notopo": (("SIZE_PASS",), False, 32, 12),
}


def seed_of(x):
    return zlib.crc32(str(x).encode()) & 0xffffffff


def propose(dom, st, rng, n, allowed, use_registered):
    extra = {}
    if use_registered:
        try:
            extra = dom.tools() or {}
        except Exception:
            extra = {}
    names = list(allowed) + list(extra)
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


def apply_any(dom, st, param, kind, args, use_registered):
    extra = {}
    if use_registered:
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


def search_episode(dom, st, param, steps, rng, nprop, allowed, use_registered):
    for turn in range(steps):
        if dom.feasible(st):
            return 1
        nxt = size_pass(dom, st, param)
        if nxt is not st:
            st = nxt
        if dom.feasible(st):
            return 1
        remaining = steps - turn - 1
        base_ok, _ = rollout(dom, st, param, remaining)
        if base_ok:
            return 1
        best_v, best = potential(dom, st), None
        scored = []
        for kind, args in propose(dom, st, rng, nprop, allowed, use_registered):
            cand = apply_any(dom, st, param, kind, args, use_registered)
            if cand is None:
                continue
            ok, rolled = rollout(dom, cand, param, remaining)
            if ok:
                return 1
            v = potential(dom, rolled)
            scored.append((v, cand))
            if v > best_v + 1e-12:
                best_v, best = v, cand
        if scored:
            scored.sort(key=lambda x: -x[0])
            anchor = scored[0][1]
            for kind, args in propose(dom, anchor, rng, min(8, nprop), allowed, use_registered):
                cand = apply_any(dom, anchor, param, kind, args, use_registered)
                if cand is None:
                    continue
                ok, rolled = rollout(dom, cand, param, remaining)
                if ok:
                    return 1
                v = potential(dom, rolled)
                if v > best_v + 1e-12:
                    best_v, best = v, cand
        if best is not None:
            st = best
    ok, _ = rollout(dom, st, param, 2)
    return 1 if ok else 0


def _cpu(t):
    key, arm = t
    try:
        if arm == "fsd_native":
            truss, goals = _load_truss_and_goals(json.load(open(key)))
            ok, _, _ = native_rollout(truss, goals, 8)
            return (arm, str(key), 1.0 if ok else 0.0)
        allowed, reg, pool, horizon = ARMS[arm]
        dom = get_domain("truss")
        st = dom.load(json.load(open(key)))
        rng = random.Random(seed_of(key))
        return (arm, str(key),
                float(search_episode(dom, st, "r", horizon, rng, pool, allowed, reg)))
    except Exception:
        return None


def _llm(t):
    key, region, token = t
    try:
        import planner13
        r = planner13.episode(json.load(open(key)), "planner", region, token, 8, 8)
        return ("llm", str(key), 1.0 if r.get("feasible") else 0.0)
    except Exception:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--procs", type=int, default=44)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/da_robust.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    files = [str(f) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]]
    cpu_arms = ([x for x in a.only.split(",") if x] or list(ARMS)) + ["fsd_native"]
    print("problems %d-%d | %d files | %d cpu arms%s"
          % (a.start, a.start + a.n, len(files), len(cpu_arms),
             "" if a.no_llm else " + llm"), flush=True)
    t0 = time.time()
    rows = []
    with Pool(a.procs) as pool:
        rows += [r for r in pool.map(_cpu, [(f, arm) for f in files for arm in cpu_arms],
                                     chunksize=2) if r]
    print("  cpu arms done, %.0fs" % (time.time() - t0), flush=True)
    if not a.no_llm:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            rows += [r for r in ex.map(_llm, [(f, a.region, token) for f in files]) if r]
        print("  llm arm done, %.0fs" % (time.time() - t0), flush=True)

    with open(a.out, "w") as fh:
        for arm, pid, y in rows:
            fh.write(json.dumps({"arm": arm, "problem_id": pid, "feasible": bool(y)}) + "\n")

    arms = sorted({r[0] for r in rows})
    by = {arm: {r[1]: r[2] for r in rows if r[0] == arm} for arm in arms}
    pids = sorted(set.intersection(*[set(by[x]) for x in arms]))
    rate = {arm: sum(by[arm][p] for p in pids) / max(len(pids), 1) for arm in arms}
    print("\npaired on %d problems\n" % len(pids))
    for arm in sorted(arms, key=lambda x: -rate[x]):
        print("  %-11s %.4f" % (arm, rate[arm]))
    print("\n  each arm against the native heuristic:")
    for arm in sorted(arms, key=lambda x: -rate[x]):
        if arm == "fsd_native":
            continue
        u, d, p = sign_test([by["fsd_native"][x] for x in pids], [by[arm][x] for x in pids])
        print("    %-11s %+.4f   discordant %d-%d   exact sign p = %.3g"
              % (arm, rate[arm] - rate["fsd_native"], u, d, p))
