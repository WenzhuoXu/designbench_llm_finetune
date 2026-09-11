"""
The cost curve in the second domain, where the domain registers nothing.

On the truss the search saturates at 287 analyses for 0.9721 against the heuristic's 0.4930 at
6, and beats it at every budget down to 44 analyses. That is a truss result until the same curve
appears somewhere else. The synthetic domain is the harder case for the framework: it registers
no tools at all, so the search has only the generic three, and on the truss two of those three
turned out to dilute the pool rather than help.

Two questions in one run. Which generic tool set is right where there is no topology to fall
back on -- all three, or SIZE_PASS alone as on the truss. And what the curve looks like against
budget, with the domain's own heuristic as the in-script control and its cost measured the same
way.

An LLM arm proposes tool calls into the same search, kept live per standing rule; it runs on
threads while the rest runs on processes, over the same instances in the same invocation.
"""
import os, sys, json, math, random, zlib, argparse, time, re
from pathlib import Path
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(PROJECT / "scripts"), str(PROJECT / "design_agent")):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_synth import SynthDomain
from da_search import size_pass, potential
from da_search2 import sample_generic, apply_generic
from probe26_presentation import sign_test
import api_grammar_2x2 as H

BUDGETS = (100, 250, 500, 1000, 2500, 5000)
POOL, HORIZON = 32, 12
GENERIC = ("SIZE_PASS", "SCALE", "TRIM")
SONNET45 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
TOOL_RE = re.compile(r"(SIZE_PASS|SCALE|TRIM)\s*\(([^)]*)\)")

LLM_SYSTEM = """You are tuning the sizes of independent components. Each has a value g_i that
rises with its size x_i at an unknown rate, and a cost weight; the configuration is acceptable
when every g_i >= 1.0 and total cost is within budget.

A sizing pass runs automatically each turn, moving every component towards its requirement.
Propose %d candidate moves, each on its own line as <tool>...</tool>, from:

  SIZE_PASS(margin)              re-run the sizing pass to a different margin, 0.95 to 1.35
  SCALE([ids], factor)           scale the listed components, 0.75 to 1.6
  TRIM(threshold, factor)        shrink every component whose value exceeds threshold

Each is simulated and the best kept, so a bad guess costs nothing. Nothing but the lines."""


def seed_of(x):
    return zlib.crc32(str(x).encode()) & 0xffffffff


def budgeted_rollout(dom, st, steps, budget):
    cur = st
    for _ in range(max(0, steps)):
        if dom.calls >= budget:
            return dom.feasible(cur), cur
        if dom.feasible(cur):
            return True, cur
        nxt = size_pass(dom, cur, "x")
        if nxt is cur:
            break
        cur = nxt
    return dom.feasible(cur), cur


def search_budgeted(dom, st, rng, budget, allowed, propose_fn=None):
    for turn in range(HORIZON):
        if dom.calls >= budget or dom.feasible(st):
            break
        nxt = size_pass(dom, st, "x")
        if nxt is not st:
            st = nxt
        if dom.feasible(st):
            break
        remaining = HORIZON - turn - 1
        ok, _ = budgeted_rollout(dom, st, remaining, budget)
        if ok:
            return 1, dom.calls
        cands = (propose_fn(dom, st, rng, POOL) if propose_fn else
                 [(rng.choice(allowed), None) for _ in range(POOL)])
        best_v, best, scored = potential(dom, st), None, []
        for kind, args in cands:
            if dom.calls >= budget:
                break
            if args is None:
                args = sample_generic(dom, st, rng, kind)
            cand = apply_generic(dom, st, "x", kind, args)
            if cand is None:
                continue
            ok, rolled = budgeted_rollout(dom, cand, remaining, budget)
            if ok:
                return 1, dom.calls
            v = potential(dom, rolled)
            scored.append((v, cand))
            if v > best_v + 1e-12:
                best_v, best = v, cand
        if scored and dom.calls < budget:
            scored.sort(key=lambda x: -x[0])
            anchor = scored[0][1]
            for _ in range(min(8, POOL)):
                if dom.calls >= budget:
                    break
                kind = rng.choice(allowed)
                cand = apply_generic(dom, anchor, "x", kind,
                                     sample_generic(dom, anchor, rng, kind))
                if cand is None:
                    continue
                ok, rolled = budgeted_rollout(dom, cand, remaining, budget)
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
        dom = SynthDomain()
        st = dom.load({"seed": key, "n": 10})
        rng = random.Random(seed_of(key))
        if arm == "heuristic":
            ok, _ = budgeted_rollout(dom, st, HORIZON, 10 ** 9)
            return (arm, "s%d" % key, 1.0 if ok else 0.0, dom.calls)
        toolset, budget = arm.split("_")[1], int(arm.split("_")[2])
        allowed = GENERIC if toolset == "gen" else ("SIZE_PASS",)
        y, used = search_budgeted(dom, st, rng, budget, allowed)
        return (arm, "s%d" % key, float(y), used)
    except Exception:
        return None


def _llm(t):
    key, region, token = t
    try:
        dom = SynthDomain()
        st = dom.load({"seed": key, "n": 10})
        rng = random.Random(seed_of(key))

        def propose_fn(d, s, r, n):
            em = d.element_margins(s)
            rows = "\n".join("  C%-3d value %8.3f   weight %6.3f   size %7.4f"
                             % (i, em[i], s["w"][i], s["x"][i]) for i in range(s["n"]))
            body = ("Components:\n" + rows
                    + "\n\ntotal cost %.3f, budget %.3f\n"
                    % (sum(s["w"][i] * s["x"][i] ** 2 for i in range(s["n"])), s["B"]))
            text = H.call([{"role": "user", "content": [{"text": body}]}],
                          LLM_SYSTEM % 8, SONNET45, region, token) or ""
            out = []
            for nm, a in TOOL_RE.findall(text)[:8]:
                nums = re.findall(r"-?\d+\.?\d*", a)
                if nm == "SIZE_PASS":
                    out.append((nm, {"margin": float(nums[0]) if nums else 1.05}))
                elif nm == "TRIM":
                    out.append((nm, {"threshold": float(nums[0]) if nums else 2.0,
                                     "factor": float(nums[1]) if len(nums) > 1 else 0.85}))
                else:
                    ids = [int(x) for x in re.findall(r"\d+", a.split(",")[0])]
                    out.append((nm, {"ids": ids or [0],
                                     "factor": float(nums[-1]) if nums else 1.1}))
            while len(out) < n:
                k = rng.choice(GENERIC)
                out.append((k, sample_generic(d, s, rng, k)))
            return out
        y, used = search_budgeted(dom, st, rng, 5000, GENERIC, propose_fn)
        return ("llm", "s%d" % key, float(y), used)
    except Exception:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nllm", type=int, default=120)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/da_cost_synth.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    arms = (["search_gen_%d" % b for b in BUDGETS]
            + ["search_size_%d" % b for b in BUDGETS] + ["heuristic"])
    print("%d instances | %d arms%s" % (a.n, len(arms), "" if a.no_llm else " + llm"), flush=True)
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = [r for r in pool.map(_work, [(i, arm) for i in range(a.n) for arm in arms],
                                    chunksize=8) if r]
    print("  cpu arms done, %.0fs" % (time.time() - t0), flush=True)
    llm_rows = []
    if not a.no_llm:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            llm_rows = [r for r in ex.map(_llm, [(i, a.region, token)
                                                 for i in range(a.nllm)]) if r]
        print("  llm arm done, %.0fs" % (time.time() - t0), flush=True)

    with open(a.out, "w") as fh:
        for arm, pid, y, c in rows + llm_rows:
            fh.write(json.dumps({"arm": arm, "problem_id": pid,
                                 "feasible": bool(y), "calls": c}) + "\n")

    by = {arm: {r[1]: r[2] for r in rows if r[0] == arm} for arm in arms}
    cost = {arm: [r[3] for r in rows if r[0] == arm] for arm in arms}
    pids = sorted(set.intersection(*[set(by[x]) for x in arms]))
    rate = {arm: sum(by[arm][p] for p in pids) / max(len(pids), 1) for arm in arms}
    mc = {arm: sum(cost[arm]) / max(len(cost[arm]), 1) for arm in arms}
    print("\npaired on %d instances\n" % len(pids))
    print("  arm                feasibility   mean evals   vs heuristic")
    for arm in arms:
        tag = ""
        if arm != "heuristic":
            u, d, p = sign_test([by["heuristic"][x] for x in pids], [by[arm][x] for x in pids])
            tag = "%+.4f  %d-%d  p=%.2g" % (rate[arm] - rate["heuristic"], u, d, p)
        print("  %-17s  %.4f       %7.0f      %s" % (arm, rate[arm], mc[arm], tag))
    if llm_rows:
        lp = sorted({r[1] for r in llm_rows} & set(pids))
        if lp:
            lr = sum(dict((r[1], r[2]) for r in llm_rows)[x] for x in lp) / len(lp)
            hr = sum(by["heuristic"][x] for x in lp) / len(lp)
            br = sum(by["search_gen_5000"][x] for x in lp) / len(lp)
            print("\n  on the %d instances the llm arm covered: llm %.4f, "
                  "search_gen_5000 %.4f, heuristic %.4f" % (len(lp), lr, br, hr))
