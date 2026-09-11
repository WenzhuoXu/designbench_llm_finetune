"""Bound the action-space asymmetry found by _verify_argspace.py.

Measured: 37% of the LLM's SCALE proposals fall outside the random sampler's support --
it names up to 7 members per move (sampler: at most min(4,ne)) and uses factors up to
2.0 after clipping (sampler: 1.60).  SCALE is the move class the report credits with the
win.  So the question is whether the LLM's margin survives a random control given the
SAME SCALE support.

  random_sizing  uniform over SIZE_PASS/SCALE/TRIM, original sampler ranges
  random_wide    identical, except SCALE draws k ~ 1..min(7,ne) ids and factor ~ U(0.75,2.0)

Everything else -- budget_search, BudgetedTruss, the mandatory SIZE_PASS(1.05), seeding,
problems -- is da_lowbudget's own code, unchanged.
"""
import sys, json, random, time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from math import comb

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts"), str(PROJECT / "design_agent")):
    if p not in sys.path:
        sys.path.insert(0, p)

import da_lowbudget as L
from da_search import TOOLS as GENERIC


def wide_propose(rng, wide):
    def f(dom, st, n):
        ne = dom.n_elements(st)
        out = []
        for _ in range(n):
            kind = rng.choice(list(GENERIC))
            if kind == "SIZE_PASS":
                out.append((kind, {"margin": rng.uniform(0.95, 1.35)}))
            elif kind == "TRIM":
                out.append((kind, {"threshold": rng.uniform(1.2, 5.0),
                                   "factor": rng.uniform(0.75, 0.95)}))
            else:
                cap, hi = (7, 2.0) if wide else (4, 1.6)
                k = rng.randint(1, max(1, min(cap, ne)))
                out.append((kind, {"ids": rng.sample(range(ne), k),
                                   "factor": rng.uniform(0.75, hi)}))
        return out
    return f


def episode(spec, arm, budget):
    pid = spec.get("problem_id", "")
    dom = L.BudgetedTruss(budget)
    log = {"turns": 0, "cands": 0, "invalid": 0, "repeat": 0, "model_calls": 0,
           "parsed": 0, "picked": Counter()}
    try:
        st = dom.load(spec)
        rng = random.Random(L.seed_of(pid))
        L.budget_search(dom, st, wide_propose(rng, arm == "random_wide"), log)
    except L.BudgetExhausted:
        pass
    except Exception as e:
        return {"problem_id": pid, "arm": arm, "budget": budget, "feasible": False,
                "calls": dom.calls, "err": type(e).__name__ + ": " + str(e)[:120]}
    return {"problem_id": pid, "arm": arm, "budget": budget, "feasible": bool(dom.found),
            "calls": dom.calls, "picked": dict(log["picked"])}


def sign_test(a, b):
    up = sum(1 for i in range(len(a)) if b[i] > a[i])
    dn = sum(1 for i in range(len(a)) if b[i] < a[i])
    m = up + dn
    if m == 0:
        return up, dn, 1.0
    kk = min(up, dn)
    return up, dn, min(1.0, sum(comb(m, i) for i in range(kk + 1)) / (2.0 ** m) * 2)


if __name__ == "__main__":
    START, N, BUDGETS = 150, 180, (10, 20)
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[START:START + N]]
    tasks = [(s, arm, B) for s in specs for B in BUDGETS
             for arm in ("random_sizing", "random_wide")]
    print("problems %d-%d | budgets %s | episodes %d" % (START, START + N, BUDGETS, len(tasks)),
          flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=24) as ex:
        rows = list(ex.map(lambda t: episode(*t), tasks))
    print("wall %.0fs  errors %d  overruns %d" % (
        time.time() - t0, sum(1 for r in rows if r.get("err")),
        sum(1 for r in rows if r["calls"] > r["budget"])), flush=True)

    old = [json.loads(l) for l in
           open(PROJECT / "results/api_guidance/da_lowbudget_conf.jsonl") if l.strip()]

    for B in BUDGETS:
        by = {a: {r["problem_id"]: (1.0 if r["feasible"] else 0.0)
                  for r in old if r["budget"] == B and r["arm"] == a}
              for a in ("llm", "random", "fsd")}
        for a in ("random_sizing", "random_wide"):
            by[a] = {r["problem_id"]: (1.0 if r["feasible"] else 0.0)
                     for r in rows if r["budget"] == B and r["arm"] == a}
        pids = sorted(set.intersection(*[set(v) for v in by.values()]))
        print("\n=== budget %d   paired n=%d" % (B, len(pids)))
        for a in ("llm", "random", "random_sizing", "random_wide", "fsd"):
            print("  %-14s feasibility %.4f" % (a, sum(by[a][p] for p in pids) / len(pids)))
        for x, y in (("random", "llm"), ("random_sizing", "llm"), ("random_wide", "llm"),
                     ("random_sizing", "random_wide"), ("random_wide", "fsd")):
            u, d, p = sign_test([by[x][q] for q in pids], [by[y][q] for q in pids])
            print("  %-14s vs %-14s  %+.4f   discordant %d-%d   p = %.4g"
                  % (y, x, sum(by[y][q] for q in pids) / len(pids)
                     - sum(by[x][q] for q in pids) / len(pids), u, d, p))
        pk = Counter()
        for r in rows:
            if r["budget"] == B and r["arm"] == "random_wide":
                for k, v in (r.get("picked") or {}).items():
                    pk[k] += v
        print("  random_wide moves taken:", dict(pk.most_common()))
