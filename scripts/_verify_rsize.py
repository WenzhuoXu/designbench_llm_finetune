"""Adversarial re-test of da_lowbudget.

Two questions:
  1. Does the recorded `random` arm reproduce exactly when re-run? (seeding / determinism)
  2. The recorded `random` proposer draws uniformly over SIX tool names, half of them
     topology (ADD_MEMBER/REMOVE_MEMBER/MOVE_JOINT), which the pick-counts show almost
     never win.  A sizing-only uniform proposer (SIZE_PASS/SCALE/TRIM) is the natural
     stronger null.  Does the LLM still beat THAT?

Reuses da_lowbudget's own budget_search / BudgetedTruss / seed_of unchanged, so the
search is byte-identical to the one that produced the headline.  Pairs against the
recorded llm and fsd rows from da_lowbudget_conf.jsonl.
"""
import os, sys, json, random, time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from math import comb

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts"), str(PROJECT / "design_agent")):
    if p not in sys.path:
        sys.path.insert(0, p)

import da_lowbudget as L          # the experiment under test, imported unchanged
import da_search2 as S2
from da_search import TOOLS as GENERIC


def sizing_propose(rng):
    """Uniform over the three SIZING tools only -- no topology draws."""
    def f(dom, st, n):
        out = []
        for _ in range(n):
            kind = rng.choice(list(GENERIC))
            out.append((kind, S2.sample_generic(dom, st, rng, kind)))
        return out
    return f


def episode(spec, arm, budget):
    pid = spec.get("problem_id", "")
    base = getattr(L._tls, "n", 0)
    dom = L.BudgetedTruss(budget)
    log = {"turns": 0, "cands": 0, "invalid": 0, "repeat": 0, "model_calls": 0,
           "parsed": 0, "picked": Counter()}
    err = None
    try:
        st = dom.load(spec)
        rng = random.Random(L.seed_of(pid))
        prop = L.random_propose(rng) if arm == "random_repro" else sizing_propose(rng)
        L.budget_search(dom, st, prop, log)
    except L.BudgetExhausted:
        pass
    except Exception as e:
        err = type(e).__name__ + ": " + str(e)[:200]
    row = {"problem_id": pid, "arm": arm, "budget": budget, "feasible": bool(dom.found),
           "calls": dom.calls, "fea": getattr(L._tls, "n", 0) - base,
           "picked": dict(log["picked"])}
    row.update({k: v for k, v in log.items() if k != "picked"})
    if err:
        row["err"] = err
    return row


def sign_test(a, b):
    up = sum(1 for i in range(len(a)) if b[i] > a[i])
    dn = sum(1 for i in range(len(a)) if b[i] < a[i])
    m = up + dn
    if m == 0:
        return up, dn, 1.0
    kk = min(up, dn)
    return up, dn, min(1.0, sum(comb(m, i) for i in range(kk + 1)) / (2.0 ** m) * 2)


if __name__ == "__main__":
    START, N = 150, 180
    BUDGETS = (10, 20)
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[START:START + N]]
    tasks = [(s, arm, B) for s in specs for B in BUDGETS
             for arm in ("random_repro", "random_sizing")]
    print("problems %d-%d | budgets %s | episodes %d" % (START, START + N, BUDGETS, len(tasks)),
          flush=True)
    t0 = time.time()
    out = str(PROJECT / "results/api_guidance/da_lowbudget_rsize.jsonl")
    rows = []
    with ThreadPoolExecutor(max_workers=24) as ex, open(out, "w") as fh:
        for i, r in enumerate(ex.map(lambda t: episode(*t), tasks)):
            rows.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 100 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs" % (time.time() - t0), flush=True)

    old = [json.loads(l) for l in
           open(PROJECT / "results/api_guidance/da_lowbudget_conf.jsonl") if l.strip()]

    print("\nerrors: %d   calls!=fea: %d   overruns: %d" % (
        sum(1 for r in rows if r.get("err")),
        sum(1 for r in rows if r["calls"] >= 0 and r["calls"] != r["fea"]),
        sum(1 for r in rows if r["calls"] > r["budget"])))

    for B in BUDGETS:
        by = {}
        for a in ("llm", "random", "fsd"):
            by[a] = {r["problem_id"]: (1.0 if r["feasible"] else 0.0)
                     for r in old if r["budget"] == B and r["arm"] == a}
        for a in ("random_repro", "random_sizing"):
            by[a] = {r["problem_id"]: (1.0 if r["feasible"] else 0.0)
                     for r in rows if r["budget"] == B and r["arm"] == a}
        pids = sorted(set.intersection(*[set(v) for v in by.values()]))
        print("\n=== budget %d   paired n=%d" % (B, len(pids)))

        same = sum(1 for p in pids if by["random"][p] == by["random_repro"][p])
        print("  RE-RUN CHECK: recorded `random` vs fresh re-run agree on %d/%d problems"
              % (same, len(pids)))

        for a in ("llm", "random", "random_repro", "random_sizing", "fsd"):
            cl = [r["calls"] for r in rows if r["budget"] == B and r["arm"] == a] or \
                 [r["calls"] for r in old if r["budget"] == B and r["arm"] == a]
            print("  %-14s feasibility %.4f   mean analyses %.1f"
                  % (a, sum(by[a][p] for p in pids) / len(pids), sum(cl) / max(len(cl), 1)))
        for x, y in (("random", "llm"), ("random_sizing", "llm"),
                     ("random", "random_sizing"), ("random_sizing", "fsd")):
            u, d, p = sign_test([by[x][q] for q in pids], [by[y][q] for q in pids])
            print("  %-14s vs %-14s  %+.4f   discordant %d-%d   p = %.4g"
                  % (y, x, sum(by[y][q] for q in pids) / len(pids)
                     - sum(by[x][q] for q in pids) / len(pids), u, d, p))
        pk = Counter()
        for r in rows:
            if r["budget"] == B and r["arm"] == "random_sizing":
                for k, v in (r.get("picked") or {}).items():
                    pk[k] += v
        print("  random_sizing moves taken:", dict(pk.most_common()))

    # conditional analyses-used, from the ORIGINAL confirmation run
    print("\n--- mean analyses used, conditioned on outcome (original conf run) ---")
    for B in BUDGETS:
        for a in ("llm", "random", "fsd"):
            sub = [r for r in old if r["budget"] == B and r["arm"] == a]
            f = [r["calls"] for r in sub if r["feasible"]]
            nf = [r["calls"] for r in sub if not r["feasible"]]
            print("  B=%-3d %-7s  feasible n=%3d mean %.1f | INFEASIBLE n=%3d mean %.2f"
                  % (B, a, len(f), sum(f) / max(len(f), 1), len(nf), sum(nf) / max(len(nf), 1)))
