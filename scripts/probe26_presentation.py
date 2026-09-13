"""
Intervene on the SIGNAL, not on the policy.

Every deployable intervention tried so far has tried to make the agent act better on a signal
it is given, and all seven failed at matched budget. They share a blind spot: the signal's
encoding was held fixed. But the law says value is carried by ORDERING FIDELITY, and in the
real setting the tool's ordering is already perfect -- it is the model that acts on a degraded
version of it (in-loop slope +3.20 against a procedural +7.14).

So the intervention belongs at the interface. The per-member table is rendered in member-id
order; the criticality ordering is present in the numbers but not in the layout, and the model
must extract it. Re-sorting the rows by criticality changes NO information -- same rows, same
member ids, same columns, essentially the same tokens -- it only makes the ordering that
matters the ordering the reader encounters.

Three arms, paired on problem and seed, all on the TRUTHFUL signal (no corruption):

  native      rows in member-id order                        (the current interface)
  worst_first rows sorted by shown margin ascending
  best_first  rows sorted by shown margin descending

best_first separates two accounts: if what helps is making the ordering salient at all, both
sorted arms move; if what helps is putting the critical members where they are read first,
only worst_first moves. A procedural consumer is invariant to row order by construction, so
any effect here is model-side.

Endpoints. Primary: feasibility, paired exact sign test, worst_first vs native. Secondary, and
more sensitive: the rate at which the model's first candidate actually addresses the true worst
member -- the acted ordering fidelity the law is about.
"""
import os, sys, json, math, random, zlib, argparse, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
# Repo root from this file's location; DesignBench is a sibling by default
# (the Bridges-2 layout) and can be pointed elsewhere with DESIGNBENCH_ROOT.
PROJECT = Path(__file__).resolve().parents[1]
DESIGNBENCH = Path(os.environ.get("DESIGNBENCH_ROOT", PROJECT.parent / "DesignBench"))
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals
from llm_finetune.data.processors import designbench_prompt as DP
import api_grammar_2x2 as H
from probe15_matrix import fmt, true_fos, SUP

SONNET45 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
ARMS = ("native", "worst_first", "best_first")


def build_table_ordered(truss, disp, order):
    """probe15's member table with the ROW SEQUENCE permuted. Content per row is unchanged."""
    rows = ["", "Per-member state (r = outer radius, t = wall thickness, m):",
            "  id    FOS_buckling    FOS_yielding         r         t"]
    for i in order:
        m = truss.members[i]
        sp = getattr(getattr(m, "shape", None), "_params", None) or {}
        r_, t_ = sp.get("r"), sp.get("t")
        rows.append(f"  M{i:<4d}{fmt(disp[i][0])}    {fmt(disp[i][1])}  "
                    f"{(f'{float(r_):.5f}' if r_ is not None else '  n/a  ')}  "
                    f"{(f'{float(t_):.5f}' if t_ is not None else '  n/a  ')}")
    rows.append("A member with FOS far above the required 1.5 is carrying more "
                "material than it needs.")
    return "\n".join(rows)


def row_order(arm, margins):
    n = len(margins)
    if arm == "native":
        return list(range(n))
    o = sorted(range(n), key=lambda i: margins[i])
    return o if arm == "worst_first" else o[::-1]


def episode(spec, model, arm, region, token, k, max_steps, seed):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    pid = spec.get("problem_id", "")
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    system = DP.TRUSS_SYSTEM_PROMPT + (
        f"\n\nEach turn, propose {k} DIFFERENT candidate moves rather than one. Think "
        f"briefly, then emit exactly {k} lines of the form <action>...</action>. They "
        f"must be genuinely different candidates -- only one will be executed."
    ) + H.COMPOUND_RULE
    history, turns = [], []
    feasible = False
    for _ in range(max_steps):
        n = len(truss.members)
        tf = true_fos(truss)                                  # truthful signal throughout
        margins = [min(a / gb, b / gy) for a, b in tf]
        j_true = min(range(n), key=lambda j: margins[j])
        tbl = build_table_ordered(truss, tf, row_order(arm, margins))
        obs = DP.format_eval_result(state) + tbl
        convo = [{"role": "user", "content": [{"text":
                  DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                  + DP.format_eval_result(initial) + (tbl if not history else "")}]}]
        for h in history:
            convo.append({"role": "assistant", "content": [{"text": h["t"]}]})
            convo.append({"role": "user", "content": [{"text":
                          "[Simulation Result]\nSTRUCTURAL ANALYSIS RESULT:\n" + h["o"]}]})
        text = H.call(convo, system, model, region, token)
        cands = H.parse_candidates(text, k, True)
        if not cands:
            history.append({"t": f"No <action> parsed. Emit exactly {k} lines.", "o": obs})
            continue
        sup = []
        for c in cands:
            s = set()
            for a_, b_, d_ in SUP.findall(c):
                if a_:
                    s.add(int(a_))
                if d_:
                    s.add(int(d_))
                for x in (b_ or "").replace(" ", "").split(","):
                    if x.isdigit():
                        s.add(int(x))
            sup.append(sorted(s))
        turns.append({"j_true": j_true, "n": n, "supports": sup,
                      "hit_first": bool(sup and j_true in sup[0]),
                      "hit_any": any(j_true in s for s in sup)})
        scored = []
        for c in cands:
            nt = H.apply_candidate(truss, c)
            if nt is None:
                continue
            scored.append((nt, _analyze_truss(nt, goals)))
        if not scored:
            history.append({"t": text.strip(), "o": obs})
            continue
        B = float(goals.get("maximum_mass", float("inf")))

        def key(t):
            st = t[1]
            fb = float(st.get("fos_buckling", 0) or 0); fy = float(st.get("fos_yielding", 0) or 0)
            ms = float(st.get("mass", 9e9) or 9e9)
            return min(fb / gb, fy / gy) - (0.5 if ms > B else 0.0)
        truss, state = max(scored, key=key)
        nf = true_fos(truss)
        nm = [min(a / gb, b / gy) for a, b in nf]
        history.append({"t": text.strip(),
                        "o": DP.format_eval_result(state)
                             + build_table_ordered(truss, nf, row_order(arm, nm))})
        if state.get("is_feasible"):
            feasible = True
            break
    return {"problem_id": pid, "model": model, "arm": arm, "feasible": feasible, "turns": turns}


def work(t):
    spec, model, arm, region, token, k, ms, seed = t
    try:
        return episode(spec, model, arm, region, token, k, ms, seed)
    except Exception as e:
        return {"problem_id": spec.get("problem_id"), "model": model, "arm": arm,
                "feasible": False, "turns": [], "err": str(e)[:200]}


def sign_test(a, b):
    from math import comb
    up = sum(1 for i in range(len(a)) if b[i] > a[i])
    dn = sum(1 for i in range(len(a)) if b[i] < a[i])
    m = up + dn
    if m == 0:
        return up, dn, 1.0
    kk = min(up, dn)
    return up, dn, min(1.0, sum(comb(m, i) for i in range(kk + 1)) / (2.0 ** m) * 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/presentation.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[:a.n]]
    tasks = [(s, SONNET45, arm, a.region, token, a.k, a.max_steps, a.seed)
             for s in specs for arm in ARMS]
    print("arms %d | episodes %d | truthful signal throughout" % (len(ARMS), len(tasks)), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w") as fh:
        for i, r in enumerate(ex.map(work, tasks)):
            out.append(r)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 50 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs\n" % (time.time() - t0))

    by = {arm: {r["problem_id"]: r for r in out if r["arm"] == arm} for arm in ARMS}
    pids = sorted(set.intersection(*[set(by[arm]) for arm in ARMS]))
    print("paired on %d problems\n" % len(pids))
    print("  arm           feasible   addresses true worst (1st cand)   (any cand)")
    stat = {}
    for arm in ARMS:
        f = [1.0 if by[arm][p]["feasible"] else 0.0 for p in pids]
        tn = [t for p in pids for t in by[arm][p]["turns"]]
        h1 = sum(t["hit_first"] for t in tn) / max(len(tn), 1)
        ha = sum(t["hit_any"] for t in tn) / max(len(tn), 1)
        stat[arm] = f
        print("  %-12s  %.4f        %.4f  (n=%d turns)          %.4f"
              % (arm, sum(f) / len(f), h1, len(tn), ha))
    print()
    for arm in ("worst_first", "best_first"):
        u, d, p = sign_test(stat["native"], stat[arm])
        print("  %-12s vs native: %+.4f  discordant %d-%d  exact sign p = %.3g"
              % (arm, sum(stat[arm]) / len(pids) - sum(stat["native"]) / len(pids), u, d, p))
