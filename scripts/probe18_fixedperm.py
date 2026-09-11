"""
EXPERIMENT 18 -- resolve a design discrepancy of my own making.

probe13 measured, on Sonnet 4.5, that the agent follows a FALSE address as reliably as a
true one (0.912 vs 0.852, p=0.10, null). probe15 measured the opposite on four families
(follow-rate drops 0.39-0.47, p ~ 1e-10). The two are not comparable, and the reason is a
bug in probe15, not a finding:

  probe13  ONE permutation per problem, fixed across all turns and reused in the history
           the model sees. This matches the project's own placebo arm, which seeds
           member_table with a per-problem CRC.
  probe15  the permutation is REDRAWN every turn, and redrawn AGAIN when the history entry
           is built -- so the same member's safety factors jump between rows from turn to
           turn, and the history disagrees with the observation it followed.

A redrawn permutation is a mixture over unknown permutations, which is a strictly stronger
garbling AND is detectable across turns in a way a fixed one is not. So probe15's family
numbers measure INCONSISTENT corruption, which is a different condition.

This runs the family sweep with the permutation fixed per episode, at probe13's turn budget,
so the key contrast is measured on all four families under the design probe13 used.
"""
import os, sys, json, math, random, zlib, argparse, time, re, statistics, collections
from pathlib import Path
from math import comb, sqrt, erf
from concurrent.futures import ThreadPoolExecutor
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals
from llm_finetune.data.processors import designbench_prompt as DP
import api_grammar_2x2 as H
from probe15_matrix import build_table, true_fos, spearman, SUP


def episode(spec, model, permute, region, token, k, max_steps):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    pid = spec.get("problem_id", "")
    n0 = len(truss.members)
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    # ONE permutation, fixed for the whole episode, used for the observation AND the history
    rng = random.Random(zlib.crc32(pid.encode()))
    perm = list(range(n0))
    if permute:
        rng.shuffle(perm)

    def disp_of(t):
        tf = true_fos(t)
        return [tf[perm[i]] for i in range(len(tf))] if permute else tf

    system = DP.TRUSS_SYSTEM_PROMPT + (
        f"\n\nEach turn, propose {k} DIFFERENT candidate moves rather than one. Think "
        f"briefly, then emit exactly {k} lines of the form <action>...</action>. They "
        f"must be genuinely different candidates -- only one will be executed."
    ) + H.COMPOUND_RULE
    history, turns = [], []
    feasible = False
    for _ in range(max_steps):
        n = len(truss.members)
        if n != n0:
            break
        tf = true_fos(truss)
        disp = disp_of(truss)
        tm = [min(a / gb, b / gy) for a, b in tf]
        dm = [min(a / gb, b / gy) for a, b in disp]
        j_true = min(range(n), key=lambda j: tm[j])
        row_shown = min(range(n), key=lambda j: dm[j])
        rho = spearman([min(x, 50) for x in tm], [min(x, 50) for x in dm])
        tbl = build_table(truss, disp)
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
            for a, b, d in SUP.findall(c):
                if a:
                    s.add(int(a))
                if d:
                    s.add(int(d))
                for x in (b or "").replace(" ", "").split(","):
                    if x.isdigit():
                        s.add(int(x))
            sup.append(sorted(s))
        turns.append({"j_true": j_true, "row_shown": row_shown, "n": n,
                      "rho": None if rho is None else round(rho, 4), "supports": sup})
        scored = []
        for c in cands:
            nt = H.apply_candidate(truss, c)
            if nt is None or len(nt.members) != n0:
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
        history.append({"t": text.strip(),
                        "o": DP.format_eval_result(state) + build_table(truss, disp_of(truss))})
        if state.get("is_feasible"):
            feasible = True
            break
    return {"problem_id": pid, "model": model, "permute": int(permute),
            "feasible": feasible, "turns": turns}


def work(t):
    spec, model, permute, region, token, k, ms = t
    try:
        return episode(spec, model, permute, region, token, k, ms)
    except Exception as e:
        return {"problem_id": spec.get("problem_id"), "model": model,
                "permute": int(permute), "turns": [], "err": str(e)[:150]}


FAM = ["us.anthropic.claude-sonnet-4-5-20250929-v1:0",
       "us.anthropic.claude-sonnet-4-20250514-v1:0",
       "us.anthropic.claude-haiku-4-5-20251001-v1:0",
       "us.anthropic.claude-opus-4-1-20250805-v1:0"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/fixedperm.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[:a.n]]
    tasks = [(s, m, pp, a.region, token, a.k, a.max_steps)
             for s in specs for m in FAM for pp in (False, True)]
    print("episodes %d" % len(tasks), flush=True)
    rows = []
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w") as fh:
        for i, r in enumerate(ex.map(work, tasks)):
            rows.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 100 == 0:
                print("  %d/%d %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs" % (time.time() - t0))

    def ep(r):
        ht = hr = tot = 0; ch = 0.0
        for t in r["turns"]:
            for s in t["supports"]:
                if not s:
                    continue
                tot += 1
                ht += int(t["j_true"] in s); hr += int(t["row_shown"] in s)
                ch += min(1.0, len(s) / max(t["n"], 1))
        return None if not tot else {"true": ht/tot, "row": hr/tot, "chance": ch/tot,
                                     "feas": int(r["feasible"]), "n": tot}
    cell = collections.defaultdict(dict)
    for r in rows:
        e = ep(r)
        if e:
            cell[(r["model"], r["permute"])][r["problem_id"]] = e

    def norm_p(z):
        return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

    def paired(A, B, ks):
        d = [A[k] - B[k] for k in ks]
        m = sum(d)/len(d); sd = statistics.stdev(d) if len(d) > 1 else 0.0
        se = sd/sqrt(len(d)) if sd else 0.0
        hi = sum(1 for x in d if x > 0); lo = sum(1 for x in d if x < 0); dd = hi+lo
        sp = 1.0 if dd == 0 else min(1.0, 2*sum(comb(dd,i) for i in range(0, min(hi,lo)+1))/2**dd)
        return m, se, hi, lo, sp

    print("\n=== FIXED permutation, per-episode. All four families. ===")
    print("  model                        n   truthful->true  perm->row  perm->true  chance  feas T/P")
    for m in FAM:
        d0 = cell.get((m, 0), {}); d1 = cell.get((m, 1), {})
        ks = sorted(set(d0) & set(d1))
        if len(ks) < 5:
            print("  %-28s insufficient (%d)" % (m.split("anthropic.")[-1][:26], len(ks)))
            continue
        t0_ = statistics.mean(d0[k]["true"] for k in ks)
        r1 = statistics.mean(d1[k]["row"] for k in ks)
        t1 = statistics.mean(d1[k]["true"] for k in ks)
        c1 = statistics.mean(d1[k]["chance"] for k in ks)
        print("  %-28s %-3d %.3f           %.3f      %.3f       %.3f   %.2f/%.2f"
              % (m.split("anthropic.")[-1][:26], len(ks), t0_, r1, t1, c1,
                 statistics.mean(d0[k]["feas"] for k in ks),
                 statistics.mean(d1[k]["feas"] for k in ks)))
        mm, se, hi, lo, sp = paired({k: d1[k]["row"] for k in ks},
                                    {k: d0[k]["true"] for k in ks}, ks)
        # TOST equivalence against a +-0.10 margin on the follow-rate difference
        z_lo = (mm + 0.10) / se if se else float("inf")
        z_hi = (mm - 0.10) / se if se else float("-inf")
        p_tost = max(norm_p(z_lo) / 2 if mm > -0.10 else 1.0,
                     norm_p(z_hi) / 2 if mm < 0.10 else 1.0)
        print("      follows FALSE address vs TRUE: %+.3f (se %.3f, sign %d-%d p=%.2e) | "
              "TOST |d|<0.10: p=%.3f %s" % (mm, se, hi, lo, sp, p_tost,
                                            "EQUIVALENT" if p_tost < 0.05 else "not equivalent"))
        bb, _, hb, lb, spb = paired({k: d1[k]["true"] for k in ks},
                                    {k: d1[k]["chance"] for k in ks}, ks)
        print("      permuted true-worst vs chance: %+.3f (sign %d-%d p=%.2e)" % (bb, hb, lb, spb))
