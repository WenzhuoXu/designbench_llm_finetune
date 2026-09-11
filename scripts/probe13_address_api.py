"""
EXPERIMENT 13 -- the address test, with the model in the loop.

The placebo arm scores 0.059 against 0.416 with NO table (-35.6pp, disc 73-1, p=7.9e-21).
probe10 showed the simulator-side twin: exact per-element factors at PERMUTED addresses are
worth +2.5pp (p=0.125, null) against +26.0pp correctly addressed.

The mechanism claim is that the policy TRANSCRIBES the addresses it is shown and carries no
independent prior that could override a false one. The shuffle makes that decidable, because
row i displays member perm[i]'s FOS while r/t stay truthful. So if j* is the TRUE worst
member, the row a transcriber would act on is perm^{-1}(j*), and the true worst is j*.

  P(acts on perm^{-1}(j*))  high  -> transcription: the policy reads addresses off the table
  P(acts on j*)             chance -> it has no independent physical judgement

Both are logged per turn, in both arms, on the SAME problems. The per_member arm is the
positive control and should put both quantities at the same place (perm = identity).

Cost control: max_steps is capped low. The endpoint here is per-TURN address behaviour,
which is already saturated in a few turns; episode outcomes are measured elsewhere.
"""
import os, sys, json, copy, math, random, zlib, argparse, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals
from llm_finetune.data.processors import designbench_prompt as DP
import api_grammar_2x2 as H          # reuse call/member_table/parse_candidates/apply_candidate

LOCK = threading.Lock()


def perm_for(problem_id, n):
    """Reproduce exactly the permutation member_table() applies."""
    rng = random.Random(zlib.crc32(problem_id.encode()))
    perm = list(range(n))
    rng.shuffle(perm)
    return perm


def own_margin(m, gb, gy):
    def v(a):
        try:
            x = float(getattr(m, a, float("inf")) or float("inf"))
        except Exception:
            x = float("inf")
        return x if x == x else float("inf")
    return min(v("fos_buckling") / gb, v("fos_yielding") / gy)


def episode(spec, obs_mode, model, region, token, k, max_steps):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    pid = spec.get("problem_id", "")
    n = len(truss.members)
    show_table = True
    seed = zlib.crc32(pid.encode()) if obs_mode == "shuffled" else None
    perm = perm_for(pid, n) if obs_mode == "shuffled" else list(range(n))
    inv = [0] * n
    for i, j in enumerate(perm):
        inv[j] = i                                   # inv[j] = the row that displays member j
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    tbl = lambda t: H.member_table(t, seed) if show_table else ""
    system = DP.TRUSS_SYSTEM_PROMPT + (
        f"\n\nEach turn, propose {k} DIFFERENT candidate moves rather than one. Think "
        f"briefly, then emit exactly {k} lines of the form <action>...</action>. They "
        f"must be genuinely different candidates -- only one will be executed."
    ) + H.COMPOUND_RULE

    history, turns = [], []
    for _ in range(max_steps):
        margins = [own_margin(m, gb, gy) for m in truss.members]
        j_true = min(range(n), key=lambda j: margins[j])       # TRUE worst member
        row_shown = inv[j_true]                                # row a transcriber acts on
        obs = DP.format_eval_result(state) + tbl(truss)
        convo = [{"role": "user", "content": [{"text":
                  DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                  + DP.format_eval_result(initial)
                  + (tbl(truss) if not history else "")}]}]
        for h in history:
            convo.append({"role": "assistant", "content": [{"text": h["thinking"]}]})
            convo.append({"role": "user", "content": [{"text":
                          "[Simulation Result]\nSTRUCTURAL ANALYSIS RESULT:\n" + h["obs"]}]})
        text = H.call(convo, system, model, region, token)
        cands = H.parse_candidates(text, k, True)
        if not cands:
            history.append({"thinking": f"No <action> parsed. Emit exactly {k} "
                                        "<action>...</action> lines.", "obs": obs})
            continue
        sup = []
        for c in cands:
            s = set()
            for part in c.split(H.MACRO_SEP):
                mm = __import__("re").findall(r"SCALE_PARAM\(\s*(\d+)|SCALE_MULTI_PARAM\(\s*\[([0-9,\s]*)\]", part)
                for a, b in mm:
                    if a:
                        s.add(int(a))
                    for x in (b or "").replace(" ", "").split(","):
                        if x.isdigit():
                            s.add(int(x))
            sup.append(sorted(s))
        turns.append({"j_true": j_true, "row_shown": row_shown, "n": n,
                      "margins": [round(x, 4) if math.isfinite(x) else None for x in margins],
                      "supports": sup})
        scored = []
        for c in cands:
            nt = H.apply_candidate(truss, c)
            if nt is None:
                continue
            st = _analyze_truss(nt, goals)
            scored.append((nt, st))
        if not scored:
            history.append({"thinking": text.strip(), "obs": obs})
            continue
        # advance on the candidate with the best min-margin under budget (selector is not
        # the object of study here; any consistent rule keeps the trajectory realistic)
        B = float(goals.get("maximum_mass", float("inf")))
        def key(t):
            st = t[1]
            fb = float(st.get("fos_buckling", 0) or 0); fy = float(st.get("fos_yielding", 0) or 0)
            ms = float(st.get("mass", 9e9) or 9e9)
            return (min(fb / gb, fy / gy) - (0.5 if ms > B else 0.0))
        truss, state = max(scored, key=key)
        history.append({"thinking": text.strip(),
                        "obs": DP.format_eval_result(state) + tbl(truss)})
        if state.get("is_feasible"):
            break
    return {"problem_id": pid, "obs": obs_mode, "turns": turns}


def work(t):
    spec, obs_mode, model, region, token, k, ms = t
    try:
        return episode(spec, obs_mode, model, region, token, k, ms)
    except Exception as e:
        return {"problem_id": spec.get("problem_id"), "obs": obs_mode, "turns": [], "err": str(e)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/address_test.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    files = sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[:a.n]
    specs = [json.load(open(f)) for f in files]
    tasks = [(s, m, a.model, a.region, token, a.k, a.max_steps)
             for s in specs for m in ("per_member", "shuffled")]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w") as fh:
        for i, r in enumerate(ex.map(work, tasks)):
            rows.append(r)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 20 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs" % (time.time() - t0))

    print("\n=== ADDRESS TEST ===")
    for mode in ("per_member", "shuffled"):
        R = [r for r in rows if r["obs"] == mode]
        T = [t for r in R for t in r["turns"]]
        if not T:
            print("  %s: no turns (errors: %s)" % (mode, [r.get("err") for r in R if r.get("err")][:2]))
            continue
        hit_true = hit_row = tot = 0
        chance = 0.0
        for t in T:
            for s in t["supports"]:
                if not s:
                    continue
                tot += 1
                hit_true += int(t["j_true"] in s)
                hit_row += int(t["row_shown"] in s)
                chance += min(1.0, len(s) / max(t["n"], 1))
        print("  %-11s turns %-4d candidates %-5d" % (mode, len(T), tot))
        print("      P(acts on the TRUE worst member j*)      = %.4f" % (hit_true / tot))
        print("      P(acts on the DISPLAYED-worst row)       = %.4f" % (hit_row / tot))
        print("      chance, given support sizes              = %.4f" % (chance / tot))
