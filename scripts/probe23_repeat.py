"""
EXPERIMENT 23 -- what is the run-to-run band on an API arm, and does Bedrock accept a seed?

Every multi-turn arm this session was ONE draw per problem at temperature 0.7. Pairing by
problem controls the problem; nothing controlled the model's own sampling noise inside a
cell, and that band was never measured. Until it is, a reported "-1.2pp, null" is
uninterpretable, because the instrument's resolution is unknown.

Part 1: does the Converse API honour a seed? Try additionalModelRequestFields={"seed": ...}
and compare repeated generations at temperature 0.7, with and without.

Part 2: run the SAME arm R independent times over the same problems and report
  - the spread of the arm mean across repeats (this is the run-to-run band)
  - per-problem agreement across repeats (how often the same problem flips)
  - the implied paired MDE at n problems x 1 draw, versus x R draws
"""
import os, sys, json, math, argparse, time, statistics, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
P = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(P), str(DB), str(P / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import urllib.request, urllib.error
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals
from llm_finetune.data.processors import designbench_prompt as DP
import api_grammar_2x2 as H
from probe15_matrix import build_table, true_fos

ENDPOINT = "https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse"


def call_seeded(messages, system, model, region, token, seed=None, temperature=0.7,
                max_tokens=300):
    url = ENDPOINT.format(region=region, model=model)
    body = {"messages": messages,
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature}}
    if system:
        body["system"] = [{"text": system}]
    if seed is not None:
        body["additionalModelRequestFields"] = {"seed": seed}
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.loads(r.read().decode())
        return "".join(c.get("text", "") for c in payload["output"]["message"]["content"]), None
    except urllib.error.HTTPError as e:
        return None, "HTTP %s: %s" % (e.code, e.read().decode()[:200])
    except Exception as e:
        return None, str(e)[:200]


def episode(spec, model, region, token, k, max_steps):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    B = float(goals.get("maximum_mass", float("inf")))
    system = DP.TRUSS_SYSTEM_PROMPT + (
        f"\n\nEach turn, propose {k} DIFFERENT candidate moves rather than one. Think "
        f"briefly, then emit exactly {k} lines of the form <action>...</action>. They "
        f"must be genuinely different candidates -- only one will be executed."
    ) + H.COMPOUND_RULE
    history = []
    for _ in range(max_steps):
        tbl = build_table(truss, true_fos(truss))
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
        scored = []
        for c in cands:
            nt = H.apply_candidate(truss, c)
            if nt is None:
                continue
            try:
                st = _analyze_truss(nt, goals)
            except Exception:
                continue
            scored.append((nt, st))
        if not scored:
            history.append({"t": text.strip(), "o": obs})
            continue
        def key(t):
            st = t[1]
            fb = float(st.get("fos_buckling", 0) or 0); fy = float(st.get("fos_yielding", 0) or 0)
            ms = float(st.get("mass", 9e9) or 9e9)
            return min(fb / gb, fy / gy) - (0.5 if ms > B else 0.0)
        truss, state = max(scored, key=key)
        history.append({"t": text.strip(),
                        "o": DP.format_eval_result(state) + build_table(truss, true_fos(truss))})
        if state.get("is_feasible"):
            return 1
    return 0


def work(t):
    spec, rep, model, region, token, k, ms = t
    try:
        return {"problem_id": spec["problem_id"], "rep": rep,
                "feasible": episode(spec, model, region, token, k, ms)}
    except Exception as e:
        return {"problem_id": spec.get("problem_id"), "rep": rep, "err": str(e)[:120]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(P / "results/api_guidance/repeat.jsonl"))
    a = ap.parse_args()
    tok = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()

    print("=== PART 1: does Converse accept a seed? ===")
    msg = [{"role": "user", "content": [{"text": "Name three colours, comma separated. Vary your answer."}]}]
    for label, sd in (("no seed", None), ("seed=42", 42)):
        outs, err = [], None
        for _ in range(3):
            o, e = call_seeded(msg, "", a.model, a.region, tok, seed=sd)
            if e:
                err = e; break
            outs.append((o or "").strip()[:60])
        if err:
            print("  %-9s REJECTED -> %s" % (label, err[:150]))
        else:
            print("  %-9s accepted | identical across 3 draws: %s" % (label, len(set(outs)) == 1))
            for o in outs:
                print("      %r" % o)

    print("\n=== PART 2: run-to-run band, same arm repeated ===")
    specs = [json.load(open(f)) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:a.n]]
    tasks = [(s, r, a.model, a.region, tok, a.k, a.max_steps)
             for r in range(a.reps) for s in specs]
    print("episodes %d (%d problems x %d repeats)" % (len(tasks), a.n, a.reps), flush=True)
    rows = []
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w") as fh:
        for i, r in enumerate(ex.map(work, tasks)):
            rows.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 50 == 0:
                print("  %d/%d %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs" % (time.time() - t0))

    per_rep = collections.defaultdict(dict)
    for r in rows:
        if "feasible" in r:
            per_rep[r["rep"]][r["problem_id"]] = r["feasible"]
    common = sorted(set.intersection(*[set(d) for d in per_rep.values()])) if per_rep else []
    N = len(common)
    means = [statistics.mean(per_rep[rp][p] for p in common) for rp in sorted(per_rep)]
    print("\nproblems common to all repeats: %d" % N)
    print("arm mean per repeat: %s" % ", ".join("%.3f" % m for m in means))
    print("spread: min %.3f max %.3f range %.1fpp | sd across repeats %.4f (%.1fpp)"
          % (min(means), max(means), 100 * (max(means) - min(means)),
             statistics.stdev(means) if len(means) > 1 else 0.0,
             100 * (statistics.stdev(means) if len(means) > 1 else 0.0)))

    flips = [sum(per_rep[rp][p] for rp in per_rep) for p in common]
    R = len(per_rep)
    always = sum(1 for f in flips if f == R); never = sum(1 for f in flips if f == 0)
    print("problems solved in ALL %d repeats: %d (%.1f%%) | NEVER: %d (%.1f%%) | "
          "unstable: %d (%.1f%%)"
          % (R, always, 100 * always / N, never, 100 * never / N,
             N - always - never, 100 * (N - always - never) / N))

    # per-problem Bernoulli variance -> paired MDE
    pvar = statistics.mean([f / R * (1 - f / R) for f in flips])
    sd_paired_1 = math.sqrt(2 * pvar)
    print("\nmean within-problem variance across repeats: %.4f" % pvar)
    print("implied paired sd at 1 draw/cell: %.4f" % sd_paired_1)
    for nn in (40, 80, 200):
        print("  paired MDE from SAMPLING NOISE ALONE at n=%-3d, 1 draw/cell: %.1fpp | "
              "at %d draws/cell: %.1fpp"
              % (nn, 100 * 2.8 * sd_paired_1 / math.sqrt(nn), a.reps,
                 100 * 2.8 * sd_paired_1 / math.sqrt(nn * a.reps)))
