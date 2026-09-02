#!/usr/bin/env python
"""Does ranking an LLM's own proposals by the potential beat picking one at random?

This is the SELECTION route -- the only one of the three uses of Phi that has
survived measurement so far (truss weak GRPO rollouts +10.9pp p<1e-6, battery
+20.8pp p=2.75e-4). Both of those used a weak or procedural proposer. The open
question is whether it still helps when the proposer is STRONG, because on the
old saturated benchmark the effect with a strong policy was exactly +0.000
(discordant 0/0) -- there was no headroom left to capture.

`problems_hard` has headroom: greedy 0.116 against depth-1 search 0.734.

Protocol, paired by problem, 20 turns:
  each turn the model proposes K distinct candidate actions in ONE call;
  every candidate is simulated (FEA is ~1ms once warm, so this is free);
    arm `phi`     applies argmax Phi over the K
    arm `random`  applies a uniform draw from the SAME K
  `oracle` records whether any of the K reached feasibility -- the ceiling.

Both arms therefore pay the same API cost and the same simulator cost; they
differ only in which of the K they keep. Credentials come from the environment.
"""
from __future__ import annotations
import argparse, copy, json, os, random, re, sys, threading, time
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

import urllib.request, urllib.error
# Lazily imported by urllib/ssl otherwise; a cold import inside a worker thread
# holds the CPython import lock and freezes the whole pool on this filesystem.
import ssl, socket, http.client, encodings.idna, hashlib, base64, email.utils  # noqa: F401
from llm_finetune.envs.truss_env import (
    _analyze_truss, _apply_action, _load_truss_and_goals, parse_grammar_action,
)
from llm_finetune.data.processors import designbench_prompt as DP
from llm_finetune.training.rl.posterior.potential import (
    compute_potential_v2, program_from_truss_spec,
)

ENDPOINT = "https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse"
_lock = threading.Lock()

PROPOSE = (
    "\n\nEach turn, propose {k} DIFFERENT candidate actions rather than one. "
    "Think briefly, then emit exactly {k} lines, each of the form\n"
    "  <action>GRAMMAR_ACTION(...)</action>\n"
    "They must be genuinely different candidates -- vary which members you touch, "
    "which parameter, and the direction and size of the change -- because only one "
    "of them will be executed. Do not number them or add commentary between them."
)


def call(messages, system, model, region, token, max_tokens=1600, temperature=0.7, retries=4):
    url = ENDPOINT.format(region=region, model=model)
    body = {"messages": messages,
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature}}
    if system:
        body["system"] = [{"text": system}]
    data = json.dumps(body).encode()
    for a in range(retries):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Authorization": f"Bearer {token}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.loads(r.read().decode())
            return "".join(c.get("text", "")
                           for c in payload["output"]["message"]["content"])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503):
                time.sleep(2 ** a + random.random()); continue
            return ""
        except Exception:
            time.sleep(2 ** a + random.random())
    return ""


_ACTION_TAG = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)


def parse_candidates(text, k):
    """Every distinct valid grammar action in the reply, capped at k."""
    out, seen = [], set()
    for raw in _ACTION_TAG.findall(text):
        a = parse_grammar_action(raw) or parse_grammar_action(f"<action>{raw}</action>")
        if a and a not in seen:
            seen.add(a); out.append(a)
    if not out:                      # fall back to scanning the bare text
        a = parse_grammar_action(text)
        if a: out.append(a)
    return out[:k]


def run_episode(spec, arm, model, region, token, k=8, max_steps=20,
                alpha=5.0, tau=0.05, seed=0):
    rng = random.Random(seed)
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    program = program_from_truss_spec(spec, initial_mass=state.get("mass"))
    history = []
    n_props, n_turns, oracle_hit, n_sim = 0, 0, False, 0

    system = DP.TRUSS_SYSTEM_PROMPT + PROPOSE.format(k=k)
    for _ in range(max_steps):
        convo = [{"role": "user", "content": [{"text":
                  DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                  + DP.format_eval_result(initial)}]}]
        for h in history:
            convo.append({"role": "assistant", "content": [{"text": h["thinking"]}]})
            convo.append({"role": "user", "content": [{"text":
                          "[Simulation Result]\nSTRUCTURAL ANALYSIS RESULT:\n"
                          + DP.format_eval_result(h["fea_result"])}]})
        text = call(convo, system, model, region, token)
        cands = parse_candidates(text, k)
        if not cands:
            history.append({"thinking": "No <action> parsed. Emit exactly "
                                        f"{k} <action>...</action> lines.",
                            "fea_result": state})
            continue
        n_props += len(cands); n_turns += 1

        # Simulate every candidate. ~1ms each once trussme is warm, so the two
        # arms pay an identical simulator cost regardless of which they keep.
        evaluated = []
        for a in cands:
            # _apply_action mutates the truss IN PLACE and returns it, so every
            # candidate must be tried on its own copy -- otherwise the K actions
            # compose onto one truss instead of being K alternatives.
            # scripts/eval_llm_search.py:112 does the same.
            try:
                t2 = _apply_action(copy.deepcopy(truss), a)
                s2 = _analyze_truss(t2, goals)
            except Exception:
                continue
            n_sim += 1
            evaluated.append((a, t2, s2, compute_potential_v2(s2, program,
                                                              alpha=alpha, tau=tau)))
        if not evaluated:
            history.append({"thinking": text.strip(), "fea_result": state})
            continue
        if any(s2.get("is_feasible") and program.is_valid(s2) for _, _, s2, _ in evaluated):
            oracle_hit = True

        if arm == "phi":
            pick = max(range(len(evaluated)), key=lambda i: evaluated[i][3])
        else:
            pick = rng.randrange(len(evaluated))
        _a, truss, state, _phi = evaluated[pick]
        history.append({"thinking": text.strip(), "fea_result": state})
        if state.get("is_feasible") and program.is_valid(state):
            break

    return {
        "problem_id": spec.get("problem_id"), "arm": arm, "k": k,
        "feasible": bool(state.get("is_feasible")) and bool(program.is_valid(state)),
        "oracle_any_candidate_feasible": oracle_hit,
        "turns": n_turns, "mean_candidates": n_props / max(n_turns, 1),
        "n_simulator_calls": n_sim,
        "final_mass": state.get("mass"),
        "mass_ratio": (state.get("mass") / program.objective_ref) if program.objective_ref else None,
        "final_fos_b": state.get("fos_buckling"), "final_fos_y": state.get("fos_yielding"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-dir", default=str(DESIGNBENCH / "data/problems_hard"))
    ap.add_argument("--split-file", default=str(PROJECT / "data/splits/truss_hard_v1.json"))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/phi_vs_random.jsonl"))
    a = ap.parse_args()

    region = os.environ.get("AWS_REGION", "us-west-2")
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not token:
        raise SystemExit("AWS_BEARER_TOKEN_BEDROCK not set in environment")

    wanted = set(json.load(open(a.split_file))[a.split]) if a.split_file else None
    specs = []
    for f in sorted(Path(a.problems_dir).glob("*.json")):
        try: d = json.load(open(f))
        except Exception: continue
        if not isinstance(d, dict) or "topology" not in d: continue
        d.setdefault("problem_id", f.stem)
        if wanted is None or d["problem_id"] in wanted:
            specs.append(d)
    random.Random(a.seed).shuffle(specs)
    specs = specs[: a.n]
    print(f"{len(specs)} problems x 2 arms, K={a.k}, model={a.model}", flush=True)

    # Warm every lazy path on the MAIN thread. trussme imports matplotlib.pyplot,
    # whose cold import takes minutes here and holds the import lock.
    t0 = time.time()
    _t, _g = _load_truss_and_goals(specs[0])
    _st = _analyze_truss(_t, _g)
    compute_potential_v2(_st, program_from_truss_spec(specs[0], initial_mass=_st.get("mass")))
    _probe = call([{"role": "user", "content": [{"text": "Reply with OK."}]}],
                  None, a.model, region, token, max_tokens=8)
    print("warmup %.1fs, api=%s" % (time.time() - t0, "ok" if _probe.strip() else "FAILED"),
          flush=True)
    if not _probe.strip():
        raise SystemExit("Bedrock probe empty -- aborting before spending the run")

    jobs = [(s, arm) for s in specs for arm in ("phi", "random")]
    out = []
    from concurrent.futures import ThreadPoolExecutor
    def work(j):
        s_, arm_ = j
        t = time.time()
        try:
            r = run_episode(s_, arm_, a.model, region, token, k=a.k,
                            max_steps=a.max_steps, seed=a.seed)
        except Exception as e:
            r = {"problem_id": s_.get("problem_id"), "arm": arm_, "error": str(e)[:200]}
        with _lock:
            print("    [%5.1fs] %s %-6s feasible=%s oracle=%s turns=%s cand=%.1f %s"
                  % (time.time() - t, s_.get("problem_id"), arm_, r.get("feasible"),
                     r.get("oracle_any_candidate_feasible"), r.get("turns"),
                     r.get("mean_candidates") or 0.0, r.get("error", "")), flush=True)
        return r
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(work, jobs), 1):
            out.append(r)
            if i % 20 == 0: print(f"  {i}/{len(jobs)}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")

    import math, statistics as st
    by = {}
    for r in out: by.setdefault(r.get("problem_id"), {})[r.get("arm")] = r
    pairs = [(v["phi"], v["random"]) for v in by.values() if "phi" in v and "random" in v]
    print(f"\npaired on {len(pairs)} problems")
    for lab, i in (("phi", 0), ("random", 1)):
        f_ = sum(1 for p in pairs if p[i].get("feasible"))
        print(f"  {lab:7s} feasible {f_:3d}/{len(pairs)} = {f_/max(len(pairs),1):.3f}   "
              f"sim/ep {st.mean([p[i].get('n_simulator_calls') or 0 for p in pairs]):.0f}")
    orc = sum(1 for p in pairs if p[0].get("oracle_any_candidate_feasible")
                                or p[1].get("oracle_any_candidate_feasible"))
    print(f"  oracle (any candidate ever feasible) {orc}/{len(pairs)} = {orc/max(len(pairs),1):.3f}")
    n01 = sum(1 for p in pairs if p[0].get("feasible") and not p[1].get("feasible"))
    n10 = sum(1 for p in pairs if p[1].get("feasible") and not p[0].get("feasible"))
    n = n01 + n10
    pv = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    print(f"  discordant phi-only={n01} random-only={n10}  exact McNemar p={pv:.4g}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
