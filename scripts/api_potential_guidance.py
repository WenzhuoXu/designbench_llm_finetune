#!/usr/bin/env python
"""Does showing an LLM the POTENTIAL improve the actions it proposes?

The potential has been used two ways in this project: as a GRPO reward (provably
invisible -- the shaping sum telescopes to a group constant) and as a post-hoc
ranker over sampled actions. It has never been shown TO the model. The
observation the policy reads contains mass, FOS, status and constraint
violations, but no Phi.

This is the missing A/B, and it needs no GPU: one API model, the same problems,
the same 20-turn protocol, differing in ONE thing -- whether each turn's
observation carries the potential and how the last action moved it.

  control : the standard observation
  guided  : the same, plus
              Design potential Phi: <value>   (higher is better)
              Your last action changed Phi by <delta>  (+ improved / - worsened)
            and, once per episode, what Phi measures.

Phi here is free: every turn already runs the simulator, so Phi(s) costs nothing
extra. This is potential-as-FEEDBACK, not potential-as-lookahead -- no candidate
is simulated, so the guided arm has exactly the same simulator budget as the
control. That is the point: if it helps, it is the cheapest possible use of the
potential.

Credentials come from the environment only (AWS_REGION, AWS_BEARER_TOKEN_BEDROCK)
and are never written to disk.
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, threading, time
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

import urllib.request, urllib.error
# Every one of these is imported LAZILY on first use by urllib/ssl. On this
# Lustre mount a cold import stat()s for whole seconds while holding the CPython
# import lock, so a worker thread that triggers one freezes the entire pool.
# Resolving them here, single-threaded, is the difference between the run
# finishing and the run deadlocking.
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


def call(messages, system, model, region, token, max_tokens=1200, temperature=0.0, retries=4):
    """POST to Bedrock Converse with stdlib only.

    `requests` hangs indefinitely on import on this filesystem, so this uses
    urllib.request -- same request, no third-party dependency.
    """
    url = ENDPOINT.format(region=region, model=model)
    body = {"messages": messages, "inferenceConfig": {"maxTokens": max_tokens,
                                                      "temperature": temperature}}
    if system:
        body["system"] = [{"text": system}]
    data = json.dumps(body).encode()
    for a in range(retries):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Authorization": f"Bearer {token}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
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


PHI_EXPLAIN = (
    "\n\nYou will also be shown a scalar DESIGN POTENTIAL after each action. It "
    "summarises how good the current design is: higher is better. It combines how "
    "far the design is from satisfying every constraint with how heavy it is. Use "
    "the reported change to judge whether your last action helped, and adjust."
)


def run_episode(spec, arm, model, region, token, max_steps=20, alpha=5.0, tau=0.05):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    program = program_from_truss_spec(spec, initial_mass=state.get("mass"))
    phi_prev = compute_potential_v2(state, program, alpha=alpha, tau=tau)
    history, n_parse_ok, n_steps = [], 0, 0

    for turn in range(max_steps):
        # Built directly rather than via DP.build_messages: that helper needs a
        # ChatFormatter to render prior assistant turns for a LOCAL tokenizer,
        # and an API model takes plain message dicts. Same content, no formatter.
        system = DP.TRUSS_SYSTEM_PROMPT + (PHI_EXPLAIN if arm == "guided" else "")
        convo = [{"role": "user", "content": [{"text":
                  DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                  + DP.format_eval_result(initial)}]}]
        for h in history:
            convo.append({"role": "assistant", "content": [{"text": h.get("thinking") or ""}]})
            convo.append({"role": "user", "content": [{"text":
                          "[Simulation Result]\nSTRUCTURAL ANALYSIS RESULT:\n"
                          + DP.format_eval_result(h["fea_result"])}]})
        if arm == "guided" and convo:
            phi_now = compute_potential_v2(state, program, alpha=alpha, tau=tau)
            extra = f"\n\nDesign potential Phi: {phi_now:.4f}  (higher is better)"
            if history:
                d = phi_now - phi_prev
                extra += (f"\nYour last action changed Phi by {d:+.4f} "
                          f"({'improved' if d > 0 else 'worsened' if d < 0 else 'no change'}).")
            convo[-1]["content"][0]["text"] += extra
        text = call(convo, system, model, region, token)
        parsed = parse_grammar_action(text)
        if parsed is None:
            history.append({"action": "", "fea_result": state,
                            "thinking": "No valid <action> was produced last turn. Emit exactly "
                                        "one <action>...</action> block using the grammar above."})
            continue
        n_parse_ok += 1
        phi_prev = compute_potential_v2(state, program, alpha=alpha, tau=tau)
        try:
            truss = _apply_action(truss, parsed)
            state = _analyze_truss(truss, goals)
        except Exception:
            pass
        n_steps += 1
        history.append({"action": parsed, "fea_result": state, "thinking": text.strip()})
        if state.get("is_feasible") and program.is_valid(state):
            break

    return {
        "problem_id": spec.get("problem_id"), "arm": arm,
        "feasible": bool(state.get("is_feasible")) and bool(program.is_valid(state)),
        "n_steps": n_steps, "turns_used": len(history),
        "parse_rate": n_parse_ok / max(len(history), 1),
        "final_mass": state.get("mass"), "final_fos_b": state.get("fos_buckling"),
        "final_fos_y": state.get("fos_yielding"),
        "mass_ratio": (state.get("mass") / program.objective_ref) if program.objective_ref else None,
        "phi_final": compute_potential_v2(state, program, alpha=alpha, tau=tau),
        "phi_initial": compute_potential_v2(initial, program, alpha=alpha, tau=tau),
        "difficulty": (spec.get("_metadata") or {}).get("difficulty"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-dir", default=str(DESIGNBENCH / "data/problems_hard"))
    ap.add_argument("--split-file", default=str(PROJECT / "data/splits/truss_hard_v1.json"))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/run.jsonl"))
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
    print(f"{len(specs)} problems x 2 arms, model={a.model}", flush=True)

    # Warm-up on the MAIN thread: one full turn's worth of every code path a
    # worker will touch (FEA, potential, HTTP+TLS), so no worker can be the one
    # to trigger a cold import while holding the import lock.
    t0 = time.time()
    _t, _g = _load_truss_and_goals(specs[0])
    _st = _analyze_truss(_t, _g)
    compute_potential_v2(_st, program_from_truss_spec(specs[0], initial_mass=_st.get("mass")))
    _probe = call([{"role": "user", "content": [{"text": "Reply with OK."}]}],
                  None, a.model, region, token, max_tokens=8)
    print("warmup %.1fs, api=%s" % (time.time() - t0, "ok" if _probe.strip() else "FAILED"), flush=True)
    if not _probe.strip():
        raise SystemExit("Bedrock probe returned empty -- aborting before spending the run")

    jobs = [(s, arm) for s in specs for arm in ("control", "guided")]
    out = []
    from concurrent.futures import ThreadPoolExecutor
    def work(j):
        s_, arm_ = j
        t = time.time()
        try:
            r = run_episode(s_, arm_, a.model, region, token, max_steps=a.max_steps)
        except Exception as e:
            r = {"problem_id": s_.get("problem_id"), "arm": arm_, "error": str(e)[:200]}
        with _lock:
            print("    [%5.1fs] %s %-7s feasible=%s steps=%s parse=%s %s"
                  % (time.time() - t, s_.get("problem_id"), arm_, r.get("feasible"),
                     r.get("n_steps"), r.get("parse_rate"), r.get("error", "")), flush=True)
        return r
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(work, jobs), 1):
            out.append(r)
            if i % 10 == 0: print(f"  {i}/{len(jobs)}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")

    import math, statistics as st
    ok = [r for r in out if "error" not in r]
    by = {arm: [r for r in ok if r["arm"] == arm] for arm in ("control", "guided")}
    C = {r["problem_id"]: r for r in by["control"]}
    G = {r["problem_id"]: r for r in by["guided"]}
    ids = sorted(set(C) & set(G))
    n01 = sum(1 for i in ids if not C[i]["feasible"] and G[i]["feasible"])
    n10 = sum(1 for i in ids if C[i]["feasible"] and not G[i]["feasible"])
    N = n01 + n10
    p = (sum(math.comb(N, k) for k in range(min(n01, n10) + 1)) / 2 ** N * 2) if N else 1.0
    print(f"\npaired on {len(ids)} problems")
    for arm, D in (("control", C), ("guided", G)):
        f_ = sum(1 for i in ids if D[i]["feasible"])
        print(f"  {arm:8s} feasible {f_:3d}/{len(ids)} = {f_/max(len(ids),1):.3f}   "
              f"parse={st.mean([D[i]['parse_rate'] for i in ids]):.3f}  "
              f"steps={st.mean([D[i]['n_steps'] for i in ids]):.1f}  "
              f"dPhi={st.mean([D[i]['phi_final']-D[i]['phi_initial'] for i in ids]):+.3f}")
    print(f"  discordant guided-only={n01} control-only={n10}  McNemar p={min(p,1.0):.4f}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
