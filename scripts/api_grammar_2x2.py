#!/usr/bin/env python
"""Is the LLM design agent limited by its INTERFACE rather than its reasoning?

Measured context. Phi-selection is finished: it converts an available feasible
candidate into a solved problem 319/320 times. At MATCHED grammar the LLM (0.438)
is already above depth-1 procedural enumeration (0.404). What separates them is
that procedural search is armed with `fsd_macro` -- a closed-form per-member
resize -- worth 30.8pp on its own (0.712 -> 0.404 when removed).

Two things stop the policy reproducing that move, and they are independent:

  OBSERVATION  `format_eval_result` shows the MINIMUM buckling/yielding FOS over the
               whole structure and the id of the worst member. On hard_problem_0002
               that is two scalars for a ten-variable sizing problem -- and it hides
               that member M2 carries 4e17 yielding margin, i.e. free mass.
  ACTION       `SCALE_PARAM` moves one member; `SCALE_MULTI_PARAM` applies ONE shared
               factor. A fully-stressed resize needs a DIFFERENT factor per member,
               so the grammar needs one turn per member for a move the physics wants
               in one.

So: 2x2, observation x action, Phi-selection at fixed K in every cell.

  (aggregate, single)     the status quo -- reproduces the existing K=8 result
  (per_member, single)    information alone
  (aggregate, compound)   vocabulary alone
  (per_member, compound)  both -- the cell that can express a fully-stressed move

Every cell pays the SAME simulator budget: K candidates evaluated per turn, and a
compound candidate is ONE decision costing ONE simulator call, exactly as
`search_ladder.apply_action` treats a macro. The FSD formula is deliberately NOT
given -- the model has to derive the move from the table, or not.
"""
from __future__ import annotations
import argparse, copy, json, math, os, random, re, sys, threading, time, zlib
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

import urllib.request, urllib.error
# Lazily imported by urllib/ssl otherwise; a cold import inside a worker thread holds
# the CPython import lock and freezes the whole pool on this filesystem.
import ssl, socket, http.client, encodings.idna, hashlib, base64, email.utils  # noqa: F401
from llm_finetune.envs.truss_env import (
    _analyze_truss, _load_truss_and_goals, normalize_action, parse_grammar_action,
)
from llm_finetune.data.processors import designbench_prompt as DP
from llm_finetune.training.rl.posterior.potential import (
    compute_potential_v2, program_from_truss_spec,
)

ENDPOINT = "https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse"
MACRO_SEP = " ; "          # same separator search_ladder.py uses
_lock = threading.Lock()


def call(messages, system, model, region, token, max_tokens=2000, temperature=0.7, retries=4):
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
            with urllib.request.urlopen(req, timeout=150) as r:
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


def member_table(truss, shuffle_seed=None) -> str:
    """The per-member state the simulator already has and the policy never sees.

    fsd_macro() reads exactly these numbers off the member objects. Showing them
    costs nothing extra -- the FEA that produced them has already run.
    """
    members = list(getattr(truss, "members", []) or [])
    # PLACEBO: with a seed, permute the FOS pairs ACROSS members. The table keeps its
    # exact format, length and token count, and r/t stay truthful -- only the mapping
    # member -> its own utilisation is destroyed. If the per-member arm wins merely
    # because a formatted table prompts more careful reasoning, this scores like it;
    # if it wins on information, this collapses to the aggregate arm.
    fos = [(getattr(m, "fos_buckling", None), getattr(m, "fos_yielding", None))
           for m in members]
    if shuffle_seed is not None and len(fos) > 1:
        rng = random.Random(shuffle_seed)
        perm = list(range(len(fos)))
        rng.shuffle(perm)
        fos = [fos[j] for j in perm]
    rows = ["", "Per-member state (r = outer radius, t = wall thickness, m):",
            "  id    FOS_buckling    FOS_yielding         r         t"]
    for i, m in enumerate(members):
        sp = getattr(getattr(m, "shape", None), "_params", None) or {}
        def _f(v):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return "     n/a"
            if not math.isfinite(v):
                return "     inf"
            return f"{v:8.2f}" if v < 1e5 else f"{v:8.1e}"
        r_, t_ = sp.get("r"), sp.get("t")
        rows.append(f"  M{i:<4d}{_f(fos[i][0])}    "
                    f"{_f(fos[i][1])}  "
                    f"{(f'{float(r_):.5f}' if r_ is not None else '  n/a  ')}  "
                    f"{(f'{float(t_):.5f}' if t_ is not None else '  n/a  ')}")
    rows.append("A member with FOS far above the required 1.5 is carrying more "
                "material than it needs.")
    return "\n".join(rows)


# The compound arms used to be the ONLY ones given a worked <action> example. Sonnet
# parsed 100% either way so it never showed, but Haiku produced NO parseable action in
# ~2/3 of single-arm episodes (parts/cand 0.34 and 0.20 instead of 1.00; sims/ep 27 vs
# 116). That is a format-compliance advantage masquerading as a compound-action effect.
# Both arms now get an equivalent example, so the only difference is what an action may
# CONTAIN, not how well the format is demonstrated.
SINGLE_RULE = (
    "\n\nEach <action> contains exactly ONE grammar action, for example\n"
    "  <action>SCALE_PARAM(3, r, 1.15)</action>\n"
)

COMPOUND_RULE = (
    "\n\nOne <action> may contain SEVERAL grammar actions joined by ' ; '. They are "
    "applied together as a SINGLE move and cost a single analysis, for example\n"
    "  <action>SCALE_PARAM(0, r, 0.7) ; SCALE_PARAM(3, r, 1.4) ; SCALE_PARAM(5, r, 0.8)</action>\n"
    "This lets you give each member its OWN factor in one move rather than one member "
    "per turn. Use it when different members need to change by different amounts."
)


def parse_candidates(text, k, compound):
    """Distinct valid candidates. A candidate may be a ' ; '-joined compound move."""
    out, seen = [], set()
    for raw in re.findall(r"<action>(.*?)</action>", text, re.DOTALL | re.IGNORECASE):
        raw = raw.strip()
        if compound and MACRO_SEP.strip() in raw:
            parts = [p.strip() for p in raw.split(";") if p.strip()]
            good = [parse_grammar_action(p) for p in parts]
            good = [g for g in good if g]
            if not good:
                continue
            cand = MACRO_SEP.join(good)
        else:
            cand = parse_grammar_action(raw)
        if cand and cand not in seen:
            seen.add(cand); out.append(cand)
    if not out:
        one = parse_grammar_action(text)
        if one:
            out.append(one)
    return out[:k]


def apply_candidate(truss, cand):
    """Apply one candidate (possibly compound) to a COPY. Mirrors search_ladder.apply_action."""
    from validation.truss_executor import execute_grammar_action
    nt = copy.deepcopy(truss)
    ok = False
    for part in cand.split(MACRO_SEP):
        part = part.strip()
        if not part:
            continue
        try:
            ok = bool(execute_grammar_action(nt, normalize_action(nt, part))) or ok
        except Exception:
            pass
    return nt if ok else None


def run_episode(spec, obs_mode, act_mode, model, region, token, k=8, max_steps=20,
                alpha=5.0, tau=0.05):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    program = program_from_truss_spec(spec, initial_mass=state.get("mass"))
    compound = act_mode == "compound"
    # "shuffled" = the placebo table; both it and per_member show a table.
    show_table = obs_mode in ("per_member", "shuffled")
    # zlib.crc32, not hash(): PYTHONHASHSEED is randomised per process, so hash()
    # would give a different permutation on every run and the placebo would be
    # unreproducible.
    seed = (zlib.crc32(spec.get("problem_id", "").encode()) if obs_mode == "shuffled"
            else None)
    tbl = lambda t: member_table(t, seed) if show_table else ""
    history = []
    n_prop, n_turns, n_sim, oracle = 0, 0, 0, False
    n_parts = []

    system = DP.TRUSS_SYSTEM_PROMPT + (
        f"\n\nEach turn, propose {k} DIFFERENT candidate moves rather than one. Think "
        f"briefly, then emit exactly {k} lines of the form <action>...</action>. They "
        f"must be genuinely different candidates -- only one will be executed."
    ) + (COMPOUND_RULE if compound else SINGLE_RULE)

    for _ in range(max_steps):
        obs = DP.format_eval_result(state) + tbl(truss)
        convo = [{"role": "user", "content": [{"text":
                  DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                  + DP.format_eval_result(initial)
                  + (tbl(truss) if not history else "")}]}]
        for h in history:
            convo.append({"role": "assistant", "content": [{"text": h["thinking"]}]})
            convo.append({"role": "user", "content": [{"text":
                          "[Simulation Result]\nSTRUCTURAL ANALYSIS RESULT:\n" + h["obs"]}]})
        text = call(convo, system, model, region, token)
        cands = parse_candidates(text, k, compound)
        if not cands:
            history.append({"thinking": f"No <action> parsed. Emit exactly {k} "
                                        "<action>...</action> lines.", "obs": obs})
            continue
        n_prop += len(cands); n_turns += 1
        n_parts += [len(c.split(MACRO_SEP)) for c in cands]

        scored = []
        for c in cands:
            nt = apply_candidate(truss, c)
            if nt is None:
                continue
            st = _analyze_truss(nt, goals)          # one simulator call per candidate
            n_sim += 1
            scored.append((c, nt, st, compute_potential_v2(st, program, alpha=alpha, tau=tau)))
        if not scored:
            history.append({"thinking": text.strip(), "obs": obs})
            continue
        if any(s.get("is_feasible") and program.is_valid(s) for _, _, s, _ in scored):
            oracle = True
        _c, truss, state, _p = max(scored, key=lambda x: x[3])
        history.append({"thinking": text.strip(),
                        "obs": DP.format_eval_result(state)
                               + tbl(truss)})
        if state.get("is_feasible") and program.is_valid(state):
            break

    return {
        "problem_id": spec.get("problem_id"), "obs": obs_mode, "act": act_mode, "k": k,
        "feasible": bool(state.get("is_feasible")) and bool(program.is_valid(state)),
        "oracle_any_candidate_feasible": oracle,
        "turns": n_turns, "n_simulator_calls": n_sim,
        "mean_candidates": n_prop / max(n_turns, 1),
        "mean_parts_per_candidate": (sum(n_parts) / len(n_parts)) if n_parts else 0.0,
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
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/grammar_2x2.jsonl"))
    ap.add_argument("--cells", default="aggregate/single,per_member/single,"
                                       "aggregate/compound,per_member/compound",
                    help="Comma-separated cells. 'shuffled' is the PLACEBO observation: "
                         "the same table with FOS permuted across members.")
    ap.add_argument("--resume-from", default="", help="jsonl of already-finished episodes to skip")
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
    ALL_CELLS = {
        "aggregate/single": ("aggregate", "single"),
        "per_member/single": ("per_member", "single"),
        "aggregate/compound": ("aggregate", "compound"),
        "per_member/compound": ("per_member", "compound"),
        "shuffled/compound": ("shuffled", "compound"),
        "shuffled/single": ("shuffled", "single"),
    }
    CELLS = [ALL_CELLS[c] for c in a.cells.split(",") if c in ALL_CELLS]
    print(f"{len(specs)} problems x {len(CELLS)} cells, K={a.k}, model={a.model}", flush=True)

    # Warm every lazy path on the MAIN thread -- trussme imports matplotlib.pyplot,
    # whose cold import holds the CPython import lock and freezes the pool.
    t0 = time.time()
    _t, _g = _load_truss_and_goals(specs[0])
    _st = _analyze_truss(_t, _g)
    compute_potential_v2(_st, program_from_truss_spec(specs[0], initial_mass=_st.get("mass")))
    member_table(_t)
    apply_candidate(_t, "SCALE_PARAM(0, r, 1.05)")
    _probe = call([{"role": "user", "content": [{"text": "Reply with OK."}]}],
                  None, a.model, region, token, max_tokens=8)
    print("warmup %.1fs, api=%s" % (time.time() - t0, "ok" if _probe.strip() else "FAILED"), flush=True)
    if not _probe.strip():
        raise SystemExit("Bedrock probe empty -- aborting before spending the run")

    # A session restart killed the first attempt at 271/808. Episodes stream to disk,
    # so re-running the finished ones is pure waste -- skip them and carry them into
    # the final summary.
    done, out = set(), []
    if a.resume_from and Path(a.resume_from).exists():
        for line in open(a.resume_from):
            try: r = json.loads(line)
            except Exception: continue
            if "feasible" in r:
                done.add((r.get("problem_id"), r.get("obs"), r.get("act")))
                out.append(r)
        print(f"resuming: {len(done)} episodes already complete", flush=True)
    jobs = [(s, o, c) for s in specs for (o, c) in CELLS
            if (s.get("problem_id"), o, c) not in done]
    print(f"{len(jobs)} episodes to run", flush=True)
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    fh = open(outp, "w")
    for r in out:
        fh.write(json.dumps(r) + "\n")
    fh.flush()
    from concurrent.futures import ThreadPoolExecutor
    def work(j):
        s_, o_, c_ = j
        t = time.time()
        try:
            r = run_episode(s_, o_, c_, a.model, region, token, k=a.k, max_steps=a.max_steps)
        except Exception as e:
            r = {"problem_id": s_.get("problem_id"), "obs": o_, "act": c_, "error": str(e)[:200]}
        with _lock:
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print("    [%5.1fs] %-20s %-10s/%-8s feasible=%s parts=%.1f turns=%s %s"
                  % (time.time() - t, s_.get("problem_id"), o_, c_, r.get("feasible"),
                     r.get("mean_parts_per_candidate") or 0.0, r.get("turns"),
                     r.get("error", "")), flush=True)
        return r
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(work, jobs), 1):
            out.append(r)
            if i % 40 == 0: print(f"  {i}/{len(jobs)}", flush=True)
    fh.close()

    by = {}
    n_err = 0
    for r in out:
        # An episode that raised carries `error` and no `feasible`; including it made
        # the summary die with KeyError AFTER all 320 episodes had been written.
        if "error" in r or "feasible" not in r:
            n_err += 1
            continue
        by.setdefault(r.get("problem_id"), {})[(r.get("obs"), r.get("act"))] = r
    if n_err:
        print(f"dropped {n_err} errored episode(s)", flush=True)
    full = [v for v in by.values() if len(v) == len(CELLS)]
    print(f"\ncomplete on {len(full)}/{len(specs)} problems (all four cells)")
    print(f"{'observation':>12} {'action':>9} {'feasible':>9} {'oracle':>8} {'parts':>6} {'sims/ep':>8}")
    for cell in CELLS:
        fe = sum(1 for v in full if v[cell]["feasible"]) / max(len(full), 1)
        orc = sum(1 for v in full if v[cell].get("oracle_any_candidate_feasible")) / max(len(full), 1)
        pr = sum(v[cell].get("mean_parts_per_candidate") or 0 for v in full) / max(len(full), 1)
        sm = sum(v[cell].get("n_simulator_calls") or 0 for v in full) / max(len(full), 1)
        print(f"{cell[0]:>12} {cell[1]:>9} {fe:9.3f} {orc:8.3f} {pr:6.2f} {sm:8.0f}")
    base = ("aggregate", "single")
    for cell in CELLS[1:]:
        n01 = sum(1 for v in full if v[cell]["feasible"] and not v[base]["feasible"])
        n10 = sum(1 for v in full if v[base]["feasible"] and not v[cell]["feasible"])
        n = n01 + n10
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0
        print(f"  {cell[0]}/{cell[1]} vs status quo: discordant {n01} vs {n10}, exact McNemar p={p:.4g}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
