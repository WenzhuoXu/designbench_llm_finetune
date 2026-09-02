#!/usr/bin/env python
"""Does the truss interface result generalise? The same 2x2, on batteries.

Truss finding (Turn 16, replicated): feasibility 0.380 -> 0.608 from changing only
what the agent can SEE (per-member state vs two aggregate scalars) and SAY (a
compound heterogeneous move vs one parameter per turn). The effect was an
INTERACTION: compound actions HURT without per-component state (-5.8pp) and HELP
with it (+12.8pp).

Battery is the right generality test rather than a replication, because the domain
differs in exactly the way that matters:

  * There is NO `fsd_macro` equivalent. Nothing here solves the sizing subproblem
    in closed form, so the LLM is not competing against injected analytic physics.
  * There are 9 named scalar design parameters, not N structural components hidden
    behind a min(). The observation bottleneck therefore has to be constructed
    honestly: `aggregate` shows the objective and the single BINDING constraint
    (the analogue of truss's min-FOS + worst member); `full` shows every constraint
    with its limit and slack, plus every design parameter with its bounds.
  * The action limitation is IDENTICAL: `SCALE_PARAM` moves one parameter and
    `SCALE_MULTI_PARAM` applies ONE SHARED factor to a group, so a heterogeneous
    per-parameter move is inexpressible in a single turn.

Prediction under the truss interaction: in battery the `full` observation is closer
to adequate, so compound actions should help there rather than hurt.

PyBaMM costs ~3.4s per evaluation (truss FEA is ~1ms), so this is CPU-bound and
belongs on RM-shared, not a login node. Each cell pays the same simulator budget:
K candidates per turn, a compound candidate being ONE decision and ONE evaluation.
"""
from __future__ import annotations
import argparse, json, math, os, random, re, sys, time
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import urllib.request, urllib.error
import ssl, socket, http.client, encodings.idna, hashlib, base64, email.utils  # noqa: F401

ENDPOINT = "https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse"
MACRO_SEP = " ; "

_SIM = None          # one simulator per worker process (PyBaMM is not thread-safe)


def sim():
    global _SIM
    if _SIM is None:
        from battery_env import BatterySimulator
        _SIM = BatterySimulator(fidelity="spme_lean", current_a=8.0)
    return _SIM


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


SYSTEM = """You are a battery cell design engineer. You iteratively modify a
lithium-ion cell design until it satisfies every constraint.

Available actions:
  SCALE_PARAM(param, factor)                  multiply one parameter by factor
  SCALE_MULTI_PARAM([p1,p2], factor)          multiply several by the SAME factor

Parameters (short names):
  neg_thickness, pos_thickness, sep_thickness   electrode / separator thickness
  neg_porosity, pos_porosity                    electrode porosity
  neg_radius, pos_radius                        particle radius
  neg_am_fraction, pos_am_fraction              active material volume fraction

Think briefly, then emit your candidate moves, each as <action>...</action>."""


def obs_text(program, state, params, bounds, mode):
    """`aggregate` mirrors the truss status quo: objective + the binding constraint
    only. `full` exposes every constraint's slack and every parameter's headroom --
    the information a coordinated move needs in order to be aimed."""
    obj = state.get(program.objective_key)
    lines = ["Objective %s: %.4f (lower is better)" % (program.objective_key, float(obj))
             if obj is not None else "Objective: n/a"]
    viol = program.violations(state, tau=0.0)
    feasible = bool(program.is_feasible(state))
    lines.append("Status: %s" % ("FEASIBLE" if feasible else "INFEASIBLE"))
    if mode == "aggregate":
        worst = max(viol.items(), key=lambda kv: kv[1], default=(None, 0.0))
        if worst[0] is not None and worst[1] > 0:
            lines.append("Worst violated constraint: %s (relative violation %.4f)"
                         % (worst[0], worst[1]))
    else:
        lines.append("\nAll constraints:")
        for c in program.constraints:
            v = float(state.get(c.key, float("nan")))
            lim = float(c.limit)
            slack = ((lim - v) / max(abs(lim), 1e-12) if c.sense == "upper"
                     else (v - lim) / max(abs(lim), 1e-12))
            lines.append("  %-26s %12.4f  %s %-10.4f  slack %+.4f%s"
                         % (c.key, v, "<=" if c.sense == "upper" else ">=", lim,
                            slack, "  VIOLATED" if slack < 0 else ""))
        lines.append("\nCurrent design parameters (value, and its allowed range):")
        from battery_env import short
        for k, v in params.items():
            lo, hi = bounds.get(k, (float("-inf"), float("inf")))
            room_up = (hi / v) if v else float("inf")
            room_dn = (lo / v) if v else 0.0
            lines.append("  %-16s %12.6g   range [%.6g, %.6g]  (can scale x%.2f up, x%.2f down)"
                         % (short(k), v, lo, hi, room_up, room_dn))
        lines.append("A constraint with large positive slack is over-satisfied; the "
                     "parameters driving it are candidates for trading away.")
    return "\n".join(lines)


COMPOUND_RULE = (
    "\n\nOne <action> may contain SEVERAL actions joined by ' ; '. They apply "
    "together as a SINGLE move costing a single simulation, for example\n"
    "  <action>SCALE_PARAM(neg_thickness, 0.9) ; SCALE_PARAM(pos_porosity, 1.15) ; "
    "SCALE_PARAM(neg_radius, 0.8)</action>\n"
    "This lets you give each parameter its OWN factor in one move rather than one "
    "parameter per turn. Use it when different parameters need different changes."
)

_TAG = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)


def parse_candidates(text, k, compound):
    from battery_env import execute_action  # validity is decided by execution
    out, seen = [], set()
    for raw in _TAG.findall(text):
        raw = " ; ".join(p.strip() for p in raw.strip().split(";") if p.strip()) \
              if (compound and ";" in raw) else raw.strip()
        if not raw or raw in seen:
            continue
        seen.add(raw); out.append(raw)
    return out[:k]


def apply_candidate(params, cand):
    from battery_env import execute_action
    cur = dict(params)
    ok = False
    for part in cand.split(MACRO_SEP):
        part = part.strip()
        if not part:
            continue
        nxt = execute_action(cur, part)
        if nxt is not None:
            cur = nxt; ok = True
    return cur if ok else None


def in_bounds(params, bounds):
    for k, v in params.items():
        lo, hi = bounds.get(k, (float("-inf"), float("inf")))
        if v < lo or v > hi:
            return False
    return True


def run_episode(job):
    path, obs_mode, act_mode, model, region, token, k, max_steps, alpha = job
    from battery_ladder import load_problem, param_bounds, initial_params, make_program
    from llm_finetune.training.rl.posterior.potential import compute_potential_v2
    t0 = time.time()
    spec = load_problem(Path(path))
    pid = spec.get("problem_id", Path(path).stem)
    try:
        s = sim()
        bounds = param_bounds(spec)
        params = initial_params(spec, s)
        program = make_program(spec, use_scales=True)
        state = s.evaluate(params)
        obj0 = state.get(program.objective_key)
        compound = act_mode == "compound"
        system = SYSTEM + (COMPOUND_RULE if compound else "")
        history, n_sim, n_turns, n_parts, oracle = [], 0, 0, [], False

        for _ in range(max_steps):
            obs = obs_text(program, state, params, bounds, obs_mode)
            convo = [{"role": "user", "content": [{"text":
                      "PROBLEM: %s\nMinimise %s subject to the constraints below.\n\n"
                      "INITIAL STATE:\n%s" % (pid, program.objective_key, obs)}]}] \
                if not history else \
                [{"role": "user", "content": [{"text":
                  "PROBLEM: %s\nMinimise %s subject to the constraints below.\n\n"
                  "INITIAL STATE:\n%s" % (pid, program.objective_key, history[0]["obs0"])}]}]
            for h in history:
                convo.append({"role": "assistant", "content": [{"text": h["thinking"]}]})
                convo.append({"role": "user", "content": [{"text":
                              "[Simulation Result]\n" + h["obs"]}]})
            convo[-1]["content"][0]["text"] += (
                "\n\nPropose %d DIFFERENT candidate moves, each as <action>...</action>." % k)
            text = call(convo, system, model, region, token)
            cands = parse_candidates(text, k, compound)
            if not cands:
                history.append({"thinking": "No <action> parsed.", "obs": obs,
                                "obs0": history[0]["obs0"] if history else obs})
                continue
            n_turns += 1
            n_parts += [len(c.split(MACRO_SEP)) for c in cands]
            scored = []
            for c in cands:
                np_ = apply_candidate(params, c)
                if np_ is None or not in_bounds(np_, bounds):
                    continue
                st = s.evaluate(np_)          # one simulation per candidate
                n_sim += 1
                scored.append((np_, st, compute_potential_v2(st, program, alpha=alpha)))
            if not scored:
                history.append({"thinking": text.strip(), "obs": obs,
                                "obs0": history[0]["obs0"] if history else obs})
                continue
            if any(program.is_feasible(st) for _, st, _ in scored):
                oracle = True
            params, state, _ = max(scored, key=lambda x: x[2])
            new_obs = obs_text(program, state, params, bounds, obs_mode)
            history.append({"thinking": text.strip(), "obs": new_obs,
                            "obs0": history[0]["obs0"] if history else obs})
            if program.is_feasible(state):
                break

        return {"problem_id": pid, "obs": obs_mode, "act": act_mode, "k": k,
                "feasible": bool(program.is_feasible(state)),
                "oracle_any_candidate_feasible": oracle,
                "turns": n_turns, "n_simulator_calls": n_sim,
                "mean_parts_per_candidate": (sum(n_parts) / len(n_parts)) if n_parts else 0.0,
                "objective": float(state.get(program.objective_key, float("nan"))),
                "objective_initial": float(obj0) if obj0 is not None else float("nan"),
                "wall_s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"problem_id": pid, "obs": obs_mode, "act": act_mode,
                "error": "%s: %s" % (type(e).__name__, str(e)[:180]),
                "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-dir", default=str(PROJECT / "data/battery_problems"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/battery_2x2.jsonl"))
    ap.add_argument("--cells", default="",
                    help="comma-separated obs:act cells to run, e.g. 'aggregate:single'. "
                         "Empty runs the full 2x2. Used to price difficulty cheaply "
                         "before committing to all four cells.")
    a = ap.parse_args()

    region = os.environ.get("AWS_REGION", "us-west-2")
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not token:
        raise SystemExit("AWS_BEARER_TOKEN_BEDROCK not set in environment")

    files = sorted(Path(a.problems_dir).glob("*.json"))
    random.Random(a.seed).shuffle(files)
    files = files[: a.n]
    CELLS = [("aggregate", "single"), ("full", "single"),
             ("aggregate", "compound"), ("full", "compound")]
    if a.cells:
        want = {tuple(c.split(":")) for c in a.cells.split(",") if ":" in c}
        bad = want - set(CELLS)
        if bad:
            raise SystemExit("unknown cell(s): %s" % sorted(bad))
        CELLS = [c for c in CELLS if c in want]
    print("%d battery problems x %d cells, K=%d, max_steps=%d"
          % (len(files), len(CELLS), a.k, a.max_steps), flush=True)

    jobs = [(str(f), o, c, a.model, region, token, a.k, a.max_steps, a.alpha)
            for f in files for (o, c) in CELLS]
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    fh = open(outp, "w")
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_episode, jobs), 1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print("  [%4d/%d %6.0fs] %-18s %-10s/%-8s feasible=%s parts=%.1f sims=%s %s"
                  % (i, len(jobs), time.time() - t0, r.get("problem_id"), r.get("obs"),
                     r.get("act"), r.get("feasible"), r.get("mean_parts_per_candidate") or 0.0,
                     r.get("n_simulator_calls"), r.get("error", "")), flush=True)
    fh.close()
    print("wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
