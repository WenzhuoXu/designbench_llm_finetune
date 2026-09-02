#!/usr/bin/env python
"""Stop the agent walking away from good states.

Trace evidence (results/api_guidance/traces_sonnet.jsonl, 410 turns / 40 problems):

  * solved episodes take 6.1 turns; FAILED episodes take all 20, every one.
  * failed trajectories oscillate across the constraint boundary --
    hard_problem_0320 fos_b: 0.21 1.45 1.75 0.86 0.38 1.62 0.45 1.62 ... crossing
    the 1.5 requirement six times while mass hovers just over budget.
  * the CHOSEN candidate's Phi is non-monotone in 12/12 failed episodes; 0320 goes
    -1.61 -> -5.31 in one step, and three of four shown END below the best Phi they
    ever reached.

The last point is a defect in the search, not in Phi or in the proposer: argmax over
K candidates is taken unconditionally, so when every candidate is worse than the
current state the agent still moves to the least-bad one. There is no incumbent and
no floor.

Arms:
  d1            argmax over K, always move                     (current baseline)
  incumbent     track best-Phi state; only move if the best candidate BEATS it,
                otherwise stay and re-propose from the incumbent
  incumbent_fb  as `incumbent`, plus the context states the incumbent and, when the
                last move regressed, which constraint it gave up

Same K, same simulator budget per turn, same Phi. `incumbent` isolates the algorithm
fix; `incumbent_fb` adds telling the model about its own regression.
"""
from __future__ import annotations
import argparse, json, os, random, sys, threading, time
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import urllib.request, urllib.error
import ssl, socket, http.client, encodings.idna, hashlib, base64, email.utils  # noqa: F401
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals
from llm_finetune.data.processors import designbench_prompt as DP
from llm_finetune.training.rl.posterior.potential import (
    compute_potential_v2, program_from_truss_spec,
)
from api_grammar_2x2 import member_table, COMPOUND_RULE, parse_candidates, apply_candidate, call
from api_lookahead_d2 import propose, expand

_lock = threading.Lock()


def regression_note(prev, cur, best):
    """What the last move gave up, in the model's own terms."""
    bits = []
    for key, label, lim in (("fos_buckling", "buckling FOS", 1.5),
                            ("fos_yielding", "yielding FOS", 1.5)):
        p, c = prev.get(key), cur.get(key)
        if p is None or c is None:
            continue
        if p >= lim > c:
            bits.append("%s fell from %.2f to %.2f, below the required %.1f"
                        % (label, p, c, lim))
    pm, cm = prev.get("mass"), cur.get("mass")
    if pm is not None and cm is not None and cm > pm * 1.02:
        bits.append("mass rose from %.1f to %.1f" % (pm, cm))
    head = ("Best design so far: mass %.1f, buckling FOS %.2f, yielding FOS %.2f."
            % (best.get("mass", float("nan")), best.get("fos_buckling", float("nan")),
               best.get("fos_yielding", float("nan"))))
    if not bits:
        return head
    return head + " Your last move was a REGRESSION: " + "; ".join(bits) + \
        ". Do not give up a constraint you had already satisfied."


def run_episode(spec, arm, model, region, token, k=8, max_steps=20, alpha=5.0, tau=0.05):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    program = program_from_truss_spec(spec, initial_mass=state.get("mass"))
    phi = compute_potential_v2(state, program, alpha=alpha, tau=tau)

    # `fb` is the missing cell: regression feedback WITHOUT the incumbent floor.
    # incumbent_fb beat incumbent by +8.9pp (p=0.015) while the floor itself cost
    # -5.9pp vs d1, so feedback without the floor is the combination not yet tested.
    use_inc = arm in ("incumbent", "incumbent_fb")
    use_fb = arm in ("incumbent_fb", "fb")
    # veto: block only CATASTROPHIC steps. Traces: failed episodes carry 1.43 steps/ep
    # with chosen-Phi drop < -3, solved ones 0.06/ep (p10 of Delta-Phi is -0.91 solved vs
    # -2.33 failed). Small downhill steps are on the path to success and must pass;
    # the incumbent floor blocked those too and lost 5.9pp. On a catastrophic turn the
    # whole candidate set came from one bad generation, so re-propose from the CURRENT
    # state (fresh call) rather than reverting to the incumbent. Accept after 2 retries.
    veto_thr = -3.0 if arm == "veto" else None
    veto_retries = 2
    best_truss, best_state, best_phi = truss, dict(state), phi
    history, n_turns, n_sim, n_llm, n_stay = [], 0, 0, 0, 0

    for _ in range(max_steps):
        text, cands = propose(spec, initial, history, truss, k, model, region, token)
        n_llm += 1
        if not cands:
            history.append({"thinking": f"No <action> parsed. Emit exactly {k} "
                                        "<action>...</action> lines.",
                            "obs": DP.format_eval_result(state) + member_table(truss)})
            continue
        n_turns += 1
        scored = expand(truss, goals, program, cands, alpha, tau)
        n_sim += len(scored)
        if not scored:
            history.append({"thinking": text.strip(),
                            "obs": DP.format_eval_result(state) + member_table(truss)})
            continue

        feas = [x for x in scored if x[2].get("is_feasible") and program.is_valid(x[2])]
        if feas:
            _c, truss, state, phi = max(feas, key=lambda x: x[3])
            history.append({"thinking": text.strip(),
                            "obs": DP.format_eval_result(state) + member_table(truss)})
            break

        cand = max(scored, key=lambda x: x[3])
        if veto_thr is not None:
            tries = 0
            while cand[3] - phi < veto_thr and tries < veto_retries:
                tries += 1; n_stay += 1
                text, cands = propose(spec, initial, history, truss, k, model, region, token)
                n_llm += 1
                sc2 = expand(truss, goals, program, cands, alpha, tau) if cands else []
                n_sim += len(sc2)
                if sc2:
                    f2 = [x for x in sc2 if x[2].get("is_feasible") and program.is_valid(x[2])]
                    if f2:
                        scored, cand = sc2, max(f2, key=lambda x: x[3]); break
                    scored, cand = sc2, max(sc2, key=lambda x: x[3])
        if use_inc and cand[3] <= best_phi:
            # Every candidate is worse than the best state seen. Do NOT step downhill:
            # return to the incumbent and try again from there. This is the whole fix
            # for the oscillation -- the baseline moves here regardless.
            n_stay += 1
            truss, state, phi = best_truss, dict(best_state), best_phi
            note = ("None of the %d candidates improved on the best design so far. "
                    "Returning to it. " % len(scored))
            obs = DP.format_eval_result(state) + member_table(truss)
            if use_fb:
                obs = obs + "\n\n" + note + regression_note(cand[2], state, best_state)
            history.append({"thinking": text.strip(), "obs": obs})
            continue

        prev_state = dict(state)
        _c, truss, state, phi = cand
        if phi > best_phi:
            best_truss, best_state, best_phi = truss, dict(state), phi
        obs = DP.format_eval_result(state) + member_table(truss)
        if use_fb:
            obs = obs + "\n\n" + regression_note(prev_state, state, best_state)
        history.append({"thinking": text.strip(), "obs": obs})
        if state.get("is_feasible") and program.is_valid(state):
            break

    return {
        "problem_id": spec.get("problem_id"), "arm": arm,
        "feasible": bool(state.get("is_feasible")) and bool(program.is_valid(state)),
        "turns": n_turns, "n_llm_calls": n_llm, "n_simulator_calls": n_sim,
        "n_stay": n_stay, "final_phi": float(phi), "best_phi": float(best_phi),
        "mass_ratio": (state.get("mass") / program.objective_ref) if program.objective_ref else None,
        "final_fos_b": state.get("fos_buckling"), "final_fos_y": state.get("fos_yielding"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-dir", default=str(DESIGNBENCH / "data/problems_hard"))
    ap.add_argument("--split-file", default=str(PROJECT / "data/splits/truss_hard_v1.json"))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n", type=int, default=202)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="d1,incumbent,incumbent_fb")
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/incumbent.jsonl"))
    a = ap.parse_args()

    region = os.environ.get("AWS_REGION", "us-west-2")
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not token:
        raise SystemExit("AWS_BEARER_TOKEN_BEDROCK not set")

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
    ARMS = [x for x in a.arms.split(",") if x]
    print("%d problems x %s  k=%d" % (len(specs), ARMS, a.k), flush=True)

    t0 = time.time()
    _t, _g = _load_truss_and_goals(specs[0]); _st = _analyze_truss(_t, _g)
    compute_potential_v2(_st, program_from_truss_spec(specs[0], initial_mass=_st.get("mass")))
    member_table(_t); apply_candidate(_t, "SCALE_PARAM(0, r, 1.05)")
    _p = call([{"role": "user", "content": [{"text": "Reply with OK."}]}],
              None, a.model, region, token, max_tokens=8)
    print("warmup %.1fs api=%s" % (time.time()-t0, "ok" if _p.strip() else "FAILED"), flush=True)
    if not _p.strip():
        raise SystemExit("Bedrock probe empty")

    jobs = [(s, arm) for s in specs for arm in ARMS]
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    fh = open(outp, "w")
    from concurrent.futures import ThreadPoolExecutor
    def work(j):
        s_, arm_ = j
        t = time.time()
        try:
            r = run_episode(s_, arm_, a.model, region, token, k=a.k, max_steps=a.max_steps)
        except Exception as e:
            r = {"problem_id": s_.get("problem_id"), "arm": arm_, "error": str(e)[:200]}
        with _lock:
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print("    [%5.1fs] %-20s %-13s feasible=%s turns=%s stay=%s %s"
                  % (time.time()-t, s_.get("problem_id"), arm_, r.get("feasible"),
                     r.get("turns"), r.get("n_stay"), r.get("error","")), flush=True)
        return r
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, _ in enumerate(ex.map(work, jobs), 1):
            if i % 50 == 0: print("  %d/%d" % (i, len(jobs)), flush=True)
    fh.close()
    print("wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
