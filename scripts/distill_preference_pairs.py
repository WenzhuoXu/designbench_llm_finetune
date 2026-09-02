#!/usr/bin/env python
"""Extract preference pairs from the potential-guided search.

Distilling the search by SFT transfers the winner and throws away the comparison.
The search's competence is an argmax over counterfactuals -- "of the ~140 actions
available at this state, this one raises Phi most" -- and a demonstration shows
the model only which action won, never the margin or the alternatives. Measured
on the held-out split, SFT on 123 traces moved feasibility 61.8% -> 55.9% and left
mass at 0.871 against the teacher's 0.620: nothing transferred.

Two facts make preferences the better extraction:

  * every state yields MANY comparisons rather than one demonstration, so the
    signal per problem is far larger -- the binding constraint on the SFT set was
    41 distinct problems, and this reads the same problems much harder;
  * a preference is defined even when the trajectory FAILS, so the 25 training
    problems the search never solved (and every state of every failed rollout)
    become usable data instead of being discarded.

Both turns carry the same reasoning -- the state facts the simulator reports,
which constraint binds and which element it blames -- and differ ONLY in the
action. That isolates the ranking: the model cannot prefer a turn because its
prose claims a bigger improvement, because neither turn claims one.

CPU only.

    python scripts/distill_preference_pairs.py \\
        --split-file data/splits/truss_v1_auto.json --split train \\
        --out results/distill/search_preferences.jsonl --workers 16
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm_finetune.data.processors import designbench_prompt  # noqa: E402
from llm_finetune.training.rl.posterior.potential import (  # noqa: E402
    compute_potential_v2,
    program_from_truss_spec,
)
from scripts.distill_search_traces import _binding  # noqa: E402
from scripts.search_ladder import (  # noqa: E402
    candidate_actions, fea, make_truss, param_bounds, step,
)


def _state_reasoning(state: dict, program) -> str:
    """The facts the simulator reports. Identical for both sides of a pair."""
    binding, element = _binding(state, program)
    if not binding:
        return ("All constraints are satisfied; the remaining objective is to shed mass "
                "without breaking any of them.")
    readable = {
        "fos_buckling": f"buckling FOS is {state.get('fos_buckling', 0.0):.2f}, below the "
                        f"{program.limit_for('fos_buckling') or 1.5:.1f} minimum",
        "fos_yielding": f"yielding FOS is {state.get('fos_yielding', 0.0):.2f}, below the "
                        f"{program.limit_for('fos_yielding') or 1.5:.1f} minimum",
        "mass": f"mass is {state.get('mass', 0.0):.1f} kg, over its limit",
        "deflection": f"deflection is {state.get('deflection', 0.0):.3e} m, over its limit",
    }
    head = readable.get(binding, f"{binding} is violating its limit")
    tail = f", and the simulator reports {element} as the governing element." if element else "."
    return f"The binding constraint is {binding.replace('_', ' ')}: {head}{tail}"


_FACTOR = re.compile(r"[:,]\s*([0-9.]+)\s*\]?\)\s*$")


def _action_kind(action: str) -> str:
    return action.split("(", 1)[0].strip()


_PARAM = re.compile(r"[,\[]\s*([rt])\s*[,:]")


def _move_signature(action: str) -> tuple:
    """(action type, parameter, scale factor) -- everything except WHICH element."""
    m = _FACTOR.search(action.strip())
    factor = round(float(m.group(1)), 4) if m else None
    p = _PARAM.search(action)
    return (_action_kind(action), p.group(1) if p else "?", factor)


def _direction(action: str) -> str:
    """Does this action grow or shrink the member(s) it touches?"""
    m = _FACTOR.search(action.strip())
    if not m:
        return "other"
    try:
        return "grow" if float(m.group(1)) > 1.0 else "shrink"
    except ValueError:
        return "other"


def _turn(reasoning: str, action: str) -> str:
    return f"<think>\n{reasoning}\n</think>\n<action>{action}</action>"


def pairs_for_problem(spec: dict, *, max_steps: int, alpha: float, tau: float,
                      pairs_per_state: int, min_gap: float, rng: random.Random) -> list[dict]:
    goals = spec.get("goals", {}) or {}
    bounds = param_bounds(spec)
    truss = make_truss(spec)
    state = fea(truss, goals)
    program = program_from_truss_spec(spec, initial_mass=state.get("mass"))
    phi_kwargs = dict(alpha=alpha, tau=tau)

    initial_state = dict(state)
    action_history: list[dict] = []
    out: list[dict] = []
    feasible_steps = infeasible_steps = 0

    for _ in range(max_steps):
        acts = candidate_actions(
            truss, bounds, macros=False,
            target_fos_b=program.limit_for("fos_buckling") or 1.5,
            target_fos_y=program.limit_for("fos_yielding") or 1.5,
        )
        scored = []
        for action in acts:
            res = step(truss, goals, action, program)
            if res is None:
                continue
            nt, ns = res
            scored.append((compute_potential_v2(ns, program, **phi_kwargs), action, nt, ns))
        if len(scored) < 2:
            break
        scored.sort(key=lambda item: item[0], reverse=True)
        best_phi, best_action, best_truss, best_state = scored[0]

        reasoning = _state_reasoning(state, program)
        # The prompt is the raw DesignBench message list, so the same
        # warmstart_reasoning transform the RL rollout uses can be applied to it.
        messages = [
            {"role": "system", "content": designbench_prompt.build_problem_text(spec)},
            {"role": "system", "content": "INITIAL STATE ANALYSIS:\n"
                                          + designbench_prompt.format_eval_result(initial_state)},
        ]
        for turn in action_history:
            messages.append({"role": "assistant",
                             "content": _turn(turn["reasoning"], turn["action"])})
            messages.append({"role": "system",
                             "content": "STRUCTURAL ANALYSIS RESULT:\n"
                                        + designbench_prompt.format_eval_result(turn["fea_result"])})

        # Rejected candidates: meaningfully worse, drawn from across the ranking
        # rather than only the bottom, and -- critically -- DIRECTION-MATCHED.
        #
        # Without matching, the pairs are trivially separable. The greedy
        # trajectory reaches feasibility in one or two steps and then spends the
        # remaining ~18 shedding mass, so most states are post-feasibility, where
        # the best action shrinks a member and almost every worse one grows it.
        # Measured on the first version: the chosen action had scale factor > 1 in
        # 13.2% of pairs and the rejected in 87.7%. A model can score ~88% by
        # learning "prefer shrinking" and nothing about which member or how much,
        # and DPO duly hit training accuracy 1.00 within ten steps.
        #
        # Restricting the rejected candidate to the SAME direction as the chosen
        # one makes the preference about the ranking rather than the sign.
        # Match on ACTION TYPE as well as direction. Direction matching alone left
        # SCALE_MULTI_PARAM as the chosen action 36.9% of the time against 6.0% for
        # rejected -- a 31-point gap worth ~65% accuracy to a model that learned
        # nothing but "prefer the multi-member action". Matching both leaves only
        # the question the ranking is actually about: which member, and how much.
        # SAME MOVE, DIFFERENT ELEMENT. The rejected action must match the chosen
        # one in action type, parameter and scale factor, differing only in which
        # member it touches.
        #
        # Anything looser leaves the pair separable by surface features. Matching
        # direction alone left a 31-point SCALE_MULTI_PARAM asymmetry; adding type
        # still left the scale factor free, and a majority-guess over
        # (type, direction, factor) signatures scored **91%** -- the pairs were
        # being decided by how much, not by where. Fixing the move reduces that
        # ceiling to chance, so any accuracy above 50% is knowledge of which
        # element to act on: precisely the critical-element question that the
        # feedback field answers for free at depth 0.
        worse = [s for s in scored[1:] if best_phi - s[0] > min_gap]
        best_sig = _move_signature(best_action)
        same_move = [s for s in worse if _move_signature(s[1]) == best_sig]
        pool = same_move
        if pool:
            # Spread the rejected picks across the ranking of same-move
            # alternatives so the model sees a gradient, not just the worst one.
            picks = []
            k = min(pairs_per_state, len(pool))
            for j in range(k):
                lo = j * len(pool) // k
                hi = max((j + 1) * len(pool) // k, lo + 1)
                picks.append(rng.choice(pool[lo:hi]))
            for phi_bad, bad_action, _, _ in picks:
                out.append({
                    "problem_id": spec.get("problem_id"),
                    "step": len(action_history),
                    "messages": messages,
                    "chosen": _turn(reasoning, best_action),
                    "rejected": _turn(reasoning, bad_action),
                    "phi_chosen": best_phi,
                    "phi_rejected": phi_bad,
                    "phi_gap": best_phi - phi_bad,
                    "n_candidates": len(scored),
                })

        action_history.append({"action": best_action, "reasoning": reasoning,
                               "fea_result": best_state})
        was_feasible = program.is_feasible(state)
        truss, state = best_truss, best_state
        if program.is_feasible(state):
            feasible_steps += 1
            # Stop once the mass-shedding phase has contributed as many states as
            # the constraint-fixing phase, so neither regime dominates the data.
            if feasible_steps > max(infeasible_steps, 2):
                break
        else:
            infeasible_steps += 1
    return out


def _worker(args_tuple):
    path, max_steps, alpha, tau, pairs_per_state, min_gap, seed = args_tuple
    spec = json.load(open(path))
    if not (isinstance(spec, dict) and "topology" in spec):
        return []
    spec.setdefault("problem_id", Path(path).stem)
    try:
        return pairs_for_problem(spec, max_steps=max_steps, alpha=alpha, tau=tau,
                                 pairs_per_state=pairs_per_state, min_gap=min_gap,
                                 rng=random.Random(seed))
    except Exception as exc:  # noqa: BLE001
        print(f"  {spec.get('problem_id')}: {type(exc).__name__}: {exc}", flush=True)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(DESIGNBENCH / "data/problems"))
    ap.add_argument("--split-file", default=str(PROJECT / "data/splits/truss_v1_auto.json"))
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--out", default=str(PROJECT / "results/distill/search_preferences.jsonl"))
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--pairs-per-state", type=int, default=3)
    ap.add_argument("--min-gap", type=float, default=0.01,
                    help="Minimum Phi gap for a pair to be informative rather than noise.")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    wanted = set(json.load(open(args.split_file))[args.split]) if args.split_file else None
    files, seen = [], set()
    for f in sorted(Path(args.problems).glob("*.json")):
        try:
            spec = json.load(open(f))
        except Exception:
            continue
        if not (isinstance(spec, dict) and "topology" in spec):
            continue
        pid = spec.get("problem_id", f.stem)
        if pid in seen or (wanted is not None and pid not in wanted):
            continue
        seen.add(pid)
        files.append(str(f))
    print(f"{len(files)} problems on split '{args.split}'")

    tasks = [(f, args.max_steps, args.alpha, args.tau, args.pairs_per_state,
              args.min_gap, i) for i, f in enumerate(files)]
    t0 = time.time()
    pairs = []
    import multiprocessing as mp
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, got in enumerate(pool.imap_unordered(_worker, tasks), 1):
            pairs.extend(got)
            if i % 20 == 0:
                print(f"  {i}/{len(tasks)}  {len(pairs)} pairs  ({time.time()-t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    import statistics as st
    gaps = [p["phi_gap"] for p in pairs]
    print(f"\nwrote {len(pairs)} preference pairs -> {out}  ({time.time()-t0:.0f}s)")
    print(f"  distinct problems : {len({p['problem_id'] for p in pairs})}")
    print(f"  distinct states   : {len({(p['problem_id'], p['step']) for p in pairs})}")
    print(f"  Phi gap           : median {st.median(gaps):.4f}  p10 {sorted(gaps)[len(gaps)//10]:.4f}")
    print(f"  candidates ranked : median {st.median([p['n_candidates'] for p in pairs]):.0f} per state")
    print(f"  (SFT set for comparison: 123 traces, 41 distinct problems, 2563 turns)")


if __name__ == "__main__":
    main()
