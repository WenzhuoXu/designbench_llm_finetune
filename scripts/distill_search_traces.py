#!/usr/bin/env python
"""Distil the potential-guided search into SFT traces.

Measured on the 130 truss problems, the trained policy sits at median mass/ref
0.917 while a one-step lookahead under the goal-aligned potential reaches 0.451
-- and beats the policy on 26/27 problems they both solve. The policy has not
internalised the potential. The gap is the objective.

The search buys that quality with simulator calls (~2700 per problem against the
policy's 20). This script turns those calls into supervision: it runs the search
on the training split, and writes each decision as a conversation turn whose
reasoning is the search's OWN justification -- which member binds, how many
candidates were compared, what the chosen action did to the potential, and by
what margin it won. Nothing is invented: every number in the reasoning is read
off the search that produced the action.

Output is DesignBench SFT JSONL (system/system/assistant/system/... with
``<think>...Action: X</think>``), so it feeds the existing warmstart pipeline
unchanged. Prompt text is produced by the same ``designbench_prompt`` helpers the
RL rollout and the evaluator use, so the distilled distribution matches the one
the policy is later asked to act in.

CPU only, ~11 s per problem.

    python scripts/distill_search_traces.py \\
        --split-file data/splits/truss_v1.json --split train \\
        --out results/distill/search_policy_train.jsonl --workers 16
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
from scripts.search_ladder import (  # noqa: E402
    MACRO_SEP,
    candidate_actions,
    fea,
    make_truss,
    param_bounds,
    step,
)

_ORDINAL = {0: "largest", 1: "second largest", 2: "third largest"}


def _describe_action(action: str, state: dict) -> str:
    """A short human reading of a grammar action (or macro)."""
    if MACRO_SEP in action:
        n = len(action.split(MACRO_SEP))
        return f"resize all {n} governed members together, each by its own utilisation"
    head = action.split("(", 1)[0]
    inner = action[action.find("(") + 1:action.rfind(")")]
    parts = [x.strip() for x in inner.split(",")]
    if head == "SCALE_PARAM" and len(parts) == 3:
        name = {"r": "radius", "t": "thickness"}.get(parts[1], parts[1])
        pct = (float(parts[2]) - 1.0) * 100.0
        verb = "increase" if pct >= 0 else "reduce"
        return f"{verb} member M{parts[0]}'s {name} by {abs(pct):.0f}%"
    if head == "SCALE_MULTI_PARAM":
        return "scale the same parameter on every member"
    return action


def _binding(state: dict, program) -> tuple[str, str]:
    """Which constraint binds, and the element the simulator blames for it."""
    worst_name, worst_gap, tag = "", 0.0, ""
    for name, violation in program.violations(state, tau=0.0).items():
        if violation > worst_gap:
            worst_name, worst_gap = name, violation
    if worst_name == "fos_buckling":
        mid = state.get("min_fos_buckling_member_id")
        tag = f"M{mid}" if isinstance(mid, int) else ""
    elif worst_name == "fos_yielding":
        mid = state.get("min_fos_yielding_member_id")
        tag = f"M{mid}" if isinstance(mid, int) else ""
    return worst_name, tag


def _reasoning(state, next_state, program, action, rank, n_cands, phi_before, phi_after,
               margin, mass_cap, step_idx) -> str:
    """The search's own justification, in words. Every figure is measured."""
    binding, element = _binding(state, program)
    lines = []
    readable = {
        "fos_buckling": f"buckling FOS is {state.get('fos_buckling', 0.0):.2f}, below the "
                        f"{program.limit_for('fos_buckling') or 1.5:.1f} minimum",
        "fos_yielding": f"yielding FOS is {state.get('fos_yielding', 0.0):.2f}, below the "
                        f"{program.limit_for('fos_yielding') or 1.5:.1f} minimum",
        "mass": f"mass is {state.get('mass', 0.0):.1f} kg, over the "
                f"{mass_cap:.1f} kg limit" if mass_cap else "mass is over its limit",
        "deflection": f"deflection is {state.get('deflection', 0.0):.3e} m, over the limit",
    }
    if binding:
        head = readable.get(binding, f"{binding} is violating its limit")
        lines.append(f"The binding constraint is {binding.replace('_', ' ')}: {head}"
                     + (f", and the simulator reports {element} as the governing member." if element
                        else "."))
    else:
        lines.append("All constraints are satisfied, so the remaining objective is to shed mass "
                     "without breaking any of them.")

    lines.append(
        f"Comparing {n_cands} candidate modifications one step ahead, "
        f"{_describe_action(action, state)} gives the {_ORDINAL.get(rank, f'{rank + 1}th largest')} "
        f"improvement in the constrained objective"
        + (f", ahead of the next best by {margin:.3f}." if margin > 1e-6
           else "; several candidates are within noise of it.")
    )
    lines.append(
        f"It moves the objective from {phi_before:.2f} to {phi_after:.2f}: "
        f"buckling FOS {state.get('fos_buckling', 0.0):.2f} to {next_state.get('fos_buckling', 0.0):.2f}, "
        f"mass {state.get('mass', 0.0):.1f} to {next_state.get('mass', 0.0):.1f} kg"
        + (f" against the {mass_cap:.1f} kg cap." if mass_cap else ".")
    )
    return "\n".join(lines)


def solve_and_trace(spec: dict, max_steps: int, alpha: float, tau: float,
                    keep_infeasible: bool = False, macros: bool = False) -> dict | None:
    """Run the lookahead search, recording a conversation as it goes."""
    goals = spec.get("goals", {}) or {}
    bounds = param_bounds(spec)
    truss = make_truss(spec)
    state = fea(truss, goals)
    program = program_from_truss_spec(spec, initial_mass=state.get("mass"))
    mass_cap = program.limit_for("mass")
    phi_kwargs = dict(alpha=alpha, tau=tau)

    messages = [
        {"role": "system", "content": designbench_prompt.build_problem_text(spec)},
        {"role": "system", "content": "INITIAL STATE ANALYSIS:\n"
                                      + designbench_prompt.format_eval_result(state)},
    ]
    best = None
    steps = 0
    for step_idx in range(max_steps):
        # macros=False by default: a distillation target must be an action the
        # policy can actually emit. The fully-stressed macro rescales every member
        # by a different factor in one move; the grammar has no such action, so
        # distilling it would teach a string the environment cannot execute.
        acts = candidate_actions(
            truss, bounds, macros=macros,
            target_fos_b=program.limit_for("fos_buckling") or 1.5,
            target_fos_y=program.limit_for("fos_yielding") or 1.5,
        )
        scored = []
        for action in acts:
            out = step(truss, goals, action, program)
            if out is None:
                continue
            nt, ns = out
            scored.append((compute_potential_v2(ns, program, **phi_kwargs), action, nt, ns))
        if not scored:
            break
        scored.sort(key=lambda item: item[0], reverse=True)
        phi_before = compute_potential_v2(state, program, **phi_kwargs)
        phi_after, action, next_truss, next_state = scored[0]
        margin = phi_after - scored[1][0] if len(scored) > 1 else 0.0
        if phi_after <= phi_before + 1e-9:
            break  # no candidate improves the objective; stop rather than pad the trace

        messages.append({
            "role": "assistant",
            "content": "<think>" + _reasoning(
                state, next_state, program, action, 0, len(scored),
                phi_before, phi_after, margin, mass_cap, step_idx)
            + f"\nAction: {action}</think>",
        })
        truss, state = next_truss, next_state
        steps += 1
        messages.append({
            "role": "system",
            "content": "STRUCTURAL ANALYSIS RESULT:\n"
                       + designbench_prompt.format_eval_result(state),
        })
        if program.is_feasible(state) and (best is None or state["mass"] < best[0]):
            best = (state["mass"], steps, len(messages))

    if best is None and not keep_infeasible:
        return None
    if best is not None:
        # Cut the conversation at the lightest feasible state reached.
        messages = messages[:best[2]]
        messages.append({"role": "assistant",
                         "content": "<answer>Feasible design reached; "
                                    f"mass {best[0]:.1f} kg.</answer>"})
    return {
        "problem_id": spec.get("problem_id"),
        "trace_id": f"{spec.get('problem_id')}_searchpolicy",
        "strategy_type": "potential_guided_lookahead",
        "trace_quality": 1.0,
        "messages": messages,
        "structured": {
            "feasible": best is not None,
            "n_steps": best[1] if best else steps,
            "final_mass": best[0] if best else state.get("mass"),
            "mass_ref": program.objective_ref,
            "mass_ratio": (best[0] / program.objective_ref) if best and program.objective_ref else None,
            "initial_mass": messages and None,
        },
    }


def _worker(args_tuple):
    path, max_steps, alpha, tau, keep, macros = args_tuple
    spec = json.load(open(path))
    if not (isinstance(spec, dict) and "topology" in spec):
        return None
    spec.setdefault("problem_id", Path(path).stem)
    try:
        return solve_and_trace(spec, max_steps, alpha, tau, keep, macros)
    except Exception as exc:  # noqa: BLE001
        print(f"  {spec.get('problem_id')}: {type(exc).__name__}: {exc}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(DESIGNBENCH / "data/problems"))
    ap.add_argument("--split-file", default=str(PROJECT / "data/splits/truss_v1.json"))
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--out", default=str(PROJECT / "results/distill/search_policy_train.jsonl"))
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--keep-infeasible", action="store_true")
    ap.add_argument("--allow-macros", action="store_true",
                    help="Permit compound fully-stressed moves. Off by default: the "
                         "environment grammar cannot express them, so they are not "
                         "valid distillation targets.")
    args = ap.parse_args()

    wanted = None
    if args.split_file:
        wanted = set(json.load(open(args.split_file))[args.split])
    files = []
    seen = set()
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
    tasks = [(f, args.max_steps, args.alpha, args.tau, args.keep_infeasible, args.allow_macros)
             for f in files]
    t0 = time.time()
    traces = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            for i, trace in enumerate(pool.imap_unordered(_worker, tasks), 1):
                if trace:
                    traces.append(trace)
                if i % 20 == 0:
                    print(f"  {i}/{len(tasks)}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        for task in tasks:
            trace = _worker(task)
            if trace:
                traces.append(trace)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")

    solved = [t for t in traces if t["structured"]["feasible"]]
    ratios = [t["structured"]["mass_ratio"] for t in solved if t["structured"]["mass_ratio"]]
    turns = [t["structured"]["n_steps"] for t in solved]
    import statistics as st
    print(f"\nwrote {len(traces)} traces -> {out}  ({time.time()-t0:.0f}s)")
    print(f"  feasible          : {len(solved)}/{len(files)} ({len(solved)/max(len(files),1):.1%})")
    if ratios:
        print(f"  median mass/ref   : {st.median(ratios):.3f}   (trained policy: 0.917)")
    if turns:
        print(f"  median turns      : {st.median(turns):.0f}  max {max(turns)}")
        print(f"  assistant turns   : {sum(len([m for m in t['messages'] if m['role']=='assistant']) for t in traces)}")


if __name__ == "__main__":
    main()
