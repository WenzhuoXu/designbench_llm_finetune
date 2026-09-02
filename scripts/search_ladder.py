#!/usr/bin/env python
"""Search ladder: how far does potential-guided search get with NO LLM?

Runs several fixed (non-LLM) policies over the DesignBench problem set on CPU
and reports feasibility, steps and final mass. It answers three questions that
the GPU ablations could not:

  Q1 (framing)   How much of the benchmark falls to the classical sizing
                 heuristic alone (greedy critical-member scaling)?  Until this
                 number exists, "an LLM solved 54%" is unanchored.
  Q2 (theory)    Does one-step lookahead over the SAME action set beat the
                 greedy heuristic?  This is the innovation -- estimating an
                 action's potential by simulating it -- isolated from the LLM.
  Q3 (potential) Does the potential's *specification* decide the answer?  The
                 same search is run under the shipped potential (v1, hardcoded
                 constraint set) and the goal-aligned Lagrangian (v2).

It also logs, at every state the greedy policy visits, whether the
lookahead-best action targets the critical member reported by the FEA feedback
-- the correlation between "state feedback" and "search" that the study needs.

CPU only.  Usage:
    python scripts/search_ladder.py --out results/search_ladder --workers 32
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm_finetune.training.rl.posterior.potential import (  # noqa: E402
    DesignProgram,
    compute_potential,
    compute_potential_v2,
    program_from_truss_spec,
)

SCALE_FACTORS = (0.85, 1.1, 1.25, 1.5, 2.0)
PARAMS = ("r", "t")


# ── environment helpers ──────────────────────────────────────────────────────

def load_problem(path: Path) -> dict | None:
    spec = json.load(open(path))
    return spec if isinstance(spec, dict) and "topology" in spec else None


def make_truss(spec: dict):
    from validation.truss_executor import load_truss_from_problem
    return load_truss_from_problem(spec)


def fea(truss, goals: dict) -> dict:
    from validation.truss_executor import analyze_truss
    state = analyze_truss(truss, goals)
    out = dict(state)
    for key in ("mass", "fos_buckling", "fos_yielding", "deflection"):
        try:
            value = float(out.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        out[key] = value if math.isfinite(value) else (0.0 if "fos" in key else 1e9)
    return out


MACRO_SEP = " ; "


def apply_action(truss, action: str) -> bool:
    """Apply one grammar action, or a MACRO: several actions joined by MACRO_SEP.

    A macro is one decision, evaluated with one simulator call. Without macros the
    search is strictly weaker than the fully-stressed-design heuristic, which
    resizes EVERY member by its own utilisation in a single move -- a comparison
    the search cannot win on action space alone, whatever the potential says.
    """
    from validation.truss_executor import execute_grammar_action
    from llm_finetune.envs.truss_env import normalize_action
    ok = True
    for part in action.split(MACRO_SEP):
        part = part.strip()
        if not part:
            continue
        try:
            ok = bool(execute_grammar_action(truss, normalize_action(truss, part))) and ok
        except Exception:
            return False
    return ok


def member_params(truss) -> list[dict]:
    out = []
    for m in truss.members:
        params = getattr(getattr(m, "shape", None), "_params", None) or {}
        out.append({k: float(v) for k, v in params.items() if isinstance(v, (int, float))})
    return out


def critical_members(state: dict) -> tuple[list[int], list[int]]:
    def _ids(key):
        v = state.get(key)
        if v is None or v == "":
            return []
        return [int(x) for x in (v if isinstance(v, list) else [v])]
    return _ids("min_fos_buckling_member_id"), _ids("min_fos_yielding_member_id")


def param_bounds(spec: dict) -> dict[str, tuple[float, float]]:
    opt = spec.get("optimization") or {}
    sp = opt.get("shape_params") or {}
    bounds = {}
    for name, rng in sp.items():
        if isinstance(rng, dict) and "min" in rng and "max" in rng:
            bounds[name] = (float(rng["min"]), float(rng["max"]))
    return bounds


# ── candidate generation ─────────────────────────────────────────────────────

def candidate_actions(truss, bounds: dict[str, tuple[float, float]], include_global: bool = True,
                      macros: bool = True, target_fos_b: float = 1.5,
                      target_fos_y: float = 1.5) -> list[str]:
    """Sizing actions that stay inside the problem's own parameter bounds."""
    dims = member_params(truss)
    n = len(dims)
    actions: list[str] = []
    for mid, params in enumerate(dims):
        for name in PARAMS:
            cur = params.get(name)
            if cur is None:
                continue
            lo, hi = bounds.get(name, (0.0, math.inf))
            for f in SCALE_FACTORS:
                new = cur * f
                if new < lo or new > hi:
                    continue
                actions.append(f"SCALE_PARAM({mid}, {name}, {f})")
    if include_global and n > 1:
        ids = "[" + ",".join(str(i) for i in range(n)) + "]"
        for name in PARAMS:
            lo, hi = bounds.get(name, (0.0, math.inf))
            for f in SCALE_FACTORS:
                vals = [d.get(name) for d in dims if d.get(name) is not None]
                if not vals:
                    continue
                if min(vals) * f < lo or max(vals) * f > hi:
                    continue
                actions.append(f"SCALE_MULTI_PARAM({ids}, [{name}:{f}])")
    if macros:
        for margin in (1.0, 1.05, 1.15, 1.35):
            macro = fsd_macro(truss, bounds, target_fos_b, target_fos_y, margin)
            if macro:
                actions.append(macro)
    return actions


def fsd_macro(truss, bounds: dict[str, tuple[float, float]], target_fos_b: float,
              target_fos_y: float, margin: float) -> str:
    """One fully-stressed-design resize, expressed as a macro grammar action.

    Buckling FOS scales roughly as r^3 through the second moment of area and
    yielding FOS as the area, so member i needs
    max((target_b/fos_b_i)^(1/3), target_y/fos_y_i). Under-stressed members shrink,
    which is how FSD reaches feasibility at low mass.
    """
    dims = member_params(truss)
    lo, hi = bounds.get("r", (0.0, math.inf))
    parts = []
    for i, m in enumerate(truss.members):
        cur = dims[i].get("r")
        if cur is None or cur <= 0:
            continue
        try:
            fb, fy = float(m.fos_buckling), float(m.fos_yielding)
        except (TypeError, ValueError):
            fb = fy = float("inf")
        need_b = (target_fos_b * margin / fb) ** (1.0 / 3.0) if math.isfinite(fb) and fb > 0 else 0.7
        need_y = (target_fos_y * margin / fy) if math.isfinite(fy) and fy > 0 else 0.7
        f = min(2.0, max(0.7, max(need_b, need_y)))
        f = min(f, hi / cur)
        f = max(f, lo / cur)
        if abs(f - 1.0) < 1e-3:
            continue
        parts.append(f"SCALE_PARAM({i}, r, {f:.6f})")
    return MACRO_SEP.join(parts)


def step(truss, goals: dict, action: str, program=None):
    """Apply one action to a copy; return (new_truss, new_state) or None."""
    nt = copy.deepcopy(truss)
    if not apply_action(nt, action):
        return None
    st = fea(nt, goals)
    # Reject states outside the simulator's domain of validity. Two exploits were
    # found by letting the search run without this: negative mass (pipe area
    # pi*t*(2r-t) < 0 when t > 2r, reachable inside the declared bounds), and
    # mechanisms (thin a member until forces vanish -- FOS explodes, mass drops,
    # deflection reaches 1e14 m, and the mass-constrained family states no
    # deflection limit to catch it). The second accounted for 17% of "successes".
    if not math.isfinite(st.get("mass", 0.0)) or st.get("mass", 0.0) <= 0.0:
        return None
    if program is not None and not program.is_valid(st):
        return None
    return nt, st


# ── potentials ───────────────────────────────────────────────────────────────

def phi_v1(state: dict, cons: DesignProgram, initial_mass: float, alpha: float) -> float:
    return compute_potential(state, initial_mass=initial_mass, alpha=alpha)


# Set once from the CLI; the PHI table keeps its signature.
_PHI_OPTS = {"tau": 0.05, "feasibility_offset": 0.0}


def phi_v2(state: dict, cons: DesignProgram, initial_mass: float, alpha: float) -> float:
    return compute_potential_v2(state, cons, alpha=alpha, **_PHI_OPTS)


PHI = {"v1": phi_v1, "v2": phi_v2}


# ── policies ─────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    problem_id: str
    policy: str
    feasible: bool
    steps: int
    final_mass: float
    initial_mass: float
    mass_ref: float
    final_fos_b: float
    final_fos_y: float
    fea_calls: int
    family: str


def run_greedy_critical(truss, spec, goals, cons, max_steps, factor=1.3, diagnostics=None,
                        alpha=5.0, bounds=None):
    """Classical fully-stressed sizing: scale the buckling-critical member."""
    state = fea(truss, goals)
    initial_mass = state["mass"]
    calls = 1
    taken = 0
    for i in range(max_steps):
        if cons.is_feasible(state):
            break
        crit_b, crit_y = critical_members(state)
        target = crit_b[0] if crit_b else (crit_y[0] if crit_y else 0)
        # which parameter helps: buckling responds to r (moment of inertia),
        # yielding to area; scale r when buckling binds, t otherwise.
        binding_b = state["fos_buckling"] < (cons.limit_for("fos_buckling") or 1.5)
        param = "r" if binding_b else "t"
        action = f"SCALE_PARAM({target}, {param}, {factor})"
        if diagnostics is not None:
            diagnostics.append(_lookahead_diagnostic(
                truss, spec, goals, cons, state, initial_mass, alpha, bounds, i))
        out = step(truss, goals, action, cons)
        calls += 1
        if out is None:
            break
        truss, state = out
        taken += 1
    return truss, state, calls, initial_mass, taken


def _lookahead_diagnostic(truss, spec, goals, cons, state, initial_mass, alpha, bounds, depth):
    """Does the 1-step-lookahead best action target the FEA-reported critical member?"""
    acts = candidate_actions(truss, bounds or {}, macros=False)
    if not acts:
        return None
    scored = []
    for a in acts:
        out = step(truss, goals, a, cons)
        if out is None:
            continue
        _, ns = out
        scored.append((
            a,
            phi_v1(ns, cons, initial_mass, alpha),
            phi_v2(ns, cons, initial_mass, alpha),
            bool(cons.is_feasible(ns)),
        ))
    if not scored:
        return None
    crit_b, crit_y = critical_members(state)
    crit = set(crit_b) | set(crit_y)

    def _mid(action: str):
        inner = action[action.find("(") + 1:]
        head = inner.split(",")[0].strip()
        return int(head) if head.isdigit() else None

    def _summary(idx):
        """Ranking view over all candidates and over member-targeted ones only."""
        ranked = sorted(scored, key=lambda s: s[idx], reverse=True)
        best = ranked[0]
        margin = best[idx] - ranked[1][idx] if len(ranked) > 1 else 0.0
        bid = _mid(best[0])

        member_only = [s for s in scored if _mid(s[0]) is not None]
        member_only.sort(key=lambda s: s[idx], reverse=True)
        mo = {}
        if member_only:
            mbest = member_only[0]
            mbid = _mid(mbest[0])
            mo_margin = mbest[idx] - member_only[1][idx] if len(member_only) > 1 else 0.0
            # Where does the best CRITICAL-member action rank among member actions?
            crit_ranks = [i for i, s in enumerate(member_only) if _mid(s[0]) in crit]
            n_members_touched = len({_mid(s[0]) for s in member_only})
            mo = {
                "best_member_action": mbest[0],
                "best_member_targets_critical": mbid in crit,
                "member_margin": mo_margin,
                "crit_best_rank": (min(crit_ranks) if crit_ranks else None),
                "n_member_candidates": len(member_only),
                "n_members_touched": n_members_touched,
                # Value gap between the lookahead-best member action and the best
                # action on the critical member: how much the feedback hint costs.
                "crit_value_gap": (mbest[idx] - member_only[min(crit_ranks)][idx]) if crit_ranks else None,
            }
        return {
            "best_action": best[0],
            "best_targets_critical": (bid in crit) if bid is not None else None,
            "best_is_global": bid is None,
            "margin": margin,
            "best_reaches_feasible": best[3],
        } | mo

    return {
        "depth": depth,
        "n_candidates": len(scored),
        "critical_members": sorted(crit),
        "fos_buckling": state["fos_buckling"],
        "fos_yielding": state["fos_yielding"],
        "mass_ratio": state["mass"] / max(cons.objective_ref, 1e-9),
        "any_candidate_feasible": any(s[3] for s in scored),
        "v1": _summary(1),
        "v2": _summary(2),
    }


def run_greedy_fsd(truss, spec, goals, cons, max_steps, bounds, target_margin=1.05):
    """Classical fully-stressed design (the honest engineering baseline).

    Each iteration resizes EVERY member from its own utilisation: buckling FOS
    scales roughly as r^3 through the second moment of area, yielding FOS as the
    area, so the required factor is max((target/fos_b)^(1/3), target/fos_y).
    Over-designed members shrink, which is why FSD reaches feasibility at low
    mass -- the greedy critical-member loop can only ever add material.
    One FEA call per iteration, not per candidate.
    """
    state = fea(truss, goals)
    initial_mass = state["mass"]
    calls, taken = 1, 0
    best = (state["mass"], copy.deepcopy(truss), state) if cons.is_feasible(state) else None
    tgt_b = (cons.limit_for("fos_buckling") or 1.5) * target_margin
    tgt_y = (cons.limit_for("fos_yielding") or 1.5) * target_margin
    for _ in range(max_steps):
        dims = member_params(truss)
        factors = []
        for i, m in enumerate(truss.members):
            try:
                fb = float(m.fos_buckling)
                fy = float(m.fos_yielding)
            except (TypeError, ValueError):
                fb = fy = float("inf")
            need_b = (tgt_b / fb) ** (1.0 / 3.0) if math.isfinite(fb) and fb > 0 else 0.7
            need_y = (tgt_y / fy) if math.isfinite(fy) and fy > 0 else 0.7
            f = max(need_b, need_y)
            factors.append(min(2.0, max(0.7, f)))
        moved = False
        for i, f in enumerate(factors):
            if abs(f - 1.0) < 1e-3:
                continue
            cur = dims[i].get("r")
            if cur is None:
                continue
            lo, hi = bounds.get("r", (0.0, math.inf))
            f_eff = min(f, hi / cur) if cur > 0 else f
            f_eff = max(f_eff, lo / cur) if cur > 0 else f_eff
            if abs(f_eff - 1.0) < 1e-3:
                continue
            if apply_action(truss, f"SCALE_PARAM({i}, r, {f_eff:.6f})"):
                moved = True
        state = fea(truss, goals)
        calls += 1
        taken += 1
        if cons.is_feasible(state) and (best is None or state["mass"] < best[0]):
            best = (state["mass"], copy.deepcopy(truss), state)
        if not moved:
            break
    if best is not None:
        return best[1], best[2], calls, initial_mass, taken
    return truss, state, calls, initial_mass, taken


def run_lookahead(truss, spec, goals, cons, max_steps, phi_name, alpha, bounds, depth=1, beam=5,
                  stop_on_feasible=True, macros=True):
    """Greedy (or depth-2 beam) search maximising the chosen potential."""
    phi = PHI[phi_name]
    state = fea(truss, goals)
    initial_mass = state["mass"]
    calls = 1
    taken = 0
    best = (state["mass"], copy.deepcopy(truss), state) if cons.is_feasible(state) else None
    for _ in range(max_steps):
        if cons.is_feasible(state) and stop_on_feasible:
            break
        acts = candidate_actions(
            truss, bounds, macros=macros,
            target_fos_b=cons.limit_for("fos_buckling") or 1.5,
            target_fos_y=cons.limit_for("fos_yielding") or 1.5,
        )
        if not acts:
            break
        children = []
        for a in acts:
            out = step(truss, goals, a, cons)
            calls += 1
            if out is None:
                continue
            nt, ns = out
            children.append((phi(ns, cons, initial_mass, alpha), a, nt, ns))
        if not children:
            break
        # A candidate that reaches feasibility ends the episode immediately.
        feas = [c for c in children if cons.is_feasible(c[3])]
        if feas:
            feas.sort(key=lambda c: c[3]["mass"])
            if best is None or feas[0][3]["mass"] < best[0]:
                best = (feas[0][3]["mass"], copy.deepcopy(feas[0][2]), feas[0][3])
            if stop_on_feasible:
                _, _, truss, state = feas[0]
                taken += 1
                break
        children.sort(key=lambda c: c[0], reverse=True)
        if depth >= 2:
            rescored = []
            for val, a, nt, ns in children[:beam]:
                sub = candidate_actions(
                    nt, bounds, macros=macros,
                    target_fos_b=cons.limit_for("fos_buckling") or 1.5,
                    target_fos_y=cons.limit_for("fos_yielding") or 1.5,
                )
                best_child = -math.inf
                child_feasible = False
                for a2 in sub:
                    out2 = step(nt, goals, a2, cons)
                    calls += 1
                    if out2 is None:
                        continue
                    _, ns2 = out2
                    if cons.is_feasible(ns2):
                        child_feasible = True
                    best_child = max(best_child, phi(ns2, cons, initial_mass, alpha))
                if best_child == -math.inf:
                    best_child = val
                rescored.append((val + 0.99 * best_child + (5.0 if child_feasible else 0.0), a, nt, ns))
            rescored.sort(key=lambda c: c[0], reverse=True)
            children = rescored
        _, _, truss, state = children[0]
        taken += 1
        if cons.is_feasible(state) and (best is None or state["mass"] < best[0]):
            best = (state["mass"], copy.deepcopy(truss), state)
    if best is not None:
        return best[1], best[2], calls, initial_mass, taken
    return truss, state, calls, initial_mass, taken


def run_random(truss, spec, goals, cons, max_steps, alpha, bounds, seed=0):
    """Control: uniformly random legal sizing action each step."""
    import random
    rng = random.Random(seed)
    state = fea(truss, goals)
    initial_mass = state["mass"]
    calls, taken = 1, 0
    for _ in range(max_steps):
        if cons.is_feasible(state):
            break
        acts = candidate_actions(truss, bounds)
        if not acts:
            break
        out = step(truss, goals, rng.choice(acts), cons)
        calls += 1
        if out is None:
            continue
        truss, state = out
        taken += 1
    return truss, state, calls, initial_mass, taken


# ── driver ───────────────────────────────────────────────────────────────────

def run_problem(args_tuple):
    path, max_steps, alpha, policies, want_diag = args_tuple
    spec = load_problem(Path(path))
    if spec is None:
        return [], []
    pid = spec.get("problem_id", Path(path).stem)
    goals = spec.get("goals", {}) or {}
    bounds = param_bounds(spec)
    family = "mass" if goals.get("maximum_mass") else ("deflection" if goals.get("maximum_deflection") else "fos_only")

    base = make_truss(spec)
    s0 = fea(base, goals)
    cons = program_from_truss_spec(spec, initial_mass=s0["mass"])

    results, diags = [], []
    for policy in policies:
        t = copy.deepcopy(base)
        diag_sink = [] if (want_diag and policy == "greedy_critical") else None
        try:
            if policy == "greedy_critical":
                tf, sf, calls, m0, taken = run_greedy_critical(
                    t, spec, goals, cons, max_steps, diagnostics=diag_sink, alpha=alpha, bounds=bounds)
            elif policy == "greedy_fsd":
                tf, sf, calls, m0, taken = run_greedy_fsd(
                    t, spec, goals, cons, max_steps, bounds)
            elif policy == "random":
                tf, sf, calls, m0, taken = run_random(
                    t, spec, goals, cons, max_steps, alpha, bounds, seed=abs(hash(pid)) % 10000)
            elif policy.startswith("lookahead"):
                parts = policy.split("_")
                phi_name, dstr = parts[1], parts[2]
                minmass = "minmass" in parts
                # "legal": restrict the search to actions the ENVIRONMENT's grammar
                # can express -- one SCALE_PARAM, or one SCALE_MULTI_PARAM with a
                # single shared factor. The fully-stressed macro (a different factor
                # per member, applied between two simulator calls) is a compound move
                # the policy would need one turn per member to reproduce, so including
                # it makes any comparison against the LLM a budget comparison.
                legal = "legal" in parts
                tf, sf, calls, m0, taken = run_lookahead(
                    t, spec, goals, cons, max_steps, phi_name, alpha, bounds,
                    depth=int(dstr[1:]), stop_on_feasible=not minmass, macros=not legal)
            else:
                continue
        except Exception as exc:  # keep the sweep alive; record the failure
            results.append(asdict(RunResult(pid, policy, False, -1, float("nan"), s0["mass"],
                                            cons.objective_ref, float("nan"), float("nan"), 0, family))
                           | {"error": f"{type(exc).__name__}: {exc}"})
            continue
        results.append(asdict(RunResult(
            problem_id=pid, policy=policy, feasible=bool(cons.is_feasible(sf)),
            steps=taken, final_mass=sf["mass"], initial_mass=m0, mass_ref=cons.objective_ref,
            final_fos_b=sf["fos_buckling"], final_fos_y=sf["fos_yielding"],
            fea_calls=calls, family=family)))
        if diag_sink:
            for d in diag_sink:
                if d:
                    diags.append({"problem_id": pid, "family": family} | d)
    return results, diags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(DESIGNBENCH / "data/problems"))
    ap.add_argument("--out", default=str(PROJECT / "results/search_ladder"))
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--tau", type=float, default=0.05,
                    help="Hinge softness in relative units.")
    ap.add_argument("--feasibility-offset", type=float, default=0.0,
                    help="Rank any violating design below any satisfying one. On batteries this "
                         "took the ladder from 91.7%% to 100%% feasible with 33%% fewer sims; "
                         "this flag measures whether trusses need it too.")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--policies", default="greedy_critical,lookahead_v1_d1,lookahead_v2_d1,lookahead_v2_d2")
    ap.add_argument("--no-diagnostics", action="store_true")
    args = ap.parse_args()
    _PHI_OPTS["tau"] = args.tau
    _PHI_OPTS["feasibility_offset"] = args.feasibility_offset

    files = sorted(Path(args.problems).glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    policies = [p for p in args.policies.split(",") if p]
    tasks = [(str(f), args.max_steps, args.alpha, policies, not args.no_diagnostics) for f in files]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_results, all_diags = [], []
    # Stream each problem's rows as they complete. A depth-2 sweep over 580 problems
    # can outlive its walltime, and buffering everything until the end meant a
    # timeout discarded the entire run.
    inc = open(out_dir / "runs.jsonl", "w")
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            for i, (res, diag) in enumerate(pool.imap_unordered(run_problem, tasks), 1):
                all_results.extend(res)
                all_diags.extend(diag)
                for r in res:
                    inc.write(json.dumps(r) + "\n")
                inc.flush()
                if i % 5 == 0:
                    print(f"  {i}/{len(tasks)} problems  ({time.time()-t0:.0f}s)", flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            res, diag = run_problem(task)
            all_results.extend(res)
            all_diags.extend(diag)
            print(f"  {i}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)

    inc.close()  # rows were already streamed above
    with open(out_dir / "lookahead_diagnostics.jsonl", "w") as f:
        for d in all_diags:
            f.write(json.dumps(d) + "\n")

    print(f"\nDone in {time.time()-t0:.0f}s -> {out_dir}")
    summarize(all_results, all_diags)


def summarize(results, diags):
    import statistics as st
    from collections import defaultdict
    by = defaultdict(list)
    for r in results:
        by[(r["policy"], r["family"])].append(r)
        by[(r["policy"], "ALL")].append(r)

    print(f"\n{'policy':<20} {'family':<11} {'n':>4} {'feas':>7} {'mass/ref(feas)':>15} {'FEA':>7}")
    print("-" * 72)
    for (policy, family), rows in sorted(by.items()):
        n = len(rows)
        feas = [r for r in rows if r["feasible"]]
        mr = [r["final_mass"] / r["mass_ref"] for r in feas if r["mass_ref"] > 0 and math.isfinite(r["final_mass"])]
        calls = st.mean([r["fea_calls"] for r in rows]) if rows else 0
        print(f"{policy:<20} {family:<11} {n:>4} {len(feas)/max(n,1):>6.1%} "
              f"{(st.mean(mr) if mr else float('nan')):>15.3f} {calls:>7.0f}")

    if diags:
        print(f"\nLookahead-vs-feedback agreement over {len(diags)} visited states:")
        for tag in ("v1", "v2"):
            glob = [d[tag]["best_is_global"] for d in diags]
            mo = [d[tag] for d in diags if "best_member_targets_critical" in d[tag]]
            hits = [d["best_member_targets_critical"] for d in mo]
            ranks = [d["crit_best_rank"] for d in mo if d["crit_best_rank"] is not None]
            gaps = [d["crit_value_gap"] for d in mo if d["crit_value_gap"] is not None]
            margins = [d["member_margin"] for d in mo]
            print(f"  Phi_{tag}:")
            print(f"    global (all-member) action is the overall argmax : {sum(glob)/len(glob):>6.1%}")
            print(f"    among MEMBER actions, argmax targets critical mbr: "
                  f"{(sum(hits)/len(hits) if hits else float('nan')):>6.1%}  (n={len(hits)})")
            if ranks:
                top3 = sum(1 for r in ranks if r < 3) / len(ranks)
                print(f"    rank of best critical-member action           : "
                      f"median {st.median(ranks):>4.0f}, in top-3 {top3:.1%}")
            if gaps:
                print(f"    Phi lost by following the critical-member hint: "
                      f"mean {st.mean(gaps):.4f}, median {st.median(gaps):.4f}")
            if margins:
                print(f"    argmax margin over runner-up (member actions) : "
                      f"median {st.median(margins):.4f}")


if __name__ == "__main__":
    main()
