"""The three inclusion gates. A domain-variant's data enters the dataset only
after all three pass; per-problem failures inside a passing variant are
excluded individually and counted.

G1 distribution-variance gate  - the generated problem SET is diverse enough
G2 phi-measurable gate         - Phi can actually rank candidates (margin
                                 outside the noise band; not constant)
G3 data-quality gate           - harvested pairs/trajectories are clean,
                                 replayable, and their Phi gaps clear the band
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Optional


def _cv(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    if abs(mean) < 1e-12:
        return None
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(max(var, 0.0)) / abs(mean)


def _spec_hash(spec: dict) -> str:
    body = {k: v for k, v in spec.items() if k != "problem_id"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def distribution_variance_gate(
    specs: List[dict],
    initial_violations: List[float],
    initial_objectives: List[float],
    initially_infeasible: List[bool],
    *,
    min_families: int = 3,
    min_family_frac: float = 0.08,
    min_violation_cv: float = 0.25,
    min_objective_cv: float = 0.10,
    min_infeasible_frac: float = 0.5,
) -> dict:
    """G1: the problem set must span, not cluster.

    Checks: no duplicate specs; >= min_families families each carrying a
    non-trivial share; spread in initial violation (the difficulty proxy) and
    in the initial objective; and most problems must START infeasible (a set
    that starts solved generates no restoration data).
    """
    n = len(specs)
    hashes = [_spec_hash(s) for s in specs]
    n_dupes = n - len(set(hashes))

    families: Dict[str, int] = {}
    for s in specs:
        fam = str(s.get("family", "unknown"))
        families[fam] = families.get(fam, 0) + 1
    family_fracs = {f: c / max(n, 1) for f, c in families.items()}
    thin_families = [f for f, frac in family_fracs.items() if frac < min_family_frac]

    viol_cv = _cv(initial_violations)
    obj_cv = _cv(initial_objectives)
    infeasible_frac = (sum(1 for b in initially_infeasible if b) / max(n, 1))

    # tercile occupancy of the difficulty proxy
    finite_v = sorted(v for v in initial_violations if v is not None and math.isfinite(v))
    terciles_ok = False
    if len(finite_v) >= 9:
        lo = finite_v[len(finite_v) // 3]
        hi = finite_v[(2 * len(finite_v)) // 3]
        terciles_ok = lo < hi  # a genuinely graded difficulty axis

    checks = {
        "no_duplicates": n_dupes == 0,
        "enough_families": len(families) >= min_families and not thin_families,
        "violation_spread": viol_cv is not None and viol_cv >= min_violation_cv,
        "objective_spread": obj_cv is not None and obj_cv >= min_objective_cv,
        "mostly_infeasible_start": infeasible_frac >= min_infeasible_frac,
        "difficulty_terciles_populated": terciles_ok,
    }
    return {
        "gate": "G1_distribution_variance",
        "passed": all(checks.values()),
        "checks": checks,
        "n_problems": n,
        "n_duplicates": n_dupes,
        "families": families,
        "thin_families": thin_families,
        "initial_violation_cv": viol_cv,
        "initial_objective_cv": obj_cv,
        "initially_infeasible_frac": infeasible_frac,
    }


def phi_measurable_gate(
    per_problem: List[dict],
    *,
    min_state_frac: float = 0.6,
    min_problem_frac: float = 0.8,
) -> dict:
    """G2: Phi must be able to RANK. A problem is measurable when >=
    min_state_frac of its visited states had a top-2 Phi margin outside the
    noise band (and Phi non-constant across candidates). Problems where Phi is
    constant at the initial state are degenerate (the truss-mechanism analog)
    and are excluded outright. The variant passes when >= min_problem_frac of
    non-degenerate problems are measurable."""
    rows = []
    for row in per_problem:
        frac = row.get("measurable_frac")
        rows.append({
            "problem_id": row.get("problem_id"),
            "measurable_frac": frac,
            "degenerate": bool(row.get("degenerate")),
            "measurable": (frac is not None and frac >= min_state_frac
                           and not row.get("degenerate")),
        })
    degenerate = [r["problem_id"] for r in rows if r["degenerate"]]
    considered = [r for r in rows if not r["degenerate"]]
    measurable = [r["problem_id"] for r in considered if r["measurable"]]
    frac = len(measurable) / max(len(considered), 1)
    return {
        "gate": "G2_phi_measurable",
        "passed": bool(considered) and frac >= min_problem_frac,
        "measurable_problem_frac": frac,
        "n_considered": len(considered),
        "n_measurable": len(measurable),
        "degenerate_problems": degenerate,
        "excluded_problems": degenerate + [r["problem_id"] for r in considered
                                           if not r["measurable"]],
        "included_problems": measurable,
    }


def data_quality_gate(
    pair_stats: dict,
    traj_stats: dict,
    *,
    min_pair_yield: float = 0.7,
    min_traj_replay_ok: float = 0.95,
) -> dict:
    """G3: harvested data must be clean.

    pair_stats: {n_emitted, n_rejected_gap, n_rejected_invalid, n_rejected_nan}
    traj_stats: {n_traj, n_replay_ok, n_replay_failed, n_nan}
    A pair is emitted only when its Phi gap clears the noise band and both
    sides sit inside the validity envelope; a trajectory only when a full
    seeded replay reproduces its final metrics. This gate checks the YIELD of
    those filters - a domain rejecting most of its own harvest is telling us
    its noise band or envelope is mis-set, and its data is not trusted."""
    n_considered = (pair_stats.get("n_emitted", 0)
                    + pair_stats.get("n_rejected_gap", 0)
                    + pair_stats.get("n_rejected_invalid", 0)
                    + pair_stats.get("n_rejected_nan", 0))
    pair_yield = pair_stats.get("n_emitted", 0) / max(n_considered, 1)
    n_traj = traj_stats.get("n_traj", 0)
    replay_ok_frac = traj_stats.get("n_replay_ok", 0) / max(n_traj, 1)
    checks = {
        "pair_yield": pair_yield >= min_pair_yield,
        "no_nan_pairs": pair_stats.get("n_rejected_nan", 0) == 0,
        "trajectories_replay": n_traj == 0 or replay_ok_frac >= min_traj_replay_ok,
        "some_data": pair_stats.get("n_emitted", 0) > 0,
    }
    return {
        "gate": "G3_data_quality",
        "passed": all(checks.values()),
        "checks": checks,
        "pair_yield": pair_yield,
        "pair_stats": pair_stats,
        "traj_replay_ok_frac": replay_ok_frac if n_traj else None,
        "traj_stats": traj_stats,
    }
