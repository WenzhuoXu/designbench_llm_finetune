"""Harvest training data from phi_select episodes.

Two artifacts, structured (rendering to chat format is a downstream step):

  pairs.jsonl        same-state preference pairs from the ranked candidate
                     lists. Both sides share the state; they differ only in
                     the action, so a preference cannot be won by prose. Pairs
                     are emitted from FAILURE states too (T6 lesson: this is
                     the data format that transfers the decision rule).
  trajectories.jsonl replay-verified expert trajectories with per-turn
                     search-derived justification fields.
"""

from __future__ import annotations

import json
import math
from typing import IO, List, Optional

from .domain import Domain, PhiEvaluator, apply_action
from .ladder import EpisodeResult

MAX_PAIRS_PER_STATE = 4
MAX_PAIRS_PER_PROBLEM = 60


def _finite_dict(d: dict) -> bool:
    for v in d.values():
        if isinstance(v, float) and not math.isfinite(v):
            return False
    return True


def _action_key(action: dict) -> str:
    return json.dumps(action, sort_keys=True, default=_json_default)


def harvest_pairs(result: EpisodeResult, *, k_sigma: float = 2.0,
                  eps: float = 1e-9, dedup_time_key: Optional[str] = None) -> (list, dict):
    """Emit preference pairs from an episode's recorded turns.

    Pairing scheme per state: (best, worst), (best, median), and adjacent
    top pairs - each emitted only when the Phi gap clears the noise band of
    the two sides and both sides are envelope-valid.

    dedup_time_key (time-stepping domains, docs/corpus_native_domains.md):
    the state metric that carries simulated time. When given, a pair whose
    (problem, that metric, chosen action, rejected action) was already
    emitted is dropped and counted in stats["n_dropped_duplicate"]: a turn
    that re-recorded the same step would otherwise harvest the same pairs
    again. v1/v2 callers leave it None and their stats are unchanged."""
    pairs = []
    stats = {"n_emitted": 0, "n_rejected_gap": 0,
             "n_rejected_invalid": 0, "n_rejected_nan": 0}
    seen = None
    if dedup_time_key is not None:
        stats["n_dropped_duplicate"] = 0
        seen = set()

    for rec in result.turn_records:
        ranked = rec.top + rec.spread
        if len(ranked) < 2:
            continue
        combos = []
        if len(rec.top) >= 1 and rec.spread:
            combos.append((rec.top[0], rec.spread[-1]))   # best vs worst
            combos.append((rec.top[0], rec.spread[0]))    # best vs median
        for a, b in zip(rec.top, rec.top[1:]):            # adjacent top pairs
            combos.append((a, b))

        emitted_here = 0
        for hi, lo in combos:
            if emitted_here >= MAX_PAIRS_PER_STATE:
                break
            if hi is lo or hi["action"] == lo["action"]:
                continue
            if not (_finite_dict(hi["metrics"]) and _finite_dict(lo["metrics"])):
                stats["n_rejected_nan"] += 1
                continue
            if hi["valid_frac"] < 0.8 or lo["valid_frac"] < 0.8:
                stats["n_rejected_invalid"] += 1
                continue
            gap = hi["phi_mean"] - lo["phi_mean"]
            band = k_sigma * math.sqrt(hi["phi_se"] ** 2 + lo["phi_se"] ** 2)
            if gap <= max(band, eps):
                stats["n_rejected_gap"] += 1
                continue
            if seen is not None:
                key = (result.problem_id, rec.state_metrics.get(dedup_time_key),
                       _action_key(hi["action"]), _action_key(lo["action"]))
                if key in seen:
                    stats["n_dropped_duplicate"] += 1
                    continue
                seen.add(key)
            pairs.append({
                "problem_id": result.problem_id,
                "turn": rec.turn,
                "state_metrics": rec.state_metrics,
                "state_phi": rec.state_phi,
                "binding_constraint": rec.binding_constraint,
                "cand_count": rec.cand_count,
                "chosen": hi,
                "rejected": lo,
                "phi_gap": gap,
                "noise_band": band,
            })
            stats["n_emitted"] += 1
            emitted_here += 1
        if len(pairs) >= MAX_PAIRS_PER_PROBLEM:
            break
    return pairs, stats


def replay_trajectory(domain: Domain, spec: dict, ev: PhiEvaluator,
                      result: EpisodeResult, *, tol: float = 1e-6) -> bool:
    """Re-execute the action sequence with the SAME seeds and confirm the
    final aggregated metrics reproduce. Deterministic domains must match to
    tol; stochastic ones must match exactly under identical seed streams."""
    try:
        st = domain.load(spec)
        for a in result.actions:
            apply_action(domain, st, a)     # legacy {i,p,v,v_old} or {verb,args}
        if not result.seeds:
            return not result.actions
        final = ev.score(st, seed0=result.seeds[-1])
        if result.turn_records:
            pass  # per-turn checks are covered by the final-state comparison
        recorded_obj = result.final_objective
        replayed_obj = final.agg_metrics.get(ev.program.objective_key)
        if recorded_obj is None or replayed_obj is None:
            return final.feasible == result.final_feasible
        scale = max(1.0, abs(recorded_obj))
        return (abs(float(replayed_obj) - float(recorded_obj)) <= tol * scale
                and final.feasible == result.final_feasible)
    except Exception:
        return False


def harvest_trajectory(domain: Domain, spec: dict, ev: PhiEvaluator,
                       result: EpisodeResult) -> (Optional[dict], bool):
    """Build one replay-verified trajectory record, or (None, replay_ok)."""
    replay_ok = replay_trajectory(domain, spec, ev, result)
    if not (result.feasible_ever and result.actions):
        return None, replay_ok
    if not replay_ok:
        return None, False
    turns = []
    for rec in result.turn_records:
        if rec.chosen is None:
            continue
        turns.append({
            "turn": rec.turn,
            "state_metrics": rec.state_metrics,
            "state_phi": rec.state_phi,
            "binding_constraint": rec.binding_constraint,
            "cand_count": rec.cand_count,
            "action": rec.chosen,
            "margin": rec.margin,
            "noise_band": rec.noise_band,
            "measurable": rec.measurable,
        })
    return {
        "problem_id": result.problem_id,
        "final_feasible": result.final_feasible,
        "final_objective": result.final_objective,
        "best_feasible_objective": result.best_feasible_objective,
        "reference": result.reference,
        "turns": turns,
        "actions": result.actions,
        "seeds": result.seeds,
        "sim_calls": result.sim_calls,
    }, True


def write_jsonl(fh: IO, rows: List[dict]) -> None:
    for row in rows:
        fh.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)
