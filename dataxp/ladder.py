"""Non-LLM policy ladder over a Domain: the acceptance test every
domain-variant runs BEFORE any training data is harvested, plus the envelope
red-team (the D9 lesson made a gate).

Policies:
  random      no signal
  greedy      the domain's depth-0 feedback hint (coupling anchor)
  blind       enumeration, halt at first feasible candidate (no ranking)
  phi_select  argmax E[Phi] over enumerated candidates (depth 1)
  phi_d2      two-step lookahead over the top-M depth-1 candidates
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from dataclasses import replace as dc_replace

from .domain import (
    SET_PARAM,
    Candidate,
    Domain,
    Move,
    PhiEvaluator,
    PhiScore,
    apply_action,
    as_move,
    stable_seed,
)

EPS = 1e-9


@dataclass
class ScoredCandidate:
    """A scored legal move. `move` is a Move; for v1/v2 domains it is the
    SET_PARAM move of the Candidate, and as_dict() is unchanged."""
    move: Move
    score: PhiScore

    def as_dict(self) -> dict:
        return {
            "action": self.move.as_dict(),
            "phi_mean": float(self.score.mean),
            "phi_se": float(self.score.se),
            "n_rep": int(self.score.n_rep),
            "feasible": bool(self.score.feasible),
            "valid_frac": float(self.score.valid_frac),
            "metrics": {k: float(v) for k, v in self.score.agg_metrics.items()},
        }


@dataclass
class TurnRecord:
    turn: int
    state_metrics: dict
    state_phi: float
    cand_count: int
    chosen: Optional[dict]
    measurable: bool
    margin: float
    noise_band: float
    hint_match: Optional[bool]
    binding_constraint: Optional[str]
    top: List[dict] = field(default_factory=list)     # top-6 scored candidates
    spread: List[dict] = field(default_factory=list)  # median + worst


@dataclass
class EpisodeResult:
    policy: str
    problem_id: str
    feasible_ever: bool
    final_feasible: bool
    final_objective: Optional[float]
    best_feasible_objective: Optional[float]
    reference: Optional[float]
    turns_used: int
    sim_calls: int
    degenerate: bool
    measurable_frac: Optional[float]
    hint_match_frac: Optional[float]
    turn_records: List[TurnRecord] = field(default_factory=list)
    actions: List[dict] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "policy": self.policy,
            "problem_id": self.problem_id,
            "feasible_ever": self.feasible_ever,
            "final_feasible": self.final_feasible,
            "final_objective": self.final_objective,
            "best_feasible_objective": self.best_feasible_objective,
            "reference": self.reference,
            "obj_over_ref": (
                float(self.best_feasible_objective) / float(self.reference)
                if (self.best_feasible_objective is not None
                    and self.reference not in (None, 0)) else None),
            "turns_used": self.turns_used,
            "sim_calls": self.sim_calls,
            "degenerate": self.degenerate,
            "measurable_frac": self.measurable_frac,
            "hint_match_frac": self.hint_match_frac,
        }


def _apply(domain: Domain, st, move: Move):
    child = domain.clone(st)
    domain.apply(child, move)
    return child


def _legal(domain: Domain, st, rng, max_candidates=None) -> List[Move]:
    """domain.legal_moves, forwarding max_candidates only when asked (the
    default call is the exact v1/v2 enumerate_candidates call)."""
    if max_candidates is None:
        return list(domain.legal_moves(st, rng))
    return list(domain.legal_moves(st, rng, max_candidates=max_candidates))


def _binding_constraint(program, metrics) -> Optional[str]:
    viol = program.violations(metrics, tau=0.0)
    if not viol:
        return None
    key, value = max(viol.items(), key=lambda kv: kv[1])
    return key if value > 0 else None


def run_episode(domain: Domain, spec: dict, ev: PhiEvaluator, policy: str,
                *, turns: int = 20, record: bool = False,
                base_seed: int = 0) -> EpisodeResult:
    pid = str(spec.get("problem_id", "unknown"))
    rng = np.random.default_rng(stable_seed(pid, policy, base_seed))
    st = domain.load(spec)
    sim0 = ev.sim_calls

    cur = ev.score(st, seed0=stable_seed(pid, policy, "s", 0))
    objective_key = ev.program.objective_key
    feasible_ever = cur.feasible
    best_obj = _obj_of(cur, objective_key) if cur.feasible else None

    turn_records: List[TurnRecord] = []
    actions: List[dict] = []
    seeds: List[int] = []
    degenerate = False
    measurable_flags: List[bool] = []
    hint_flags: List[bool] = []
    no_improve = 0
    t = 0

    for t in range(turns):
        seed_t = stable_seed(pid, policy, "s", t + 1)
        chosen: Optional[Move] = None

        if policy == "random":
            cands = _legal(domain, st, rng)
            if cands:
                chosen = cands[int(rng.integers(len(cands)))]

        elif policy == "greedy":
            # v2 domains with typed moves (and native-verb domains) may name
            # the greedy move directly, as a Candidate or a Move
            hm = getattr(domain, "hint_move", None)
            if callable(hm):
                chosen = as_move(hm(st))
            h = None if chosen is not None else domain.hint(st)
            if h is not None:
                i, p, direction = h
                curval = domain.get(st, i, p)
                if curval is not None and math.isfinite(curval):
                    f = max(domain.factors) if direction == "up" else min(domain.factors)
                    lo, hi = domain.bounds(st, p)
                    v = min(hi, max(lo, curval * f))
                    if abs(v - curval) > 1e-12 * max(1.0, abs(curval)):
                        chosen = Move.from_candidate(
                            Candidate(i=i, p=p, v=float(v), v_old=float(curval)))
            if chosen is None:
                cands = _legal(domain, st, rng)
                if cands:
                    chosen = cands[int(rng.integers(len(cands)))]

        elif policy == "blind":
            cands = _legal(domain, st, rng)
            for ci, cand in enumerate(cands):
                sc = ev.score(_apply(domain, st, cand),
                              seed0=stable_seed(pid, policy, t, ci))
                if sc.feasible:
                    chosen = cand
                    break
            if chosen is None and cands:
                chosen = cands[int(rng.integers(len(cands)))]

        elif policy in ("phi_select", "phi_d2"):
            cands = _legal(domain, st, rng)
            if not cands:
                break
            scored = []
            for ci, cand in enumerate(cands):
                sc = ev.score(_apply(domain, st, cand),
                              seed0=stable_seed(pid, policy, t, ci))
                scored.append(ScoredCandidate(move=cand, score=sc))
            scored.sort(key=lambda s: s.score.mean, reverse=True)

            # adaptive refinement of the top two until the margin is decided
            while (len(scored) >= 2 and domain.stochastic
                   and not ev.decided(scored[0].score, scored[1].score)
                   and (scored[0].score.n_rep < ev.n_rep_max
                        or scored[1].score.n_rep < ev.n_rep_max)):
                for k in (0, 1):
                    s = scored[k]
                    if s.score.n_rep < ev.n_rep_max:
                        s.score = ev.refine(
                            _apply(domain, st, s.move), s.score,
                            seed0=stable_seed(pid, policy, t, "ref", k))
                scored.sort(key=lambda s: s.score.mean, reverse=True)

            # The measurability statistic (top vs median) has to be estimated
            # well enough to be trusted: "unmeasurable" must mean the state
            # cannot be decided at the MAXIMUM affordable replicate budget, not
            # merely at the minimum. Scored at n_rep_min the standard errors
            # swamp the gap, and the noisy variants reported ~0 measurable
            # states and emitted almost no pairs - a calibration artefact, not
            # a property of those domains.
            if domain.stochastic and len(scored) >= 3:
                for _ in range(4):
                    mid_i = len(scored) // 2
                    a_s, b_s = scored[0].score, scored[mid_i].score
                    band_m = 2.0 * math.sqrt(a_s.se ** 2 + b_s.se ** 2)
                    if abs(a_s.mean - b_s.mean) > max(band_m, EPS):
                        break
                    if a_s.n_rep >= ev.n_rep_max and b_s.n_rep >= ev.n_rep_max:
                        break
                    for k in (0, mid_i):
                        s = scored[k]
                        if s.score.n_rep < ev.n_rep_max:
                            s.score = ev.refine(
                                _apply(domain, st, s.move), s.score,
                                seed0=stable_seed(pid, policy, t, "meas", k))
                    scored.sort(key=lambda s: s.score.mean, reverse=True)

            if policy == "phi_d2" and len(scored) > 1:
                top_m = scored[:6]
                for k, s in enumerate(top_m):
                    child = _apply(domain, st, s.move)
                    sub = _legal(domain, child, rng, max_candidates=24)
                    best_child = s.score.mean
                    for cj, cand2 in enumerate(sub):
                        sc2 = ev.score(_apply(domain, child, cand2),
                                       seed0=stable_seed(pid, policy, t, k, cj))
                        best_child = max(best_child, sc2.mean)
                    s.d2 = best_child  # depth-2 value; depth-1 stats stay in s.score
                top_m.sort(key=lambda s: getattr(s, "d2", s.score.mean), reverse=True)
                scored = top_m + scored[6:]

            spread = max(s.score.mean for s in scored) - min(s.score.mean for s in scored)
            if len(scored) >= 2:
                margin = scored[0].score.mean - scored[1].score.mean
                band = 2.0 * math.sqrt(scored[0].score.se ** 2 + scored[1].score.se ** 2)
            else:
                margin, band = float("inf"), 0.0
            # Measurability asks whether Phi carries RANKING SIGNAL at this
            # state, which is the top candidate against a TYPICAL one - not the
            # top two. Symmetric structures (a 4-fold symmetric tower, a
            # balanced grid) routinely make the top two candidates exactly
            # equivalent, and a tie between two equally-good actions is a free
            # decision, not an unrankable state. The top-2 margin is still
            # recorded: it is the reliability statistic that gates PAIR
            # emission, where a tie genuinely carries no preference.
            mid_score = scored[len(scored) // 2].score
            top_gap = scored[0].score.mean - mid_score.mean
            top_band = 2.0 * math.sqrt(scored[0].score.se ** 2 + mid_score.se ** 2)
            measurable = spread > EPS and top_gap > max(top_band, EPS)
            measurable_flags.append(measurable)
            if t == 0 and spread <= EPS:
                degenerate = True

            h = domain.hint(st)
            if h is not None:
                top = scored[0].move
                hint_flags.append(top.verb == SET_PARAM and top.args["i"] == h[0]
                                  and top.args["p"] == h[1])

            valid = [s for s in scored if s.score.valid_frac >= ev.valid_frac_min]
            pick = (valid or scored)[0]
            inc_mean = cur.mean
            if domain.stochastic and getattr(domain, "rescore_incumbent", False):
                # v2 stochastic domains: compare against a FRESH estimate of
                # the incumbent at the pick's replicate count. The recorded
                # incumbent score is one draw; a lucky draw otherwise stalls
                # the search for many turns (inventory2: 2 of 5 failures).
                # cur itself is unchanged, so replay stays exact.
                inc_mean = ev.score(st, seed0=stable_seed(pid, policy, "inc", t),
                                    n_rep=max(cur.n_rep, pick.score.n_rep)).mean
            improve = pick.score.mean > inc_mean + EPS
            # Time-stepping domains (advances_time): when no child improves on
            # the incumbent, the null move (grid2op: DO_NOTHING) is applied so
            # the world advances instead of the same state being scored and
            # recorded again next turn. It is recorded as the turn's action so
            # the trajectory's turns and actions stay aligned for replay. v1/v2
            # domains have no null move and keep the state, as before.
            # (not on the stop branch below - feasible and a second non-improving
            # turn - where nothing is applied: the record then says None as for
            # v1/v2, so a trajectory never ends on an unexecuted null move.)
            null: Optional[Move] = None
            if (not improve and getattr(domain, "advances_time", False)
                    and not (cur.feasible and no_improve + 1 >= 2)):
                null = as_move(domain.null_move(st))
            if record:
                mid = scored[len(scored) // 2]
                turn_records.append(TurnRecord(
                    turn=t,
                    state_metrics={k: float(v) for k, v in cur.agg_metrics.items()},
                    state_phi=float(cur.mean),
                    cand_count=len(scored),
                    chosen=(pick.move.as_dict() if improve
                            else (null.as_dict() if null is not None else None)),
                    measurable=measurable,
                    margin=float(margin),
                    noise_band=float(band),
                    hint_match=(hint_flags[-1] if h is not None else None),
                    binding_constraint=_binding_constraint(ev.program, cur.agg_metrics),
                    top=[s.as_dict() for s in scored[:6]],
                    spread=[mid.as_dict(), scored[-1].as_dict()],
                ))
            if improve:
                chosen = pick.move
                no_improve = 0
            else:
                no_improve += 1
                if cur.feasible and no_improve >= 2:
                    break
                chosen = null
        else:
            raise ValueError("unknown policy %r" % policy)

        if chosen is not None:
            st = _apply(domain, st, chosen)
            actions.append(chosen.as_dict())
            seeds.append(seed_t)
            cur = ev.score(st, seed0=seed_t)
            if cur.feasible:
                feasible_ever = True
                obj = _obj_of(cur, objective_key)
                if obj is not None and (best_obj is None or obj < best_obj):
                    best_obj = obj
            if policy == "blind" and cur.feasible:
                break

    return EpisodeResult(
        policy=policy,
        problem_id=pid,
        feasible_ever=feasible_ever,
        final_feasible=cur.feasible,
        final_objective=_obj_of(cur, objective_key),
        best_feasible_objective=best_obj,
        reference=_ref_of(domain, spec),
        turns_used=t + 1,
        sim_calls=ev.sim_calls - sim0,
        degenerate=degenerate,
        measurable_frac=(sum(measurable_flags) / len(measurable_flags)
                         if measurable_flags else None),
        hint_match_frac=(sum(hint_flags) / len(hint_flags)
                         if hint_flags else None),
        turn_records=turn_records,
        actions=actions,
        seeds=seeds,
    )


def _obj_of(score: PhiScore, objective_key: Optional[str]) -> Optional[float]:
    if objective_key is None:
        return None
    v = score.agg_metrics.get(objective_key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _ref_of(domain: Domain, spec: dict) -> Optional[float]:
    try:
        r = domain.reference(spec)
        return float(r) if r is not None and math.isfinite(float(r)) else None
    except Exception:
        return None


def redteam_envelope(domain: Domain, spec: dict, *, turns: int = 20) -> dict:
    """Run phi_select with the validity envelope STRIPPED from the potential,
    then judge the end state under the full program. Any run that ends
    'feasible' by the stripped program but invalid under the envelope is a
    caught exploit; a domain whose stripped search finds none has either a
    tight parameterisation or an untested envelope - both worth knowing."""
    ev = PhiEvaluator(domain, spec)
    stripped = dc_replace(ev.program, validity_bounds=(),
                          objective_must_be_positive=False)
    full_program = ev.program
    ev.program = stripped
    result = run_episode(domain, spec, ev, "phi_select", turns=turns,
                         base_seed=13)
    st = domain.load(spec)
    for a in result.actions:
        apply_action(domain, st, a)
    final = domain.evaluate(domain.clone(st), seed=stable_seed("redteam", spec.get("problem_id")))
    stripped_feasible = stripped.is_feasible(final)
    valid_under_envelope = full_program.is_valid(final)
    return {
        "problem_id": str(spec.get("problem_id")),
        "stripped_feasible": bool(stripped_feasible),
        "valid_under_envelope": bool(valid_under_envelope),
        "exploit_caught": bool(stripped_feasible and not valid_under_envelope),
        "final_metrics": {k: float(v) for k, v in final.items()
                          if isinstance(v, (int, float)) and math.isfinite(float(v))},
    }
