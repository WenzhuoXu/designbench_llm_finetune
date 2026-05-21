"""Tree-expansion utilities for one-step online posterior evaluation."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from llm_finetune.training.rl.posterior.features import (
    ACTION_CLASS_ORDER,
    action_class,
    shannon_entropy,
)


@dataclass
class CandidateAction:
    action: str
    score: float = 0.0
    logprob: float = 0.0
    source: str = "model"
    metadata: dict = field(default_factory=dict)

    @property
    def cls(self) -> str:
        return action_class(self.action)


@dataclass
class EvaluatedCandidate:
    candidate: CandidateAction
    next_state: dict
    immediate_reward: float
    total_value: float
    depth_used: int
    subtree_fea_calls: int = 0
    elapsed_s: float = 0.0


@dataclass
class TreeSearchResult:
    evaluated: list[EvaluatedCandidate]
    baseline: float
    tree_best_action: str
    tree_best_class: str
    class_entropy: float
    class_coverage: int
    adaptive_expanded: bool
    total_fea_calls: int
    total_elapsed_s: float
    trace: dict = field(default_factory=dict)

    @property
    def action_values(self) -> dict[str, float]:
        return {item.candidate.action: item.total_value for item in self.evaluated}


TransitionFn = Callable[[Any, str], tuple[Any, float, int]]
ContinuationFn = Callable[[Any, int], tuple[float, int, dict]]
SamplerFn = Callable[[Any, int, int, Optional[str]], list[CandidateAction]]


def _serialize_candidate(candidate: CandidateAction) -> dict:
    return asdict(candidate) | {"cls": candidate.cls}


def _serialize_state(node_or_state: Any) -> Any:
    state = getattr(node_or_state, "current_state", node_or_state)
    if isinstance(state, dict):
        return dict(state)
    return state


def stratified_select_candidates(
    candidates: list[CandidateAction],
    branching: int,
    *,
    ensure_full_class_coverage: bool = False,
) -> list[CandidateAction]:
    """Select candidates with optional class coverage preference."""
    if branching <= 0 or not candidates:
        return []

    unique: list[CandidateAction] = []
    seen_actions = set()
    for candidate in candidates:
        if candidate.action in seen_actions:
            continue
        seen_actions.add(candidate.action)
        unique.append(candidate)

    unique.sort(key=lambda c: (c.score, c.logprob), reverse=True)
    if not ensure_full_class_coverage:
        return unique[:branching]

    selected: list[CandidateAction] = []
    covered = set()
    for cls in ACTION_CLASS_ORDER:
        match = next((cand for cand in unique if cand.cls == cls), None)
        if match is not None:
            selected.append(match)
            covered.add(match.action)
        if len(selected) >= branching:
            return selected[:branching]

    for candidate in unique:
        if candidate.action in covered:
            continue
        selected.append(candidate)
        covered.add(candidate.action)
        if len(selected) >= branching:
            break
    return selected


def class_entropy(candidates: list[CandidateAction]) -> float:
    if not candidates:
        return 0.0
    by_class: dict[str, int] = {}
    for candidate in candidates:
        by_class[candidate.cls] = by_class.get(candidate.cls, 0) + 1
    total = sum(by_class.values())
    return shannon_entropy(count / total for count in by_class.values())


def evaluate_tree(
    *,
    root_node: Any,
    root_candidates: list[CandidateAction],
    transition_fn: TransitionFn,
    continuation_fn: ContinuationFn,
    sampler_fn: SamplerFn,
    depth: int,
    level1_branching: int,
    deeper_branching: int,
    gamma: float = 0.99,
    adaptive_entropy_threshold: float | None = None,
    stratified: bool = True,
) -> TreeSearchResult:
    start = time.perf_counter()
    entropy = class_entropy(root_candidates)
    adaptive_expanded = adaptive_entropy_threshold is None or entropy > adaptive_entropy_threshold
    first_branching = level1_branching if adaptive_expanded else min(level1_branching, 1)
    chosen = stratified_select_candidates(
        root_candidates,
        first_branching,
        ensure_full_class_coverage=stratified and first_branching >= len(ACTION_CLASS_ORDER),
    )

    evaluated: list[EvaluatedCandidate] = []
    evaluated_traces: list[dict] = []
    total_fea_calls = 0
    for candidate in chosen:
        next_node, immediate_reward, fea_calls = transition_fn(root_node, candidate.action)
        total_fea_calls += fea_calls
        child_start = time.perf_counter()
        if depth <= 0:
            continuation_value, child_calls, continuation_trace = continuation_fn(next_node, 0)
            depth_used = 0
        else:
            continuation_value, child_calls, continuation_trace = _recursive_value(
                node=next_node,
                depth=depth - 1,
                branching=deeper_branching,
                transition_fn=transition_fn,
                continuation_fn=continuation_fn,
                sampler_fn=sampler_fn,
                gamma=gamma,
                stratified=stratified,
            )
            depth_used = depth
        total_fea_calls += child_calls
        total_value = immediate_reward + gamma * continuation_value
        candidate_elapsed = time.perf_counter() - child_start
        evaluated.append(
            EvaluatedCandidate(
                candidate=candidate,
                next_state=getattr(next_node, "current_state", next_node),
                immediate_reward=immediate_reward,
                total_value=total_value,
                depth_used=depth_used,
                subtree_fea_calls=fea_calls + child_calls,
                elapsed_s=candidate_elapsed,
            )
        )
        evaluated_traces.append(
            {
                "candidate": _serialize_candidate(candidate),
                "immediate_reward": immediate_reward,
                "continuation_value": continuation_value,
                "total_value": total_value,
                "depth_used": depth_used,
                "subtree_fea_calls": fea_calls + child_calls,
                "elapsed_s": candidate_elapsed,
                "next_state": _serialize_state(next_node),
                "continuation_trace": continuation_trace,
            }
        )

    if not evaluated:
        return TreeSearchResult(
            evaluated=[],
            baseline=0.0,
            tree_best_action="",
            tree_best_class="UNKNOWN",
            class_entropy=entropy,
            class_coverage=0,
            adaptive_expanded=adaptive_expanded,
            total_fea_calls=0,
            total_elapsed_s=time.perf_counter() - start,
            trace={
                "root_state": _serialize_state(root_node),
                "root_candidates": [_serialize_candidate(candidate) for candidate in root_candidates],
                "selected_root_candidates": [],
                "evaluated_candidates": [],
                "class_entropy": entropy,
                "adaptive_expanded": adaptive_expanded,
                "selected_root_branching": first_branching,
            },
        )

    evaluated.sort(key=lambda item: item.total_value, reverse=True)
    baseline = sum(item.total_value for item in evaluated) / len(evaluated)
    best = evaluated[0]
    return TreeSearchResult(
        evaluated=evaluated,
        baseline=baseline,
        tree_best_action=best.candidate.action,
        tree_best_class=best.candidate.cls,
        class_entropy=entropy,
        class_coverage=len({item.candidate.cls for item in evaluated}),
        adaptive_expanded=adaptive_expanded,
        total_fea_calls=total_fea_calls,
        total_elapsed_s=time.perf_counter() - start,
        trace={
            "root_state": _serialize_state(root_node),
            "root_candidates": [_serialize_candidate(candidate) for candidate in root_candidates],
            "selected_root_candidates": [_serialize_candidate(candidate) for candidate in chosen],
            "evaluated_candidates": evaluated_traces,
            "class_entropy": entropy,
            "adaptive_expanded": adaptive_expanded,
            "selected_root_branching": first_branching,
        },
    )


def _recursive_value(
    *,
    node: Any,
    depth: int,
    branching: int,
    transition_fn: TransitionFn,
    continuation_fn: ContinuationFn,
    sampler_fn: SamplerFn,
    gamma: float,
    stratified: bool,
) -> tuple[float, int, dict]:
    if depth <= 0:
        return continuation_fn(node, 0)

    sampled = sampler_fn(node, branching, depth, None)
    chosen = stratified_select_candidates(
        sampled,
        branching,
        ensure_full_class_coverage=stratified and branching >= len(ACTION_CLASS_ORDER),
    )
    if not chosen:
        continuation_value, continuation_calls, continuation_trace = continuation_fn(node, depth)
        return (
            continuation_value,
            continuation_calls,
            {
                "mode": "fallback_continuation",
                "depth": depth,
                "state": _serialize_state(node),
                "sampled_candidates": [_serialize_candidate(candidate) for candidate in sampled],
                "selected_candidates": [],
                "continuation_trace": continuation_trace,
            },
        )

    best_value = -math.inf
    total_calls = 0
    branch_traces: list[dict] = []
    best_action = ""
    for candidate in chosen:
        next_node, immediate_reward, fea_calls = transition_fn(node, candidate.action)
        continuation_value, child_calls, child_trace = _recursive_value(
            node=next_node,
            depth=depth - 1,
            branching=branching,
            transition_fn=transition_fn,
            continuation_fn=continuation_fn,
            sampler_fn=sampler_fn,
            gamma=gamma,
            stratified=stratified,
        )
        total_calls += fea_calls + child_calls
        total_value = immediate_reward + gamma * continuation_value
        if total_value > best_value:
            best_value = total_value
            best_action = candidate.action
        branch_traces.append(
            {
                "candidate": _serialize_candidate(candidate),
                "immediate_reward": immediate_reward,
                "continuation_value": continuation_value,
                "total_value": total_value,
                "subtree_fea_calls": fea_calls + child_calls,
                "next_state": _serialize_state(next_node),
                "child_trace": child_trace,
            }
        )

    if best_value == -math.inf:
        continuation_value, continuation_calls, continuation_trace = continuation_fn(node, depth)
        return (
            continuation_value,
            continuation_calls,
            {
                "mode": "empty_recursive_fallback",
                "depth": depth,
                "state": _serialize_state(node),
                "sampled_candidates": [_serialize_candidate(candidate) for candidate in sampled],
                "selected_candidates": [_serialize_candidate(candidate) for candidate in chosen],
                "continuation_trace": continuation_trace,
            },
        )
    return (
        best_value,
        total_calls,
        {
            "mode": "recursive",
            "depth": depth,
            "branching": branching,
            "state": _serialize_state(node),
            "sampled_candidates": [_serialize_candidate(candidate) for candidate in sampled],
            "selected_candidates": [_serialize_candidate(candidate) for candidate in chosen],
            "best_action": best_action,
            "branches": branch_traces,
        },
    )
