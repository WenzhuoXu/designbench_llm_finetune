import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math
from dataclasses import dataclass

import pytest

from llm_finetune.data.processors.chat_formatter import ChatFormatter, ThinkingMode
from llm_finetune.training.rl.posterior.features import (
    ACTION_CLASS_ORDER,
    action_class,
    action_locality,
    difficulty_score,
    extract_state_features,
)
from llm_finetune.training.rl.posterior.potential import compute_potential, compute_step_reward
from llm_finetune.training.rl.posterior.state_context import PosteriorStateContext, PosteriorTurn
from llm_finetune.training.rl.posterior.tree_expansion import (
    CandidateAction,
    evaluate_tree,
    stratified_select_candidates,
)


class DummyTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=True, return_tensors=None):
        text = "\n".join(f"{m['role']}:{m['content']}" for m in conversation)
        if not tokenize:
            return text
        ids = [ord(ch) % 255 for ch in text]
        if return_tensors == "pt":
            import torch
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(chr(tid) for tid in token_ids)


def test_extract_state_features_matches_spec_shape():
    state = {
        "mass": 180.0,
        "fos_buckling": 1.2,
        "fos_yielding": 1.7,
        "deflection": 0.012,
        "is_feasible": False,
        "member_dimensions": {"0": {}, "1": {}},
        "step": 3,
    }
    features = extract_state_features(state, initial_mass=200.0, mean_horizon=9.0)
    assert len(features) == 14
    assert features[0] == pytest.approx(1.2 / 1.5)
    assert features[3] == pytest.approx(180.0 / 200.0)
    assert features[12] == 2.0
    assert features[13] == pytest.approx(3 / 9)


def test_action_taxonomy_helpers():
    assert action_class("SCALE_PARAM(3, radius, 1.1)") == "SCALE_PARAM"
    assert action_locality("MOVE_JOINT(1, [0,0], [1,0])") == "mid"
    assert action_locality("ADD_MEMBER(0, 1, mat, shape, 0.1, 0.01)") == "non_local"


def test_difficulty_score_increases_for_bad_constraint_state():
    easy = {"fos_buckling": 1.8, "fos_yielding": 1.7, "deflection": 0.005}
    hard = {"fos_buckling": 0.7, "fos_yielding": 0.8, "deflection": 0.02}
    assert difficulty_score(hard) > difficulty_score(easy)


def test_potential_and_step_reward_favor_better_state():
    prev_state = {
        "mass": 200.0,
        "fos_buckling": 0.8,
        "fos_yielding": 0.9,
        "deflection": 0.02,
    }
    next_state = {
        "mass": 190.0,
        "fos_buckling": 1.3,
        "fos_yielding": 1.4,
        "deflection": 0.012,
    }
    assert compute_potential(next_state, initial_mass=200.0) > compute_potential(prev_state, initial_mass=200.0)
    assert compute_step_reward(prev_state, next_state, initial_mass=200.0) > 0.0


def test_state_context_rebuilds_intermediate_messages():
    formatter = ChatFormatter(
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.QWEN3,
        model_id="Qwen/Qwen3-14B",
    )
    context = PosteriorStateContext(
        problem_spec={"problem_id": "p1", "description": "Test problem", "goals": {}},
        current_state={"mass": 10.0, "step": 1},
        initial_state={"mass": 10.0, "step": 0},
        action_history=["SCALE_PARAM(0, radius, 1.1)"],
        turns=[
            PosteriorTurn(
                action="SCALE_PARAM(0, radius, 1.1)",
                thinking="Buckling is low.",
                fea_result={"mass": 10.5, "fos_buckling": 1.0, "fos_yielding": 1.1, "is_feasible": False},
            )
        ],
        prompt_text="PROBLEM: Test",
    )
    messages = context.build_messages(formatter)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert any("FEA Result" in msg["content"] for msg in messages if msg["role"] == "user")


def test_stratified_selection_prefers_class_coverage():
    candidates = [
        CandidateAction("SCALE_PARAM(0, radius, 1.1)", score=0.9),
        CandidateAction("SCALE_PARAM(1, radius, 1.2)", score=0.8),
        CandidateAction("ADD_MEMBER(0, 1, M, P, 0.1, 0.01)", score=0.7),
        CandidateAction("REMOVE_MEMBER(2)", score=0.6),
        CandidateAction("MOVE_JOINT(1, [0,0], [1,0])", score=0.5),
    ]
    selected = stratified_select_candidates(candidates, 4, ensure_full_class_coverage=True)
    assert len(selected) == 4
    assert len({candidate.cls for candidate in selected}) == 4


@dataclass
class DummyNode:
    name: str
    current_state: dict


GRAPH = {
    ("root", "SCALE_PARAM(0, radius, 1.1)"): ("a", 1.0),
    ("root", "ADD_MEMBER(0, 1, M, P, 0.1, 0.01)"): ("b", 0.4),
    ("a", "MOVE_JOINT(0, [0,0], [1,0])"): ("a1", 0.6),
    ("a", "REMOVE_MEMBER(0)"): ("a2", 0.1),
    ("b", "MOVE_JOINT(0, [0,0], [1,0])"): ("b1", 0.2),
    ("b", "REMOVE_MEMBER(0)"): ("b2", 0.3),
}


def _dummy_transition(node: DummyNode, action: str):
    next_name, reward = GRAPH[(node.name, action)]
    return DummyNode(next_name, {"step": node.current_state.get("step", 0) + 1}), reward, 1


def _dummy_continuation(node: DummyNode, depth_remaining: int):
    del depth_remaining
    terminal_bonus = {"a1": 1.0, "a2": 0.1, "b1": 0.0, "b2": 0.2}.get(node.name, 0.0)
    return terminal_bonus, 0


def _dummy_sampler(node: DummyNode, branching: int, depth: int, preferred_class):
    del branching, depth, preferred_class
    if node.name == "a":
        return [
            CandidateAction("MOVE_JOINT(0, [0,0], [1,0])", score=0.9),
            CandidateAction("REMOVE_MEMBER(0)", score=0.1),
        ]
    if node.name == "b":
        return [
            CandidateAction("MOVE_JOINT(0, [0,0], [1,0])", score=0.2),
            CandidateAction("REMOVE_MEMBER(0)", score=0.3),
        ]
    return []


def test_tree_expansion_prefers_better_long_horizon_branch():
    root = DummyNode("root", {"step": 0})
    root_candidates = [
        CandidateAction("SCALE_PARAM(0, radius, 1.1)", score=0.8),
        CandidateAction("ADD_MEMBER(0, 1, M, P, 0.1, 0.01)", score=0.9),
    ]
    result = evaluate_tree(
        root_node=root,
        root_candidates=root_candidates,
        transition_fn=_dummy_transition,
        continuation_fn=_dummy_continuation,
        sampler_fn=_dummy_sampler,
        depth=1,
        level1_branching=2,
        deeper_branching=2,
        gamma=0.99,
        adaptive_entropy_threshold=None,
        stratified=False,
    )
    assert result.tree_best_action == "SCALE_PARAM(0, radius, 1.1)"
    # depth=1, level1_branching=2 → 2 root transitions (one per candidate); no child FEA at depth-1=0
    assert result.total_fea_calls >= 2
