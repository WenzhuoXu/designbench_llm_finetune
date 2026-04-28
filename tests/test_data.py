"""
Tests for data pipeline: datasets, processors, collators.

Tests chat formatting, trace processing, MCTS data loading, and collation.
Many tests use mock tokenizers to avoid requiring model downloads.

Run with: pytest tests/test_data.py -v
"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from unittest.mock import MagicMock, patch

from llm_finetune.data.processors.chat_formatter import ChatFormatter, ThinkingMode
from llm_finetune.data.processors.mcts_processor import MCTSProcessor, MCTSSample
from llm_finetune.data.collators import DataCollatorForSFT, DataCollatorForRL
from llm_finetune.training.sft.targets import FullSequenceTarget


# ── Mock tokenizer ────────────────────────────────────────────────────────────

def make_tokenizer():
    """Minimal mock tokenizer for testing."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 2
    tok.pad_token = "<pad>"
    tok.eos_token = "</s>"
    # apply_chat_template returns simple token IDs
    tok.apply_chat_template.side_effect = lambda msgs, **kwargs: list(range(10, 10 + len(msgs) * 5))
    tok.decode.side_effect = lambda ids, **kwargs: "text " * len(ids)
    tok.convert_tokens_to_ids.return_value = 999
    tok.unk_token_id = 999
    return tok


# ── ChatFormatter tests ───────────────────────────────────────────────────────

def test_chat_formatter_from_model_id_qwen3():
    tok = make_tokenizer()
    formatter = ChatFormatter.from_model_id("Qwen/Qwen3-14B", tok)
    assert formatter.thinking_mode == ThinkingMode.QWEN3

def test_chat_formatter_from_model_id_deepseek():
    tok = make_tokenizer()
    formatter = ChatFormatter.from_model_id("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", tok)
    assert formatter.thinking_mode == ThinkingMode.DEEPSEEK_R1

def test_chat_formatter_from_model_id_llama():
    tok = make_tokenizer()
    formatter = ChatFormatter.from_model_id("meta-llama/Llama-4-Scout-17B-16E-Instruct", tok)
    assert formatter.thinking_mode == ThinkingMode.NONE

def test_chat_formatter_build_messages():
    tok = make_tokenizer()
    formatter = ChatFormatter.from_model_id("Qwen/Qwen2.5-14B-Instruct", tok)
    messages = formatter.build_messages("Optimize this truss.")
    assert len(messages) >= 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Optimize this truss." in messages[1]["content"]

def test_chat_formatter_build_messages_with_history():
    tok = make_tokenizer()
    formatter = ChatFormatter.from_model_id("Qwen/Qwen2.5-14B-Instruct", tok)
    history = [
        {"action": "SCALE_PARAM(3, radius, 1.1)", "fea_result": {"mass": 180.0, "is_feasible": False}},
    ]
    messages = formatter.build_messages("Optimize this truss.", action_history=history)
    # Should have: system, user, assistant, user (fea feedback)
    assert len(messages) >= 4

def test_chat_formatter_thinking_action_none():
    tok = make_tokenizer()
    formatter = ChatFormatter.from_model_id("Qwen/Qwen2.5-14B-Instruct", tok)
    _, action = formatter.extract_thinking("SCALE_PARAM(3, radius, 1.1)")
    assert action == "SCALE_PARAM(3, radius, 1.1)"

def test_chat_formatter_thinking_extraction():
    tok = make_tokenizer()
    formatter = ChatFormatter.from_model_id("Qwen/Qwen3-14B", tok)
    text = "<think>I should scale member 3</think>\nSCALE_PARAM(3, radius, 1.2)"
    thinking, action = formatter.extract_thinking(text)
    assert "scale member 3" in thinking
    assert action == "SCALE_PARAM(3, radius, 1.2)"

def test_chat_formatter_fea_feedback():
    tok = make_tokenizer()
    formatter = ChatFormatter.from_model_id("Qwen/Qwen2.5-14B-Instruct", tok)
    fea = {"mass": 175.3, "fos_buckling": 1.45, "fos_yielding": 1.8, "is_feasible": False}
    feedback = formatter._format_fea_feedback(fea)
    assert "175.3" in feedback or "175" in feedback
    assert "INFEASIBLE" in feedback


# ── MCTSProcessor tests ───────────────────────────────────────────────────────

def make_mock_tree_json():
    """Create a minimal tree JSON for testing."""
    return {
        "problem_id": "auto_problem_090",
        "nodes": {
            "root": {
                "mass": 200.0, "fos_buckling": 0.8, "fos_yielding": 1.2,
                "deflection": 0.015, "is_feasible": False, "depth": 0,
                "member_dimensions": {},
            },
            "child1": {
                "mass": 190.0, "fos_buckling": 1.0, "fos_yielding": 1.4,
                "deflection": 0.013, "is_feasible": False, "depth": 1,
                "member_dimensions": {},
            },
            "child2": {
                "mass": 185.0, "fos_buckling": 1.6, "fos_yielding": 1.7,
                "deflection": 0.011, "is_feasible": True, "depth": 2,
                "member_dimensions": {},
            },
        },
        "edges": [
            {"parent_id": "root", "child_id": "child1",
             "action_string": "SCALE_PARAM(3, radius, 1.15)", "action_type": "SCALE_PARAM"},
            {"parent_id": "child1", "child_id": "child2",
             "action_string": "ADD_MEMBER(0, 5, 6061_T6_Aluminum, Pipe, 0.03, 0.004)",
             "action_type": "ADD_MEMBER"},
        ],
    }


def test_mcts_processor_extract_samples():
    processor = MCTSProcessor()
    tree_data = make_mock_tree_json()
    samples = list(processor._extract_samples(tree_data))
    assert len(samples) == 2
    assert all(isinstance(s, MCTSSample) for s in samples)

def test_mcts_processor_problem_id():
    processor = MCTSProcessor()
    tree_data = make_mock_tree_json()
    samples = list(processor._extract_samples(tree_data))
    assert all(s.problem_id == "auto_problem_090" for s in samples)

def test_mcts_processor_subtree_value():
    processor = MCTSProcessor()
    tree_data = make_mock_tree_json()
    # child1 has child2 which is feasible — subtree value should be positive
    value = processor.get_subtree_value("child1", tree_data)
    assert value > 0

def test_mcts_processor_from_tree_file():
    processor = MCTSProcessor()
    tree_data = make_mock_tree_json()

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(tree_data, f)
        tmp_path = f.name

    samples = processor.process_tree_file(tmp_path)
    assert len(samples) > 0
    Path(tmp_path).unlink()

def test_mcts_processor_solution_only():
    processor = MCTSProcessor(solution_paths_only=True)
    tree_data = make_mock_tree_json()
    samples = list(processor._extract_samples(tree_data))
    # Both edges lead to subtrees with solutions (child2 is feasible)
    assert all(s.subtree_reaches_solution for s in samples)


# ── DataCollatorForSFT tests ──────────────────────────────────────────────────

def test_collator_sft_pads_to_max():
    tok = make_tokenizer()
    target = FullSequenceTarget()
    collator = DataCollatorForSFT(tokenizer=tok, target=target, pad_to_multiple_of=None)

    features = [
        {"input_ids": torch.tensor([1, 2, 3]),
         "labels": torch.tensor([1, 2, 3]),
         "attention_mask": torch.tensor([1, 1, 1])},
        {"input_ids": torch.tensor([1, 2, 3, 4, 5]),
         "labels": torch.tensor([1, 2, 3, 4, 5]),
         "attention_mask": torch.tensor([1, 1, 1, 1, 1])},
    ]
    batch = collator(features)
    assert batch["input_ids"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5)
    assert batch["attention_mask"].shape == (2, 5)
    # Padded positions should be 0 in attention_mask
    assert batch["attention_mask"][0, 3].item() == 0

def test_collator_sft_pad_to_multiple():
    tok = make_tokenizer()
    collator = DataCollatorForSFT(tokenizer=tok, target=None, pad_to_multiple_of=8)
    features = [
        {"input_ids": torch.tensor([1, 2, 3]),
         "labels": torch.tensor([1, 2, 3]),
         "attention_mask": torch.tensor([1, 1, 1])},
    ]
    batch = collator(features)
    assert batch["input_ids"].shape[1] % 8 == 0


# ── DataCollatorForRL tests ───────────────────────────────────────────────────

def test_collator_rl_left_pads():
    tok = make_tokenizer()
    collator = DataCollatorForRL(tokenizer=tok, pad_to_multiple_of=None)
    features = [
        {"input_ids": torch.tensor([1, 2, 3]),
         "attention_mask": torch.tensor([1, 1, 1]),
         "problem_id": "p1", "problem_spec": {}},
        {"input_ids": torch.tensor([4, 5]),
         "attention_mask": torch.tensor([1, 1]),
         "problem_id": "p2", "problem_spec": {}},
    ]
    batch = collator(features)
    assert batch["input_ids"].shape == (2, 3)
    # Left-pad: shorter sequence padded at start
    assert batch["attention_mask"][1, 0].item() == 0   # left pad position
    assert batch["attention_mask"][1, 1].item() == 1   # real token
    assert batch["attention_mask"][1, 2].item() == 1   # real token


# ── SFTDataset tests ──────────────────────────────────────────────────────────

def test_sft_dataset_from_examples():
    from llm_finetune.data.datasets.sft_dataset import SFTDataset
    tok = make_tokenizer()
    examples = [
        {"input_ids": list(range(20)), "labels": list(range(20)),
         "attention_mask": [1]*20, "problem_id": "p1",
         "trace_id": "t1", "trace_quality": 0.9, "reaches_solution": True, "n_actions": 5},
    ]
    dataset = SFTDataset(examples=examples, tokenizer=tok, max_seq_len=100)
    assert len(dataset) == 1
    item = dataset[0]
    assert "input_ids" in item
    assert item["input_ids"].dtype == torch.long

def test_sft_dataset_truncation():
    from llm_finetune.data.datasets.sft_dataset import SFTDataset
    tok = make_tokenizer()
    examples = [
        {"input_ids": list(range(50)), "labels": list(range(50)),
         "attention_mask": [1]*50, "problem_id": "p1",
         "trace_id": "t1", "trace_quality": 0.9, "reaches_solution": False, "n_actions": 3},
    ]
    dataset = SFTDataset(examples=examples, tokenizer=tok, max_seq_len=30)
    item = dataset[0]
    assert item["input_ids"].shape[0] <= 30
