"""
SFT Dataset: loads DesignBench SFT JSONL → tokenized training examples.

Data source: DesignBench/data/sft/{train,dev}.jsonl — pre-built multi-turn
conversations with <think> tags and FEA feedback between steps.

Supports:
  - Sequence packing (multiple traces per sample to maximize GPU utilization)
  - Quality filtering via trace_quality field
  - transform_example() hook for target-specific message reformatting

Usage:
    dataset = SFTDataset.from_sft_jsonl(
        path="DesignBench/data/sft/train.jsonl",
        tokenizer=tokenizer,
        formatter=formatter,
        target=WarmstartReasoningTarget(),
    )
    # Use with TRL SFTTrainer or standard DataLoader
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from llm_finetune.data.processors.chat_formatter import ChatFormatter

log = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """PyTorch Dataset of tokenized SFT examples from DesignBench traces."""

    # TRL 1.0+ checks for column_names to detect pre-tokenized datasets.
    # Setting to None lets TRL infer column names from the first example.
    column_names: list[str] = None  # type: ignore[assignment]

    def __init__(
        self,
        examples: list[dict],
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 8192,
    ):
        """
        Args:
            examples: List of dicts with keys: input_ids, labels, attention_mask.
            tokenizer: Used for padding/EOS.
            max_seq_len: Hard cap on sequence length.
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        input_ids = ex["input_ids"][: self.max_seq_len]
        labels = ex["labels"][: self.max_seq_len]
        attention_mask = ex["attention_mask"][: self.max_seq_len]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    @classmethod
    def from_sft_jsonl(
        cls,
        path: str | Path,
        tokenizer: PreTrainedTokenizer,
        formatter: ChatFormatter,
        target,  # SFTTarget instance
        max_seq_len: int = 8192,
        min_trace_quality: float = 0.0,
    ) -> "SFTDataset":
        """Create SFTDataset from DesignBench SFT JSONL (train.jsonl / dev.jsonl).

        Uses SFTJsonlProcessor which:
        1. Reads pre-built message lists from the JSONL file
        2. Calls target.transform_example() to apply message transforms
        3. Tokenizes via ChatFormatter

        Split is handled by file selection (train.jsonl vs dev.jsonl),
        not by random splitting.

        Args:
            path: Path to train.jsonl or dev.jsonl.
            tokenizer: Model tokenizer.
            formatter: ChatFormatter for this model.
            target: SFTTarget instance (transform_example + loss masking).
            max_seq_len: Maximum sequence length.
            min_trace_quality: Filter examples below this quality.

        Returns:
            SFTDataset ready for training.
        """
        from llm_finetune.data.processors.sft_jsonl_processor import SFTJsonlProcessor

        processor = SFTJsonlProcessor(
            tokenizer=tokenizer,
            formatter=formatter,
            target=target,
            max_seq_len=max_seq_len,
            min_trace_quality=min_trace_quality,
        )
        examples = processor.process_jsonl(path)

        log.info(
            f"SFTDataset (jsonl): {len(examples)} examples "
            f"(avg len={_avg_len(examples):.0f} tokens)"
        )
        return cls(examples=examples, tokenizer=tokenizer, max_seq_len=max_seq_len)

    def get_stats(self) -> dict:
        """Return dataset statistics for logging."""
        lengths = [len(e["input_ids"]) for e in self.examples]
        qualities = [e.get("trace_quality", 0.0) for e in self.examples]
        n_solution = sum(1 for e in self.examples if e.get("reaches_solution", False))
        return {
            "n_examples": len(self.examples),
            "mean_length": sum(lengths) / len(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "mean_quality": sum(qualities) / len(qualities) if qualities else 0,
            "solution_rate": n_solution / len(self.examples) if self.examples else 0,
        }


def _avg_len(examples: list[dict]) -> float:
    if not examples:
        return 0.0
    return sum(len(e["input_ids"]) for e in examples) / len(examples)
