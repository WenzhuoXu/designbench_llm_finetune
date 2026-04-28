"""
DataCollator implementations for SFT and RL training.

DataCollatorForSFT:
  - Pads variable-length sequences to batch maximum
  - Applies SFTTarget masking (labels = -100 for non-supervised tokens)
  - Right-padding for SFT (standard for causal LM training)

DataCollatorForRL:
  - Left-padding for prompt tensors (required for generation)
  - Handles variable-length prompts in GRPO batches
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForSFT:
    """Collator for SFT training with target masking.

    Applies the SFTTarget's get_loss_mask() to set labels=-100 for
    tokens that should not be supervised (e.g., prompt tokens, thinking
    tokens, or everything except grammar actions).

    Compatible with TRL SFTTrainer.
    """
    tokenizer: PreTrainedTokenizer
    target: Any  # SFTTarget instance
    pad_to_multiple_of: Optional[int] = 8  # Tensor core alignment

    def __call__(self, features: list[dict]) -> dict:
        # Get max length in batch
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        pad_id = self.tokenizer.pad_token_id or 0

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for f in features:
            ids = f["input_ids"]
            lbls = f["labels"]
            mask = f["attention_mask"]

            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
                lbls = lbls.tolist()
                mask = mask.tolist()

            # Apply SFT target masking before padding
            # (target operates on unpadded sequences)
            if self.target is not None:
                ids_t = torch.tensor(ids, dtype=torch.long)
                lbls_t = torch.tensor(lbls, dtype=torch.long)
                loss_mask = self.target.get_loss_mask(ids_t, lbls_t, self.tokenizer)
                masked_labels = torch.where(
                    loss_mask.bool(), lbls_t, torch.full_like(lbls_t, -100)
                ).tolist()
            else:
                masked_labels = lbls

            # Right-pad to max_len
            pad_len = max_len - len(ids)
            batch_input_ids.append(ids + [pad_id] * pad_len)
            batch_labels.append(masked_labels + [-100] * pad_len)
            batch_attention_mask.append(mask + [0] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }


@dataclass
class DataCollatorForRL:
    """Collator for RL prompt batches (left-padding for generation).

    Left-padding is required for autoregressive generation with batch size > 1,
    so that all sequences end at the same position (no right-pad tokens in
    the generated prefix).
    """
    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(f["input_ids"].shape[0] for f in features)
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        pad_id = self.tokenizer.pad_token_id or 0

        batch_input_ids = []
        batch_attention_mask = []
        problem_ids = []
        problem_specs = []

        for f in features:
            ids = f["input_ids"].tolist() if isinstance(f["input_ids"], torch.Tensor) else f["input_ids"]
            mask = f["attention_mask"].tolist() if isinstance(f["attention_mask"], torch.Tensor) else f["attention_mask"]

            # Left-pad
            pad_len = max_len - len(ids)
            batch_input_ids.append([pad_id] * pad_len + ids)
            batch_attention_mask.append([0] * pad_len + mask)
            problem_ids.append(f.get("problem_id", ""))
            problem_specs.append(f.get("problem_spec", {}))

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "problem_ids": problem_ids,
            "problem_specs": problem_specs,
        }
