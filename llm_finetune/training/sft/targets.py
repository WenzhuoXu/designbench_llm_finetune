"""
SFT Target Functions — Research Hook.

Controls what tokens are supervised during SFT (i.e., where loss is computed).
This is a primary research variable: different supervision targets produce
different learned behaviors.

To add a new target:
  1. Subclass SFTTarget
  2. Implement get_loss_mask() and transform_example()
  3. Register in TARGET_REGISTRY

Usage:
    target = TARGET_REGISTRY["warmstart_reasoning"]()
    mask = target.get_loss_mask(input_ids, labels, tokenizer)
    # mask[i] = 1 → compute loss at position i
    # mask[i] = 0 → ignore (labels[i] = -100 in DataCollatorForSFT)

    # Or via config (in train_sft.py):
    target = build_target_from_config(cfg.data.target_fn)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import torch
from transformers import PreTrainedTokenizer


class SFTTarget(ABC):
    """Abstract base class for SFT supervision targets.

    Research hook: subclass this to implement custom supervision strategies.

    The key method is get_loss_mask(): it receives the full tokenized sequence
    and returns a binary mask (1=compute loss, 0=ignore). This mask is applied
    in DataCollatorForSFT before the labels are passed to the model.
    """

    @abstractmethod
    def get_loss_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
    ) -> torch.Tensor:
        """Compute loss mask for a single sequence.

        Args:
            input_ids: Token IDs, shape (seq_len,).
            labels: Label IDs (same as input_ids for full supervision), shape (seq_len,).
            tokenizer: Model tokenizer (for token → string lookups).

        Returns:
            Binary tensor, shape (seq_len,). 1=compute loss, 0=ignore.
            All positions where labels=-100 will be ignored regardless of this mask.
        """
        ...

    @abstractmethod
    def transform_example(
        self, example: dict, tokenizer: PreTrainedTokenizer
    ) -> dict:
        """Transform a raw training example before tokenization.

        Research hook: override to change data representation before tokenization.
        For example: reformat messages, augment with reasoning, remap roles.

        Args:
            example: Raw example dict (from JSONL or other data source).
            tokenizer: Model tokenizer.

        Returns:
            Transformed example dict.
        """
        ...

    def name(self) -> str:
        return self.__class__.__name__


class FullSequenceTarget(SFTTarget):
    """Supervise all non-masked tokens. Simple baseline.

    Standard SFT: the model learns to generate the full response including
    thinking tokens, reasoning text, and the final grammar action.
    No selective masking — loss is computed on every token where labels != -100.
    """

    def get_loss_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
    ) -> torch.Tensor:
        return (labels != -100).long()

    def transform_example(self, example: dict, tokenizer: PreTrainedTokenizer) -> dict:
        return example


class WarmstartReasoningTarget(SFTTarget):
    """Warmstart target for GRPO pre-training.

    Reformats DesignBench SFT messages to teach:
      1. Output format: <think>...</think>\\n<action>ACTION(...)</action>
      2. Feedback reading: model sees simulation results as user turns
      3. Turn-taking: system prompt → assistant action → user feedback → repeat

    transform_example() applies WarmstartTransform to restructure messages.
    get_loss_mask() supervises <think>, <action>, and <answer> spans.

    This target is intentionally lightweight — it teaches format, not strategy.
    The RL stage (GRPO) learns the actual design policy.
    """

    def __init__(self):
        from llm_finetune.data.processors.warmstart_transform import WarmstartTransform
        self._transform = WarmstartTransform()

    def transform_example(self, example: dict, tokenizer: PreTrainedTokenizer) -> dict:
        """Reformat messages: extract actions from <think>, remap roles."""
        messages = example.get("messages")
        if messages:
            example = dict(example)  # shallow copy to avoid mutating original
            example["messages"] = self._transform.transform_messages(messages)
        return example

    def get_loss_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
    ) -> torch.Tensor:
        """Supervise <think>...</think>, <action>...</action>, and <answer>...</answer> spans."""
        mask = torch.zeros_like(labels)
        full_text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)

        # Find all supervised character spans
        supervised_spans = []

        # <think>...</think> blocks
        for m in re.finditer(r"<think>(.*?)</think>", full_text, re.DOTALL):
            supervised_spans.append((m.start(), m.end()))

        # <action>...</action> blocks
        for m in re.finditer(r"<action>(.*?)</action>", full_text, re.DOTALL):
            supervised_spans.append((m.start(), m.end()))

        # <answer>...</answer> blocks
        for m in re.finditer(r"<answer>(.*?)</answer>", full_text, re.DOTALL):
            supervised_spans.append((m.start(), m.end()))

        if not supervised_spans:
            # Fallback: supervise all assistant tokens
            return (labels != -100).long()

        # Map character spans → token positions
        char_to_token = _build_char_to_token_map(input_ids, tokenizer)
        for span_start, span_end in supervised_spans:
            for char_pos in range(span_start, span_end):
                token_idx = char_to_token.get(char_pos)
                if token_idx is not None and labels[token_idx].item() != -100:
                    mask[token_idx] = 1

        return mask


# ── Utilities ────────────────────────────────────────────────────────────────

def _build_char_to_token_map(
    input_ids: torch.Tensor, tokenizer: PreTrainedTokenizer
) -> dict[int, int]:
    """Build a map from character position → token index."""
    char_to_token: dict[int, int] = {}
    char_pos = 0
    for token_idx, tid in enumerate(input_ids.tolist()):
        tok_text = tokenizer.decode([tid], skip_special_tokens=False)
        for _ in tok_text:
            char_to_token[char_pos] = token_idx
            char_pos += 1
    return char_to_token


# ── Registry ──────────────────────────────────────────────────────────────────

TARGET_REGISTRY: dict[str, type[SFTTarget]] = {
    "warmstart_reasoning": WarmstartReasoningTarget,
    "full_sequence": FullSequenceTarget,
}


def build_target_from_config(target_name: str, **kwargs) -> SFTTarget:
    """Build a SFTTarget instance from a config name.

    Args:
        target_name: Key in TARGET_REGISTRY (e.g., "warmstart_reasoning").
        **kwargs: Additional kwargs passed to the target constructor.

    Returns:
        SFTTarget instance.

    Raises:
        ValueError if target_name is not registered.
    """
    if target_name not in TARGET_REGISTRY:
        raise ValueError(
            f"Unknown SFT target: {target_name!r}. "
            f"Available: {list(TARGET_REGISTRY.keys())}"
        )
    return TARGET_REGISTRY[target_name](**kwargs)
