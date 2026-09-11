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

import logging
import re
from abc import ABC, abstractmethod

import torch
from transformers import PreTrainedTokenizer

from llm_finetune.data.grammar import find_actions, validate_action

log = logging.getLogger(__name__)


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


class AssistantOnlyTarget(SFTTarget):
    """Supervise the model's own turns and nothing else.

    The distillation corpus alternates rendered state (user) with the search's
    chosen tool call (assistant). Supervising the state as well would train the
    model to predict simulator output it will never be asked to produce, and at
    serving time is handed for free -- roughly four fifths of the tokens in this
    corpus. Worse, a model rewarded for continuing an observation learns to
    invent one; the failure mode is a confident hallucinated margin table.

    The mask is read off ChatML role markers, so it needs a template that emits
    <|im_start|>role ... <|im_end|> (Qwen, Phi, DeepSeek-R1-Distill). On a
    template without them the mask falls back to every non-masked token and says
    so, rather than silently supervising the wrong spans.
    """

    _warned = False

    def get_loss_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
    ) -> torch.Tensor:
        valid = (labels != -100).long()
        start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        unk = getattr(tokenizer, "unk_token_id", None)
        if start_id is None or end_id is None or start_id == unk or end_id == unk:
            if not AssistantOnlyTarget._warned:
                log.warning(
                    "assistant_only: tokenizer has no ChatML role markers; "
                    "falling back to full-sequence supervision"
                )
                AssistantOnlyTarget._warned = True
            return valid

        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        mask = [0] * len(ids)
        i = 0
        while i < len(ids):
            if ids[i] != start_id:
                i += 1
                continue
            # Role sits between the marker and the newline that opens the body.
            j = i + 1
            role = ""
            while j < len(ids) and "\n" not in role and j - i < 8:
                role += tokenizer.decode([ids[j]])
                j += 1
            k = j
            while k < len(ids) and ids[k] != end_id:
                k += 1
            if role.strip().startswith("assistant"):
                # Include <|im_end|>: the model has to learn where to stop.
                for t in range(j, min(k + 1, len(ids))):
                    mask[t] = 1
            i = k + 1

        out = torch.tensor(mask, dtype=torch.long, device=getattr(input_ids, "device", None))
        return out * valid

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


class GoldCurriculumWarmstartTarget(WarmstartReasoningTarget):
    """Gold warmstart target for compute-efficient GRPO preparation.

    This target turns each successful DesignBench trace into many one-turn SFT
    examples. Each example contains the full context up to a single assistant
    decision and supervises exactly that next response. The curriculum keeps
    SFT light enough for LoRA/2-GPU runs while teaching strict executable action
    syntax before GRPO.

    Stages:
      1. Action-only grammar and turn-taking.
      2. Short reasoning plus one canonical action.
      3. Stage 2 plus terminal answers, but only after feasible feedback.
    """

    def __init__(self, curriculum_stage: int = 2):
        super().__init__()
        self.curriculum_stage = int(curriculum_stage)

    def get_loss_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
    ) -> torch.Tensor:
        """Supervise only the sliced target assistant turn and its stop token."""
        mask = torch.zeros_like(labels)
        full_text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)

        span = _final_assistant_content_span(full_text)
        if span is None:
            return super().get_loss_mask(input_ids, labels, tokenizer)

        char_to_token = _build_char_to_token_map(input_ids, tokenizer)
        span_start, span_end = span
        for char_pos in range(span_start, span_end):
            token_idx = char_to_token.get(char_pos)
            if token_idx is not None and labels[token_idx].item() != -100:
                mask[token_idx] = 1

        return mask

    def transform_example(self, example: dict, tokenizer: PreTrainedTokenizer) -> list[dict]:
        messages = example.get("messages")
        if not messages:
            return []

        transformed_messages = self._transform.transform_messages(messages)
        turn_examples: list[dict] = []
        action_step = 0

        for idx, msg in enumerate(transformed_messages):
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            base_meta = {
                "problem_id": example.get("problem_id", ""),
                "trace_id": example.get("trace_id", ""),
                "trace_quality": example.get("trace_quality", 1.0),
                "strategy_type": example.get("strategy_type", ""),
                "step_index": action_step,
            }

            if "<answer>" in content:
                if self.curriculum_stage >= 3 and _previous_feedback_is_feasible(
                    transformed_messages[:idx]
                ):
                    turn_examples.append({
                        **base_meta,
                        "messages": transformed_messages[: idx + 1],
                        "gold_stage": "terminal_answer",
                        "action_type": "ANSWER",
                        "reaches_solution": True,
                    })
                continue

            actions = find_actions(content)
            if len(actions) != 1:
                continue
            validation = validate_action(actions[0])
            if not validation.is_valid:
                continue

            action_step += 1
            target_content = _gold_target_content(
                content,
                validation.action,
                curriculum_stage=self.curriculum_stage,
            )
            turn_examples.append({
                **base_meta,
                "messages": transformed_messages[:idx] + [
                    {"role": "assistant", "content": target_content}
                ],
                "gold_stage": (
                    "action_only" if self.curriculum_stage <= 1 else "reasoned_action"
                ),
                "action": validation.action,
                "canonical_action": validation.canonical_action,
                "action_type": validation.action_type,
                "reaches_solution": False,
            })

        return turn_examples


# ── Utilities ────────────────────────────────────────────────────────────────

def _gold_target_content(content: str, action: str, *, curriculum_stage: int) -> str:
    if curriculum_stage <= 1:
        return f"<action>{action}</action>"

    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if not think_match:
        return f"<action>{action}</action>"
    thinking = " ".join(think_match.group(1).strip().split())
    if not thinking:
        return f"<action>{action}</action>"
    return f"<think>\n{thinking}\n</think>\n<action>{action}</action>"


def _previous_feedback_is_feasible(messages: list[dict]) -> bool:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "").upper()
        if "INFEASIBLE" in content:
            return False
        return "FEASIBLE" in content
    return False


def _final_assistant_content_span(full_text: str) -> tuple[int, int] | None:
    """Return the final assistant payload span, including the closing chat token."""
    marker = "<|im_start|>assistant\n"
    start = full_text.rfind(marker)
    if start < 0:
        return None

    content_start = start + len(marker)
    end_marker = "<|im_end|>"
    end = full_text.find(end_marker, content_start)
    if end < 0:
        return content_start, len(full_text)
    return content_start, end + len(end_marker)

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
    "gold_curriculum_warmstart": GoldCurriculumWarmstartTarget,
    "warmstart_reasoning": WarmstartReasoningTarget,
    "full_sequence": FullSequenceTarget,
    "assistant_only": AssistantOnlyTarget,
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
