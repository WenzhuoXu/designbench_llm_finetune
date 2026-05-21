"""
JSONL processor for DesignBench SFT data.

Loads pre-built multi-turn conversations from SFT JSONL files, applies
target.transform_example() (message reformatting hook), and tokenizes
via ChatFormatter.

Usage:
    processor = SFTJsonlProcessor(tokenizer, formatter, target, max_seq_len=8192)
    examples = processor.process_jsonl("DesignBench/data/sft/train.jsonl")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from transformers import PreTrainedTokenizer

from llm_finetune.data.processors.chat_formatter import ChatFormatter
from llm_finetune.training.sft.targets import SFTTarget

log = logging.getLogger(__name__)


class SFTJsonlProcessor:
    """Processes DesignBench SFT JSONL data into tokenized training examples.

    Key difference from TraceProcessor: this processor calls
    target.transform_example() on each example, activating the research hook
    that allows targets to modify message content before tokenization.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        formatter: ChatFormatter,
        target: SFTTarget,
        max_seq_len: int = 8192,
        min_trace_quality: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.target = target
        self.max_seq_len = max_seq_len
        self.min_trace_quality = min_trace_quality

    def process_jsonl(self, path: str | Path) -> list[dict]:
        """Load and process all examples from an SFT JSONL file.

        Returns:
            List of dicts with keys: input_ids, labels, attention_mask,
            problem_id, trace_id, trace_quality.
        """
        path = Path(path)
        log.info(f"Loading SFT JSONL from {path} ...")

        processed = []
        skipped = 0
        total = 0

        with open(path) as f:
            for line in f:
                total += 1
                line = line.strip()
                if not line:
                    continue

                raw = json.loads(line)
                results = self._process_example(raw)
                if not results:
                    skipped += 1
                    continue
                processed.extend(results)

        log.info(
            f"Processed {len(processed)}/{total} examples "
            f"({skipped} skipped by quality filter)"
        )
        return processed

    def _process_example(self, raw: dict) -> Optional[list[dict]]:
        """Process a single JSONL example.

        Steps:
            1. Quality filter
            2. Call target.transform_example() — applies message transforms
            3. Tokenize messages via formatter.apply_template()
            4. Return tokenized result with metadata
        """
        trace_quality = raw.get("trace_quality", 1.0)
        if trace_quality < self.min_trace_quality:
            return None

        messages = raw.get("messages")
        if not messages:
            return None

        # Apply target transform (this is the research hook)
        # For WarmstartReasoningTarget, this reformats messages
        # For other targets, this is typically a no-op pass-through
        transformed = self.target.transform_example(raw, self.tokenizer)
        transformed_examples = transformed if isinstance(transformed, list) else [transformed]

        processed: list[dict] = []
        for transformed_ex in transformed_examples:
            ex_messages = transformed_ex.get("messages", messages)
            if not ex_messages:
                continue

            # Tokenize with chat template
            input_ids = self.formatter.apply_template(
                ex_messages,
                add_generation_prompt=False,
                tokenize=True,
            )

            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[: self.max_seq_len]

            # Labels: clone input_ids (SFTTarget.get_loss_mask() masks later in collator)
            labels = list(input_ids)

            metadata = {
                key: value
                for key, value in transformed_ex.items()
                if key not in {"messages", "structured"}
            }
            processed.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
                "problem_id": raw.get("problem_id", ""),
                "trace_id": raw.get("trace_id", ""),
                "trace_quality": trace_quality,
                "reaches_solution": transformed_ex.get("reaches_solution", True),
                "n_actions": sum(1 for m in ex_messages if m["role"] == "assistant"),
                **metadata,
            })

        return processed or None
