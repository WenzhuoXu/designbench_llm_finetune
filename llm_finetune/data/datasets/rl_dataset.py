"""
RL Prompt Dataset: problem specs → prompt tensors for GRPO rollout generation.

Each item in this dataset is a prompt (system + user message, no assistant response)
representing a truss design problem. The GRPO trainer generates K completions per
prompt using vLLM, executes them through TrussRolloutEnv to get rewards, and uses
the (prompt, completions, rewards) group for policy gradient updates.

Usage:
    dataset = RLPromptDataset.from_problems_dir(
        problems_dir="data/problems/",
        tokenizer=tokenizer,
        formatter=formatter,
        n_epochs=1,
    )
    # Feed to GRPOTrainer — it handles rollout generation internally
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from llm_finetune.data.processors.chat_formatter import ChatFormatter

log = logging.getLogger(__name__)

DESIGNBENCH_PROBLEMS_DIR = Path(
    "/ocean/projects/mch250030p/wxu7/DesignBench/data/problems"
)


class RLPromptDataset(Dataset):
    """Dataset of prompt-only tensors for GRPO rollout generation.

    Each item represents one design problem (with a specific initial state)
    that will be used to generate K rollout trajectories during GRPO training.

    Built from problem spec files in DesignBench/data/problems/.
    """

    def __init__(
        self,
        prompts: list[dict],
        tokenizer: PreTrainedTokenizer,
        max_prompt_len: int = 2048,
    ):
        """
        Args:
            prompts: List of dicts with keys: input_ids, problem_id, problem_spec.
            tokenizer: Used for left-padding (RL uses left-pad for generation).
            max_prompt_len: Maximum prompt length (truncated from left if exceeded).
        """
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        item = self.prompts[idx]
        input_ids = item["input_ids"][-self.max_prompt_len:]  # left-truncate
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "problem_id": item["problem_id"],
            # problem_spec is kept as-is (not a tensor) for env.reset()
            "problem_spec": item.get("problem_spec", {}),
        }

    @classmethod
    def from_problems_dir(
        cls,
        problems_dir: str | Path,
        tokenizer: PreTrainedTokenizer,
        formatter: ChatFormatter,
        max_prompt_len: int = 2048,
        problem_ids: Optional[list[str]] = None,
        repeat: int = 1,
    ) -> "RLPromptDataset":
        """Build dataset from problem spec JSON files.

        Args:
            problems_dir: Directory containing auto_problem_NNN.json files.
            tokenizer: Model tokenizer.
            formatter: ChatFormatter for prompt formatting.
            max_prompt_len: Maximum prompt length.
            problem_ids: Optional filter to specific problem IDs.
            repeat: Repeat each problem N times (for more rollouts per problem).

        Returns:
            RLPromptDataset.
        """
        problems_dir = Path(problems_dir)
        problem_files = sorted(problems_dir.glob("auto_problem_*.json"))
        if not problem_files:
            problem_files = sorted(problems_dir.glob("*.json"))

        log.info(f"Loading {len(problem_files)} problem specs from {problems_dir}")
        prompts = []
        for pf in problem_files:
            with open(pf) as f:
                spec = json.load(f)
            pid = spec.get("problem_id", pf.stem)
            if problem_ids and pid not in problem_ids:
                continue
            for _ in range(repeat):
                prompt_ids = cls._build_prompt(spec, formatter, tokenizer)
                prompts.append({
                    "input_ids": prompt_ids,
                    "problem_id": pid,
                    "problem_spec": spec,
                })

        log.info(f"RLPromptDataset: {len(prompts)} prompts")
        return cls(prompts=prompts, tokenizer=tokenizer, max_prompt_len=max_prompt_len)

    @staticmethod
    def _build_prompt(
        spec: dict,
        formatter: ChatFormatter,
        tokenizer: PreTrainedTokenizer,
    ) -> list[int]:
        """Build tokenized prompt from a problem spec."""
        # Format problem as human-readable text
        problem_text = _spec_to_problem_text(spec)
        messages = formatter.build_messages(problem_text=problem_text)
        return formatter.apply_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )


def _spec_to_problem_text(spec: dict) -> str:
    """Convert a problem spec JSON to human-readable problem description."""
    lines = [f"PROBLEM: {spec.get('description', spec.get('problem_id', 'Truss optimization'))}"]

    # Initial structure
    members = spec.get("topology", {}).get("members", [])
    if members:
        lines.append(f"\nINITIAL STRUCTURE: {len(members)} members")
        for m in members[:5]:  # show first 5 to keep prompt short
            j1, j2 = m.get("joint_1", "?"), m.get("joint_2", "?")
            shape = m.get("shape", {})
            r = shape.get("r", "?")
            t = shape.get("t", "?")
            lines.append(f"  Member {m.get('id', '?')}: joints ({j1},{j2}), r={r}, t={t}")
        if len(members) > 5:
            lines.append(f"  ... ({len(members) - 5} more members)")

    # Loading
    loads = spec.get("loading", [])
    if loads:
        lines.append(f"\nLOADS: {len(loads)} point loads")

    # Goals
    goals = spec.get("goals", {})
    if goals:
        lines.append("\nDESIGN GOALS:")
        for k, v in goals.items():
            lines.append(f"  {k}: {v}")

    lines.append("\nBegin your iterative optimization.")
    return "\n".join(lines)
