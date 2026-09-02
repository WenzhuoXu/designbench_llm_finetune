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
        return {
            # TRL GRPOTrainer expects "prompt" as a raw text string and tokenizes
            # it internally — this is the required key.
            "prompt": item["prompt_text"],
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
        problem_weights: Optional[dict] = None,
    ) -> "RLPromptDataset":
        """Build dataset from problem spec JSON files.

        Args:
            problems_dir: Directory containing auto_problem_NNN.json files.
            tokenizer: Model tokenizer.
            formatter: ChatFormatter for prompt formatting.
            max_prompt_len: Maximum prompt length.
            problem_ids: Optional filter to specific problem IDs.
            repeat: Repeat each problem N times (for more rollouts per problem).
            problem_weights: Optional per-problem sampling weights. Under
                scale_rewards='group' only the ORDER of a group's K rollouts
                reaches the gradient, so a group whose rollouts are uniformly
                feasible or uniformly infeasible carries no feasibility signal.
                Measured on the training split, only 33% of groups come out
                mixed -- and mixedness is strongly predicted by the initial
                violation (0.65 for the lowest tercile, 0.03 for the highest).
                Weighting the repeat count by that prediction roughly doubles the
                number of informative gradient events per step at no simulator
                cost. See scripts/group_ordering_divergence.py.

        Returns:
            RLPromptDataset.
        """
        problems_dir = Path(problems_dir)
        if problem_ids:
            # An explicit id list is authoritative: globbing auto_problem_* first
            # would silently drop every other problem in the split.
            problem_files = sorted(problems_dir.glob("*.json"))
        else:
            problem_files = sorted(problems_dir.glob("auto_problem_*.json"))
            if not problem_files:
                problem_files = sorted(problems_dir.glob("*.json"))

        log.info(f"Loading {len(problem_files)} problem specs from {problems_dir}")
        prompts = []
        seen_ids: set[str] = set()
        for pf in problem_files:
            with open(pf) as f:
                spec = json.load(f)
            # The directory holds a few files that are not problem specs (e.g.
            # test_problems.json is a LIST). Globbing "*.json" for an explicit id
            # list surfaces them, so skip anything that is not a spec, and skip
            # duplicate problem_ids (two files share auto_problem_000).
            if not isinstance(spec, dict) or "topology" not in spec:
                continue
            pid = spec.get("problem_id", pf.stem)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            if problem_ids and pid not in problem_ids:
                continue
            problem_text = _spec_to_problem_text(spec)
            messages = formatter.build_messages(problem_text=problem_text)
            # Build both text (for TRL) and token IDs (kept for reference)
            prompt_text = formatter.apply_template(messages, add_generation_prompt=True, tokenize=False)
            n_repeat = repeat
            if problem_weights:
                mean_w = sum(problem_weights.values()) / max(len(problem_weights), 1)
                w = float(problem_weights.get(pid, mean_w))
                if mean_w > 0:
                    n_repeat = max(1, int(round(repeat * w / mean_w)))
            for _ in range(n_repeat):
                prompts.append({
                    "prompt_text": prompt_text,
                    "problem_id": pid,
                    "problem_spec": spec,
                })

        if problem_weights:
            counts: dict[str, int] = {}
            for item in prompts:
                counts[item["problem_id"]] = counts.get(item["problem_id"], 0) + 1
            if counts:
                log.info(
                    f"informativeness weighting: repeats range "
                    f"{min(counts.values())}-{max(counts.values())} across "
                    f"{len(counts)} problems"
                )
        log.info(f"RLPromptDataset: {len(prompts)} prompts")
        return cls(prompts=prompts, tokenizer=tokenizer, max_prompt_len=max_prompt_len)


def _spec_to_problem_text(spec: dict) -> str:
    """Convert a problem spec JSON to human-readable problem description."""
    lines = [f"PROBLEM: {spec.get('description', spec.get('problem_id', 'Truss optimization'))}"]

    # Initial structure
    members = spec.get("topology", {}).get("members", [])
    if members:
        lines.append(f"\nINITIAL STRUCTURE: {len(members)} members")
        for idx, m in enumerate(members[:5]):  # show first 5 to keep prompt short
            joints = m.get("joints", [])
            j1 = joints[0] if len(joints) > 0 else m.get("joint_1", "?")
            j2 = joints[1] if len(joints) > 1 else m.get("joint_2", "?")
            shape = m.get("shape", {})
            r = shape.get("r", "?")
            t = shape.get("t", "?")
            member_id = m.get("id", idx)
            lines.append(f"  Member {member_id}: joints ({j1},{j2}), r={r}, t={t}")
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
