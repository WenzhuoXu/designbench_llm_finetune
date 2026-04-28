"""
MCTS Dataset: wraps DesignBench ReasoningTree structures for research.

This is the primary research hook for data structure experimentation.
The dataset exposes DesignBench's tree data in multiple formats to support:

1. Flat training:   (state_text, action) pairs for next-action prediction
2. Path training:   complete solution trajectories (SFT on successful paths)
3. Value training:  (state, value) pairs for value function / PRM training
4. Subtree batches: graph-structured samples for GNN or tree-structured models
5. Curriculum:      sorted by difficulty (depth, quality, solution status)

Research hook: override MCTSDataset.build_sample_text() to change how tree
nodes are represented as text (e.g., add member-level analysis, force diagrams,
structural intuition).

Usage:
    dataset = MCTSDataset.from_tree_dir(
        tree_dir="data/modification_trees/",
        tokenizer=tokenizer,
        formatter=formatter,
        sampling_strategy="path",  # flat | path | subtree | curriculum
    )
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from llm_finetune.data.processors.chat_formatter import ChatFormatter
from llm_finetune.data.processors.mcts_processor import MCTSProcessor, MCTSSample, PathSample

log = logging.getLogger(__name__)

DESIGNBENCH_TREES_DIR = Path(
    "/ocean/projects/mch250030p/wxu7/DesignBench/data/modification_trees"
)


class SamplingStrategy(Enum):
    FLAT = "flat"           # All (state, action) pairs equally weighted
    PATH = "path"           # Only solution-reaching paths
    SUBTREE = "subtree"     # Contiguous subtree batches
    CURRICULUM = "curriculum"  # Ordered by depth → quality → difficulty


class MCTSDataset(Dataset):
    """PyTorch Dataset over DesignBench modification trees.

    This is a primary research hook. The tree structure supports:
      - AlphaZero-style value/policy training
      - Process Reward Model (PRM) training using per-step FEA feedback
      - Curriculum learning over tree depth and quality
      - MCTS-guided data augmentation

    Research hooks:
      - Override build_sample_text() to change state text representation
      - Override compute_target_value() to change value function definition
      - Override sample_curriculum() to change curriculum ordering
    """

    def __init__(
        self,
        samples: list[MCTSSample],
        tokenizer: PreTrainedTokenizer,
        formatter: ChatFormatter,
        max_seq_len: int = 2048,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        state_text = self.build_sample_text(sample)
        target_value = self.compute_target_value(sample)
        action_text = sample.action

        # Tokenize (state_text → action prediction)
        messages = [
            {"role": "system", "content": "You are an expert truss structural engineer."},
            {"role": "user", "content": state_text},
            {"role": "assistant", "content": action_text},
        ]
        input_ids = self.formatter.apply_template(
            messages, add_generation_prompt=False, tokenize=True
        )[: self.max_seq_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "target_value": torch.tensor(target_value, dtype=torch.float32),
            "subtree_value": torch.tensor(sample.subtree_value, dtype=torch.float32),
            "depth": sample.depth,
            "is_feasible": sample.is_feasible,
            "problem_id": sample.problem_id,
            "action_type": sample.action_type,
        }

    def build_sample_text(self, sample: MCTSSample) -> str:
        """Build text representation of the current design state.

        Research hook: override to add richer state representations:
          - Member-level stress/strain analysis
          - Critical member identification
          - Structural reasoning hints
          - Force flow diagrams (text-based)
        """
        lines = [
            f"[Design State at depth={sample.depth}]",
            f"  Mass:         {sample.mass:.4f} kg",
            f"  FOS_buckling: {sample.fos_buckling:.4f}",
            f"  FOS_yielding: {sample.fos_yielding:.4f}",
            f"  Deflection:   {sample.deflection:.6f} m",
            f"  Status:       {'FEASIBLE' if sample.is_feasible else 'INFEASIBLE'}",
        ]

        if sample.member_dimensions:
            lines.append("  Key members:")
            for mid, dims in list(sample.member_dimensions.items())[:5]:
                r = dims.get("params", {}).get("r", "?")
                t = dims.get("params", {}).get("t", "?")
                j = dims.get("joints", "?")
                lines.append(f"    Member {mid} (joints {j}): r={r:.4f}, t={t:.4f}")

        lines.append("\nWhat grammar action should be applied next?")
        return "\n".join(lines)

    def compute_target_value(self, sample: MCTSSample) -> float:
        """Compute target value for value function training.

        Research hook: override to implement different value functions:
          - Binary: 1.0 if subtree_reaches_solution else 0.0
          - Continuous: normalized FOS improvement
          - Discounted: γ^depth * feasibility_signal
          - PRM-based: learned process reward
        """
        if sample.subtree_reaches_solution:
            # Normalize by target FOS (1.5 for DesignBench)
            target_fos = 1.5
            value = min(sample.subtree_value / target_fos, 1.0)
        else:
            value = max(0.0, sample.subtree_value / 1.5)
        return value

    def sample_curriculum(self, n: Optional[int] = None) -> "MCTSDataset":
        """Return a curriculum-ordered subset of this dataset.

        Research hook: override to implement custom curriculum strategies:
          - Easy-to-hard (shallow trees first)
          - Hard-to-easy (deep, non-solution paths first)
          - Solution-focused (solution paths only early in training)
          - Mixed (interleaved solution and non-solution paths)
        """
        sorted_samples = sorted(
            self.samples,
            key=lambda s: (
                not s.subtree_reaches_solution,  # solution paths first
                -s.subtree_value,               # higher value first
                s.depth,                         # shallower first
            ),
        )
        if n is not None:
            sorted_samples = sorted_samples[:n]
        return MCTSDataset(
            samples=sorted_samples,
            tokenizer=self.tokenizer,
            formatter=self.formatter,
            max_seq_len=self.max_seq_len,
        )

    def get_path_dataset(self) -> "MCTSPathDataset":
        """Convert to path-level dataset (one item per solution trajectory)."""
        return MCTSPathDataset(
            path_samples=self._extract_paths(),
            tokenizer=self.tokenizer,
            formatter=self.formatter,
            max_seq_len=self.max_seq_len,
        )

    def _extract_paths(self) -> list:
        """Extract solution paths from flat MCTS samples."""
        # Group samples by problem_id, then extract chains
        # This is a simplified implementation — override for richer path extraction
        from collections import defaultdict
        by_problem: dict = defaultdict(list)
        for s in self.samples:
            by_problem[s.problem_id].append(s)

        paths = []
        for pid, problem_samples in by_problem.items():
            # Sort by depth to build paths
            problem_samples.sort(key=lambda s: s.depth)
            if any(s.subtree_reaches_solution for s in problem_samples):
                paths.append(problem_samples)
        return paths

    @classmethod
    def from_tree_dir(
        cls,
        tree_dir: str | Path,
        tokenizer: PreTrainedTokenizer,
        formatter: ChatFormatter,
        sampling_strategy: str = "flat",
        max_seq_len: int = 2048,
        min_subtree_value: float = 0.0,
        solution_only: bool = False,
        max_depth: int = 10,
        seed: int = 42,
    ) -> "MCTSDataset":
        """Create MCTSDataset from directory of tree JSON files.

        Args:
            tree_dir: Directory with {problem_id}_tree.json files.
            tokenizer: Model tokenizer.
            formatter: ChatFormatter.
            sampling_strategy: "flat" | "path" | "curriculum" | "subtree".
            max_seq_len: Max sequence length.
            min_subtree_value: Filter samples below this value.
            solution_only: Only include samples from solution subtrees.
            max_depth: Maximum tree depth to include.
            seed: Random seed for shuffling.

        Returns:
            MCTSDataset ready for training.
        """
        processor = MCTSProcessor(
            min_subtree_value=min_subtree_value,
            solution_paths_only=solution_only,
            max_depth=max_depth,
        )

        tree_dir = Path(tree_dir)
        if not tree_dir.exists():
            log.warning(f"Tree dir {tree_dir} does not exist. Using empty dataset.")
            return cls(samples=[], tokenizer=tokenizer, formatter=formatter, max_seq_len=max_seq_len)

        samples = processor.process_tree_dir(tree_dir)

        if sampling_strategy == "curriculum":
            dataset = cls(samples=samples, tokenizer=tokenizer, formatter=formatter, max_seq_len=max_seq_len)
            return dataset.sample_curriculum()

        if sampling_strategy in ("path", "solution_path"):
            samples = [s for s in samples if s.subtree_reaches_solution]

        # Shuffle for flat/path strategies
        rng = random.Random(seed)
        rng.shuffle(samples)

        log.info(f"MCTSDataset ({sampling_strategy}): {len(samples)} samples from {tree_dir}")
        return cls(samples=samples, tokenizer=tokenizer, formatter=formatter, max_seq_len=max_seq_len)

    def get_stats(self) -> dict:
        if not self.samples:
            return {}
        depths = [s.depth for s in self.samples]
        values = [s.subtree_value for s in self.samples]
        n_solution = sum(1 for s in self.samples if s.subtree_reaches_solution)
        return {
            "n_samples": len(self.samples),
            "mean_depth": sum(depths) / len(depths),
            "max_depth": max(depths),
            "mean_value": sum(values) / len(values),
            "solution_rate": n_solution / len(self.samples),
            "action_types": _count_action_types(self.samples),
        }


class MCTSPathDataset(Dataset):
    """Dataset over complete solution trajectories (path-level).

    Each item is a multi-turn conversation for one full trajectory
    (root → leaf), suitable for SFT or trajectory-level RL.
    """

    def __init__(
        self,
        path_samples: list,
        tokenizer: PreTrainedTokenizer,
        formatter: ChatFormatter,
        max_seq_len: int = 8192,
    ):
        self.path_samples = path_samples
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.path_samples)

    def __getitem__(self, idx: int) -> dict:
        path = self.path_samples[idx]
        # Build multi-turn conversation from path
        messages = self._path_to_messages(path)
        input_ids = self.formatter.apply_template(
            messages, add_generation_prompt=False, tokenize=True
        )[: self.max_seq_len]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }

    def _path_to_messages(self, path: list[MCTSSample]) -> list[dict]:
        """Convert a list of MCTS samples (a path) to a chat message list."""
        if not path:
            return []
        first = path[0]
        messages = [
            {
                "role": "system",
                "content": "You are an expert truss structural engineer.",
            },
            {
                "role": "user",
                "content": f"Optimize this truss design (problem {first.problem_id}).",
            },
        ]
        for step in path:
            messages.append({"role": "assistant", "content": step.action})
            messages.append({
                "role": "user",
                "content": (
                    f"[FEA] mass={step.child_mass:.3f}, "
                    f"fos_b={step.child_fos_buckling:.3f}, "
                    f"fos_y={step.child_fos_yielding:.3f}, "
                    f"feasible={step.child_is_feasible}"
                ),
            })
        return messages


def _count_action_types(samples: list[MCTSSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in samples:
        counts[s.action_type] = counts.get(s.action_type, 0) + 1
    return counts
