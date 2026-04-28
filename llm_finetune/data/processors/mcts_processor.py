"""
Converts DesignBench ReasoningTree (MCTS-like) structures into training samples.

DesignBench generates modification trees via local search, stored as:
  data/modification_trees/auto_problem_NNN_tree.json

Each tree is a ReasoningTree with:
  - nodes: dict[node_id → TrussModificationNode] (design state + FEA metrics)
  - edges: list[TrussDesignEdge] (grammar action + parent/child node IDs)

This processor extracts MCTS-compatible training samples:
  1. Flat: all (state, action, child_state) tuples
  2. Path: only solution-reaching paths (SolutionTrace objects)
  3. Subtree: contiguous subtree batches (for graph-structured training)
  4. Value: (state, subtree_value) pairs for value function training

Research hook: MCTSSample is the data unit — extend it for richer representations
like action visit counts, PUCT priors, or multi-step rollout returns.

Usage:
    processor = MCTSProcessor()
    samples = processor.process_tree_file("data/modification_trees/auto_problem_090_tree.json")
    # Or load traces:
    path_samples = processor.process_traces_file("data/modification_trees/auto_problem_090_traces.json")
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger(__name__)

# Add DesignBench to path for data structure imports
DESIGNBENCH_PATH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")


@dataclass
class MCTSSample:
    """A single MCTS training sample derived from a DesignBench modification tree.

    Research hook: extend this dataclass to add fields like:
        - policy_prior: distribution over actions at this state
        - value_estimate: V(s) from MCTS rollouts
        - ucb_score: UCB score used to select this node
        - n_visits: number of MCTS visits to this node
        - Returns: G_t for policy gradient targets
    """
    # State representation
    problem_id: str
    node_id: str
    parent_node_id: Optional[str]

    # Design state (current FEA metrics)
    mass: float
    fos_buckling: float
    fos_yielding: float
    deflection: float
    is_feasible: bool

    # Action taken from this state (edge label)
    action: str              # grammar action string, e.g. "SCALE_PARAM(3, radius, 1.15)"
    action_type: str         # e.g. "SCALE_PARAM", "ADD_MEMBER"

    # Child state (post-action FEA metrics)
    child_mass: float
    child_fos_buckling: float
    child_fos_yielding: float
    child_is_feasible: bool
    child_node_id: str

    # Tree-level annotations
    depth: int               # depth of parent node in tree
    subtree_value: float     # best FOS improvement reachable from child node
    subtree_reaches_solution: bool  # any descendant is feasible?

    # Member dimensions for fine-grained state (optional, may be empty)
    member_dimensions: dict = field(default_factory=dict)

    # Formatted text for LLM training
    state_text: str = ""     # natural language description of current state
    action_text: str = ""    # formatted grammar action


@dataclass
class PathSample:
    """A complete solution path from root to feasible design.

    Derived from SolutionTrace objects. Used for SFT-style training on
    solution trajectories, or for RL advantage estimation.

    Research hook: extend with MCTS-style returns G_t = Σ r_t for each step.
    """
    problem_id: str
    trace_id: str
    trace_quality: float
    reaches_solution: bool
    strategy_type: str        # "DFS", "BFS", "Mixed"

    # Sequence of (state, action) pairs along the path
    states: list[dict]        # FEA state at each step
    actions: list[str]        # grammar action at each step
    rewards: list[float]      # reward signal at each step (if computed)

    # Final outcome
    final_mass: float
    final_fos_buckling: float
    final_fos_yielding: float
    final_is_feasible: bool

    # Formatted for LLM training
    messages: list[dict] = field(default_factory=list)  # multi-turn chat format


class MCTSProcessor:
    """Extracts MCTS training samples from DesignBench modification trees."""

    def __init__(
        self,
        min_subtree_value: float = 0.0,
        solution_paths_only: bool = False,
        max_depth: int = 10,
        include_member_dimensions: bool = True,
    ):
        """
        Args:
            min_subtree_value: Filter samples where subtree_value < threshold.
            solution_paths_only: If True, only include samples from solution paths.
            max_depth: Maximum tree depth to include.
            include_member_dimensions: Whether to include full member dim dict.
        """
        self.min_subtree_value = min_subtree_value
        self.solution_paths_only = solution_paths_only
        self.max_depth = max_depth
        self.include_member_dimensions = include_member_dimensions
        self._ensure_designbench_path()

    def process_tree_file(self, tree_path: str | Path) -> list[MCTSSample]:
        """Process a single tree JSON file into flat MCTS samples.

        Args:
            tree_path: Path to {problem_id}_tree.json

        Returns:
            List of MCTSSample, one per tree edge.
        """
        tree_data = self._load_json(tree_path)
        return list(self._extract_samples(tree_data))

    def process_tree_dir(self, tree_dir: str | Path) -> list[MCTSSample]:
        """Process all tree JSON files in a directory."""
        tree_dir = Path(tree_dir)
        all_samples = []
        tree_files = sorted(tree_dir.glob("*_tree.json"))
        log.info(f"Processing {len(tree_files)} tree files from {tree_dir}")
        for f in tree_files:
            try:
                samples = self.process_tree_file(f)
                all_samples.extend(samples)
            except Exception as e:
                log.warning(f"Failed to process {f}: {e}")
        log.info(f"Extracted {len(all_samples)} MCTS samples from {len(tree_files)} trees")
        return all_samples

    def process_traces_file(self, traces_path: str | Path) -> list[PathSample]:
        """Process a traces JSON file into PathSample objects.

        Args:
            traces_path: Path to {problem_id}_traces.json

        Returns:
            List of PathSample, one per solution trace.
        """
        traces_data = self._load_json(traces_path)
        samples = []
        for trace in traces_data:
            sample = self._trace_to_path_sample(trace)
            if sample is not None:
                samples.append(sample)
        return samples

    def get_subtree_value(self, node_id: str, tree_data: dict) -> float:
        """Compute subtree value: best FOS improvement reachable from this node.

        Research hook: override this to implement different value functions,
        e.g. discounted returns, PRM-based values, or learned value functions.
        """
        nodes = tree_data.get("nodes", {})
        edges = tree_data.get("edges", [])

        # Collect all descendant nodes
        descendants = self._get_descendants(node_id, edges)
        descendants.add(node_id)

        best_fos = -float("inf")
        for nid in descendants:
            node = nodes.get(nid, {})
            fos_b = node.get("fos_buckling", 0.0) or 0.0
            fos_y = node.get("fos_yielding", 0.0) or 0.0
            combined = min(fos_b, fos_y)  # both must exceed 1.5 for feasibility
            best_fos = max(best_fos, combined)

        return best_fos if best_fos > -float("inf") else 0.0

    def _extract_samples(self, tree_data: dict) -> Iterator[MCTSSample]:
        """Extract MCTS samples from raw tree JSON data."""
        problem_id = tree_data.get("problem_id", "unknown")
        nodes = tree_data.get("nodes", {})
        edges = tree_data.get("edges", [])

        for edge in edges:
            parent_id = str(edge.get("parent_id") or edge.get("from_node_id", ""))
            child_id = str(edge.get("child_id") or edge.get("to_node_id", ""))
            action = edge.get("action_string", edge.get("action", ""))
            action_type = edge.get("action_type", self._infer_action_type(action))

            parent_node = nodes.get(parent_id, {})
            child_node = nodes.get(child_id, {})

            if not parent_node or not child_node:
                continue

            depth = parent_node.get("depth", 0)
            if depth > self.max_depth:
                continue

            subtree_value = self.get_subtree_value(child_id, tree_data)
            if subtree_value < self.min_subtree_value:
                continue

            subtree_reaches_solution = self._subtree_has_solution(child_id, edges, nodes)
            if self.solution_paths_only and not subtree_reaches_solution:
                continue

            member_dims = {}
            if self.include_member_dimensions:
                member_dims = parent_node.get("member_dimensions", {})

            sample = MCTSSample(
                problem_id=problem_id,
                node_id=parent_id,
                parent_node_id=parent_node.get("parent_id"),
                mass=float(parent_node.get("mass", 0.0) or 0.0),
                fos_buckling=float(parent_node.get("fos_buckling", 0.0) or 0.0),
                fos_yielding=float(parent_node.get("fos_yielding", 0.0) or 0.0),
                deflection=float(parent_node.get("deflection", 0.0) or 0.0),
                is_feasible=bool(parent_node.get("is_feasible", False)),
                action=action,
                action_type=action_type,
                child_mass=float(child_node.get("mass", 0.0) or 0.0),
                child_fos_buckling=float(child_node.get("fos_buckling", 0.0) or 0.0),
                child_fos_yielding=float(child_node.get("fos_yielding", 0.0) or 0.0),
                child_is_feasible=bool(child_node.get("is_feasible", False)),
                child_node_id=child_id,
                depth=depth,
                subtree_value=subtree_value,
                subtree_reaches_solution=subtree_reaches_solution,
                member_dimensions=member_dims,
            )
            yield sample

    def _trace_to_path_sample(self, trace: dict) -> Optional[PathSample]:
        """Convert a SolutionTrace JSON dict to PathSample."""
        if self.solution_paths_only and not trace.get("reaches_solution", False):
            return None

        states = trace.get("state_sequence", trace.get("node_sequence", []))
        actions = trace.get("action_sequence", trace.get("gold_action_sequence", []))

        if not actions:
            return None

        final_state = states[-1] if states else {}

        return PathSample(
            problem_id=trace.get("problem_id", ""),
            trace_id=trace.get("trace_id", ""),
            trace_quality=float(trace.get("trace_quality", 0.0)),
            reaches_solution=bool(trace.get("reaches_solution", False)),
            strategy_type=trace.get("strategy_type", "Unknown"),
            states=[s if isinstance(s, dict) else {} for s in states],
            actions=actions,
            rewards=[],  # populated by reward functions during RL training
            final_mass=float(final_state.get("mass", 0.0) or 0.0),
            final_fos_buckling=float(final_state.get("fos_buckling", 0.0) or 0.0),
            final_fos_yielding=float(final_state.get("fos_yielding", 0.0) or 0.0),
            final_is_feasible=bool(final_state.get("is_feasible", False)),
        )

    def _get_descendants(self, node_id: str, edges: list[dict]) -> set[str]:
        """BFS to collect all descendant node IDs."""
        children_map: dict[str, list[str]] = {}
        for edge in edges:
            parent = str(edge.get("parent_id") or edge.get("from_node_id", ""))
            child = str(edge.get("child_id") or edge.get("to_node_id", ""))
            children_map.setdefault(parent, []).append(child)

        visited = set()
        queue = list(children_map.get(node_id, []))
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            queue.extend(children_map.get(nid, []))
        return visited

    def _subtree_has_solution(
        self, node_id: str, edges: list[dict], nodes: dict
    ) -> bool:
        """Check if any descendant node is feasible."""
        if nodes.get(node_id, {}).get("is_feasible", False):
            return True
        for desc_id in self._get_descendants(node_id, edges):
            if nodes.get(desc_id, {}).get("is_feasible", False):
                return True
        return False

    def _infer_action_type(self, action: str) -> str:
        """Extract action type from grammar action string."""
        if action.startswith("SCALE_MULTI_PARAM"):
            return "SCALE_MULTI_PARAM"
        for prefix in ["SCALE_PARAM", "ADD_MEMBER", "MODIFY_PARAM",
                       "REMOVE_MEMBER", "MOVE_JOINT"]:
            if action.startswith(prefix):
                return prefix
        return "UNKNOWN"

    def _load_json(self, path: str | Path) -> dict:
        path = Path(path)
        with open(path) as f:
            return json.load(f)

    def _ensure_designbench_path(self) -> None:
        """Add DesignBench to sys.path for data structure imports."""
        db_path = str(DESIGNBENCH_PATH)
        if db_path not in sys.path:
            sys.path.insert(0, db_path)
