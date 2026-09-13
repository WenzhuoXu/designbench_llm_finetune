"""
TrussRolloutEnv: DesignBench FEA environment for RL rollouts.

Wraps DesignBench's truss_executor.py functions to provide a gym-like
interface for GRPO rollout execution.

Key design points:
  - Stateless: each reset() creates a fresh Truss from problem spec
  - Parallel: uses multiprocessing pool for batched rollout execution
    (104 CPUs on H100 node → run 50+ FEA calls in parallel per batch)
  - Fast: each FEA call takes 5-15ms; pool of 32 workers → ~1000 FEA/s
  - Serializable: Truss state can be pickled for distributed workers

Usage:
    env = TrussRolloutEnv(max_steps=20, n_workers=32)

    # Single rollout
    state = env.reset(problem_spec)
    for action in actions:
        state, reward, done, info = env.step(action)

    # Run full completion (for GRPO reward computation)
    rollout = env.run_completion(problem_spec, completion_text)

    # Batched rollout (parallel)
    rollouts = env.run_batch(problem_specs, completions)
"""

from __future__ import annotations

import logging
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from llm_finetune.data.grammar import ACTION_PATTERN as COMBINED_PATTERN
from llm_finetune.data.grammar import extract_action

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llm_finetune.training.rl.rewards import RolloutResult

# DesignBench path for FEA imports
DESIGNBENCH_PATH = Path(os.environ.get(
    "DESIGNBENCH_ROOT", Path(__file__).resolve().parents[2].parent / "DesignBench"))

@dataclass
class StepResult:
    """Result of a single environment step."""
    state: dict                    # FEA state after action
    reward: float                  # Immediate reward (from reward function)
    done: bool                     # Episode ended (solved or max_steps)
    success: bool                  # Action parsed and executed successfully
    action_str: str                # Parsed grammar action (may differ from raw output)
    error: str = ""                # Error message if step failed
    info: dict = field(default_factory=dict)  # Additional info (FOS deltas, etc.)


class TrussRolloutEnv:
    """FEA-backed environment for truss design rollouts.

    The environment executes grammar actions through DesignBench's trussme
    FEA solver. Each action modifies the truss structure and triggers an
    analysis that returns updated structural metrics.

    Designed for parallel GRPO rollouts:
      - ProcessPoolExecutor with n_workers processes
      - Each worker has its own trussme state (stateless per call)
      - Supports batch execution for efficiency
    """

    def __init__(
        self,
        max_steps: int = 20,
        n_workers: int = 16,
        timeout_per_step: float = 30.0,
        reward_fn=None,   # Optional RewardFunction — if None, env returns 0.0
    ):
        """
        Args:
            max_steps: Maximum grammar actions per episode.
            n_workers: Number of parallel FEA workers (default 16; up to 104 on H100 node).
            timeout_per_step: Timeout in seconds per FEA call.
            reward_fn: Optional RewardFunction for step-level rewards.
        """
        self.max_steps = max_steps
        self.n_workers = n_workers
        self.timeout_per_step = timeout_per_step
        self.reward_fn = reward_fn
        self._executor: Optional[ProcessPoolExecutor] = None
        self._ensure_designbench_path()

    def reset(self, problem_spec: dict) -> dict:
        """Initialize a new episode from a problem spec.

        Args:
            problem_spec: Problem JSON (from problems/auto_problem_NNN.json).

        Returns:
            Initial state dict: {mass, fos_buckling, fos_yielding, deflection,
                                  is_feasible, member_dimensions, step=0}.
        """
        try:
            truss, goals = _load_truss_and_goals(problem_spec)
            state = _analyze_truss(truss, goals)
            state["step"] = 0
            state["problem_spec"] = problem_spec
            return state
        except Exception as e:
            log.error(f"reset() failed: {e}")
            return {"error": str(e), "step": 0, "is_feasible": False}

    def step(self, truss_state: Any, action_str: str, problem_spec: dict) -> StepResult:
        """Execute a grammar action on the current truss state.

        Args:
            truss_state: Serialized truss state (from previous reset/step).
            action_str: Raw model output (may contain thinking tokens).
            problem_spec: Problem specification dict.

        Returns:
            StepResult with new state, reward, done flag.
        """
        # Parse grammar action from model output
        parsed_action = parse_grammar_action(action_str)
        if parsed_action is None:
            return StepResult(
                state=truss_state if isinstance(truss_state, dict) else {},
                reward=0.0,
                done=False,
                success=False,
                action_str=action_str,
                error="Grammar parse failed",
            )

        try:
            result = _execute_step_in_worker(problem_spec, truss_state, parsed_action)
            new_state = result["state"]
            new_state["step"] = (truss_state.get("step", 0) + 1 if isinstance(truss_state, dict) else 1)
            done = new_state.get("is_feasible", False) or new_state["step"] >= self.max_steps
            return StepResult(
                state=new_state,
                reward=0.0,  # reward computed from full rollout in GRPO
                done=done,
                success=True,
                action_str=parsed_action,
                info=result.get("info", {}),
            )
        except Exception as e:
            log.debug(f"step() error for action {parsed_action!r}: {e}")
            return StepResult(
                state=truss_state if isinstance(truss_state, dict) else {},
                reward=0.0,
                done=False,
                success=False,
                action_str=parsed_action,
                error=str(e),
            )

    def run_completion(
        self,
        problem_spec: dict,
        completion: str,
    ) -> "RolloutResult":
        """Execute a full model completion as a rollout episode.

        Parses all grammar actions from the completion text and executes
        them sequentially through the FEA simulator.

        Args:
            problem_spec: Problem specification dict.
            completion: Full model completion (may include thinking tokens).

        Returns:
            RolloutResult with full state history, action sequence, and metrics.
        """
        from llm_finetune.training.rl.rewards import RolloutResult

        initial_state = self.reset(problem_spec)
        if "error" in initial_state:
            return RolloutResult(
                problem_id=problem_spec.get("problem_id", ""),
                action_sequence=[],
                raw_outputs=[completion],
                state_history=[],
                final_state={},
                initial_state=initial_state,
                token_counts=[],
                parse_success=[],
                reaches_solution=False,
                error_message=initial_state["error"],
            )

        # Split completion into steps (each step = one action)
        steps = self._split_completion_into_steps(completion)
        state_history = [initial_state]
        action_sequence = []
        parse_success = []
        current_state = initial_state

        try:
            truss, goals = _load_truss_and_goals(problem_spec)
        except Exception as e:
            return RolloutResult(
                problem_id=problem_spec.get("problem_id", ""),
                action_sequence=[],
                raw_outputs=[completion],
                state_history=[initial_state],
                final_state=initial_state,
                initial_state=initial_state,
                token_counts=[],
                parse_success=[],
                reaches_solution=False,
                error_message=str(e),
            )

        n_fea_calls = 0
        for step_text in steps[:self.max_steps]:
            parsed = parse_grammar_action(step_text)
            if parsed is None:
                parse_success.append(False)
                action_sequence.append(step_text[:100])
                state_history.append(deepcopy(current_state))
                continue

            try:
                truss = _apply_action(truss, parsed)
                state = _analyze_truss(truss, goals)
                n_fea_calls += 1
                state["step"] = len(state_history)
                state_history.append(state)
                action_sequence.append(parsed)
                parse_success.append(True)
                current_state = state
                if state.get("is_feasible", False):
                    break
            except Exception as e:
                log.debug(f"Action {parsed!r} failed: {e}")
                parse_success.append(False)
                action_sequence.append(parsed)
                state_history.append(deepcopy(current_state))

        final_state = state_history[-1] if state_history else initial_state
        return RolloutResult(
            problem_id=problem_spec.get("problem_id", ""),
            action_sequence=action_sequence,
            raw_outputs=[completion],
            state_history=state_history,
            final_state=final_state,
            initial_state=initial_state,
            token_counts=[len(s.split()) for s in steps],
            parse_success=parse_success,
            reaches_solution=final_state.get("is_feasible", False),
            n_fea_calls=n_fea_calls,
            n_steps=len(action_sequence),
        )

    def run_batch(
        self,
        problem_specs: list[dict],
        completions: list[str],
    ) -> list["RolloutResult"]:
        """Execute a batch of rollouts in parallel using ProcessPoolExecutor.

        Uses n_workers processes from the 104 available CPUs on the H100 node.
        Each worker executes one complete rollout (problem_spec + completion → FEA).

        Args:
            problem_specs: List of problem specifications.
            completions: List of model completions (one per spec).

        Returns:
            List of RolloutResult, one per (spec, completion) pair.
        """
        assert len(problem_specs) == len(completions)

        if self.n_workers <= 1:
            return [
                self.run_completion(spec, comp)
                for spec, comp in zip(problem_specs, completions)
            ]

        results = [None] * len(problem_specs)
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_idx = {
                executor.submit(_run_completion_worker, spec, comp): i
                for i, (spec, comp) in enumerate(zip(problem_specs, completions))
            }
            for future in as_completed(future_to_idx, timeout=self.timeout_per_step * self.max_steps):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=5.0)
                except Exception as e:
                    log.warning(f"Worker failed for rollout {idx}: {e}")
                    results[idx] = _make_failed_rollout(problem_specs[idx], str(e))

        return [r for r in results if r is not None]

    def _split_completion_into_steps(self, completion: str) -> list[str]:
        """Split a multi-step completion into individual step texts.

        Each step is separated by FEA feedback markers or can be identified
        by grammar action boundaries.
        """
        # Try splitting on common step delimiters
        for delimiter in ["[FEA Result]", "[FEA executed", "\n\n"]:
            if delimiter in completion:
                parts = completion.split(delimiter)
                return [p.strip() for p in parts if p.strip()]

        # Fall back: one step per action line
        lines = completion.split("\n")
        steps = []
        current = []
        for line in lines:
            current.append(line)
            if COMBINED_PATTERN.search(line):
                steps.append("\n".join(current))
                current = []
        if current:
            steps.append("\n".join(current))
        return steps if steps else [completion]

    def _ensure_designbench_path(self) -> None:
        db_path = str(DESIGNBENCH_PATH)
        if db_path not in sys.path:
            sys.path.insert(0, db_path)


def parse_grammar_action(text: str) -> Optional[str]:
    """Extract the first semantically valid grammar action from model output text.

    Handles:
    - Text with <think>…</think> preamble
    - Multiple actions on separate lines
    - Actions embedded in longer text
    - Invalid placeholders/ranges by returning None

    Args:
        text: Raw model output.

    Returns:
        First grammar action string, or None if none found.
    """
    return extract_action(text)


# ── Worker functions (module-level for pickling) ──────────────────────────────

def _ensure_path():
    db_path = str(DESIGNBENCH_PATH)
    if db_path not in sys.path:
        sys.path.insert(0, db_path)


def _load_truss_and_goals(problem_spec: dict):
    """Load Truss object from problem spec dict."""
    _ensure_path()
    try:
        from validation.truss_executor import load_truss_from_problem
        truss = load_truss_from_problem(problem_spec)
        goals = problem_spec.get("goals", {})
        return truss, goals
    except ImportError:
        raise ImportError(
            "DesignBench not importable. Ensure DESIGNBENCH_PATH is correct and "
            "the DesignBench conda environment is active."
        )


_ALL_MEMBERS = re.compile(
    r"^\s*SCALE_PARAM\s*\(\s*all_members\s*,\s*(\w+)\s*,\s*([\d.]+)\s*\)\s*$",
    re.IGNORECASE,
)


# Flat form: ADD_MEMBER(j1, j2, material, Shape, p1, p2[, p3...])
# Executor form: ADD_MEMBER(j1, j2, material, Shape(a=p1, b=p2))
_ADD_FLAT = re.compile(
    r"^\s*ADD_MEMBER\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*([^,()]+?)\s*,\s*(\w+)\s*,\s*"
    r"([\d.eE+-]+(?:\s*,\s*[\d.eE+-]+)*)\s*\)\s*$",
    re.IGNORECASE,
)
# Shape parameter names, in the order the flat form lists them.
_SHAPE_PARAMS = {
    "pipe": ("r", "t"),
    "box": ("w", "h", "t"),
    "bar": ("r",),
    "square": ("side",),
}


def normalize_action(truss, action_str: str) -> str:
    """Rewrite whole-structure actions into a form the executor accepts.

    ``SCALE_PARAM(all_members, thickness, 1.224)`` appears in **27.5% of the gold
    SFT actions** (3295 of 11964) and in the few-shot example of the system
    prompt, but DesignBench's executor matches the member id with ``(\d+)``
    (validation/truss_executor.py:214), so every one of them falls through to
    "unknown action" and silently does nothing. A quarter of the supervised signal
    has been teaching an action that cannot execute, which is a large part of the
    ~0.66 execute rates this study attributed to "grammar drift".

    The intent is unambiguous and is exactly the whole-structure move that both
    fully-stressed design and the potential-guided search converge on, so it is
    expanded here rather than discarded. Rewriting in our own action layer leaves
    the benchmark repository untouched.
    """
    text = action_str or ""

    match = _ALL_MEMBERS.match(text)
    if match:
        n = len(getattr(truss, "members", []) or [])
        if n == 0:
            return text
        param, factor = match.group(1), match.group(2)
        ids = ",".join(str(i) for i in range(n))
        return f"SCALE_MULTI_PARAM([{ids}], [{param}:{factor}])"

    # ADD_MEMBER's flat form is what the gold SFT data and CLAUDE.md both use
    # (ADD_MEMBER(1, 6, A36_Steel, Pipe, 0.0232, 0.0039)), but the executor's
    # regex requires the shape's parameters inside the shape call
    # (validation/truss_executor.py:277). Roughly 26.5% of gold actions are in
    # the flat form, so they parsed as "unknown action" and silently added
    # nothing -- which together with all_members means over half the supervised
    # actions never touched the design.
    match = _ADD_FLAT.match(text)
    if match:
        j1, j2, material, shape = match.group(1), match.group(2), match.group(3), match.group(4)
        values = [v.strip() for v in match.group(5).split(",")]
        names = _SHAPE_PARAMS.get(shape.lower())
        if names and len(values) >= len(names):
            params = ", ".join(f"{k}={v}" for k, v in zip(names, values))
            return f"ADD_MEMBER({j1}, {j2}, {material}, {shape}({params}))"
    return text


def _apply_action(truss, action_str: str):
    """Apply a grammar action to a truss object (in-place)."""
    _ensure_path()
    from validation.truss_executor import execute_grammar_action
    execute_grammar_action(truss, normalize_action(truss, action_str))
    return truss


def _analyze_truss(truss, goals: dict) -> dict:
    """Run FEA and return state dict."""
    _ensure_path()
    from validation.truss_executor import analyze_truss
    return _sanitize_fea_state(analyze_truss(truss, goals), goals)


def _finite_or_default(value: object, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return numeric if math.isfinite(numeric) else default


def _sanitize_fea_state(state: dict, goals: dict) -> dict:
    """Clamp non-finite FEA outputs into bounded, explicitly bad states.

    DesignBench occasionally returns inf/nan for structurally invalid designs.
    RL can tolerate bad rollouts, but not non-finite scalars in rewards.
    """
    clean = dict(state or {})
    reasons: list[str] = []

    max_mass = _finite_or_default((goals or {}).get("maximum_mass"), 0.0)
    penalty_mass = max(max_mass * 10.0, 1.0) if max_mass > 0 else 1_000_000.0
    mass = _finite_or_default(clean.get("mass"), penalty_mass)
    if mass <= 0.0 or mass >= penalty_mass:
        reasons.append("mass")
        mass = penalty_mass
    clean["mass"] = mass

    fos_cap = 15.0
    for key in ("fos_buckling", "fos_yielding"):
        raw = clean.get(key)
        value = _finite_or_default(raw, 0.0)
        if value != raw or value < 0.0 or value > fos_cap:
            reasons.append(key)
        clean[key] = max(0.0, min(fos_cap, value))

    max_deflection = _finite_or_default((goals or {}).get("maximum_deflection"), 0.01)
    max_deflection = max(max_deflection, 1e-6)
    deflection_cap = max_deflection * 100.0
    deflection = _finite_or_default(clean.get("deflection"), deflection_cap)
    if deflection < 0.0 or deflection > deflection_cap:
        reasons.append("deflection")
    clean["deflection"] = max(0.0, min(deflection_cap, deflection))

    if reasons:
        clean["is_feasible"] = False
        clean["fea_sanitized"] = True
        clean["fea_sanitize_reasons"] = sorted(set(reasons))
    else:
        clean.setdefault("fea_sanitized", False)

    return clean


def _execute_step_in_worker(
    problem_spec: dict, current_state: dict, action_str: str
) -> dict:
    """Execute a single step in a worker process (picklable)."""
    _ensure_path()
    from validation.truss_executor import (
        analyze_truss,
        execute_grammar_action,
        load_truss_from_problem,
    )
    # Rebuild truss from spec (stateless: re-apply all previous actions)
    truss = load_truss_from_problem(problem_spec)
    execute_grammar_action(truss, action_str)
    goals = problem_spec.get("goals", {})
    state = analyze_truss(truss, goals)
    return {"state": state, "info": {}}


def _run_completion_worker(problem_spec: dict, completion: str):
    """Worker function for parallel batch rollout execution."""
    env = TrussRolloutEnv(max_steps=20, n_workers=1)
    return env.run_completion(problem_spec, completion)


def _make_failed_rollout(problem_spec: dict, error_msg: str):
    from llm_finetune.training.rl.rewards import RolloutResult
    return RolloutResult(
        problem_id=problem_spec.get("problem_id", ""),
        action_sequence=[],
        raw_outputs=[],
        state_history=[],
        final_state={},
        initial_state={},
        token_counts=[],
        parse_success=[],
        reaches_solution=False,
        error_message=error_msg,
    )
