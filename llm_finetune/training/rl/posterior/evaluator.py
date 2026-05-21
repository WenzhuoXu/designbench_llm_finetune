"""One-step online posterior evaluator with live LLM candidates."""

from __future__ import annotations

import json
import logging
import random
import hashlib
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from llm_finetune.data.datasets.rl_dataset import RLPromptDataset, _spec_to_problem_text
from llm_finetune.data.processors.chat_formatter import ChatFormatter
from llm_finetune.envs.truss_env import (
    _analyze_truss,
    _apply_action,
    _load_truss_and_goals,
    parse_grammar_action,
)
from llm_finetune.training.rl.posterior.features import (
    action_class,
    difficulty_score,
    extract_state_features,
)
from llm_finetune.training.rl.posterior.potential import compute_potential, compute_step_reward
from llm_finetune.training.rl.posterior.state_context import PosteriorStateContext, PosteriorTurn
from llm_finetune.training.rl.posterior.tree_expansion import (
    CandidateAction,
    TreeSearchResult,
    evaluate_tree,
)

log = logging.getLogger(__name__)


def _node_token(action: str) -> str:
    cls = action_class(action) if action else "ROOT"
    digest = hashlib.sha1(action.encode("utf-8")).hexdigest()[:10] if action else "root"
    return f"{cls}:{digest}"


def _node_id(problem_id: str, ablation_key: str, step_t: Optional[int], actions: list[str]) -> str:
    payload = json.dumps(
        {
            "problem_id": problem_id,
            "ablation_key": ablation_key,
            "step_t": step_t,
            "actions": actions,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class PosteriorEvalConfig:
    sampling_mode: str = "rollin_policy"
    n_states: int = 8
    rollout_steps: int = 2
    rollout_mixed_random_prob: float = 0.25
    candidate_count: int = 4
    candidate_pool_size: int = 12
    max_new_tokens: int = 3072
    temperature: float = 0.8
    depth: int = 1
    reference_depth: int = 2
    level1_branching: int = 5
    deeper_branching: int = 3
    reference_level1_branching: int = 5
    reference_deeper_branching: int = 5
    gamma: float = 0.99
    alpha: float = 5.0
    max_steps: int = 9
    adaptive_entropy_threshold: float = 0.5
    continuation_width: int = 4
    max_continuation_steps: int = 9  # cap on _greedy_continuation rollout depth per leaf
    stratified: bool = True
    seed: int = 42


@dataclass
class PosteriorEvalRecord:
    problem_id: str
    sampling_mode: str
    rollout_depth: int
    state_step: int
    difficulty: float
    current_action_history: list[str]
    top_model_action: str
    top_model_class: str
    tree_best_action: str
    tree_best_class: str
    reference_best_action: str
    reference_best_class: str
    rho_tree_agreement: int
    ranking_match_vs_reference: int
    optimal_class_missed: int
    class_entropy: float
    class_coverage: int
    adaptive_expanded: bool
    tree_total_fea_calls: int
    reference_total_fea_calls: int
    tree_elapsed_s: float
    reference_elapsed_s: float
    tree_action_values: dict[str, float] = field(default_factory=dict)
    reference_action_values: dict[str, float] = field(default_factory=dict)
    feature_vector: list[float] = field(default_factory=list)
    tree_trace: dict = field(default_factory=dict)
    reference_trace: dict = field(default_factory=dict)


class _AllSeqsEOSCriteria(StoppingCriteria):
    """Stop a batched generate() call once every sequence has produced an EOS token.

    Without this, HF runs the batch until max_new_tokens even if all sequences
    already finished early, wasting GPU time.
    """

    def __init__(self, eos_token_ids: int | list[int], prompt_len: int) -> None:
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        self._eos: set[int] = set(eos_token_ids)
        self._prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated = input_ids[:, self._prompt_len:]
        return all(
            any(tok.item() in self._eos for tok in seq)
            for seq in generated
        )


class OneStepOnlinePosteriorEvaluator:
    """Live one-step evaluator for posterior tree-expansion experiments."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        formatter: ChatFormatter,
        dataset: RLPromptDataset,
        config: Optional[PosteriorEvalConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.dataset = dataset
        self.config = config or PosteriorEvalConfig()
        self.random = random.Random(self.config.seed)
        self.generation_log_path: Optional[Path] = None
        self.generation_ablation_key = ""
        self.generation_step_t: Optional[int] = None
        self.llm_call_budget: Optional[int] = None
        self.llm_calls_completed = 0
        self._progress_lock = threading.Lock()
        self._progress = {
            "stage": "initializing",
            "stage_started_at": time.perf_counter(),
            "context_index": 0,
            "context_total": 0,
            "problem_id": "",
            "records_written": 0,
        }
        if hasattr(self.model, "eval"):
            self.model.eval()

    def update_progress(self, **kwargs) -> None:
        with self._progress_lock:
            stage = kwargs.get("stage")
            if stage is not None and stage != self._progress.get("stage"):
                self._progress["stage"] = stage
                self._progress["stage_started_at"] = time.perf_counter()
            for key, value in kwargs.items():
                if key != "stage":
                    self._progress[key] = value

    def snapshot_progress(self) -> dict:
        with self._progress_lock:
            snapshot = dict(self._progress)
        snapshot["stage_elapsed_s"] = time.perf_counter() - snapshot["stage_started_at"]
        return snapshot

    def configure_generation_logging(
        self,
        *,
        output_path: str | Path | None = None,
        ablation_key: Optional[str] = None,
        step_t: Optional[int] = None,
        llm_call_budget: Optional[int] = None,
    ) -> None:
        """Configure JSONL logging for every LLM generation attempt.

        The eval script logs root candidates, but MCTS expansion and continuation
        candidates are generated inside this evaluator. Keeping the logging hook
        here makes those nested model calls visible too.
        """
        if output_path is not None:
            self.generation_log_path = Path(output_path)
        if ablation_key is not None:
            self.generation_ablation_key = ablation_key
        if step_t is not None:
            self.generation_step_t = step_t
        if llm_call_budget is not None:
            self.llm_call_budget = llm_call_budget

    def _log_generation_attempts(
        self,
        *,
        context: PosteriorStateContext,
        messages: list[dict],
        attempts: list[dict],
        stage_label: str,
    ) -> None:
        if self.generation_log_path is None or not attempts:
            return
        step_t = self.generation_step_t
        branch_path = (
            list(context.action_history[step_t:])
            if step_t is not None and step_t <= len(context.action_history)
            else []
        )
        actual_action_history = (
            list(context.action_history[:step_t]) if step_t is not None else []
        )
        context_node_id = _node_id(
            context.problem_id,
            self.generation_ablation_key,
            step_t,
            actual_action_history + branch_path,
        )
        context_parent_node_id = (
            _node_id(
                context.problem_id,
                self.generation_ablation_key,
                step_t,
                actual_action_history + branch_path[:-1],
            )
            if branch_path
            else None
        )
        context_node_tokens = [_node_token(action) for action in branch_path]
        stage_lower = stage_label.lower()
        if "tree expansion" in stage_lower:
            branch_origin = "mcts_tree_expansion"
        elif "greedy continuation" in stage_lower:
            branch_origin = "mcts_rollout_continuation"
        elif "pilot" in stage_lower:
            branch_origin = "pilot_root"
        elif step_t is not None and context.n_steps == step_t:
            branch_origin = "actual_trajectory_root"
        elif step_t is not None and context.n_steps > step_t:
            branch_origin = "mcts_simulated_branch"
        else:
            branch_origin = "unknown_generation_context"
        self.generation_log_path.parent.mkdir(parents=True, exist_ok=True)
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = str(message.get("content", ""))
                break
        with self.generation_log_path.open("a", encoding="utf-8") as f:
            for attempt in attempts:
                raw = str(attempt.get("raw_text", ""))
                parsed_action = str(attempt.get("parsed_action", ""))
                node_actions = (
                    actual_action_history + branch_path + [parsed_action]
                    if parsed_action
                    else actual_action_history + branch_path
                )
                node_id = _node_id(
                    context.problem_id,
                    self.generation_ablation_key,
                    step_t,
                    node_actions,
                )
                parent_node_id = context_node_id if parsed_action else context_parent_node_id
                thinking = ""
                if "<think>" in raw and "</think>" in raw:
                    thinking = raw.split("<think>", 1)[1].split("</think>", 1)[0].strip()
                record = {
                    "problem_id": context.problem_id,
                    "ablation_key": self.generation_ablation_key,
                    "step_t": self.generation_step_t,
                    "context_n_steps": context.n_steps,
                    "record_type": "llm_generation",
                    "stage_label": stage_label,
                    "branch_origin": branch_origin,
                    "node_id": node_id,
                    "parent_node_id": parent_node_id,
                    "node_depth": len(branch_path) + (1 if parsed_action else 0),
                    "node_action_tokens": (
                        context_node_tokens + [_node_token(parsed_action)]
                        if parsed_action
                        else context_node_tokens
                    ),
                    "context_node_id": context_node_id,
                    "context_parent_node_id": context_parent_node_id,
                    "is_actual_trajectory_context": bool(
                        step_t is not None and context.n_steps == step_t
                    ),
                    "is_mcts_simulation_context": bool(
                        step_t is not None and context.n_steps > step_t
                    ),
                    "mcts_depth_from_step": (
                        context.n_steps - step_t if step_t is not None else None
                    ),
                    "actual_action_history": actual_action_history,
                    "branch_path_from_step": branch_path,
                    "context_action_history": list(context.action_history),
                    "selected_for_actual_transition": False,
                    "candidate_rank": int(attempt.get("rank", 0) or 0),
                    "source": str(attempt.get("source", "")),
                    "parse_success": bool(attempt.get("parse_success", False)),
                    "parsed_action": parsed_action,
                    "action_class": action_class(parsed_action) if parsed_action else "",
                    "raw_output": raw,
                    "thinking": thinking,
                    "prompt_messages": messages,
                    "prompt_last_user_message": last_user[:400],
                }
                f.write(json.dumps(record) + "\n")

    def _branch_summary(self, context: PosteriorStateContext, stage_label: str) -> dict:
        step_t = self.generation_step_t
        branch_path = (
            list(context.action_history[step_t:])
            if step_t is not None and step_t <= len(context.action_history)
            else []
        )
        actual_action_history = (
            list(context.action_history[:step_t]) if step_t is not None else []
        )
        context_node_id = _node_id(
            context.problem_id,
            self.generation_ablation_key,
            step_t,
            actual_action_history + branch_path,
        )
        context_parent_node_id = (
            _node_id(
                context.problem_id,
                self.generation_ablation_key,
                step_t,
                actual_action_history + branch_path[:-1],
            )
            if branch_path
            else None
        )
        stage_lower = stage_label.lower()
        if "tree expansion" in stage_lower:
            branch_origin = "mcts_tree_expansion"
        elif "greedy continuation" in stage_lower:
            branch_origin = "mcts_rollout_continuation"
        elif "pilot" in stage_lower:
            branch_origin = "pilot_root"
        elif step_t is not None and context.n_steps == step_t:
            branch_origin = "actual_trajectory_root"
        elif step_t is not None and context.n_steps > step_t:
            branch_origin = "mcts_simulated_branch"
        else:
            branch_origin = "unknown_generation_context"
        return {
            "branch_origin": branch_origin,
            "context_node_id": context_node_id,
            "context_parent_node_id": context_parent_node_id,
            "context_node_depth": len(branch_path),
            "context_node_action_tokens": [_node_token(action) for action in branch_path],
            "depth": context.n_steps - step_t if step_t is not None else None,
            "branch_path": branch_path,
        }

    def _print_generation_summary(
        self,
        *,
        context: PosteriorStateContext,
        stage_label: str,
        attempts: list[dict],
        candidates: list[CandidateAction],
        num_candidates: int,
        llm_call_units: int,
        provider_note: str,
    ) -> None:
        self.llm_calls_completed += llm_call_units
        remaining = (
            max(self.llm_call_budget - self.llm_calls_completed, 0)
            if self.llm_call_budget is not None
            else None
        )
        branch = self._branch_summary(context, stage_label)
        parsed = sum(1 for attempt in attempts if attempt.get("parse_success"))
        remaining_text = str(remaining) if remaining is not None else "unknown"
        depth_text = str(branch["depth"]) if branch["depth"] is not None else "unknown"
        path = branch["branch_path"]
        path_text = "ROOT" if not path else " -> ".join(path[-3:])
        log.info(
            "LLM progress | calls +%d (%d/%s, left=%s) | provider=%s | "
            "branch=%s context_node=%s parent=%s depth=%s step=%s problem=%s | stage=%s | "
            "parsed=%d/%d selected_candidates=%d/%d | path=%s",
            llm_call_units,
            self.llm_calls_completed,
            self.llm_call_budget if self.llm_call_budget is not None else "?",
            remaining_text,
            provider_note,
            branch["branch_origin"],
            branch["context_node_id"],
            branch["context_parent_node_id"] or "none",
            depth_text,
            self.generation_step_t,
            context.problem_id,
            stage_label,
            parsed,
            len(attempts),
            len(candidates),
            num_candidates,
            path_text,
        )

    def log_mcts_decision(
        self,
        *,
        problem_id: str,
        ablation_key: str,
        step_t: int,
        selected_action: str,
        selected_source: str,
        top_model_action: str,
        tree_best_action: str,
        candidate_actions: list[str],
        actual_action_history: Optional[list[str]] = None,
        phi_before: float,
        phi_after: float,
        reward: float,
        tree_trace: Optional[dict] = None,
    ) -> None:
        if self.generation_log_path is None:
            return
        actual_history = list(actual_action_history or [])
        node_id = _node_id(problem_id, ablation_key, step_t, actual_history + [selected_action])
        parent_node_id = _node_id(problem_id, ablation_key, step_t, actual_history)
        self.generation_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "problem_id": problem_id,
            "ablation_key": ablation_key,
            "step_t": step_t,
            "context_n_steps": step_t,
            "record_type": "actual_transition_decision",
            "stage_label": f"{ablation_key} step {step_t} decision",
            "branch_origin": "actual_trajectory_selected_transition",
            "node_id": node_id,
            "parent_node_id": parent_node_id,
            "node_depth": 1,
            "node_action_tokens": [_node_token(selected_action)] if selected_action else [],
            "context_node_id": parent_node_id,
            "is_actual_trajectory_context": True,
            "is_mcts_simulation_context": False,
            "mcts_depth_from_step": 0,
            "actual_action_history": actual_history,
            "branch_path_from_step": [selected_action] if selected_action else [],
            "context_action_history": actual_history,
            "selected_for_actual_transition": True,
            "selected_action": selected_action,
            "selected_action_class": action_class(selected_action) if selected_action else "",
            "selected_source": selected_source,
            "top_model_action": top_model_action,
            "tree_best_action": tree_best_action,
            "candidate_actions": candidate_actions,
            "phi_before": phi_before,
            "phi_after": phi_after,
            "reward": reward,
            "tree_trace": tree_trace or {},
        }
        with self.generation_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def evaluate(self) -> list[PosteriorEvalRecord]:
        records: list[PosteriorEvalRecord] = []
        contexts = self.sample_contexts()
        log.info("Posterior evaluator sampled %d contexts", len(contexts))
        started_at = time.perf_counter()
        total = len(contexts)
        for index, context in enumerate(contexts, start=1):
            context_start = time.perf_counter()
            log.info(
                "Evaluating context %d/%d: problem_id=%s step=%s history_len=%d feasible=%s",
                index,
                total,
                context.problem_id,
                context.current_state.get("step", 0),
                len(context.action_history),
                context.current_state.get("is_feasible", False),
            )
            record = self.evaluate_context(context)
            if record is not None:
                records.append(record)
                context_elapsed = time.perf_counter() - context_start
                total_elapsed = time.perf_counter() - started_at
                remaining = total - index
                eta_seconds = (total_elapsed / index) * remaining if index else 0.0
                log.info(
                    "Completed context %d/%d in %.1fs: tree_best=%s reference_best=%s "
                    "tree_fea=%d reference_fea=%d tree_elapsed=%.1fs reference_elapsed=%.1fs "
                    "records=%d eta=%.1f min",
                    index,
                    total,
                    context_elapsed,
                    record.tree_best_class,
                    record.reference_best_class,
                    record.tree_total_fea_calls,
                    record.reference_total_fea_calls,
                    record.tree_elapsed_s,
                    record.reference_elapsed_s,
                    len(records),
                    eta_seconds / 60.0,
                )
            else:
                context_elapsed = time.perf_counter() - context_start
                log.warning(
                    "Context %d/%d produced no record after %.1fs: problem_id=%s",
                    index,
                    total,
                    context_elapsed,
                    context.problem_id,
                )
        return records

    def sample_contexts(self) -> list[PosteriorStateContext]:
        self.update_progress(stage="sampling contexts")
        prompts = list(self.dataset.prompts)
        self.random.shuffle(prompts)
        contexts: list[PosteriorStateContext] = []
        for prompt_index, item in enumerate(prompts, start=1):
            if len(contexts) >= self.config.n_states:
                break
            self.update_progress(
                stage=f"sampling context candidates ({len(contexts) + 1}/{self.config.n_states})",
                problem_id=item["problem_spec"].get("problem_id", ""),
            )
            context = self._initial_context(item["problem_spec"])
            if context is None:
                continue

            if self.config.sampling_mode == "initial":
                contexts.append(context)
                continue

            rolled = self._roll_in_context(context, sample_index=prompt_index)
            if rolled is not None:
                contexts.append(rolled)
        return contexts

    def evaluate_context(self, context: PosteriorStateContext) -> Optional[PosteriorEvalRecord]:
        self.update_progress(stage="root candidate generation", problem_id=context.problem_id)
        root_candidates, top_model_action, _ = self.generate_candidates(
            context,
            num_candidates=self.config.candidate_count,
            include_greedy=True,
            stage_label="root candidate generation",
        )
        if not root_candidates or not top_model_action:
            return None

        self.update_progress(stage="tree search", problem_id=context.problem_id)
        tree_result = evaluate_tree(
            root_node=context,
            root_candidates=root_candidates,
            transition_fn=self._transition,
            continuation_fn=self._greedy_continuation,
            sampler_fn=self._sample_for_tree,
            depth=self.config.depth,
            level1_branching=self.config.level1_branching,
            deeper_branching=self.config.deeper_branching,
            gamma=self.config.gamma,
            adaptive_entropy_threshold=self.config.adaptive_entropy_threshold,
            stratified=self.config.stratified,
        )

        self.update_progress(stage="reference candidate generation", problem_id=context.problem_id)
        reference_candidates, _, _ = self.generate_candidates(
            context,
            num_candidates=max(
                self.config.reference_level1_branching,
                self.config.candidate_count,
            ),
            include_greedy=True,
            stage_label="reference candidate generation",
        )
        self.update_progress(stage="reference tree search", problem_id=context.problem_id)
        reference_result = evaluate_tree(
            root_node=context,
            root_candidates=reference_candidates,
            transition_fn=self._transition,
            continuation_fn=self._greedy_continuation,
            sampler_fn=self._sample_for_tree,
            depth=self.config.reference_depth,
            level1_branching=self.config.reference_level1_branching,
            deeper_branching=self.config.reference_deeper_branching,
            gamma=self.config.gamma,
            adaptive_entropy_threshold=None,
            stratified=True,
        )

        if not tree_result.evaluated or not reference_result.evaluated:
            return None

        self.update_progress(stage="finalizing record", problem_id=context.problem_id)
        reference_best = reference_result.evaluated[0]
        feature_vector = extract_state_features(
            context.current_state,
            initial_mass=float(context.initial_state.get("mass", 1.0) or 1.0),
            mean_horizon=self.config.max_steps,
        )
        tree_classes = {item.candidate.cls for item in tree_result.evaluated}
        return PosteriorEvalRecord(
            problem_id=context.problem_id,
            sampling_mode=self.config.sampling_mode,
            rollout_depth=self.config.depth,
            state_step=context.current_state.get("step", 0),
            difficulty=difficulty_score(context.current_state),
            current_action_history=list(context.action_history),
            top_model_action=top_model_action,
            top_model_class=action_class(top_model_action),
            tree_best_action=tree_result.tree_best_action,
            tree_best_class=tree_result.tree_best_class,
            reference_best_action=reference_best.candidate.action,
            reference_best_class=reference_best.candidate.cls,
            rho_tree_agreement=int(top_model_action == tree_result.tree_best_action),
            ranking_match_vs_reference=int(tree_result.tree_best_action == reference_best.candidate.action),
            optimal_class_missed=int(reference_best.candidate.cls not in tree_classes),
            class_entropy=tree_result.class_entropy,
            class_coverage=tree_result.class_coverage,
            adaptive_expanded=tree_result.adaptive_expanded,
            tree_total_fea_calls=tree_result.total_fea_calls,
            reference_total_fea_calls=reference_result.total_fea_calls,
            tree_elapsed_s=tree_result.total_elapsed_s,
            reference_elapsed_s=reference_result.total_elapsed_s,
            tree_action_values=tree_result.action_values,
            reference_action_values=reference_result.action_values,
            feature_vector=feature_vector,
            tree_trace=tree_result.trace,
            reference_trace=reference_result.trace,
        )

    def generate_candidates(
        self,
        context: PosteriorStateContext,
        *,
        num_candidates: int,
        include_greedy: bool,
        stage_label: str = "candidate generation",
        prompt_suffix: str = "",
    ) -> tuple[list[CandidateAction], str, list[dict]]:
        messages = context.build_messages(self.formatter)
        if prompt_suffix:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i] = {**messages[i], "content": messages[i]["content"] + prompt_suffix}
                    break
        if hasattr(self.model, "generate_candidates"):
            candidates, top_model_action, generation_attempts = self.model.generate_candidates(
                context,
                formatter=self.formatter,
                config=self.config,
                num_candidates=num_candidates,
                include_greedy=include_greedy,
                stage_label=stage_label,
                update_progress=self.update_progress,
                prompt_suffix=prompt_suffix,
            )
            self._log_generation_attempts(
                context=context,
                messages=messages,
                attempts=generation_attempts,
                stage_label=stage_label,
            )
            self._print_generation_summary(
                context=context,
                stage_label=stage_label,
                attempts=generation_attempts,
                candidates=candidates,
                num_candidates=num_candidates,
                llm_call_units=len(generation_attempts),
                provider_note="openai-concurrent" if len(generation_attempts) > 1 else "openai",
            )
            return candidates, top_model_action, generation_attempts

        self.update_progress(stage=f"{stage_label}: formatting prompt", problem_id=context.problem_id)
        prompt_ids = self.formatter.apply_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if not torch.is_tensor(prompt_ids):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        prompt_ids = prompt_ids.to(self._model_device())
        attention_mask = torch.ones_like(prompt_ids)

        candidates: list[CandidateAction] = []
        top_model_action = ""
        generation_attempts: list[dict] = []

        with torch.no_grad():
            if include_greedy:
                self.update_progress(
                    stage=f"{stage_label}: greedy decode",
                    problem_id=context.problem_id,
                )
                greedy = self.model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=self.config.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                greedy_text = self.tokenizer.decode(
                    greedy[0][prompt_ids.shape[1]:],
                    skip_special_tokens=False,
                )
                parsed = parse_grammar_action(greedy_text)
                generation_attempts.append({
                    "source": "greedy",
                    "rank": 0,
                    "raw_text": greedy_text,
                    "parsed_action": parsed or "",
                    "parse_success": bool(parsed),
                })
                if parsed:
                    top_model_action = parsed
                    candidates.append(
                        CandidateAction(
                            action=parsed,
                            score=1.0,
                            source="greedy",
                            metadata={"raw_text": greedy_text},
                        )
                    )

            self.update_progress(
                stage=f"{stage_label}: sampled decode ({max(num_candidates, self.config.candidate_pool_size)} seqs)",
                problem_id=context.problem_id,
            )
            eos_ids = self.tokenizer.eos_token_id
            sampled = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=self.config.temperature,
                num_return_sequences=max(num_candidates, self.config.candidate_pool_size),
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_ids,
                stopping_criteria=StoppingCriteriaList([
                    _AllSeqsEOSCriteria(eos_ids, prompt_ids.shape[1])
                ]),
            )
            seen = {candidate.action for candidate in candidates}
            for rank, seq in enumerate(sampled):
                sample_text = self.tokenizer.decode(
                    seq[prompt_ids.shape[1]:],
                    skip_special_tokens=False,
                )
                parsed = parse_grammar_action(sample_text)
                generation_attempts.append({
                    "source": "sample",
                    "rank": rank,
                    "raw_text": sample_text,
                    "parsed_action": parsed or "",
                    "parse_success": bool(parsed),
                })
                if not parsed or parsed in seen:
                    continue
                seen.add(parsed)
                candidates.append(
                    CandidateAction(
                        action=parsed,
                        score=max(0.0, 0.99 - 0.01 * rank),
                        source="sample",
                        metadata={"raw_text": sample_text, "rank": rank},
                    )
                )
                if len(candidates) >= num_candidates:
                    break

        self.update_progress(
            stage=f"{stage_label}: parsed {len(candidates[:num_candidates])} candidates",
            problem_id=context.problem_id,
        )
        if not top_model_action and candidates:
            top_model_action = candidates[0].action
        self._log_generation_attempts(
            context=context,
            messages=messages,
            attempts=generation_attempts,
            stage_label=stage_label,
        )
        local_call_units = 1 + int(include_greedy)
        self._print_generation_summary(
            context=context,
            stage_label=stage_label,
            attempts=generation_attempts,
            candidates=candidates[:num_candidates],
            num_candidates=num_candidates,
            llm_call_units=local_call_units,
            provider_note=f"local-batched/{max(num_candidates, self.config.candidate_pool_size)}seq",
        )
        return candidates[:num_candidates], top_model_action, generation_attempts

    def _initial_context(self, problem_spec: dict) -> Optional[PosteriorStateContext]:
        try:
            truss, goals = _load_truss_and_goals(problem_spec)
            state = _analyze_truss(truss, goals)
        except Exception as exc:
            log.warning("Failed to initialize problem %s: %s", problem_spec.get("problem_id", ""), exc)
            return None
        state["step"] = 0
        return PosteriorStateContext(
            problem_spec=problem_spec,
            current_state=state,
            initial_state=state,
            prompt_text=_spec_to_problem_text(problem_spec),
        )

    def _roll_in_context(self, context: PosteriorStateContext, sample_index: int) -> Optional[PosteriorStateContext]:
        current = context
        for rollout_step in range(self.config.rollout_steps):
            self.update_progress(
                stage=f"roll-in step {rollout_step + 1}/{self.config.rollout_steps} for sampled context {sample_index}",
                problem_id=current.problem_id,
            )
            candidates, _, _ = self.generate_candidates(
                current,
                num_candidates=4,
                include_greedy=False,
                stage_label=f"roll-in step {rollout_step + 1}/{self.config.rollout_steps}",
            )
            if not candidates:
                return current
            chosen = candidates[0]
            if self.config.sampling_mode == "rollin_mixed" and self.random.random() < self.config.rollout_mixed_random_prob:
                chosen = self.random.choice(candidates)
            next_context, _, _ = self._transition(current, chosen.action)
            current = next_context
            if current.current_state.get("is_feasible", False):
                break
        return current

    def _transition(
        self,
        context: PosteriorStateContext,
        action: str,
    ) -> tuple[PosteriorStateContext, float, int]:
        truss, goals = _load_truss_and_goals(context.problem_spec)
        for prev_action in context.action_history:
            truss = _apply_action(truss, prev_action)
        truss = _apply_action(truss, action)
        next_state = _analyze_truss(truss, goals)
        next_state["step"] = context.n_steps + 1
        reward = compute_step_reward(
            context.current_state,
            next_state,
            initial_mass=float(context.initial_state.get("mass", 1.0) or 1.0),
            gamma=self.config.gamma,
            alpha=self.config.alpha,
        )
        next_turn = PosteriorTurn(action=action, fea_result=next_state)
        next_context = PosteriorStateContext(
            problem_spec=context.problem_spec,
            current_state=next_state,
            initial_state=context.initial_state,
            action_history=context.action_history + [action],
            turns=context.turns + [next_turn],
            prompt_text=context.prompt_text,
        )
        return next_context, reward, 1

    def _sample_for_tree(
        self,
        context: PosteriorStateContext,
        branching: int,
        depth: int,
        preferred_class: Optional[str],
    ) -> list[CandidateAction]:
        del depth, preferred_class
        self.update_progress(
            stage=f"tree expansion candidate sampling (branching={branching})",
            problem_id=context.problem_id,
        )
        candidates, _, _ = self.generate_candidates(
            context,
            num_candidates=max(branching, self.config.candidate_pool_size),
            include_greedy=True,
            stage_label=f"tree expansion sampling (branching={branching})",
        )
        return candidates

    def _greedy_continuation(
        self,
        context: PosteriorStateContext,
        depth_remaining: int,
    ) -> tuple[float, int, dict]:
        del depth_remaining
        total_value = 0.0
        total_calls = 0
        current = context
        max_rollout = min(
            self.config.max_continuation_steps,
            max(0, self.config.max_steps - current.n_steps),
        )
        steps_taken = 0
        step_traces: list[dict] = []
        terminated_reason = "max_rollout_exhausted"
        for step_idx in range(max_rollout):
            if current.current_state.get("is_feasible", False):
                terminated_reason = "already_feasible"
                break
            self.update_progress(
                stage=f"greedy continuation step {step_idx + 1}/{max_rollout}",
                problem_id=current.problem_id,
            )
            candidates, _, _ = self.generate_candidates(
                current,
                num_candidates=self.config.continuation_width,
                include_greedy=True,
                stage_label=f"greedy continuation step {step_idx + 1}/{max_rollout}",
            )
            if not candidates:
                terminated_reason = "no_candidates"
                break
            best_action = None
            best_reward = None
            best_context = None
            candidate_evaluations: list[dict] = []
            for candidate in candidates:
                next_context, reward, fea_calls = self._transition(current, candidate.action)
                total_calls += fea_calls
                candidate_evaluations.append(
                    {
                        "candidate": {
                            "action": candidate.action,
                            "cls": candidate.cls,
                            "score": candidate.score,
                            "logprob": candidate.logprob,
                            "source": candidate.source,
                            "metadata": dict(candidate.metadata),
                        },
                        "reward": reward,
                        "fea_calls": fea_calls,
                        "next_state": dict(next_context.current_state),
                    }
                )
                if best_reward is None or reward > best_reward:
                    best_action = candidate.action
                    best_reward = reward
                    best_context = next_context
            if best_action is None or best_context is None or best_reward is None:
                terminated_reason = "no_best_action"
                break
            total_value += (self.config.gamma ** step_idx) * best_reward
            step_traces.append(
                {
                    "step_index": step_idx + 1,
                    "state_before": dict(current.current_state),
                    "candidate_evaluations": candidate_evaluations,
                    "chosen_action": best_action,
                    "chosen_reward": best_reward,
                    "state_after": dict(best_context.current_state),
                    "discounted_reward_contribution": (self.config.gamma ** step_idx) * best_reward,
                }
            )
            current = best_context
            steps_taken = step_idx + 1
        if current.current_state.get("is_feasible", False):
            terminal_bonus = self.config.gamma ** steps_taken * compute_potential(
                current.current_state,
                initial_mass=float(context.initial_state.get("mass", 1.0) or 1.0),
                alpha=self.config.alpha,
            )
            total_value += terminal_bonus
            terminated_reason = "reached_feasible_state"
        else:
            terminal_bonus = 0.0
        return (
            total_value,
            total_calls,
            {
                "mode": "greedy_continuation",
                "max_rollout": max_rollout,
                "steps_taken": steps_taken,
                "terminated_reason": terminated_reason,
                "total_value": total_value,
                "total_fea_calls": total_calls,
                "terminal_bonus": terminal_bonus,
                "initial_state": dict(context.current_state),
                "final_state": dict(current.current_state),
                "steps": step_traces,
            },
        )

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")


def save_records_jsonl(records: list[PosteriorEvalRecord], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")


def append_record_jsonl(record: PosteriorEvalRecord, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")
