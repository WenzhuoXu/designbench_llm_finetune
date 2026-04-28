"""One-step online posterior evaluator with live LLM candidates."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

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


@dataclass
class PosteriorEvalConfig:
    sampling_mode: str = "rollin_policy"
    n_states: int = 8
    rollout_steps: int = 2
    rollout_mixed_random_prob: float = 0.25
    candidate_count: int = 4
    candidate_pool_size: int = 12
    max_new_tokens: int = 192
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
        self.model.eval()

    def evaluate(self) -> list[PosteriorEvalRecord]:
        records: list[PosteriorEvalRecord] = []
        contexts = self.sample_contexts()
        log.info("Posterior evaluator sampled %d contexts", len(contexts))
        for context in contexts:
            record = self.evaluate_context(context)
            if record is not None:
                records.append(record)
        return records

    def sample_contexts(self) -> list[PosteriorStateContext]:
        prompts = list(self.dataset.prompts)
        self.random.shuffle(prompts)
        contexts: list[PosteriorStateContext] = []
        for item in prompts:
            if len(contexts) >= self.config.n_states:
                break
            context = self._initial_context(item["problem_spec"])
            if context is None:
                continue

            if self.config.sampling_mode == "initial":
                contexts.append(context)
                continue

            rolled = self._roll_in_context(context)
            if rolled is not None:
                contexts.append(rolled)
        return contexts

    def evaluate_context(self, context: PosteriorStateContext) -> Optional[PosteriorEvalRecord]:
        root_candidates, top_model_action = self.generate_candidates(
            context,
            num_candidates=self.config.candidate_count,
            include_greedy=True,
        )
        if not root_candidates or not top_model_action:
            return None

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

        reference_candidates, _ = self.generate_candidates(
            context,
            num_candidates=max(
                self.config.reference_level1_branching,
                self.config.candidate_count,
            ),
            include_greedy=True,
        )
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
        )

    def generate_candidates(
        self,
        context: PosteriorStateContext,
        *,
        num_candidates: int,
        include_greedy: bool,
    ) -> tuple[list[CandidateAction], str]:
        messages = context.build_messages(self.formatter)
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

        with torch.no_grad():
            if include_greedy:
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

            sampled = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=self.config.temperature,
                num_return_sequences=max(num_candidates, self.config.candidate_pool_size),
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            seen = {candidate.action for candidate in candidates}
            for rank, seq in enumerate(sampled):
                sample_text = self.tokenizer.decode(
                    seq[prompt_ids.shape[1]:],
                    skip_special_tokens=False,
                )
                parsed = parse_grammar_action(sample_text)
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

        if not top_model_action and candidates:
            top_model_action = candidates[0].action
        return candidates[:num_candidates], top_model_action

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

    def _roll_in_context(self, context: PosteriorStateContext) -> Optional[PosteriorStateContext]:
        current = context
        for _ in range(self.config.rollout_steps):
            candidates, _ = self.generate_candidates(current, num_candidates=4, include_greedy=False)
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
        candidates, _ = self.generate_candidates(
            context,
            num_candidates=max(branching, self.config.candidate_pool_size),
            include_greedy=True,
        )
        return candidates

    def _greedy_continuation(
        self,
        context: PosteriorStateContext,
        depth_remaining: int,
    ) -> tuple[float, int]:
        del depth_remaining
        total_value = 0.0
        total_calls = 0
        current = context
        max_rollout = max(0, self.config.max_steps - current.n_steps)
        steps_taken = 0
        for step_idx in range(max_rollout):
            if current.current_state.get("is_feasible", False):
                break
            candidates, _ = self.generate_candidates(
                current,
                num_candidates=self.config.continuation_width,
                include_greedy=True,
            )
            if not candidates:
                break
            best_action = None
            best_reward = None
            best_context = None
            for candidate in candidates:
                next_context, reward, fea_calls = self._transition(current, candidate.action)
                total_calls += fea_calls
                if best_reward is None or reward > best_reward:
                    best_action = candidate.action
                    best_reward = reward
                    best_context = next_context
            if best_action is None or best_context is None or best_reward is None:
                break
            total_value += (self.config.gamma ** step_idx) * best_reward
            current = best_context
            steps_taken = step_idx + 1
        if current.current_state.get("is_feasible", False):
            total_value += self.config.gamma ** steps_taken * compute_potential(
                current.current_state,
                initial_mass=float(context.initial_state.get("mass", 1.0) or 1.0),
                alpha=self.config.alpha,
            )
        return total_value, total_calls

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
