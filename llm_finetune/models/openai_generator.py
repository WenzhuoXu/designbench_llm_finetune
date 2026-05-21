"""OpenAI API candidate generation for posterior/MCTS evals."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional

from llm_finetune.data.processors.chat_formatter import ChatFormatter
from llm_finetune.envs.truss_env import parse_grammar_action
from llm_finetune.training.rl.posterior.state_context import PosteriorStateContext
from llm_finetune.training.rl.posterior.tree_expansion import CandidateAction

log = logging.getLogger(__name__)


@dataclass
class OpenAIGenerationConfig:
    model: str = "gpt-5.5"
    reasoning_effort: str = "high"
    timeout_s: float = 180.0
    max_concurrent_calls: int = 8


class OpenAICandidateGenerator:
    """Small adapter exposing the evaluator's candidate-generation interface."""

    def __init__(self, config: Optional[OpenAIGenerationConfig] = None):
        self.config = config or OpenAIGenerationConfig()
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export your OpenAI API token before running "
                "the OpenAI MCTS eval, for example: export OPENAI_API_KEY='sk-...'"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai Python package is required for api_provider=openai. "
                "Install it in my_env with: conda run -n my_env pip install openai"
            ) from exc

        self.client = OpenAI(timeout=self.config.timeout_s)
        self._supports_temperature = not self.config.model.startswith("gpt-5.5")

    def eval(self) -> "OpenAICandidateGenerator":
        return self

    def generate_candidates(
        self,
        context: PosteriorStateContext,
        *,
        formatter: ChatFormatter,
        config,
        num_candidates: int,
        include_greedy: bool,
        stage_label: str,
        update_progress: Callable[..., None],
        prompt_suffix: str = "",
    ) -> tuple[list[CandidateAction], str, list[dict]]:
        update_progress(stage=f"{stage_label}: formatting prompt", problem_id=context.problem_id)
        messages = context.build_messages(formatter)
        if prompt_suffix:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i] = {**messages[i], "content": messages[i]["content"] + prompt_suffix}
                    break

        candidates: list[CandidateAction] = []
        generation_attempts: list[dict] = []
        top_model_action = ""
        seen: set[str] = set()

        if include_greedy:
            update_progress(stage=f"{stage_label}: OpenAI greedy decode", problem_id=context.problem_id)
            raw_text = self._create_response(
                messages,
                max_output_tokens=config.max_new_tokens,
                temperature=None,
            )
            parsed = parse_grammar_action(raw_text)
            generation_attempts.append(
                {
                    "source": "openai_greedy",
                    "rank": 0,
                    "raw_text": raw_text,
                    "parsed_action": parsed or "",
                    "parse_success": bool(parsed),
                    "model": self.config.model,
                }
            )
            if parsed:
                top_model_action = parsed
                seen.add(parsed)
                candidates.append(
                    CandidateAction(
                        action=parsed,
                        score=1.0,
                        source="openai_greedy",
                        metadata={"raw_text": raw_text, "model": self.config.model},
                    )
                )

        # For paid APIs, num_candidates is a hard call budget. Do not use the
        # local-model oversampling pool here.
        pool_size = max(0, num_candidates - len(candidates))
        max_workers = max(1, min(self.config.max_concurrent_calls, pool_size))
        update_progress(
            stage=f"{stage_label}: OpenAI sampled decode ({pool_size} calls, concurrency={max_workers})",
            problem_id=context.problem_id,
        )
        sampled_results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._create_response,
                    messages,
                    max_output_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                ): rank
                for rank in range(pool_size)
            }
            for future in as_completed(futures):
                rank = futures[future]
                try:
                    sampled_results[rank] = future.result()
                except Exception as exc:
                    log.warning("OpenAI sampled decode failed at rank %d: %s", rank, exc)
                    sampled_results[rank] = ""

        for rank in range(pool_size):
            raw_text = sampled_results.get(rank, "")
            parsed = parse_grammar_action(raw_text) if raw_text else None
            generation_attempts.append(
                {
                    "source": "openai_sample",
                    "rank": rank,
                    "raw_text": raw_text,
                    "parsed_action": parsed or "",
                    "parse_success": bool(parsed),
                    "model": self.config.model,
                }
            )
            if len(candidates) >= num_candidates or not parsed or parsed in seen:
                continue
            seen.add(parsed)
            candidates.append(
                CandidateAction(
                    action=parsed,
                    score=max(0.0, 0.99 - 0.01 * rank),
                    source="openai_sample",
                    metadata={"raw_text": raw_text, "rank": rank, "model": self.config.model},
                )
            )

        update_progress(
            stage=f"{stage_label}: parsed {len(candidates[:num_candidates])} candidates",
            problem_id=context.problem_id,
        )
        if not top_model_action and candidates:
            top_model_action = candidates[0].action
        return candidates[:num_candidates], top_model_action, generation_attempts

    def _create_response(
        self,
        messages: list[dict],
        *,
        max_output_tokens: int,
        temperature: Optional[float],
    ) -> str:
        kwargs = {
            "model": self.config.model,
            "input": messages,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": self.config.reasoning_effort},
        }
        if temperature is not None and self._supports_temperature:
            kwargs["temperature"] = temperature
        try:
            response = self.client.responses.create(**kwargs)
        except Exception as exc:
            if temperature is None or not self._supports_temperature:
                raise
            log.warning("OpenAI response with temperature failed; retrying without it: %s", exc)
            self._supports_temperature = False
            kwargs.pop("temperature", None)
            response = self.client.responses.create(**kwargs)

        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text
        return str(response)


class OpenAITokenizerStub:
    """Minimal tokenizer shim for prompt dataset construction without local models."""

    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(
        self,
        conversation,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        return_tensors=None,
        **kwargs,
    ):
        del add_generation_prompt, return_tensors, kwargs
        if tokenize:
            return [0]
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in conversation)
