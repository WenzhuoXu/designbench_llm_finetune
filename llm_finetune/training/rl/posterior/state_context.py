"""Conversation reconstruction for one-step online posterior evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from llm_finetune.data.datasets.rl_dataset import _spec_to_problem_text
from llm_finetune.data.processors.chat_formatter import ChatFormatter
from llm_finetune.training.rl.posterior.features import action_class, difficulty_score


@dataclass
class PosteriorTurn:
    action: str
    fea_result: dict
    raw_output: str = ""
    thinking: str = ""


@dataclass
class PosteriorStateContext:
    """Serializable state needed to query the LLM at an intermediate step."""

    problem_spec: dict
    current_state: dict
    initial_state: dict
    action_history: list[str] = field(default_factory=list)
    turns: list[PosteriorTurn] = field(default_factory=list)
    prompt_text: Optional[str] = None
    model_scores: dict[str, float] = field(default_factory=dict)

    @property
    def problem_id(self) -> str:
        return self.problem_spec.get("problem_id", "")

    @property
    def n_steps(self) -> int:
        return len(self.action_history)

    def build_messages(self, formatter: ChatFormatter) -> list[dict]:
        history = [
            {
                "action": turn.action,
                "thinking": turn.thinking,
                "fea_result": turn.fea_result,
            }
            for turn in self.turns
        ]
        return formatter.build_messages(
            problem_text=self.prompt_text or _spec_to_problem_text(self.problem_spec),
            action_history=history,
        )

    def to_summary(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "n_steps": self.n_steps,
            "difficulty": difficulty_score(self.current_state),
            "last_action_class": action_class(self.action_history[-1]) if self.action_history else "ROOT",
            "is_feasible": bool(self.current_state.get("is_feasible", False)),
        }
