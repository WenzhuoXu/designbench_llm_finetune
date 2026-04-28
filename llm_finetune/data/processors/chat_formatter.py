"""
Chat template formatting for all DesignBench target models.

Handles model-specific chat templates and thinking token conventions:
  - Qwen3-14B:           enable_thinking=True, <think>…</think> prefix
  - DeepSeek-R1-14B:     <think>\n…\n</think> before answer
  - Phi-4-reasoning:     <think>…</think> block
  - Llama-4-Scout-17B:   standard instruct, no thinking tokens
  - Phi-4 / Gemma-3 / Qwen2.5:  standard instruct

All models use tokenizer.apply_chat_template() for consistency.

Usage:
    formatter = ChatFormatter.from_model_id("Qwen/Qwen3-14B", tokenizer)
    messages = formatter.build_messages(problem_text, action_history, fea_feedback)
    input_ids = formatter.apply_template(messages, add_generation_prompt=True)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from transformers import PreTrainedTokenizer


class ThinkingMode(Enum):
    NONE = "none"           # Standard instruct models
    QWEN3 = "qwen3"         # Qwen3: apply_chat_template with enable_thinking=True
    DEEPSEEK_R1 = "deepseek_r1"   # DeepSeek-R1: <think>\n…\n</think> block
    PHI4_REASONING = "phi4_reasoning"  # Phi-4-reasoning: <think>…</think>


# Map HF model IDs → thinking mode
MODEL_THINKING_MODE: dict[str, ThinkingMode] = {
    "Qwen/Qwen3-14B": ThinkingMode.QWEN3,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": ThinkingMode.DEEPSEEK_R1,
    "microsoft/Phi-4-reasoning": ThinkingMode.PHI4_REASONING,
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": ThinkingMode.NONE,
    "microsoft/phi-4": ThinkingMode.NONE,
    "google/gemma-3-12b-it": ThinkingMode.NONE,
    "Qwen/Qwen2.5-14B-Instruct": ThinkingMode.NONE,
}

# Truss grammar system prompt template
TRUSS_SYSTEM_PROMPT = """You are an expert structural engineer specializing in truss optimization.
You will be given a truss design problem and asked to iteratively modify the design until it satisfies all structural constraints.

Available grammar actions:
  SCALE_PARAM(member_id, param, factor)           — multiply a parameter by factor
  SCALE_MULTI_PARAM([ids], [param:factor, ...])    — scale multiple params simultaneously
  ADD_MEMBER(j1, j2, material, shape, r, t)        — add a new pipe member
  MODIFY_PARAM(member_id, param, old_val, new_val) — set parameter to exact value
  REMOVE_MEMBER(member_id)                          — remove a member
  MOVE_JOINT(joint_id, [x_old, y_old], [x_new, y_new]) — reposition a joint

Parameters: r (outer radius, meters), t (wall thickness, meters)
Material: 6061_T6_Aluminum | Shape: Pipe

After each action, you will receive updated FEA results (mass, FOS_buckling, FOS_yielding, feasible).
Respond with a single grammar action per turn. Think step-by-step about structural mechanics."""


@dataclass
class ChatFormatter:
    """Formats truss design conversations for a specific model's chat template."""
    tokenizer: PreTrainedTokenizer
    thinking_mode: ThinkingMode
    model_id: str

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        tokenizer: PreTrainedTokenizer,
        thinking_mode: Optional[ThinkingMode] = None,
    ) -> "ChatFormatter":
        if thinking_mode is None:
            thinking_mode = MODEL_THINKING_MODE.get(model_id, ThinkingMode.NONE)
        return cls(tokenizer=tokenizer, thinking_mode=thinking_mode, model_id=model_id)

    def build_messages(
        self,
        problem_text: str,
        action_history: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
    ) -> list[dict]:
        """Build OpenAI-style message list for a truss design conversation.

        Args:
            problem_text: Full problem description (system prompt content from SFT JSONL).
            action_history: List of prior turns, each:
                {"action": str, "fea_result": dict, "thinking": str (optional)}
            add_generation_prompt: Whether this is for generation (True) or training (False).

        Returns:
            List of {"role": ..., "content": ...} dicts.
        """
        messages = [
            {"role": "system", "content": TRUSS_SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ]

        if action_history:
            for turn in action_history:
                assistant_content = self._format_assistant_turn(
                    action=turn["action"],
                    thinking=turn.get("thinking", ""),
                )
                messages.append({"role": "assistant", "content": assistant_content})

                if "fea_result" in turn:
                    messages.append({
                        "role": "user",
                        "content": self._format_fea_feedback(turn["fea_result"]),
                    })

        return messages

    def apply_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        return_tensors: Optional[str] = None,
    ):
        """Apply model-specific chat template.

        For Qwen3: passes enable_thinking=True to apply_chat_template.
        For others: standard apply_chat_template call.

        Returns:
            list[int] of token IDs (if tokenize=True) or formatted string.
            Always returns a plain list when tokenize=True — handles both
            old (list) and new (BatchEncoding) transformers return formats.
        """
        kwargs = dict(
            conversation=messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_tensors=return_tensors,
        )

        if self.thinking_mode == ThinkingMode.QWEN3:
            # Qwen3 apply_chat_template supports enable_thinking natively
            try:
                kwargs["enable_thinking"] = True
            except Exception:
                pass

        result = self.tokenizer.apply_chat_template(**kwargs)

        # transformers ≥ 5.x returns BatchEncoding instead of a plain list
        # when tokenize=True. Always unwrap to list[int] for consistency.
        if tokenize and not isinstance(result, (list, str)):
            if hasattr(result, "input_ids"):
                result = result.input_ids
            elif hasattr(result, "__getitem__"):
                result = result["input_ids"]

        return result

    def _format_assistant_turn(self, action: str, thinking: str = "") -> str:
        """Format assistant response with optional thinking tokens."""
        if self.thinking_mode == ThinkingMode.NONE:
            return action.strip()

        elif self.thinking_mode == ThinkingMode.QWEN3:
            if thinking:
                return f"<think>\n{thinking.strip()}\n</think>\n{action.strip()}"
            return action.strip()

        elif self.thinking_mode == ThinkingMode.DEEPSEEK_R1:
            if thinking:
                return f"<think>\n{thinking.strip()}\n</think>\n{action.strip()}"
            return f"<think>\n</think>\n{action.strip()}"

        elif self.thinking_mode == ThinkingMode.PHI4_REASONING:
            if thinking:
                return f"<think>{thinking.strip()}</think>\n{action.strip()}"
            return action.strip()

        return action.strip()

    def _format_fea_feedback(self, fea_result: dict) -> str:
        """Format FEA result as structured feedback for the next user turn."""
        lines = ["[FEA Result]"]
        if "mass" in fea_result:
            lines.append(f"  Mass:          {fea_result['mass']:.4f} kg")
        if "fos_buckling" in fea_result:
            lines.append(f"  FOS_buckling:  {fea_result['fos_buckling']:.4f}")
        if "fos_yielding" in fea_result:
            lines.append(f"  FOS_yielding:  {fea_result['fos_yielding']:.4f}")
        if "deflection" in fea_result:
            lines.append(f"  Deflection:    {fea_result['deflection']:.6f} m")
        if "is_feasible" in fea_result:
            status = "FEASIBLE" if fea_result["is_feasible"] else "INFEASIBLE"
            lines.append(f"  Status:        {status}")
        if "error" in fea_result:
            lines.append(f"  Error:         {fea_result['error']}")
        lines.append("\nContinue optimizing. Provide your next grammar action.")
        return "\n".join(lines)

    def extract_thinking(self, text: str) -> tuple[str, str]:
        """Extract (thinking, action) from model output.

        Returns:
            (thinking_text, action_text) — thinking may be empty string.
        """
        if self.thinking_mode == ThinkingMode.NONE:
            return "", text.strip()

        # Match <think>…</think> block (greedy to get full thinking)
        match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            action = match.group(2).strip()
            return thinking, action

        return "", text.strip()
