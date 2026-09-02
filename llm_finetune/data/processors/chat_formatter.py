"""
Chat template formatting for all DesignBench target models.

Handles model-specific chat templates and thinking token conventions:
  - Qwen3 reasoning:     <think>…</think> prefix, with 2507 emitting only </think>
  - DeepSeek-R1:         <think>\n…\n</think> before answer
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

from typing import TYPE_CHECKING
if TYPE_CHECKING:  # `transformers` pulls in torch, minutes of cold import on
    # Lustre. Both uses below are annotations only, and this module already has
    # `from __future__ import annotations`, so they never evaluate at runtime.
    from transformers import PreTrainedTokenizer


class ThinkingMode(Enum):
    NONE = "none"           # Standard instruct models
    QWEN3 = "qwen3"         # Qwen3: apply_chat_template with enable_thinking=True
    DEEPSEEK_R1 = "deepseek_r1"   # DeepSeek-R1: <think>\n…\n</think> block
    PHI4_REASONING = "phi4_reasoning"  # Phi-4-reasoning: <think>…</think>


# Map HF model IDs → thinking mode
MODEL_THINKING_MODE: dict[str, ThinkingMode] = {
    "Qwen/Qwen3-14B": ThinkingMode.QWEN3,
    "Qwen/Qwen3-30B-A3B-Thinking-2507": ThinkingMode.QWEN3,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": ThinkingMode.DEEPSEEK_R1,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": ThinkingMode.DEEPSEEK_R1,
    "deepseek-ai/DeepSeek-R1-0528": ThinkingMode.DEEPSEEK_R1,
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": ThinkingMode.DEEPSEEK_R1,
    "microsoft/Phi-4-reasoning": ThinkingMode.PHI4_REASONING,
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": ThinkingMode.NONE,
    "microsoft/phi-4": ThinkingMode.NONE,
    "google/gemma-3-12b-it": ThinkingMode.NONE,
    "google/gemma-4-26B-A4B-it": ThinkingMode.NONE,
    "google/gemma-4-31B-it": ThinkingMode.NONE,
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

After each action, you will receive updated simulation results (mass, FOS_buckling, FOS_yielding, feasible).
Respond with exactly one grammar action per turn.
If you include reasoning, format it as:
<think>
...
</think>
<action>GRAMMAR_ACTION(...)</action>
Do not include any prose outside the optional <think> block and the final <action> block."""


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
        initial_fea_result: Optional[dict] = None,
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

        if initial_fea_result:
            messages.append({
                "role": "user",
                "content": self._format_fea_feedback(initial_fea_result),
            })

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

        if self.thinking_mode == ThinkingMode.QWEN3 and "2507" not in self.model_id:
            # Legacy Qwen3 templates expect enable_thinking=True explicitly.
            kwargs["enable_thinking"] = True

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
        wrapped_action = f"<action>{action.strip()}</action>"
        if self.thinking_mode == ThinkingMode.NONE:
            return wrapped_action

        elif self.thinking_mode == ThinkingMode.QWEN3:
            if thinking:
                return f"<think>\n{thinking.strip()}\n</think>\n{wrapped_action}"
            return wrapped_action

        elif self.thinking_mode == ThinkingMode.DEEPSEEK_R1:
            if thinking:
                return f"<think>\n{thinking.strip()}\n</think>\n{wrapped_action}"
            return f"<think>\n</think>\n{wrapped_action}"

        elif self.thinking_mode == ThinkingMode.PHI4_REASONING:
            if thinking:
                return f"<think>{thinking.strip()}</think>\n{wrapped_action}"
            return wrapped_action

        return wrapped_action

    def _format_fea_feedback(self, fea_result: dict) -> str:
        """Format FEA result as structured feedback for the next user turn."""
        lines = ["[Simulation Result]"]
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
        lines.append("\nContinue optimizing. Provide your next grammar action in a <action>...</action> block.")
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

        # Qwen3-2507 may omit the opening tag in generated text because the chat
        # template inserts it before decoding starts. In that case, split on the
        # closing tag and treat the prefix as reasoning content.
        if self.thinking_mode == ThinkingMode.QWEN3 and "</think>" in text:
            thinking, action = text.split("</think>", 1)
            return thinking.strip(), action.strip()

        return "", text.strip()
