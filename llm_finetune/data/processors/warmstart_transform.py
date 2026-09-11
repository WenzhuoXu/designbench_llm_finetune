"""
Warmstart message transform for SFT pre-training.

Reformats DesignBench SFT JSONL messages for warmstart training:
  1. Extracts grammar actions from inside <think> tags → wraps in <action> tags after </think>
  2. Merges problem description + initial state into a single system message
  3. Remaps FEA/simulation feedback from system → user role with [Simulation Result] prefix
  4. Cleans thinking content (removes "Modification:" / "Action:" boilerplate)

This transform is intentionally lightweight — the warmstart teaches format and
turn-taking, NOT reasoning strategy. The RL stage (GRPO) learns the actual policy.

Usage:
    transform = WarmstartTransform()
    enriched_messages = transform.transform_messages(raw_messages)
"""

from __future__ import annotations

import re
from typing import Optional

from llm_finetune.data.grammar import extract_action
from llm_finetune.data.processors.chat_formatter import TRUSS_SYSTEM_PROMPT

# Grammar action patterns (from DesignBench grammar)
ACTION_PATTERN = re.compile(
    r"(SCALE_PARAM|SCALE_MULTI_PARAM|ADD_MEMBER|MODIFY_PARAM|REMOVE_MEMBER|MOVE_JOINT|OPTIMAL_STATE)"
    r"(?:\([^)]*\))?"
)

# Pattern to extract action line from shallow <think> content
# Matches: "Action: SOME_ACTION(...)" or just "SOME_ACTION(...)"
ACTION_LINE_PATTERN = re.compile(
    r"(?:Action:\s*)?"
    r"((?:SCALE_PARAM|SCALE_MULTI_PARAM|ADD_MEMBER|MODIFY_PARAM|REMOVE_MEMBER|MOVE_JOINT|OPTIMAL_STATE)"
    r"(?:\([^)]*\))?)"
)

# Pattern to extract modification description
MODIFICATION_PATTERN = re.compile(r"Modification:\s*(.+?)(?:\n|$)")


class WarmstartTransform:
    """Reformats DesignBench SFT messages for warmstart training.

    Input message sequence (from DesignBench SFT JSONL):
        [0] system: problem description
        [1] system: initial state analysis
        [2] assistant: <think>Modification: X\nAction: Y</think>
        [3] system: FEA result
        [4] assistant: <think>Modification: X\nAction: Y</think>
        ...
        [N-1] system: final FEA result
        [N]   assistant: <answer>...</answer>

    Output message sequence:
        [0] system: truss grammar instructions
        [1] user: problem description + initial state (merged)
        [2] assistant: <think>description</think>\n<action>Y</action>
        [3] user: [Simulation Result] FEA result
        [4] assistant: <think>description</think>\n<action>Y</action>
        ...
        [N-1] user: [Simulation Result] final FEA result
        [N]   assistant: <answer>...</answer>
    """

    SIMULATION_PREFIX = "[Simulation Result]\n"

    def transform_messages(self, messages: list[dict]) -> list[dict]:
        """Transform a DesignBench SFT message list for warmstart training.

        Args:
            messages: Raw message list from SFT JSONL.

        Returns:
            Reformatted message list with correct roles and action placement.
        """
        if len(messages) < 3:
            return messages

        result: list[dict] = []

        result.append({"role": "system", "content": TRUSS_SYSTEM_PROMPT})

        # Merge messages[0] (problem) + messages[1] (initial state) into the
        # first user turn. Chat templates, especially Qwen3's, use the last
        # user turn to decide how to render assistant reasoning. A
        # system→assistant first turn causes inconsistent stripping of
        # <think> blocks and teaches a different first-action distribution.
        user_content = messages[0]["content"]
        if len(messages) > 1 and messages[1]["role"] == "system":
            user_content = user_content + "\n\n" + messages[1]["content"]
            start_idx = 2
        else:
            start_idx = 1

        result.append({"role": "user", "content": user_content})

        # Process remaining messages
        for i in range(start_idx, len(messages)):
            msg = messages[i]

            if msg["role"] == "assistant":
                result.append(self._transform_assistant(msg))
            elif msg["role"] == "system":
                # FEA/simulation feedback → user role
                result.append({
                    "role": "user",
                    "content": self.SIMULATION_PREFIX + msg["content"],
                })
            else:
                # Pass through any other roles unchanged
                result.append(msg)

        return result

    def _transform_assistant(self, msg: dict) -> dict:
        """Transform an assistant message: extract action, reformat think/action tags.

        Handles two message types:
          1. <think>Modification: X\nAction: Y</think>  → <think>X</think>\n<action>Y</action>
          2. <answer>...</answer>                         → unchanged
        """
        content = msg["content"]

        # Pass through <answer> messages unchanged
        if "<answer>" in content:
            return {"role": "assistant", "content": content}

        # Extract from <think> block
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if not think_match:
            # No <think> tags — wrap content as action if it looks like one
            if ACTION_PATTERN.search(content):
                return {"role": "assistant", "content": f"<action>{content.strip()}</action>"}
            return {"role": "assistant", "content": content}

        think_content = think_match.group(1).strip()

        # Extract the action line
        action_line = self._extract_action(think_content)

        # Extract the description (what remains after removing action)
        description = self._extract_description(think_content)

        # Build reformatted content
        if action_line and description:
            new_content = f"<think>\n{description}\n</think>\n<action>{action_line}</action>"
        elif action_line:
            new_content = f"<think>\n{action_line}\n</think>\n<action>{action_line}</action>"
        else:
            # Couldn't parse — keep original content in think tags
            new_content = f"<think>\n{think_content}\n</think>"

        return {"role": "assistant", "content": new_content}

    def _extract_action(self, think_content: str) -> Optional[str]:
        """Extract the grammar action from shallow <think> content.

        Expert trajectories carry compound moves joined by ' ; ' -- a fully-stressed pass is
        one rescale per member, eleven parts on average. ``extract_action`` returns only the
        first valid one, which would train the model to make a fraction of the move the
        expert made. Every valid part is kept and rejoined; a single-action turn is
        unaffected.
        """
        from llm_finetune.data.grammar import find_actions, validate_action

        parts = []
        for candidate in find_actions(think_content):
            result = validate_action(candidate)
            if result.is_valid and result.action not in parts:
                parts.append(result.action)
        if not parts:
            return extract_action(think_content)
        return " ; ".join(parts)

    def _extract_description(self, think_content: str) -> str:
        """Extract human-readable description from shallow <think> content.

        Removes "Modification:" prefix and "Action:" line, leaving just
        the description of what modification is being made.
        """
        # Try to extract "Modification: ..." line
        mod_match = MODIFICATION_PATTERN.search(think_content)
        if mod_match:
            return mod_match.group(1).strip()

        # Fallback: return everything before "Action:" line
        lines = think_content.split("\n")
        desc_lines = []
        for line in lines:
            if line.strip().startswith("Action:"):
                break
            if line.strip():
                desc_lines.append(line.strip())

        return " ".join(desc_lines) if desc_lines else think_content.strip()
