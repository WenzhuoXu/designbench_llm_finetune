"""Online Search Policy — theory-grounded live MCTS decisions (§3.2–3.6)."""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Optional

log = logging.getLogger(__name__)


class BudgetExhausted(Exception):
    """Raised when the FEA call budget is exhausted inside LLMCodeOptimizer."""


class OnlineSearchPolicy:
    """
    Tracks per-episode theory signals and exposes three online decisions:

      1. should_diversify()  — Δ_rank < 2γ^(D+1)σ_G (§3.3 unmet) → inject diversity prompt
      2. check_backtrack()   — §3.2 bias-bound B&B pruning → restore best context
      3. should_optimize()   — ρ ≥ threshold for w consecutive steps → LLM code optimizer
    """

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        rho_window: int = 3,
        rho_threshold: float = 0.85,
        delta_g_bar: float = 0.0,
    ):
        self.gamma = gamma
        self.rho_window = rho_window
        self.rho_threshold = rho_threshold
        self.delta_g_bar = delta_g_bar  # Δ̄_G from pilot (§3.2 bias bound)

        self._rho_history: list[int] = []
        self._best_V: float = -float("inf")
        self._best_context: Any = None
        self._n_backtracks: int = 0

    @property
    def n_backtracks(self) -> int:
        return self._n_backtracks

    def update(self, *, context: Any, V_hat: float, rho: int) -> None:
        """Call after each MCTS tree evaluation to track best state and ρ history."""
        if V_hat > self._best_V:
            self._best_V = V_hat
            self._best_context = copy.copy(context)
        self._rho_history.append(rho)

    def _bias(self, depth: int) -> float:
        return (self.gamma ** (depth + 1)) * self.delta_g_bar

    def should_diversify(self, delta_rank: float, depth_used: int, sigma_g: float) -> bool:
        """True when §3.3 reliability condition is unmet: Δ_rank < 2γ^(D+1) σ_G."""
        if sigma_g <= 0.0:
            return False
        threshold = 2.0 * (self.gamma ** (depth_used + 1)) * sigma_g
        return delta_rank < threshold

    def diversity_prompt_suffix(self, candidates: list) -> str:
        """Build a prompt suffix nudging the LLM toward underrepresented action classes."""
        from llm_finetune.training.rl.posterior.features import ACTION_CLASS_ORDER, action_class

        covered = {action_class(c.action) for c in candidates if c.action}
        missing = [cls for cls in ACTION_CLASS_ORDER if cls not in covered]
        present_summary = ", ".join(
            f"{cls}(x{sum(1 for c in candidates if action_class(c.action) == cls)})"
            for cls in ACTION_CLASS_ORDER
            if cls in covered
        )
        if missing:
            missing_str = " and ".join(missing)
            return (
                f"\n\n[SEARCH DIVERSITY HINT] Current proposals cover: {present_summary}. "
                f"Please also propose modifications involving {missing_str}. "
                f"Explore qualitatively different structural changes."
            )
        return (
            f"\n\n[SEARCH DIVERSITY HINT] Current proposals cover: {present_summary}. "
            f"The candidate actions appear too similar in structural impact. "
            f"Consider significantly different modification magnitudes or a different member entirely."
        )

    def check_backtrack(self, V_hat: float, depth_used: int) -> tuple[bool, Any]:
        """
        §3.2 Branch-and-bound pruning: backtrack when even the optimistic upper bound
        of the current state cannot reach the pessimistic lower bound of the best seen so far:
            V̂_current + bias  <  V̂_best − bias
        Returns (should_backtrack, snapshot_of_best_context).
        """
        if self._best_context is None:
            return False, None
        bias = self._bias(depth_used)
        if V_hat + bias < self._best_V - bias:
            self._n_backtracks += 1
            log.info(
                "OSP backtrack: V̂_cur+bias=%.4f < V̂_best-bias=%.4f "
                "(Δ̄_G=%.4f D=%d bias=%.4f)",
                V_hat + bias,
                self._best_V - bias,
                self.delta_g_bar,
                depth_used,
                bias,
            )
            return True, copy.copy(self._best_context)
        return False, None

    def should_optimize(self) -> bool:
        """True when ρ ≥ rho_threshold for rho_window consecutive steps."""
        if len(self._rho_history) < self.rho_window:
            return False
        return all(r >= self.rho_threshold for r in self._rho_history[-self.rho_window :])


_OPTIMIZER_PROMPT = """\
You are now in OPTIMIZER MODE. The MCTS search has converged (ρ={rho:.2f} for {w} consecutive steps).
Your task: write a Python script that minimizes the truss mass while maintaining structural feasibility.

Current design state (JSON):
{state_json}

Available API (already imported in your execution namespace):
  fea_oracle(action: str) -> dict
      Returns: {{mass, fos_buckling, fos_yielding, is_feasible}}
      Grammar actions:
        SCALE_PARAM(member_id, property, factor)
        MODIFY_PARAM(member_id, property, old_val, new_val)
        REMOVE_MEMBER(member_id)
        ADD_MEMBER(node_i, node_j, material, shape, radius, thickness)
        MOVE_JOINT(node_id, [old_x, old_y], [new_x, new_y])
      Remaining FEA budget: {budget} calls
      BudgetExhausted is raised automatically when the budget is exhausted.

Design constraints: FOS_buckling >= 1.5, FOS_yielding >= 1.5. Minimize mass.
Output ONLY executable Python code (no markdown fences, no explanation).
"""


class LLMCodeOptimizer:
    """
    Ask the LLM to write a Python optimizer when MCTS search has converged (§3.6).
    The generated code runs inside exec() with a wrapped FEA oracle injected into its namespace.
    """

    def __init__(self, rho_window: int = 3, rho_threshold: float = 0.85):
        self.rho_window = rho_window
        self.rho_threshold = rho_threshold

    def run(
        self,
        context: Any,
        evaluator: Any,
        step_budget: int,
        rho_mean: float,
    ) -> tuple[Any, str, int]:
        """
        Generate optimization code and execute it with an injected FEA oracle.
        Returns (final_context, code_str, n_fea_calls).
        """
        import json as _json

        state_json = _json.dumps(context.current_state, indent=2, default=str)
        prompt = _OPTIMIZER_PROMPT.format(
            rho=rho_mean,
            w=self.rho_window,
            state_json=state_json,
            budget=step_budget,
        )
        code_str = self._call_model(evaluator, context, prompt)
        if not code_str:
            log.warning("LLMCodeOptimizer: model returned empty code, skipping exec")
            return context, "", 0

        final_context, n_calls = self._exec_code(code_str, context, evaluator, step_budget)
        return final_context, code_str, n_calls

    def _call_model(self, evaluator: Any, context: Any, optimizer_prompt: str) -> str:
        messages = list(context.build_messages(evaluator.formatter))
        messages.append({"role": "user", "content": optimizer_prompt})

        model = evaluator.model
        if hasattr(model, "_create_response"):
            try:
                raw = model._create_response(messages, max_output_tokens=4096, temperature=None)
                return self._extract_code(raw)
            except Exception as exc:
                log.warning("LLMCodeOptimizer OpenAI call failed: %s", exc)
                return ""

        try:
            import torch

            prompt_ids = evaluator.formatter.apply_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            if not torch.is_tensor(prompt_ids):
                prompt_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
            prompt_ids = prompt_ids.to(next(model.parameters()).device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=torch.ones_like(prompt_ids),
                    do_sample=False,
                    max_new_tokens=4096,
                    pad_token_id=evaluator.tokenizer.pad_token_id,
                    eos_token_id=evaluator.tokenizer.eos_token_id,
                )
            raw = evaluator.tokenizer.decode(out[0][prompt_ids.shape[1] :], skip_special_tokens=True)
            return self._extract_code(raw)
        except Exception as exc:
            log.warning("LLMCodeOptimizer local model call failed: %s", exc)
            return ""

    @staticmethod
    def _extract_code(raw: str) -> str:
        m = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
        return m.group(1).strip() if m else raw.strip()

    @staticmethod
    def _exec_code(code_str: str, context: Any, evaluator: Any, budget: int) -> tuple[Any, int]:
        import copy as _copy
        import numpy as _np

        ctx_holder = [_copy.copy(context)]
        n_calls = [0]

        def fea_oracle(action: str) -> dict:
            if n_calls[0] >= budget:
                raise BudgetExhausted(f"FEA budget of {budget} calls exhausted")
            next_ctx, _reward, calls = evaluator._transition(ctx_holder[0], action)
            n_calls[0] += calls
            ctx_holder[0] = next_ctx
            return dict(next_ctx.current_state)

        exec_ns: dict = {
            "fea_oracle": fea_oracle,
            "BudgetExhausted": BudgetExhausted,
            "np": _np,
        }
        try:
            exec(compile(code_str, "<llm_optimizer>", "exec"), exec_ns)  # noqa: S102
        except BudgetExhausted:
            log.info("LLMCodeOptimizer: budget exhausted after %d FEA calls", n_calls[0])
        except Exception as exc:
            log.warning("LLMCodeOptimizer exec error (%s): %s", type(exc).__name__, exc)

        return ctx_holder[0], n_calls[0]
