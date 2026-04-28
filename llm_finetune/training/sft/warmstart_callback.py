"""
Warmstart readiness callback for SFT pre-training.

Monitors validation loss plateau to decide when the warmstart SFT has
produced a good enough starting point for GRPO. Early stops training
before overfitting destroys RL plasticity (Jin et al., 2025).

Format compliance (the other stopping criterion) is NOT checked inside
the training loop — doing so would require model.generate() on every rank,
which deadlocks with DeepSpeed ZeRO-3 (sharded params) and wastes 7x
compute with ZeRO-2 (replicated params). Format compliance is instead
checked post-training via scripts/check_format_compliance.py.

References:
  - DeepSeek-R1: cold-start SFT (~10k examples, few epochs) before GRPO
  - VLAA-Thinker (Chen et al., 2025): excessive SFT harms subsequent GRPO
  - Jin et al. (2025): overfitted SFT resists RL restoration
  - SASR (Chen et al., 2025): adaptive SFT→RL transition via KL monitoring
"""

from __future__ import annotations

import logging
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

log = logging.getLogger(__name__)


class WarmstartReadinessCallback(TrainerCallback):
    """Early-stops SFT warmstart when val loss improvement plateaus.

    Runs once per evaluation step. Safe for multi-GPU training: does not
    call model.generate(), does not allocate extra memory, does not
    diverge across ranks.
    """

    def __init__(
        self,
        eval_dataset=None,     # kept for API compat; unused
        tokenizer=None,        # kept for API compat; unused
        formatter=None,        # kept for API compat; unused
        eval_steps: int = 50,
        format_threshold: float = 0.90,   # unused in-loop; enforced post-training
        kl_threshold: float = 5.0,        # unused in-loop; reserved for future
        loss_delta_threshold: float = 0.05,
        n_eval_samples: int = 50,         # unused
    ):
        """
        Args:
            eval_steps: Trainer eval cadence (used for logging only).
            loss_delta_threshold: Min relative val loss improvement to continue.
                When (prev_loss - curr_loss) / prev_loss < threshold, stop.
            format_threshold / kl_threshold / n_eval_samples: kept for config
                compatibility; enforced by the post-training check script.
        """
        self.eval_steps = eval_steps
        self.format_threshold = format_threshold
        self.kl_threshold = kl_threshold
        self.loss_delta_threshold = loss_delta_threshold
        self.n_eval_samples = n_eval_samples

        self._prev_eval_loss: Optional[float] = None
        self._readiness_met = False

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        **kwargs,
    ):
        """Check val-loss-plateau stopping criterion after each eval."""
        if metrics is None:
            return

        step = state.global_step
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        # Compute relative loss improvement since previous eval
        loss_delta = None
        if self._prev_eval_loss is not None and self._prev_eval_loss > 0:
            loss_delta = (self._prev_eval_loss - eval_loss) / self._prev_eval_loss

        self._prev_eval_loss = eval_loss

        loss_plateau = (
            loss_delta is not None and loss_delta < self.loss_delta_threshold
        )

        # Log to W&B (only main process has an active run; Fix 1 gate)
        try:
            import wandb
            if wandb.run is not None:
                log_dict = {
                    "warmstart/eval_loss": eval_loss,
                    "warmstart/loss_plateau": int(loss_plateau),
                }
                if loss_delta is not None:
                    log_dict["warmstart/val_loss_delta"] = loss_delta
                wandb.log(log_dict, step=step)
        except ImportError:
            pass

        if loss_delta is not None:
            log.info(
                f"[Warmstart step={step}] eval_loss={eval_loss:.4f}, "
                f"delta={loss_delta:+.4f} (threshold={self.loss_delta_threshold}), "
                f"plateau={loss_plateau}"
            )
        else:
            log.info(
                f"[Warmstart step={step}] eval_loss={eval_loss:.4f} (first eval)"
            )

        # Early stopping decision. Setting identically on all ranks is safe —
        # HF Trainer broadcasts TrainerControl after the callback returns.
        if loss_plateau:
            log.info(
                f"[Warmstart] Val loss plateau at step {step}. "
                f"Stopping early to avoid overfitting (Jin et al., 2025). "
                f"Run scripts/check_format_compliance.py on the final checkpoint "
                f"to verify format readiness before GRPO."
            )
            self._readiness_met = True
            control.should_training_stop = True

            try:
                import wandb
                if wandb.run is not None:
                    wandb.alert(
                        title="Warmstart Early Stop",
                        text=(
                            f"Val loss plateau at step {step} "
                            f"(delta={loss_delta:.4f}). "
                            f"Run check_format_compliance.py before GRPO."
                        ),
                        level=wandb.AlertLevel.INFO,
                    )
            except Exception:
                pass

    @property
    def is_ready(self) -> bool:
        """Whether warmstart readiness (loss plateau) has been reached."""
        return self._readiness_met
