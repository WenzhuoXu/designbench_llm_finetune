"""
SFT Trainer: TRL SFTTrainer wrapper with DesignBench-specific features.

Features:
  - SFTTarget masking (loss only on selected tokens)
  - Sequence packing for throughput (eliminates padding waste)
  - Cosine LR with warmup
  - Gradient clipping with norm logging
  - Checkpoint saving + optional HF Hub push
  - DeepSpeed ZeRO-2/3 and FSDP compatible
  - Flash Attention 2 + torch.compile support (via model loader)
  - W&B + local logging integration

Usage:
    trainer = build_sft_trainer(cfg, model, tokenizer, train_dataset, eval_dataset)
    trainer.train()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedModel, PreTrainedTokenizer

from llm_finetune.data.collators import DataCollatorForSFT
from llm_finetune.training.sft.targets import SFTTarget, build_target_from_config

log = logging.getLogger(__name__)


def build_sft_trainer(
    cfg: DictConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset,
    eval_dataset=None,
    target: Optional[SFTTarget] = None,
    local_logger=None,
    wandb_logger=None,
    extra_callbacks: Optional[list] = None,
):
    """Build TRL SFTTrainer with full infrastructure.

    Args:
        cfg: Full training config (from Hydra).
        model: Loaded model (with FA2, gradient checkpointing).
        tokenizer: Model tokenizer.
        train_dataset: SFTDataset instance.
        eval_dataset: Optional validation dataset.
        target: SFTTarget instance. If None, builds from cfg.data.target_fn.
        local_logger: LocalLogger instance.
        wandb_logger: WandbLogger instance.
        extra_callbacks: Additional trainer callbacks (e.g., WarmstartReadinessCallback).

    Returns:
        TRL SFTTrainer ready to train.
    """
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        raise ImportError("trl not installed. Run: pip install trl>=0.12")

    sft_cfg = cfg.sft
    model_cfg = cfg.model
    log_cfg = cfg.get("logging", {})

    # Build target function
    if target is None:
        target_name = cfg.get("data", {}).get("target_fn", "full_sequence")
        target = build_target_from_config(target_name)
        log.info(f"SFT target: {target.name()}")

    # Data collator with target masking
    data_collator = DataCollatorForSFT(tokenizer=tokenizer, target=target)

    # Output directory
    output_dir = Path(sft_cfg.get("output_dir", "checkpoints")) / cfg.get("run_name", "sft_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    # SFTConfig — maps Hydra config → TRL TrainingArguments
    save_steps = sft_cfg.get("save_steps", 100)
    eval_steps = sft_cfg.get("eval_steps", 100)
    has_eval = eval_dataset is not None

    # load_best_model_at_end requires save_steps to be a multiple of eval_steps.
    # If they don't align, disable it to avoid a hard ValidationError.
    load_best = has_eval and (save_steps % eval_steps == 0)
    if has_eval and not load_best:
        log.warning(
            f"save_steps={save_steps} is not a multiple of eval_steps={eval_steps}; "
            "disabling load_best_model_at_end to avoid TrainingArguments validation error."
        )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        # Batch size
        per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=sft_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 8),
        # Sequence length (TRL 1.0+ uses max_length; older TRL used max_seq_length)
        max_length=sft_cfg.get("max_seq_len", 8192),
        # Learning rate schedule
        learning_rate=sft_cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=sft_cfg.get("lr_scheduler", "cosine"),
        warmup_steps=int(sft_cfg.get("warmup_ratio", 0.05) * max(1, sft_cfg.get("max_steps", -1)) if sft_cfg.get("max_steps", -1) > 0 else 0),
        # Training duration
        num_train_epochs=sft_cfg.get("num_epochs", 3),
        max_steps=sft_cfg.get("max_steps", -1),
        # Precision
        bf16=True,
        fp16=False,
        # Gradient clipping
        max_grad_norm=sft_cfg.get("max_grad_norm", 1.0),
        # Checkpointing
        save_steps=save_steps,
        save_total_limit=sft_cfg.get("save_total_limit", 3),
        load_best_model_at_end=load_best,
        # Evaluation
        eval_strategy="steps" if has_eval else "no",
        eval_steps=eval_steps if has_eval else None,
        # Logging
        logging_steps=log_cfg.get("log_every_n_steps", 10),
        # logging_dir deprecated in TRL 1.0+ — use TENSORBOARD_LOGGING_DIR env var instead
        report_to=_get_report_to(log_cfg),
        run_name=cfg.get("run_name", "designbench_sft"),
        # Optimizer
        optim=sft_cfg.get("optimizer", "adamw_torch_fused"),
        weight_decay=sft_cfg.get("weight_decay", 0.01),
        adam_beta1=0.9,
        adam_beta2=0.95,
        # Sequence packing (TRL feature: bins multiple examples into one)
        packing=sft_cfg.get("packing", True),
        # DeepSpeed
        deepspeed=_get_deepspeed_config(cfg),
        # DataLoader
        dataloader_num_workers=sft_cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory=True,
        # HF Hub
        push_to_hub=sft_cfg.get("push_to_hub", False),
        hub_model_id=sft_cfg.get("hub_model_id", None),
        # Misc
        seed=cfg.get("seed", 42),
        data_seed=cfg.get("seed", 42),
        remove_unused_columns=False,
    )

    # Custom callbacks for W&B / local logging
    callbacks = _build_callbacks(wandb_logger, local_logger, cfg)
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # TRL 1.0+: renamed from tokenizer
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    log.info(
        f"SFTTrainer built:\n"
        f"  target={target.name()}\n"
        f"  max_seq_len={training_args.max_length}\n"
        f"  packing={training_args.packing}\n"
        f"  batch_size={training_args.per_device_train_batch_size} * "
        f"{training_args.gradient_accumulation_steps} grad_accum\n"
        f"  lr={training_args.learning_rate}, scheduler={training_args.lr_scheduler_type}\n"
        f"  output_dir={output_dir}"
    )

    return trainer


def _get_report_to(log_cfg) -> list[str]:
    """Build report_to list from logging config.

    We rely on our custom DesignBenchCallback (routed through WandbLogger)
    as the single W&B sink — it logs per-step metrics AND pynvml GPU stats.
    Letting TRL/HF Trainer also report to W&B via report_to=["wandb"] would
    produce duplicate metrics on rank 0. So return "none" unless tensorboard
    is explicitly requested.
    """
    report_to = ["none"]
    if log_cfg.get("tensorboard_enabled", False):
        report_to = ["tensorboard"]
    return report_to


def _get_deepspeed_config(cfg: DictConfig) -> Optional[str]:
    """Return path to DeepSpeed config JSON, if configured."""
    parallelism = cfg.get("parallelism", {})
    ds_config = parallelism.get("deepspeed_config", None)
    if ds_config and os.path.exists(ds_config):
        return ds_config
    # Check standard locations
    for path in ["deepspeed/zero2.json", "deepspeed/zero3.json"]:
        abs_path = Path(__file__).parent.parent.parent.parent / path
        if abs_path.exists() and parallelism.get("strategy", "deepspeed_zero2") in path:
            return str(abs_path)
    return None


def _build_callbacks(wandb_logger, local_logger, cfg) -> list:
    """Build trainer callbacks for logging. All logging side effects
    are rank-gated to the main process to avoid duplicate W&B runs and
    interleaved local log files."""
    try:
        from transformers import TrainerCallback

        class DesignBenchCallback(TrainerCallback):
            """Bridges TRL trainer events to our loggers. Main-process only."""

            def on_log(self, args, state, control, logs=None, **kwargs):
                if args.local_rank not in (-1, 0):
                    return
                if logs is None:
                    return
                if local_logger is not None:
                    local_logger.log_step(state.global_step, logs)
                if wandb_logger is not None and wandb_logger.is_active:
                    wandb_logger.log_training_step(state.global_step, logs)

            def on_save(self, args, state, control, **kwargs):
                if args.local_rank not in (-1, 0):
                    return
                if wandb_logger is not None and wandb_logger.is_active:
                    ckpt_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                    if ckpt_path.exists():
                        wandb_logger.save_artifact(
                            str(ckpt_path),
                            name=f"sft-checkpoint-{state.global_step}",
                            artifact_type="model",
                        )

        return [DesignBenchCallback()]
    except ImportError:
        return []
