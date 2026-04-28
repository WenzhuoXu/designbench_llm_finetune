"""
GRPO Trainer: TRL GRPOTrainer wrapper for DesignBench RL training.

GRPO (Group Relative Policy Optimization) works as follows:
  1. For each prompt (truss problem), generate K completions (group_size=8-16)
  2. Execute each completion through TrussRolloutEnv to get rewards
  3. Normalize rewards within the group: advantage = (r - mean(group)) / std(group)
  4. Compute policy gradient loss with KL penalty from reference model
  5. Update actor weights, sync to vLLM for next iteration

Key infrastructure:
  - vLLM for fast generation (5-10x faster than HF generate)
  - Multiprocessing for parallel FEA execution (104 CPUs available)
  - DeepSpeed ZeRO-2 for actor training
  - Reference model kept on CPU (zero GPU overhead)
  - W&B Tables for rollout visualization

Usage:
    trainer = build_grpo_trainer(cfg, model, tokenizer, ref_model,
                                  train_dataset, reward_fn, cost_fn)
    trainer.train()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from llm_finetune.training.rl.rewards import RewardFunction, RolloutResult, build_reward_from_config
from llm_finetune.training.rl.costs import CostFunction, build_cost_from_config

log = logging.getLogger(__name__)


def build_grpo_trainer(
    cfg: DictConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    ref_model: Optional[PreTrainedModel],
    train_dataset,
    reward_fn: Optional[RewardFunction] = None,
    cost_fn: Optional[CostFunction] = None,
    env=None,
    local_logger=None,
    wandb_logger=None,
):
    """Build TRL GRPOTrainer with DesignBench reward/cost functions.

    Args:
        cfg: Full training config (Hydra DictConfig).
        model: Actor model (loaded with FA2, gradient checkpointing).
        tokenizer: Model tokenizer.
        ref_model: Reference model for KL penalty. If None, uses initial actor copy.
        train_dataset: RLPromptDataset — prompts for rollout generation.
        reward_fn: RewardFunction instance. Built from cfg if None.
        cost_fn: CostFunction instance (optional, used for logging).
        env: TrussRolloutEnv instance. Built from default paths if None.
        local_logger: LocalLogger instance.
        wandb_logger: WandbLogger instance.

    Returns:
        TRL GRPOTrainer ready to train.
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        raise ImportError("trl>=0.12 required for GRPOTrainer. Run: pip install trl>=0.12")

    rl_cfg = cfg.rl
    log_cfg = cfg.get("logging", {})

    # Build reward and cost functions
    if reward_fn is None:
        reward_fn = build_reward_from_config(cfg.get("rl", {}))
        log.info(f"GRPO reward function: {reward_fn.name()}")

    if cost_fn is None and cfg.get("rl", {}).get("cost_fn"):
        cost_fn = build_cost_from_config(cfg.get("rl", {}))
        log.info(f"GRPO cost function: {cost_fn.name() if cost_fn else 'None'}")

    # Build rollout environment
    if env is None:
        from llm_finetune.envs.truss_env import TrussRolloutEnv
        env = TrussRolloutEnv(
            max_steps=rl_cfg.get("max_steps", 20),
            n_workers=rl_cfg.get("n_env_workers", 8),
        )

    # Build reward function that calls the environment
    reward_callable = _build_reward_callable(
        reward_fn=reward_fn,
        cost_fn=cost_fn,
        env=env,
        wandb_logger=wandb_logger,
        rl_cfg=rl_cfg,
    )

    # Output directory
    output_dir = (
        Path(rl_cfg.get("output_dir", "checkpoints")) / cfg.get("run_name", "grpo_run")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # vLLM configuration
    use_vllm = rl_cfg.get("use_vllm", True)
    vllm_server_host = rl_cfg.get("vllm_server_host", "localhost")
    vllm_server_port = rl_cfg.get("vllm_server_port", 8000)

    if use_vllm:
        log.info(
            f"GRPO will use vLLM for rollout generation "
            f"(host={vllm_server_host}, port={vllm_server_port})"
        )

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        # GRPO-specific
        num_generations=rl_cfg.get("group_size", 8),        # K rollouts per prompt
        temperature=rl_cfg.get("temperature", 0.8),
        max_new_tokens=rl_cfg.get("max_new_tokens", 1024),
        max_prompt_length=rl_cfg.get("max_prompt_length", 2048),
        # KL regularization
        kl_coef=rl_cfg.get("kl_coef", 0.04),
        # Clipping (PPO-style)
        cliprange=rl_cfg.get("clip_ratio", 0.2),
        # Batch configuration
        per_device_train_batch_size=rl_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=rl_cfg.get("gradient_accumulation_steps", 8),
        # Training duration
        num_train_epochs=rl_cfg.get("num_epochs", 1),
        max_steps=rl_cfg.get("max_steps_train", -1),
        # LR schedule
        learning_rate=rl_cfg.get("learning_rate", 5e-7),
        lr_scheduler_type=rl_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=rl_cfg.get("warmup_ratio", 0.05),
        # Precision
        bf16=True,
        fp16=False,
        # Gradient clipping
        max_grad_norm=rl_cfg.get("max_grad_norm", 1.0),
        # vLLM integration (TRL >= 0.12)
        use_vllm=use_vllm,
        vllm_server_host=vllm_server_host if use_vllm else None,
        vllm_server_port=vllm_server_port if use_vllm else None,
        # Checkpointing
        save_steps=rl_cfg.get("save_steps", 50),
        save_total_limit=rl_cfg.get("save_total_limit", 3),
        # Logging
        logging_steps=log_cfg.get("log_every_n_steps", 10),
        report_to=["wandb"] if log_cfg.get("wandb_enabled", True) else ["none"],
        run_name=cfg.get("run_name", "designbench_grpo"),
        # Optimizer
        optim=rl_cfg.get("optimizer", "adamw_torch_fused"),
        weight_decay=rl_cfg.get("weight_decay", 0.01),
        # DeepSpeed
        deepspeed=_get_deepspeed_config(cfg),
        # DataLoader
        dataloader_num_workers=rl_cfg.get("dataloader_num_workers", 4),
        # Misc
        seed=cfg.get("seed", 42),
        remove_unused_columns=False,
    )

    callbacks = _build_grpo_callbacks(wandb_logger, local_logger, cfg)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        ref_model=ref_model,
        reward_funcs=[reward_callable],
        args=grpo_config,
        train_dataset=train_dataset,
        callbacks=callbacks,
    )

    log.info(
        f"GRPOTrainer built:\n"
        f"  reward_fn={reward_fn.name()}\n"
        f"  group_size={grpo_config.num_generations}\n"
        f"  max_new_tokens={grpo_config.max_new_tokens}\n"
        f"  kl_coef={grpo_config.kl_coef}\n"
        f"  use_vllm={use_vllm}\n"
        f"  output_dir={output_dir}"
    )

    return trainer


def _build_reward_callable(
    reward_fn: RewardFunction,
    cost_fn: Optional[CostFunction],
    env,
    wandb_logger,
    rl_cfg: DictConfig,
):
    """Build a TRL-compatible reward function that calls TrussRolloutEnv.

    TRL GRPOTrainer expects a reward function with signature:
        reward_fn(completions: list[str], prompts: list[str], **kwargs) → list[float]

    This wrapper:
    1. Parses completions to extract grammar actions (handling thinking blocks)
    2. Executes actions through TrussRolloutEnv (parallel with multiprocessing)
    3. Computes rewards using the configured RewardFunction
    4. Logs rollout results to W&B Tables
    """
    from llm_finetune.data.processors.chat_formatter import ChatFormatter

    def compute_rewards(
        completions: list[str],
        prompts: list[str],
        problem_specs: Optional[list[dict]] = None,
        **kwargs,
    ) -> list[float]:
        rewards = []
        rollout_data = []

        n = len(completions)
        if problem_specs is None:
            problem_specs = [{}] * n

        for i, (completion, prompt, spec) in enumerate(
            zip(completions, prompts, problem_specs)
        ):
            try:
                # Execute rollout through TrussRolloutEnv
                rollout = env.run_completion(
                    problem_spec=spec,
                    completion=completion,
                )
                # Compute reward
                reward = reward_fn.compute(rollout, spec)
                # Subtract cost if configured
                if cost_fn is not None:
                    cost = cost_fn.compute(rollout, spec)
                    reward -= rl_cfg.get("cost_coef", 0.1) * cost
                rewards.append(reward)
                rollout_data.append(rollout)
            except Exception as e:
                log.warning(f"Reward computation failed for rollout {i}: {e}")
                rewards.append(0.0)
                rollout_data.append(None)

        # Log rollout table to W&B
        if wandb_logger is not None and wandb_logger.is_active and rollout_data:
            valid_rollouts = [(r, s) for r, s in zip(rollout_data, problem_specs)
                             if r is not None]
            if valid_rollouts:
                rollouts, specs = zip(*valid_rollouts)
                wandb_logger.log_rollout_table(
                    rollouts=list(rollouts),
                    rewards=rewards[:len(rollouts)],
                    cost_fn=cost_fn,
                )

        return rewards

    return compute_rewards


def _get_deepspeed_config(cfg: DictConfig) -> Optional[str]:
    parallelism = cfg.get("parallelism", {})
    ds_config = parallelism.get("deepspeed_config", None)
    if ds_config and os.path.exists(ds_config):
        return ds_config
    for path in ["deepspeed/zero2.json", "deepspeed/zero3.json"]:
        abs_path = Path(__file__).parent.parent.parent.parent / path
        if abs_path.exists() and parallelism.get("strategy", "deepspeed_zero2") in path:
            return str(abs_path)
    return None


def _build_grpo_callbacks(wandb_logger, local_logger, cfg) -> list:
    """Build trainer callbacks for GRPO training logging."""
    try:
        from transformers import TrainerCallback

        class GRPOCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None:
                    return
                if local_logger is not None:
                    local_logger.log_step(state.global_step, logs)
                if wandb_logger is not None and wandb_logger.is_active:
                    wandb_logger.log_training_step(state.global_step, logs)

            def on_save(self, args, state, control, **kwargs):
                if wandb_logger is not None and wandb_logger.is_active:
                    ckpt_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                    if ckpt_path.exists():
                        wandb_logger.save_artifact(
                            str(ckpt_path),
                            name=f"grpo-checkpoint-{state.global_step}",
                            artifact_type="model",
                        )

        return [GRPOCallback()]
    except ImportError:
        return []
