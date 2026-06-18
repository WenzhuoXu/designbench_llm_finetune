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
import math
import os
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from llm_finetune.training.rl.costs import CostFunction, build_cost_from_config
from llm_finetune.training.rl.rewards import RewardFunction, build_reward_from_config

log = logging.getLogger(__name__)


MAX_GRPO_REWARD_ABS = 10_000.0
MAX_GRPO_COST = 100.0


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return numeric if math.isfinite(numeric) else default


def _bounded_scalar(
    value: object,
    *,
    lower: float,
    upper: float,
    default: float,
    label: str,
) -> float:
    numeric = _finite_float(value, default)
    if numeric != value or numeric < lower or numeric > upper:
        log.warning("%s was non-finite or out of bounds (%r); clamped", label, value)
    return max(lower, min(upper, numeric))


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
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    enable_wandb_reporting = bool(log_cfg.get("wandb_enabled", True)) and local_rank == 0

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
    # Shared mutable state for passing rho from compute_rewards to the
    # training callback. Direct wandb.log(step=k) inside compute_rewards
    # conflicts with TRL's WandbCallback step tracking and gets silently
    # dropped; emitting from on_log uses the correct global_step.
    _rho_state: dict = {"rho": 0.0}

    reward_callable = _build_reward_callable(
        reward_fn=reward_fn,
        cost_fn=cost_fn,
        env=env,
        wandb_logger=wandb_logger,
        rl_cfg=rl_cfg,
        rho_state=_rho_state,
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
        # GRPO-specific (TRL 1.0.0 names)
        num_generations=rl_cfg.get("group_size", 8),        # K rollouts per prompt
        temperature=rl_cfg.get("temperature", 0.8),
        max_completion_length=rl_cfg.get("max_new_tokens", 1024),
        # KL regularization (renamed kl_coef → beta in TRL 1.0)
        beta=rl_cfg.get("kl_coef", 0.04),
        # Clipping (renamed cliprange → epsilon in TRL 1.0)
        epsilon=rl_cfg.get("clip_ratio", 0.2),
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
        # vLLM integration — "server" mode: connect to our manually started VLLMServer (rank 0)
        use_vllm=use_vllm,
        **({"vllm_mode": "server",
            "vllm_server_host": vllm_server_host,
            "vllm_server_port": vllm_server_port} if use_vllm else {}),
        # Checkpointing
        save_steps=rl_cfg.get("save_steps", 50),
        save_total_limit=rl_cfg.get("save_total_limit", 3),
        # Logging
        logging_steps=log_cfg.get("log_every_n_steps", 10),
        report_to=["wandb"] if enable_wandb_reporting else ["none"],
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

    callbacks = _build_grpo_callbacks(wandb_logger, local_logger, cfg, _rho_state)

    # Multi-turn rollouts via TRL's rollout_func hook (generate→FEA→generate,
    # one action/turn). See docs/multiturn_grpo_design.md. Reconstructs the same
    # formatter train_grpo.py used for the dataset so turn-0 prompts match.
    rollout_func = None
    if bool(rl_cfg.get("multi_turn", False)):
        if use_vllm:
            raise ValueError(
                "rl.multi_turn=true requires rl.use_vllm=false (per-turn HF generate)."
            )
        from llm_finetune.data.processors.chat_formatter import (
            ChatFormatter,
            ThinkingMode,
            MODEL_THINKING_MODE,
        )
        from llm_finetune.training.rl.multiturn_rollout import (
            make_multiturn_rollout_func,
            register_prompt_specs,
        )
        # Resolve thinking mode by MODEL FAMILY, not the checkpoint path. With a
        # warmstart checkpoint, cfg.model.model_name_or_path is a local path not
        # in MODEL_THINKING_MODE → would default to NONE and drop Qwen3's
        # enable_thinking. The single-turn eval (base id → QWEN3) produced
        # concise, well-terminated turns; matching it here is what makes turns
        # stop at ~one action instead of running to max_turn_tokens.
        _mid = cfg.model.model_name_or_path
        _tm = MODEL_THINKING_MODE.get(_mid)
        if _tm is None:
            _low = str(_mid).lower()
            if "qwen3" in _low:
                _tm = ThinkingMode.QWEN3
            elif "deepseek-r1" in _low or "deepseek_r1" in _low:
                _tm = ThinkingMode.DEEPSEEK_R1
        mt_formatter = ChatFormatter.from_model_id(_mid, tokenizer, thinking_mode=_tm)
        log.info(f"Multi-turn rollout formatter thinking_mode={_tm}")
        n_reg = register_prompt_specs(train_dataset)
        rollout_func = make_multiturn_rollout_func(
            mt_formatter, rl_cfg, base_model_id=cfg.model.model_name_or_path
        )
        log.info(
            f"Multi-turn GRPO enabled (rollout_func): {n_reg} prompt specs registered, "
            f"max_turns={rl_cfg.get('max_turns', rl_cfg.get('max_steps', 20))}, "
            f"max_turn_tokens={rl_cfg.get('max_turn_tokens', 512)}, "
            f"max_completion_length={grpo_config.max_completion_length}"
        )

    # ref_model removed from TRL 1.0.0 constructor — created internally when beta != 0
    trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_callable],
        args=grpo_config,
        train_dataset=train_dataset,
        callbacks=callbacks,
    )
    if rollout_func is not None:
        trainer_kwargs["rollout_func"] = rollout_func
    trainer = GRPOTrainer(**trainer_kwargs)

    log.info(
        f"GRPOTrainer built:\n"
        f"  reward_fn={reward_fn.name()}\n"
        f"  group_size={grpo_config.num_generations}\n"
        f"  max_completion_length={grpo_config.max_completion_length}\n"
        f"  beta(kl)={grpo_config.beta}\n"
        f"  vllm_mode={grpo_config.vllm_mode}\n"
        f"  output_dir={output_dir}"
    )

    return trainer


def _compute_group_tree_advantages(
    rollout_data: list,
    problem_specs: list[dict],
    rl_cfg: DictConfig,
) -> list[float]:
    """Group rollouts by problem_id and set tree_metrics["tree_advantage"].

    Computes the group-relative Φ advantage within each GRPO rollout group:
        tree_advantage_k = γ·Φ(s_H^k) − mean_j(γ·Φ(s_H^j))

    This is the D=∞ trajectory-level analog of A^{(D,b)} from §3.1: the K
    rollouts sampled by GRPO are the lookahead tree at the trajectory level.

    Returns per-rollout phi_H values (for rho computation after reward scoring).
    """
    from collections import defaultdict

    from llm_finetune.training.rl.posterior.potential import compute_potential

    posterior = dict(rl_cfg.get("posterior", {})) if hasattr(rl_cfg.get("posterior", {}), "items") else {}
    alpha = float(posterior.get("alpha", 5.0))
    gamma = float(posterior.get("gamma", 0.99))

    # Group rollout indices by problem_id
    groups: dict[str, list[int]] = defaultdict(list)
    for i, rollout in enumerate(rollout_data):
        if rollout is not None:
            groups[rollout.problem_id].append(i)

    phi_H_per_rollout: list[float] = [0.0] * len(rollout_data)

    for problem_id, indices in groups.items():
        first_rollout = rollout_data[indices[0]]
        initial_mass = _finite_float((first_rollout.initial_state or {}).get("mass"), 1.0)

        phi_values: list[float] = []
        for idx in indices:
            r = rollout_data[idx]
            phi = _bounded_scalar(
                compute_potential(
                    r.final_state or {},
                    initial_mass=initial_mass,
                    alpha=alpha,
                ),
                lower=-MAX_GRPO_REWARD_ABS,
                upper=MAX_GRPO_REWARD_ABS,
                default=-MAX_GRPO_REWARD_ABS,
                label=f"phi_H[{idx}]",
            )
            phi_values.append(phi)

        mean_phi = sum(phi_values) / len(phi_values)
        for idx, phi_H in zip(indices, phi_values):
            tree_advantage = _bounded_scalar(
                gamma * phi_H - mean_phi,
                lower=-MAX_GRPO_REWARD_ABS,
                upper=MAX_GRPO_REWARD_ABS,
                default=0.0,
                label=f"tree_advantage[{idx}]",
            )
            rollout_data[idx].tree_metrics["tree_advantage"] = tree_advantage
            rollout_data[idx].tree_metrics["phi_H"] = phi_H
            phi_H_per_rollout[idx] = phi_H

    return phi_H_per_rollout


def _compute_rho_per_group(
    rollout_data: list,
    rewards: list[float],
    phi_H_per_rollout: list[float],
) -> float:
    """Compute mean rho_tree_agreement across all rollout groups in this batch.

    rho for a group = 1 if argmax(φ_H) == argmax(total_reward), else 0.
    This is the training-time proxy for ρ(t) from §3.6: does the composite
    reward rank the same trajectory as the Φ criterion?
    """
    from collections import defaultdict
    groups: dict[str, list[int]] = defaultdict(list)
    for i, rollout in enumerate(rollout_data):
        if rollout is not None:
            groups[rollout.problem_id].append(i)

    if not groups:
        return 0.0

    rho_values: list[int] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        best_phi_idx = max(indices, key=lambda i: phi_H_per_rollout[i])
        best_reward_idx = max(indices, key=lambda i: rewards[i])
        rho_values.append(int(best_phi_idx == best_reward_idx))

    return sum(rho_values) / len(rho_values) if rho_values else 0.0


def _build_reward_callable(
    reward_fn: RewardFunction,
    cost_fn: Optional[CostFunction],
    env,
    wandb_logger,
    rl_cfg: DictConfig,
    rho_state: Optional[dict] = None,
):
    """Build a TRL-compatible reward function that calls TrussRolloutEnv.

    TRL GRPOTrainer expects a reward function with signature:
        reward_fn(completions: list[str], prompts: list[str], **kwargs) → list[float]

    This wrapper:
    1. Executes completions through TrussRolloutEnv (parallel FEA)
    2. Optionally computes group-relative tree advantages (use_tree_expansion=true)
    3. Computes rewards + cost deductions
    4. Logs per-component mean+std for rewards and costs, plus rho(t)
    """
    import numpy as _np

    def _mean(vs): return float(_np.mean(vs))
    def _std(vs): return float(_np.std(vs, ddof=1)) if len(vs) > 1 else 0.0

    _call_count = [0]  # mutable counter shared across calls
    use_tree_expansion = bool(rl_cfg.get("use_tree_expansion", False))
    cost_coef = float(rl_cfg.get("cost_coef", 0.1))

    def compute_rewards(
        completions: list[str],
        prompts: list[str],
        **kwargs,
    ) -> list[float]:
        n = len(completions)
        problem_specs: list[dict] = kwargs.get("problem_spec", [{}] * n)

        # ── Phase 1: obtain rollouts ───────────────────────────────────────
        # Multi-turn path: the rollout_func already ran the generate→FEA loop
        # and threaded the actual RolloutResult per completion via the
        # 'rollout_result' extra field (TRL merges it 1:1 into reward kwargs).
        # Single-turn path: parse the one completion via run_completion.
        provided_rollouts = kwargs.get("rollout_result")
        rollout_data: list = []
        for i, (completion, spec) in enumerate(zip(completions, problem_specs)):
            if (provided_rollouts is not None and i < len(provided_rollouts)
                    and provided_rollouts[i] is not None):
                rollout_data.append(provided_rollouts[i])
                continue
            try:
                rollout = env.run_completion(problem_spec=spec, completion=completion)
                rollout_data.append(rollout)
            except Exception as e:
                log.warning(f"Rollout {i} failed: {e}")
                rollout_data.append(None)

        # ── Phase 2: group-relative tree advantages ────────────────────────
        phi_H_per_rollout: list[float] = [0.0] * n
        if use_tree_expansion:
            phi_H_per_rollout = _compute_group_tree_advantages(
                rollout_data, problem_specs, rl_cfg
            )

        # ── Phase 3: reward + cost computation ────────────────────────────
        rewards: list[float] = []
        reward_breakdowns: list[dict] = []
        cost_breakdowns: list[dict] = []

        for i, (rollout, spec) in enumerate(zip(rollout_data, problem_specs)):
            if rollout is None:
                rewards.append(0.0)
                continue
            try:
                r = _bounded_scalar(
                    reward_fn.compute(rollout, spec),
                    lower=-MAX_GRPO_REWARD_ABS,
                    upper=MAX_GRPO_REWARD_ABS,
                    default=0.0,
                    label=f"reward[{i}]",
                )
                if hasattr(reward_fn, "get_breakdown"):
                    try:
                        reward_breakdowns.append(reward_fn.get_breakdown(rollout, spec))
                    except Exception:
                        pass
                if cost_fn is not None:
                    c = _bounded_scalar(
                        cost_fn.compute(rollout, spec),
                        lower=0.0,
                        upper=MAX_GRPO_COST,
                        default=MAX_GRPO_COST,
                        label=f"cost[{i}]",
                    )
                    r -= cost_coef * c
                    if hasattr(cost_fn, "get_breakdown"):
                        try:
                            cost_breakdowns.append(cost_fn.get_breakdown(rollout, spec))
                        except Exception:
                            pass
                rewards.append(
                    _bounded_scalar(
                        r,
                        lower=-MAX_GRPO_REWARD_ABS,
                        upper=MAX_GRPO_REWARD_ABS,
                        default=0.0,
                        label=f"final_reward[{i}]",
                    )
                )
            except Exception as e:
                log.warning(f"Reward computation failed for rollout {i}: {e}")
                rewards.append(0.0)

        # ── Phase 4: rho(t) proxy ──────────────────────────────────────────
        rho = _compute_rho_per_group(rollout_data, rewards, phi_H_per_rollout)

        # ── Phase 5: logging ───────────────────────────────────────────────
        _call_count[0] += 1
        step = _call_count[0]

        if wandb_logger is not None and wandb_logger.is_active:
            valid_rollouts = [r for r in rollout_data if r is not None]
            valid_rewards = [rewards[i] for i, r in enumerate(rollout_data) if r is not None]

            if valid_rollouts:
                wandb_logger.log_rollout_table(
                    rollouts=valid_rollouts,
                    rewards=valid_rewards,
                    cost_fn=cost_fn,
                )

            # Per-component reward breakdown: mean + std across all rollouts
            if reward_breakdowns:
                merged: dict[str, list[float]] = {}
                for bd in reward_breakdowns:
                    for k, v in bd.items():
                        merged.setdefault(k, []).append(
                            _bounded_scalar(
                                v,
                                lower=-MAX_GRPO_REWARD_ABS,
                                upper=MAX_GRPO_REWARD_ABS,
                                default=0.0,
                                label=f"reward_breakdown.{k}",
                            )
                        )
                means = {k: _mean(vs) for k, vs in merged.items()}
                stds = {k: _std(vs) for k, vs in merged.items()}
                wandb_logger.log_reward_breakdown(step, means, stds)

            # Per-component cost breakdown: mean + std
            if cost_breakdowns:
                merged_c: dict[str, list[float]] = {}
                for bd in cost_breakdowns:
                    for k, v in bd.items():
                        merged_c.setdefault(k, []).append(
                            _bounded_scalar(
                                v,
                                lower=0.0,
                                upper=MAX_GRPO_COST,
                                default=MAX_GRPO_COST,
                                label=f"cost_breakdown.{k}",
                            )
                        )
                cost_means = {k: _mean(vs) for k, vs in merged_c.items()}
                cost_stds = {k: _std(vs) for k, vs in merged_c.items()}
                wandb_logger.log_cost_breakdown(step, cost_means, cost_stds)

            # rho(t) — stored in shared state, emitted from on_log callback
            # so it uses global_step and avoids W&B step-conflict drops.
            if rho_state is not None:
                rho_state["rho"] = rho

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


def _build_grpo_callbacks(wandb_logger, local_logger, cfg, rho_state: Optional[dict] = None) -> list:
    """Build trainer callbacks for GRPO training logging."""
    try:
        from transformers import TrainerCallback

        class GRPOCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None:
                    return
                # Inject rho into the log dict so it shares global_step and
                # reaches both metrics.jsonl and W&B without step conflicts.
                rho = rho_state["rho"] if rho_state is not None else 0.0
                logs_with_rho = {**logs, "rho_tree_agreement": rho}
                if local_logger is not None:
                    local_logger.log_step(state.global_step, logs_with_rho)
                if wandb_logger is not None and wandb_logger.is_active:
                    wandb_logger.log_training_step(state.global_step, logs_with_rho)

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
