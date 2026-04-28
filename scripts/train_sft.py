"""
SFT Training Entrypoint.

Uses Hydra for config composition — model, SFT hyperparams, data, parallelism,
and logging are all configured via YAML files in configs/.

Usage (local, single GPU):
    python scripts/train_sft.py model=qwen3_14b sft=warmstart data=sft_traces

Usage (multi-GPU via torchrun, see slurm/sft_h100.sbatch):
    torchrun --nproc_per_node=8 scripts/train_sft.py \
        model=qwen3_14b sft=warmstart data=sft_traces \
        parallelism=deepspeed_zero2 run_name=qwen3_sft_001

Key Hydra overrides:
    model=qwen3_14b             — which model to train
    sft=warmstart               — SFT hyperparameters
    data=sft_traces             — training data config
    data.target_fn=action_only  — SFT supervision target (research hook!)
    sft.max_steps=10            — run only 10 steps (smoke test)
    model.use_compile=true      — enable torch.compile
    model.use_lora=true         — use LoRA instead of full fine-tuning
    logging.wandb_enabled=false — disable W&B (local-only logging)
"""

import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add DesignBench to path
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/DesignBench")

log = logging.getLogger(__name__)


class _NullLogger:
    """No-op LocalLogger stub for non-main ranks."""
    def start(self) -> None: pass
    def stop(self) -> None: pass
    def log_step(self, step, metrics): pass
    def log_eval(self, step, metrics): pass


class _NullWandbLogger:
    """No-op WandbLogger stub for non-main ranks."""
    is_active = False
    def init(self) -> None: pass
    def finish(self) -> None: pass
    def log_training_step(self, step, metrics): pass
    def log_eval_metrics(self, step, metrics): pass
    def log_gpu_stats(self, step): pass
    def log_rollout_table(self, *args, **kwargs): pass
    def save_artifact(self, *args, **kwargs): pass
    def alert(self, *args, **kwargs): pass
    def watch_model(self, *args, **kwargs): pass


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent / "configs"),
    config_name="sft_config",  # configs/sft_config.yaml (see below)
)
def main(cfg: DictConfig) -> None:
    # Rank gate — only the main process constructs real loggers
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = (local_rank == 0)

    if is_main:
        log.info("=" * 60)
        log.info("DesignBench SFT Training")
        log.info("=" * 60)
        log.info(OmegaConf.to_yaml(cfg))

    run_name = cfg.get("run_name", "sft_run")

    if is_main:
        from llm_finetune.logging.local_logger import LocalLogger
        from llm_finetune.logging.wandb_logger import build_wandb_logger

        local_logger = LocalLogger(
            run_name=run_name,
            log_dir=cfg.get("logging", {}).get("log_dir", "logs"),
            log_every_n_steps=cfg.get("logging", {}).get("log_every_n_steps", 10),
            profile_steps=cfg.get("logging", {}).get("profile_steps", 0),
        )
        local_logger.start()

        wandb_logger = build_wandb_logger(cfg, run_name=run_name)
        if cfg.get("logging", {}).get("wandb_enabled", True):
            wandb_logger.init()
    else:
        local_logger = _NullLogger()
        wandb_logger = _NullWandbLogger()

    try:
        _run_training(cfg, local_logger, wandb_logger, run_name)
    finally:
        wandb_logger.finish()
        local_logger.stop()


def _run_training(cfg, local_logger, wandb_logger, run_name):
    from llm_finetune.data.datasets.sft_dataset import SFTDataset
    from llm_finetune.data.processors.chat_formatter import ChatFormatter
    from llm_finetune.models.loader import load_model_and_tokenizer
    from llm_finetune.training.sft.targets import build_target_from_config
    from llm_finetune.training.sft.trainer import build_sft_trainer

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    log.info(f"Process: local_rank={local_rank}, world_size={world_size}")

    # Load model and tokenizer
    log.info("Loading model and tokenizer ...")
    model, tokenizer = load_model_and_tokenizer(cfg.model)

    # Log model memory footprint
    from llm_finetune.models.loader import get_model_memory_footprint
    mem = get_model_memory_footprint(model)
    if mem:
        log.info(f"GPU memory after model load: {mem}")
        if wandb_logger.is_active:
            import wandb
            wandb.log({"system/gpu_mem_after_load_gb": max(
                v for k, v in mem.items() if "used_gb" in k
            )})

    # Build chat formatter
    formatter = ChatFormatter.from_model_id(
        cfg.model.model_name_or_path, tokenizer,
    )
    log.info(f"Chat formatter: thinking_mode={formatter.thinking_mode}")

    # Build SFT target (research hook)
    data_cfg = cfg.get("data", {})
    target_name = data_cfg.get("target_fn", "thinking_and_action")
    target_kwargs = OmegaConf.to_container(
        data_cfg.get("target_kwargs", {}), resolve=True
    ) if data_cfg.get("target_kwargs") else {}
    target = build_target_from_config(target_name, **target_kwargs)
    log.info(f"SFT target: {target.name()}")

    # Build datasets from DesignBench SFT JSONL
    max_seq_len = cfg.sft.get("max_seq_len", 8192)
    min_trace_quality = data_cfg.get("min_trace_quality", 0.5)
    train_path = data_cfg.get("train_jsonl")
    dev_path = data_cfg.get("dev_jsonl")
    if not train_path or not dev_path:
        raise ValueError(
            "Data config must specify train_jsonl and dev_jsonl paths "
            "(see configs/data/sft_traces.yaml)"
        )

    log.info(f"Loading training data from {train_path} ...")
    train_dataset = SFTDataset.from_sft_jsonl(
        path=train_path,
        tokenizer=tokenizer,
        formatter=formatter,
        target=target,
        max_seq_len=max_seq_len,
        min_trace_quality=min_trace_quality,
    )
    eval_dataset = SFTDataset.from_sft_jsonl(
        path=dev_path,
        tokenizer=tokenizer,
        formatter=formatter,
        target=target,
        max_seq_len=max_seq_len,
        min_trace_quality=min_trace_quality,
    )

    train_stats = train_dataset.get_stats()
    log.info(f"Train dataset: {train_stats}")
    if wandb_logger.is_active:
        import wandb
        wandb.log({"data/" + k: v for k, v in train_stats.items()})

    # Watch model gradients (rank 0 only)
    if local_rank == 0:
        wandb_logger.watch_model(model, log_freq=100)

    # Build warmstart readiness callback if early stopping is enabled
    extra_callbacks = []
    sft_cfg = cfg.get("sft", {})
    if sft_cfg.get("early_stopping", False):
        from llm_finetune.training.sft.warmstart_callback import WarmstartReadinessCallback
        warmstart_cb = WarmstartReadinessCallback(
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            formatter=formatter,
            eval_steps=sft_cfg.get("warmstart_eval_steps", 50),
            format_threshold=sft_cfg.get("warmstart_format_threshold", 0.90),
            kl_threshold=sft_cfg.get("warmstart_kl_threshold", 5.0),
            loss_delta_threshold=sft_cfg.get("warmstart_loss_delta_threshold", 0.05),
        )
        extra_callbacks.append(warmstart_cb)
        log.info(
            f"Warmstart early stopping enabled: "
            f"format>={warmstart_cb.format_threshold:.0%}, "
            f"KL<{warmstart_cb.kl_threshold}, "
            f"loss_delta<{warmstart_cb.loss_delta_threshold}"
        )

    # Build and run trainer
    trainer = build_sft_trainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        target=target,
        local_logger=local_logger,
        wandb_logger=wandb_logger,
        extra_callbacks=extra_callbacks,
    )

    log.info("Starting SFT training ...")
    train_result = trainer.train()

    log.info(f"Training complete: {train_result.metrics}")
    if wandb_logger.is_active:
        import wandb
        wandb.log({"final/" + k: v for k, v in train_result.metrics.items()})

    # Save final checkpoint
    if local_rank == 0:
        final_ckpt = Path(cfg.sft.get("output_dir", "checkpoints/sft")) / run_name / "final"
        trainer.save_model(str(final_ckpt))
        tokenizer.save_pretrained(str(final_ckpt))
        log.info(f"Final checkpoint saved to {final_ckpt}")
        wandb_logger.save_artifact(str(final_ckpt), f"sft-{run_name}-final", "model")


# ── Default Hydra config (inline, no separate sft_config.yaml needed) ─────────
# Hydra requires a config file OR defaults_list. We define defaults inline here.
# This allows `python scripts/train_sft.py model=qwen3_14b` to work directly.

import hydra.core.global_hydra as global_hydra  # noqa
from hydra._internal.utils import create_automatic_config_search_path  # noqa


def _setup_hydra_config():
    """Write the top-level Hydra config file if it doesn't exist."""
    config_dir = Path(__file__).parent.parent / "configs"
    config_file = config_dir / "sft_config.yaml"
    if not config_file.exists():
        content = """# Top-level SFT training config
# Override any field via CLI: python train_sft.py model=phi4 sft.learning_rate=1e-5

defaults:
  - model: qwen3_14b
  - sft: warmstart
  - data: sft_traces
  - parallelism: deepspeed_zero2
  - logging: logging
  - _self_

run_name: sft_run
seed: 42
"""
        config_file.write_text(content)


_setup_hydra_config()

if __name__ == "__main__":
    main()
