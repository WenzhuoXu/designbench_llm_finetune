"""
GRPO Training Entrypoint.

Launches vLLM server for fast rollout generation, then runs GRPO training
with DesignBench FEA environment for reward computation.

Usage (local, single GPU — no vLLM):
    python scripts/train_grpo.py model=qwen3_14b rl=grpo_truss rl.use_vllm=false

Usage (multi-GPU via torchrun, see slurm/grpo_h100.sbatch):
    torchrun --nproc_per_node=8 scripts/train_grpo.py \
        model=qwen3_14b rl=grpo_truss \
        parallelism=deepspeed_zero2 run_name=qwen3_grpo_001

Starting from SFT warmstart:
    torchrun --nproc_per_node=8 scripts/train_grpo.py \
        model.model_name_or_path=checkpoints/sft/qwen3_sft/final \
        rl=grpo_truss run_name=qwen3_grpo_from_sft

Key Hydra overrides (research hooks!):
    rl.reward_fn=composite           — reward function
    rl.reward_weights.feasibility=2.0 — feasibility weight
    rl.cost_fn=null                  — disable cost penalty
    rl.group_size=16                 — more rollouts per prompt
    rl.kl_coef=0.01                  — looser KL constraint
    rl.use_vllm=false               — use HF generate (slower, no vLLM needed)
"""

import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/DesignBench")

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent / "configs"),
    config_name="grpo_config",
)
def main(cfg: DictConfig) -> None:
    log.info("=" * 60)
    log.info("DesignBench GRPO RL Training")
    log.info("=" * 60)
    log.info(OmegaConf.to_yaml(cfg))

    from llm_finetune.logging.local_logger import LocalLogger
    from llm_finetune.logging.wandb_logger import build_wandb_logger

    run_name = cfg.get("run_name", "grpo_run")
    local_logger = LocalLogger(
        run_name=run_name,
        log_dir=cfg.get("logging", {}).get("log_dir", "logs"),
        log_every_n_steps=cfg.get("logging", {}).get("log_every_n_steps", 10),
    )
    local_logger.start()

    wandb_logger = build_wandb_logger(cfg, run_name=run_name)
    if cfg.get("logging", {}).get("wandb_enabled", True):
        wandb_logger.init()

    try:
        _run_grpo_training(cfg, local_logger, wandb_logger, run_name)
    finally:
        wandb_logger.finish()
        local_logger.stop()


def _run_grpo_training(cfg, local_logger, wandb_logger, run_name):
    import torch
    from llm_finetune.data.datasets.rl_dataset import RLPromptDataset
    from llm_finetune.data.processors.chat_formatter import ChatFormatter
    from llm_finetune.envs.truss_env import TrussRolloutEnv
    from llm_finetune.models.loader import load_model_and_tokenizer
    from llm_finetune.models.vllm_server import VLLMServer
    from llm_finetune.training.rl.costs import build_cost_from_config
    from llm_finetune.training.rl.grpo_trainer import build_grpo_trainer
    from llm_finetune.training.rl.rewards import build_reward_from_config

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    log.info(f"Process: local_rank={local_rank}, world_size={world_size}")

    # Load actor model
    log.info("Loading actor model ...")
    model, tokenizer = load_model_and_tokenizer(cfg.model)

    # Load reference model (kept on CPU to save GPU memory)
    # If None, TRL GRPOTrainer creates a copy automatically
    ref_model = None
    if cfg.rl.get("load_ref_model", False):
        log.info("Loading reference model on CPU ...")
        import copy
        ref_model = copy.deepcopy(model).cpu()
        ref_model.eval()

    # Build chat formatter
    formatter = ChatFormatter.from_model_id(cfg.model.model_name_or_path, tokenizer)

    # Build datasets
    data_cfg = cfg.get("data", {})
    rl_cfg = cfg.rl

    log.info("Building RL prompt dataset from problem specs ...")
    problems_dir = data_cfg.get(
        "problems_dir",
        "/ocean/projects/mch250030p/wxu7/DesignBench/data/problems",
    )
    train_dataset = RLPromptDataset.from_problems_dir(
        problems_dir=problems_dir,
        tokenizer=tokenizer,
        formatter=formatter,
        max_prompt_len=rl_cfg.get("max_prompt_length", 2048),
        repeat=data_cfg.get("repeat", 1),
    )
    log.info(f"RL dataset: {len(train_dataset)} prompts")

    # Build reward and cost functions (research hooks!)
    reward_fn = build_reward_from_config(OmegaConf.to_container(cfg.rl, resolve=True))
    cost_fn = build_cost_from_config(OmegaConf.to_container(cfg.rl, resolve=True))
    log.info(f"Reward: {reward_fn.name()}, Cost: {cost_fn.name() if cost_fn else 'None'}")

    # Build FEA environment
    env = TrussRolloutEnv(
        max_steps=rl_cfg.get("max_steps", 20),
        n_workers=rl_cfg.get("n_env_workers", 16),
    )

    # Launch vLLM server (rank 0 only)
    vllm_server = None
    if rl_cfg.get("use_vllm", True) and local_rank == 0:
        model_path = cfg.model.model_name_or_path
        cache_dir = cfg.model.get("cache_dir", "/ocean/projects/mch250030p/wxu7/hf_models")
        # Use local cache if available, else HF model ID
        local_model_path = Path(cache_dir) / "models--" + model_path.replace("/", "--")
        if not local_model_path.exists():
            local_model_path = model_path  # use HF ID

        log.info(f"Starting vLLM server (tensor_parallel={world_size}) ...")
        vllm_server = VLLMServer(
            model_path=str(local_model_path),
            tensor_parallel_size=world_size,
            port=rl_cfg.get("vllm_server_port", 8000),
            gpu_memory_utilization=0.85,
            dtype="bfloat16",
        )
        vllm_server.start(wait_ready=True)
        log.info("vLLM server ready")

    try:
        # Build GRPO trainer
        trainer = build_grpo_trainer(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            ref_model=ref_model,
            train_dataset=train_dataset,
            reward_fn=reward_fn,
            cost_fn=cost_fn,
            env=env,
            local_logger=local_logger,
            wandb_logger=wandb_logger,
        )

        log.info("Starting GRPO training ...")
        train_result = trainer.train()
        log.info(f"GRPO training complete: {train_result.metrics}")

        # Save final checkpoint
        if local_rank == 0:
            final_ckpt = (
                Path(rl_cfg.get("output_dir", "checkpoints/grpo")) / run_name / "final"
            )
            trainer.save_model(str(final_ckpt))
            tokenizer.save_pretrained(str(final_ckpt))
            log.info(f"Final GRPO checkpoint saved to {final_ckpt}")
            wandb_logger.save_artifact(str(final_ckpt), f"grpo-{run_name}-final", "model")
    finally:
        if vllm_server is not None:
            vllm_server.stop()


def _setup_grpo_hydra_config():
    config_dir = Path(__file__).parent.parent / "configs"
    config_file = config_dir / "grpo_config.yaml"
    if not config_file.exists():
        content = """# Top-level GRPO training config

defaults:
  - model: qwen3_14b
  - rl: grpo_truss
  - data: sft_traces
  - parallelism: deepspeed_zero2
  - logging: logging
  - _self_

run_name: grpo_run
seed: 42
"""
        config_file.write_text(content)


_setup_grpo_hydra_config()

if __name__ == "__main__":
    main()
