#!/bin/bash
#SBATCH --job-name=qwen3_grpo_lora_2gpu
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:h100-80:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wxu2@andrew.cmu.edu

# ============================================================
# Qwen3-14B GRPO with LoRA — 2×H100-80GB
#
# LoRA reduces trainable params from 14.8B to ~115M (r=32, 7 modules),
# allowing training on 2 GPUs instead of 8.
#
# Memory budget per GPU (H100-80GB):
#   Training state (frozen base 29.6 GB + LoRA 1.2 GB + activations): ~36 GB
#   vLLM concurrent (tp=2, weights 14.8 GB + KV cache 8 GB):          ~25 GB
#   Peak concurrent:                                                    ~61 GB ✓
#
# Uses the same grpo_posterior reward config as job 40899153.
# Fallback: increase to 4 GPUs if OOM (change ntasks, gpus, nproc_per_node,
#           VLLM_TENSOR_PARALLEL_SIZE to 4).
#
# Usage:
#   sbatch slurm/grpo_qwen3_lora_2gpu.sbatch
#   CHECKPOINT=checkpoints/sft/qwen3_14b/final sbatch slurm/grpo_qwen3_lora_2gpu.sbatch
# ============================================================

set -euo pipefail

echo "============================================================"
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start: $(date)"
echo "============================================================"

module load cuda/12.6.1
module load anaconda3/2024.10-1
conda activate my_env

PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ─── Configuration ─────────────────────────────────────────────
MODEL="${MODEL:-qwen3_14b}"
RL_CONFIG="${RL_CONFIG:-grpo_posterior}"
DATA_CONFIG="${DATA_CONFIG:-sft_traces}"
PARALLELISM="${PARALLELISM:-deepspeed_zero2}"
RUN_NAME="${RUN_NAME:-${MODEL}_grpo_lora_2gpu_${RL_CONFIG}_${SLURM_JOB_ID}}"
GPUS="${GPUS:-2}"

CHECKPOINT="${CHECKPOINT:-}"

if [ "$GPUS" -gt 4 ] && [ "${ALLOW_GT4_GPUS:-0}" != "1" ]; then
  echo "ERROR: Refusing GPUS=$GPUS. Set ALLOW_GT4_GPUS=1 only after explicit approval."
  exit 2
fi

if [ -z "$CHECKPOINT" ] && [ "${ALLOW_BASE_GRPO:-0}" != "1" ]; then
  echo "ERROR: GRPO from a base model is disabled by default."
  echo "Set CHECKPOINT=checkpoints/sft/<gold_run>/final, or set ALLOW_BASE_GRPO=1 for an explicit base-model experiment."
  exit 2
fi

echo "Model: $MODEL (LoRA r=32)"
echo "RL config: $RL_CONFIG"
echo "Checkpoint: ${CHECKPOINT:-'(base model)'}"
echo "GPUs: $GPUS"
echo "Run name: $RUN_NAME"

# ─── Environment ───────────────────────────────────────────────
export HF_HOME="/ocean/projects/mch250030p/wxu7/hf_models"
export TRANSFORMERS_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p logs/slurm "logs/$RUN_NAME"

# 2 tasks × 13 CPUs = 26 CPUs; use most for FEA parallelism
export N_FEA_WORKERS=20

# ─── Checkpoint args ───────────────────────────────────────────
CHECKPOINT_ARGS=""
if [ -n "$CHECKPOINT" ]; then
    CHECKPOINT_ARGS="model.model_name_or_path=$CHECKPOINT"
fi

# ─── Launch ────────────────────────────────────────────────────
# Derive a unique port from job ID to avoid EADDRINUSE when multiple jobs land on the same node
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))

torchrun \
  --nproc_per_node="$GPUS" \
  --master_port="$MASTER_PORT" \
  --master_addr=localhost \
  scripts/train_grpo.py \
    model="$MODEL" \
    rl="$RL_CONFIG" \
    data="$DATA_CONFIG" \
    parallelism="$PARALLELISM" \
    run_name="$RUN_NAME" \
    model.use_lora=true \
    model.lora.r=32 \
    model.lora.lora_alpha=64 \
    model.lora.lora_dropout=0.0 \
    model.device_map=null \
    rl.use_vllm=false \
    rl.max_new_tokens="${MAX_NEW_TOKENS:-2048}" \
    rl.n_env_workers="$N_FEA_WORKERS" \
    logging.wandb_project=designbench-training \
    "logging.wandb_tags=[qwen3,lora,r32,posterior,2gpu]" \
    logging.wandb_enabled=true \
    logging.save_artifact_to_wandb=true \
    logging.gpu_stats_every_n_steps=20 \
    logging.log_every_n_steps="${LOG_EVERY:-10}" \
    ${CHECKPOINT_ARGS}

echo "============================================================"
echo "GRPO LoRA training completed: $(date)"
echo "============================================================"
