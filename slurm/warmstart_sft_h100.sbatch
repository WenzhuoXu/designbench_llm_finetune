#!/bin/bash
#SBATCH --job-name=warmstart_sft
#SBATCH --partition=GPU
#SBATCH --account=mch250030p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=h100-80:8
#SBATCH --cpus-per-task=13
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

# ============================================================
# Warmstart SFT — Pre-training for GRPO RL
# 1 node × 8×H100-80GB, ~30-45 min with early stopping
# ============================================================
# Usage:
#   # Default: Qwen3-14B warmstart
#   sbatch slurm/warmstart_sft_h100.sbatch
#
#   # Different model
#   MODEL=deepseek_r1_14b sbatch slurm/warmstart_sft_h100.sbatch
#
#   # Smoke test (10 steps)
#   OVERRIDE_ARGS="sft.max_steps=10" sbatch --time=00:10:00 slurm/warmstart_sft_h100.sbatch
#
# Compute estimate (14B model, full-param, ZeRO-3):
#   - Model: 28 GB (bf16) → sharded across 8 GPUs
#   - Optimizer: 168 GB (fp32 states) → sharded
#   - Per-GPU: ~35-45 GB → comfortable on H100-80GB
#   - 5K examples × 2 epochs ÷ batch=128 ≈ 78 grad updates
#   - With early stopping: ~40-60 updates → ~30-45 min
# ============================================================

set -euo pipefail

echo "============================================================"
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs/task: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "============================================================"

# Load HPC modules
module load cuda/12.6.1
module load anaconda3/2024.10-1

# Activate conda environment
conda activate my_env

# ─── Project configuration ────────────────────────────────────
PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"

# Ensure llm_finetune is importable
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ─── Model selection (override with MODEL=<name> env var) ─────
MODEL="${MODEL:-qwen3_14b}"
SFT_CONFIG="${SFT_CONFIG:-pretrain}"
DATA_CONFIG="${DATA_CONFIG:-sft_traces}"
PARALLELISM="${PARALLELISM:-deepspeed_zero2}"
RUN_NAME="${RUN_NAME:-warmstart_${MODEL}_${SLURM_JOB_ID}}"

echo "Model: $MODEL"
echo "SFT config: $SFT_CONFIG"
echo "Data config: $DATA_CONFIG"
echo "Run name: $RUN_NAME"

# ─── HuggingFace cache ────────────────────────────────────────
# Must match cache_dir in configs/model/base.yaml
export HF_HOME="/ocean/projects/mch250030p/wxu7/hf_models"
export TRANSFORMERS_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_OFFLINE=0
# Avoid HF Xet download race condition when 8 ranks hit the cache simultaneously
export HF_HUB_DISABLE_XET=1

# ─── NCCL configuration for H100 NVLink ──────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=^lo

# ─── CUDA configuration ───────────────────────────────────────
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ─── Create log directories ───────────────────────────────────
mkdir -p logs/slurm logs/"$RUN_NAME"

# ─── Pre-download model weights (avoids 8-rank race condition) ─
# Resolve HF model ID from Hydra config, then pull via huggingface-cli
MODEL_HF_ID=$(python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('configs/model/${MODEL}.yaml')
print(c.model_name_or_path)
")
echo "Pre-downloading model: $MODEL_HF_ID"
huggingface-cli download "$MODEL_HF_ID" \
  --cache-dir "$HF_HOME" \
  --quiet || {
    echo "WARNING: huggingface-cli download failed; falling back to in-process download."
  }

# ─── Launch training with torchrun ────────────────────────────
torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  --master_addr=localhost \
  scripts/train_sft.py \
    model="$MODEL" \
    sft="$SFT_CONFIG" \
    data="$DATA_CONFIG" \
    parallelism="$PARALLELISM" \
    run_name="$RUN_NAME" \
    logging.wandb_project=designbench-training \
    ${OVERRIDE_ARGS:-}

echo "============================================================"
echo "Warmstart SFT completed: $(date)"
echo "Checkpoint: checkpoints/sft/$RUN_NAME/final"
echo "Next: CHECKPOINT=checkpoints/sft/$RUN_NAME/final sbatch slurm/grpo_h100.sbatch"
echo "============================================================"
