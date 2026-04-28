#!/bin/bash
#SBATCH --job-name=designbench_sft
#SBATCH --partition=GPU
#SBATCH --account=mch250030p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=h100-80:8
#SBATCH --cpus-per-task=13
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

# ============================================================
# DesignBench SFT Training — 1 node × 8×H100-80GB
# ============================================================
# Usage examples:
#   # Full SFT run with Qwen3-14B, warmstart config
#   sbatch slurm/sft_h100.sbatch
#
#   # Quick smoke test (10 steps only)
#   sbatch --time=00:10:00 slurm/sft_h100.sbatch --override-args "sft.max_steps=10"
#
#   # Different model
#   MODEL=deepseek_r1_14b sbatch slurm/sft_h100.sbatch
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
RUN_NAME="${RUN_NAME:-${MODEL}_sft_${SFT_CONFIG}_${SLURM_JOB_ID}}"

echo "Model: $MODEL"
echo "SFT config: $SFT_CONFIG"
echo "Run name: $RUN_NAME"

# ─── HuggingFace cache ────────────────────────────────────────
export HF_HOME="/ocean/projects/mch250030p/wxu7/hf_models"
export TRANSFORMERS_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_OFFLINE=0         # Allow downloading models
# Avoid HF Xet download race condition when 8 ranks hit the cache simultaneously
export HF_HUB_DISABLE_XET=1

# ─── NCCL configuration for H100 NVLink ──────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=2     # GPUDirect RDMA
export NCCL_SOCKET_IFNAME=^lo   # Use all non-loopback interfaces

# ─── CUDA configuration ───────────────────────────────────────
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ─── Create log directories ───────────────────────────────────
mkdir -p logs/slurm logs/"$RUN_NAME"

# ─── Launch training with torchrun ────────────────────────────
# torchrun handles distributed process group initialization
# DeepSpeed ZeRO-2 handles model/gradient/optimizer sharding
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
    "${OVERRIDE_ARGS:-}"

echo "============================================================"
echo "Training completed: $(date)"
echo "============================================================"
