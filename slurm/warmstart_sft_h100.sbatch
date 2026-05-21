#!/bin/bash
#SBATCH --job-name=warmstart_sft
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:h100-80:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

# ============================================================
# Gold warmstart SFT — Pre-training for GRPO RL
# Default: 2×H100-80GB LoRA. Never use >4 GPUs unless ALLOW_GT4_GPUS=1.
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
# Compute policy:
#   - Default 2 GPUs, LoRA enabled.
#   - Use 3-4 GPUs only by overriding GPUS and the Slurm allocation.
#   - Refuse >4 GPUs unless ALLOW_GT4_GPUS=1.
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
GPUS="${GPUS:-2}"
RUN_NAME="${RUN_NAME:-gold_warmstart_${MODEL}_${SLURM_JOB_ID}}"

if [ "$GPUS" -gt 4 ] && [ "${ALLOW_GT4_GPUS:-0}" != "1" ]; then
  echo "ERROR: Refusing GPUS=$GPUS. Set ALLOW_GT4_GPUS=1 only after explicit approval."
  exit 2
fi

echo "Model: $MODEL"
echo "SFT config: $SFT_CONFIG"
echo "Data config: $DATA_CONFIG"
echo "GPUs: $GPUS"
echo "Run name: $RUN_NAME"

# ─── HuggingFace cache ────────────────────────────────────────
# Must match cache_dir in configs/model/base.yaml
export HF_HOME="/ocean/projects/mch250030p/wxu7/hf_models"
export TRANSFORMERS_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_OFFLINE=0
# Avoid HF Xet download race condition when ranks hit the cache simultaneously
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
  --nproc_per_node="$GPUS" \
  --master_port=29500 \
  --master_addr=localhost \
  scripts/train_sft.py \
    model="$MODEL" \
    sft="$SFT_CONFIG" \
    data="$DATA_CONFIG" \
    parallelism="$PARALLELISM" \
    run_name="$RUN_NAME" \
    model.use_lora=true \
    model.lora.r=32 \
    model.lora.lora_alpha=64 \
    model.lora.lora_dropout=0.0 \
    model.device_map=null \
    logging.wandb_project=designbench-training \
    ${OVERRIDE_ARGS:-}

echo "============================================================"
echo "Warmstart SFT completed: $(date)"
echo "Checkpoint: checkpoints/sft/$RUN_NAME/final"
echo "Next: CHECKPOINT=checkpoints/sft/$RUN_NAME/final sbatch slurm/grpo_h100.sbatch"
echo "============================================================"
