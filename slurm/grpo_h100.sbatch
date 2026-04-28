#!/bin/bash
#SBATCH --job-name=designbench_grpo
#SBATCH --partition=GPU
#SBATCH --account=mch250030p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=h100-80:8
#SBATCH --cpus-per-task=13
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

# ============================================================
# DesignBench GRPO RL Training — 1 node × 8×H100-80GB
# ============================================================
# Architecture:
#   - vLLM server: launched on all 8 GPUs (tensor_parallel=8)
#     for fast rollout generation (5-10x faster than HF generate)
#   - GRPO training: DeepSpeed ZeRO-2 for actor update
#   - FEA workers: 104 CPU cores for parallel rollout execution
#
# vLLM and training share the same 8 GPUs via time-multiplexing:
#   Phase 1: vLLM generates K rollouts per prompt (GPU-intensive)
#   Phase 2: FEA executes rollouts (CPU-intensive, GPUs idle)
#   Phase 3: GRPO updates actor weights (GPU-intensive)
#   Phase 4: Weight sync back to vLLM
#
# Usage:
#   sbatch slurm/grpo_h100.sbatch
#   MODEL=qwen3_14b CHECKPOINT=checkpoints/sft/qwen3_14b sbatch slurm/grpo_h100.sbatch
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
RL_CONFIG="${RL_CONFIG:-grpo_truss}"
DATA_CONFIG="${DATA_CONFIG:-sft_traces}"
PARALLELISM="${PARALLELISM:-deepspeed_zero2}"
RUN_NAME="${RUN_NAME:-${MODEL}_grpo_${RL_CONFIG}_${SLURM_JOB_ID}}"

# If starting from an SFT checkpoint:
CHECKPOINT="${CHECKPOINT:-}"

echo "Model: $MODEL"
echo "RL config: $RL_CONFIG"
echo "Checkpoint: ${CHECKPOINT:-'(base model)'}"
echo "Run name: $RUN_NAME"

# ─── Environment ──────────────────────────────────────────────
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

mkdir -p logs/slurm logs/"$RUN_NAME"

# ─── vLLM server ───────────────────────────────────────────────
# vLLM is launched by train_grpo.py (rank 0 manages the server subprocess).
# The server uses tensor_parallel_size=8 across all H100s.
# Port 8000 is used for the OpenAI-compatible API.
# If vLLM is unavailable, the trainer falls back to HF generate (slower).
export VLLM_PORT=8000
export VLLM_TENSOR_PARALLEL_SIZE=8

# ─── FEA worker pool ───────────────────────────────────────────
# 104 CPUs available (13 × 8); use most of them for FEA parallelism
export N_FEA_WORKERS=32

# ─── Launch GRPO training ──────────────────────────────────────
CHECKPOINT_ARGS=""
if [ -n "$CHECKPOINT" ]; then
    CHECKPOINT_ARGS="model.model_name_or_path=$CHECKPOINT"
fi

torchrun \
  --nproc_per_node=8 \
  --master_port=29501 \
  --master_addr=localhost \
  scripts/train_grpo.py \
    model="$MODEL" \
    rl="$RL_CONFIG" \
    data="$DATA_CONFIG" \
    parallelism="$PARALLELISM" \
    run_name="$RUN_NAME" \
    rl.use_vllm=true \
    rl.vllm_server_port="$VLLM_PORT" \
    rl.n_env_workers="$N_FEA_WORKERS" \
    logging.wandb_project=designbench-training \
    ${CHECKPOINT_ARGS} \
    "${OVERRIDE_ARGS:-}"

echo "============================================================"
echo "GRPO training completed: $(date)"
echo "============================================================"
