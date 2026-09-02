#!/bin/bash
#SBATCH --job-name=designbench_grpo
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:h100-80:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

# ============================================================
# DesignBench GRPO RL Training — default 2×H100-80GB LoRA
# ============================================================
# Architecture:
#   - Default uses HF generate, not vLLM, to keep 2-GPU runs available.
#   - Enable vLLM explicitly with USE_VLLM=true for short validations.
#   - GRPO training: DeepSpeed ZeRO-2 for actor update
#   - FEA workers: 104 CPU cores for parallel rollout execution
#
# vLLM and training share GPUs via time-multiplexing when USE_VLLM=true:
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
GPUS="${GPUS:-2}"
USE_VLLM="${USE_VLLM:-false}"

if [ "$GPUS" -gt 4 ] && [ "${ALLOW_GT4_GPUS:-0}" != "1" ]; then
  echo "ERROR: Refusing GPUS=$GPUS. Set ALLOW_GT4_GPUS=1 only after explicit approval."
  exit 2
fi

if [ -z "$CHECKPOINT" ] && [ "${ALLOW_BASE_GRPO:-0}" != "1" ]; then
  echo "ERROR: GRPO from a base model is disabled by default."
  echo "Set CHECKPOINT=checkpoints/sft/<gold_run>/final, or set ALLOW_BASE_GRPO=1 for an explicit base-model experiment."
  exit 2
fi

echo "Model: $MODEL"
echo "RL config: $RL_CONFIG"
echo "Checkpoint: ${CHECKPOINT:-'(base model)'}"
echo "GPUs: $GPUS"
echo "Use vLLM: $USE_VLLM"
echo "Run name: $RUN_NAME"

# ─── Environment ──────────────────────────────────────────────
export HF_HOME="/ocean/projects/mch250030p/wxu7/hf_models"
export TRANSFORMERS_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET=1

# Unique rendezvous port per job: a hardcoded port made any two jobs landing
# on the same node fail instantly with EADDRINUSE (killed T5b, T5c and the
# first distillation run 11-33 s after launch).
MASTER_PORT_DERIVED=$(( 20000 + (SLURM_JOB_ID % 20000) ))
echo "rendezvous port: $MASTER_PORT_DERIVED"
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
export VLLM_TENSOR_PARALLEL_SIZE="$GPUS"

# ─── FEA worker pool ───────────────────────────────────────────
# 2 shared H100s usually provide 24 CPUs; leave room for trainer workers.
export N_FEA_WORKERS="${N_FEA_WORKERS:-20}"
# trussme (via DesignBench) imports matplotlib.pyplot; Agg skips GUI-backend
# probing, ~90s of Lustre stat()s per process on this cluster.
export MPLBACKEND=Agg

# ─── Launch GRPO training ──────────────────────────────────────
CHECKPOINT_ARGS=""
if [ -n "$CHECKPOINT" ]; then
    CHECKPOINT_ARGS="model.model_name_or_path=$CHECKPOINT"
fi

torchrun \
  --nproc_per_node="$GPUS" \
  --master_port="$MASTER_PORT_DERIVED" \
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
    rl.use_vllm="$USE_VLLM" \
    rl.vllm_server_port="$VLLM_PORT" \
    rl.n_env_workers="$N_FEA_WORKERS" \
    logging.wandb_project=designbench-training \
    ${CHECKPOINT_ARGS} \
    ${OVERRIDE_ARGS:-}

echo "============================================================"
echo "GRPO training completed: $(date)"
echo "============================================================"
