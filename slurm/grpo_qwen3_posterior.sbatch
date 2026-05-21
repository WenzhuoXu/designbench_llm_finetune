#!/bin/bash
#SBATCH --job-name=qwen3_grpo_posterior
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
# Qwen3-14B GRPO with full posterior reward suite
#
# Rewards enabled:
#   lagrangian_potential  1.0  — dense Φ shaping
#   tree_advantage        1.0  — full-trajectory Φ gain (fallback; no tree expansion)
#   step_normalized_phi   0.5  — efficiency: Φ-gain / n_steps
#   difficulty_weighted   0.3  — hard-state upweighting (λ=0.5)
#   stagnation_escape     0.3  — action-class diversity bonus
#   macro_completion      0.05 — known-sequence completion bonus
#   forward_prediction    0.10 — CoT grounding via <predict> tags
#   feasibility           0.5  — terminal feasibility signal
#   grammar_compliance    0.1  — format compliance
#
# W&B project: designbench-training
# Run: qwen3_14b_grpo_posterior_<SLURM_JOB_ID>
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
MODEL="qwen3_14b"
RL_CONFIG="grpo_posterior"
DATA_CONFIG="sft_traces"
PARALLELISM="deepspeed_zero2"
RUN_NAME="qwen3_14b_grpo_posterior_${SLURM_JOB_ID}"

CHECKPOINT="${CHECKPOINT:-}"

echo "Model: $MODEL"
echo "RL config: $RL_CONFIG"
echo "Checkpoint: ${CHECKPOINT:-'(base model)'}"
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

# ─── vLLM ──────────────────────────────────────────────────────
export VLLM_PORT=8000
export VLLM_TENSOR_PARALLEL_SIZE=8
export N_FEA_WORKERS=32

# ─── Checkpoint args ───────────────────────────────────────────
CHECKPOINT_ARGS=""
if [ -n "$CHECKPOINT" ]; then
    CHECKPOINT_ARGS="model.model_name_or_path=$CHECKPOINT"
fi

# ─── Launch ────────────────────────────────────────────────────
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
    "logging.wandb_tags=[qwen3,posterior,all_rewards,tree_advantage,step_normalized_phi,difficulty_weighted]" \
    logging.wandb_enabled=true \
    logging.save_artifact_to_wandb=true \
    logging.gpu_stats_every_n_steps=20 \
    ${CHECKPOINT_ARGS}

echo "============================================================"
echo "GRPO training completed: $(date)"
echo "============================================================"
