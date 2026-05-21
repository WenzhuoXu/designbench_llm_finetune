#!/bin/bash
#SBATCH --job-name=mcts_deepseek
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100-80:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

set -euo pipefail

echo "============================================================"
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs/task: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "============================================================"

# NOTE: flash-attn requires GLIBC_2.32; this system ships GLIBC_2.28 (RHEL 8).
# The model loader falls back to torch SDPA automatically (still fused on H100).
# To fix: on a compute node run: pip install flash-attn --no-build-isolation
module load cuda/12.6.1
module load anaconda3/2024.10-1
conda activate my_env

PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

export TRANSFORMERS_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs/slurm logs/posterior

MODEL_CONFIG="${MODEL_CONFIG:-configs/model/deepseek_r1_32b.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
MODEL_TAG="${MODEL_TAG:-deepseek_r1_32b_base}"
PROBLEMS_DIR="${PROBLEMS_DIR:-/ocean/projects/mch250030p/wxu7/DesignBench/data/problems}"
PROBLEM_IDS="${PROBLEM_IDS:-auto_problem_000 auto_problem_001}"
PILOT_STEPS="${PILOT_STEPS:-3}"
RELIABILITY_TARGET="${RELIABILITY_TARGET:-0.90}"
MAX_DEPTH="${MAX_DEPTH:-3}"
LEVEL1_BRANCHING="${LEVEL1_BRANCHING:-5}"
DEEPER_BRANCHING="${DEEPER_BRANCHING:-3}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-4}"
MAX_STEPS="${MAX_STEPS:-9}"
ALPHA="${ALPHA:-5.0}"
GAMMA="${GAMMA:-0.99}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
TEMPERATURE="${TEMPERATURE:-0.8}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-design-bench}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-mcts_ablation_${MODEL_TAG}_${SLURM_JOB_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-logs/posterior/mcts_inference_${MODEL_TAG}_${SLURM_JOB_ID}.jsonl}"

echo "Model config   : $MODEL_CONFIG"
echo "Checkpoint     : $CHECKPOINT"
echo "Problem IDs    : $PROBLEM_IDS"
echo "Pilot steps    : $PILOT_STEPS"
echo "Max depth      : $MAX_DEPTH"
echo "Candidate count: $CANDIDATE_COUNT"
echo "Max steps      : $MAX_STEPS"
echo "Output         : $OUTPUT_PATH"
echo "W&B project    : $WANDB_PROJECT / $WANDB_RUN_NAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv

# shellcheck disable=SC2086
python -u scripts/eval_mcts_inference.py \
  --model-config "$MODEL_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --problems-dir "$PROBLEMS_DIR" \
  --problem-ids $PROBLEM_IDS \
  --pilot-steps "$PILOT_STEPS" \
  --reliability-target "$RELIABILITY_TARGET" \
  --max-depth "$MAX_DEPTH" \
  --level1-branching "$LEVEL1_BRANCHING" \
  --deeper-branching "$DEEPER_BRANCHING" \
  --candidate-count "$CANDIDATE_COUNT" \
  --max-steps "$MAX_STEPS" \
  --alpha "$ALPHA" \
  --gamma "$GAMMA" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --output "$OUTPUT_PATH"

echo "============================================================"
echo "MCTS inference ablation completed: $(date)"
echo "Output: $OUTPUT_PATH"
echo "============================================================"
