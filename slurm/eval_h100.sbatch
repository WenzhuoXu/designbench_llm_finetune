#!/bin/bash
#SBATCH --job-name=designbench_eval
#SBATCH --partition=GPU
#SBATCH --account=mch250030p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=h100-80:8
#SBATCH --cpus-per-task=13
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

# ============================================================
# DesignBench Evaluation — load checkpoint and run DesignBench benchmark
# ============================================================
# Usage:
#   CHECKPOINT=checkpoints/sft/qwen3_14b/checkpoint-500 sbatch slurm/eval_h100.sbatch
# ============================================================

set -euo pipefail

echo "============================================================"
echo "Eval Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Checkpoint: ${CHECKPOINT:-'(not set — will use base model)'}"
echo "Start: $(date)"
echo "============================================================"

module load cuda/12.6.1
module load anaconda3/2024.10-1
conda activate my_env

PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
DESIGNBENCH_DIR="/ocean/projects/mch250030p/wxu7/DesignBench"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$DESIGNBENCH_DIR:${PYTHONPATH:-}"

export HF_HOME="/ocean/projects/mch250030p/wxu7/hf_models"
export TRANSFORMERS_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_DISABLE_XET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

CHECKPOINT="${CHECKPOINT:-}"
MODEL="${MODEL:-qwen3_14b}"
RUN_NAME="${RUN_NAME:-eval_${MODEL}_${SLURM_JOB_ID}}"

mkdir -p logs/slurm results/eval

python scripts/eval_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --model-config "configs/model/${MODEL}.yaml" \
  --problems-dir "${DESIGNBENCH_DIR}/data/problems" \
  --output-dir "results/eval/${RUN_NAME}" \
  --wandb-project "designbench-training" \
  --wandb-run-name "${RUN_NAME}" \
  --n-gpus 8

echo "============================================================"
echo "Evaluation completed: $(date)"
echo "============================================================"
