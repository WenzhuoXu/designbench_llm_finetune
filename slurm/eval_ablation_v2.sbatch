#!/bin/bash
#SBATCH --job-name=abl_v2_eval
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

# ============================================================
# Ablation Study v2 — checkpoint evaluation
# Runs eval_checkpoint.py on a single checkpoint using 1×H100-80GB
#
# Usage:
#   sbatch --export="ALL,CHECKPOINT=<path>,RUN_NAME=<name>" slurm/eval_ablation_v2.sbatch
# ============================================================

set -euo pipefail

echo "============================================================"
echo "Eval Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Checkpoint: ${CHECKPOINT}"
echo "Run name: ${RUN_NAME}"
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
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p logs/slurm "results/eval/${RUN_NAME}"

python scripts/eval_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --model-config "configs/model/qwen3_14b.yaml" \
  --problems-dir "${DESIGNBENCH_DIR}/data/problems" \
  --output-dir "results/eval/${RUN_NAME}" \
  --wandb-project "designbench-training" \
  --wandb-run-name "eval_${RUN_NAME}" \
  --n-gpus 1 \
  --max-steps 20 \
  --temperature 0.0

echo "============================================================"
echo "Evaluation completed: $(date)"
echo "============================================================"
