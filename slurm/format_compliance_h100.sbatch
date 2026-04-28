#!/bin/bash
#SBATCH --job-name=format_compliance
#SBATCH --partition=GPU
#SBATCH --account=mch250030p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=h100-80:8
#SBATCH --cpus-per-task=13
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

# ============================================================
# Warmstart format compliance check — runs model.generate() on
# dev-set prompts and measures <action>/<think>/<answer> tag rates.
# Saves per-sample JSON for downstream analysis and plotting.
#
# Usage:
#   CHECKPOINT=checkpoints/sft/warmstart_qwen3_14b_39083858/final \
#     sbatch slurm/format_compliance_h100.sbatch
#
# Optional overrides:
#   N=100                       # number of dev prompts (default: 50)
#   MAX_NEW_TOKENS=512          # generation budget (default: 256)
#   THRESHOLD=0.90              # GRPO-readiness threshold (default: 0.90)
# ============================================================

set -euo pipefail

PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
DESIGNBENCH_DIR="/ocean/projects/mch250030p/wxu7/DesignBench"
cd "$PROJECT_DIR"

echo "Running format compliance check on warmstart checkpoint..."
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Start: $(date)"

module load cuda/12.6.1
module load anaconda3/2024.10-1
conda activate my_env

export PYTHONPATH="$PROJECT_DIR:$DESIGNBENCH_DIR:${PYTHONPATH:-}"
export HF_HOME="/ocean/projects/mch250030p/wxu7/hf_models"
export TRANSFORMERS_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_DISABLE_XET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

CHECKPOINT="${CHECKPOINT:-checkpoints/sft/warmstart_qwen3_14b_39083858/final}"
N="${N:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
THRESHOLD="${THRESHOLD:-0.90}"

# Derive run name from checkpoint path
CKPT_NAME=$(basename "$(dirname "$CHECKPOINT")")
RESULTS_FILE="logs/compliance/${CKPT_NAME}_${SLURM_JOB_ID}.json"

mkdir -p logs/slurm logs/compliance

python scripts/check_format_compliance.py \
  --checkpoint "$CHECKPOINT" \
  --dev "${DESIGNBENCH_DIR}/data/sft/dev.jsonl" \
  --n "$N" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --threshold "$THRESHOLD" \
  --save-results "$RESULTS_FILE"

echo "Results saved to: $RESULTS_FILE"
echo "Done: $(date)"
