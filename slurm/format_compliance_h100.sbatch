#!/bin/bash
#SBATCH --job-name=format_compliance
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

# ============================================================
# Gold warmstart compliance check — runs model.generate() on
# dev-set prompts and enforces strict semantic action gates.
# Saves per-sample JSON for downstream analysis and plotting.
#
# Usage:
#   CHECKPOINT=checkpoints/sft/warmstart_qwen3_14b_39083858/final \
#     sbatch slurm/format_compliance_h100.sbatch
#
# Optional overrides:
#   N=100                       # number of dev prompts (default: 50)
#   MAX_NEW_TOKENS=512          # generation budget (default: 256)
#   THRESHOLD=0.95              # semantic compliance threshold (default: 0.95)
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
THRESHOLD="${THRESHOLD:-0.95}"
SEMANTIC_COMPLIANCE_MIN="${SEMANTIC_COMPLIANCE_MIN:-0.95}"
EXECUTABLE_ACTION_MIN="${EXECUTABLE_ACTION_MIN:-0.95}"
SINGLE_ACTION_MIN="${SINGLE_ACTION_MIN:-0.98}"
ANSWER_SPAM_MAX="${ANSWER_SPAM_MAX:-0.02}"

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
  --semantic-compliance-min "$SEMANTIC_COMPLIANCE_MIN" \
  --executable-action-min "$EXECUTABLE_ACTION_MIN" \
  --single-action-min "$SINGLE_ACTION_MIN" \
  --answer-spam-max "$ANSWER_SPAM_MAX" \
  --save-results "$RESULTS_FILE"

echo "Results saved to: $RESULTS_FILE"
echo "Done: $(date)"
