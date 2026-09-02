#!/bin/bash
#SBATCH --job-name=abl_v2_eval
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wxu2@andrew.cmu.edu

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
# trussme (via DesignBench) imports matplotlib.pyplot; Agg skips GUI-backend
# probing, ~90s of Lustre stat()s per process on this cluster.
export MPLBACKEND=Agg

mkdir -p logs/slurm "results/eval/${RUN_NAME}"

# Rollout protocol (default multi — matches the model's trained distribution and
# master-doc §3). Override per-submit: ROLLOUT=single, MAX_PROBLEMS=25, etc.
ROLLOUT="${ROLLOUT:-multi}"
MAX_PROBLEMS="${MAX_PROBLEMS:-0}"
# Held-out discipline: without a split file this evaluates problems GRPO trained on.
SAMPLE_FLAG=""
if [ -n "${N_SAMPLES:-}" ]; then SAMPLE_FLAG="--n-samples ${N_SAMPLES} --sample-seed ${SAMPLE_SEED:-0} --batch-size ${BATCH_SIZE:-8}"; fi
BATCH_FLAG=""
if [ "${BATCHED:-0}" = "1" ]; then BATCH_FLAG="--batched --batch-size ${BATCH_SIZE:-8}"; fi
SPLIT_FLAG=""
if [ -n "${SPLIT_FILE:-}" ]; then SPLIT_FLAG="--split-file ${SPLIT_FILE} --split ${SPLIT:-eval}"; fi
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_STEPS="${MAX_STEPS:-20}"
FEW_SHOT_FLAG=""; [ "${FEW_SHOT:-0}" = "1" ] && FEW_SHOT_FLAG="--few-shot"
EXTRACT_FLAG=""; [ "${EXTRACT_ACTION:-0}" = "1" ] && EXTRACT_FLAG="--extract-action"

python scripts/eval_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --model-config "${MODEL_CONFIG:-configs/model/qwen3_14b.yaml}" \
  --problems-dir "${PROBLEMS_DIR:-${DESIGNBENCH_DIR}/data/problems}" \
  --output-dir "results/eval/${RUN_NAME}" \
  --wandb-project "designbench-training" \
  --wandb-run-name "eval_${RUN_NAME}" \
  --n-gpus 1 \
  --rollout "${ROLLOUT}" \
  --max-steps "${MAX_STEPS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-problems "${MAX_PROBLEMS}" \
  ${SPLIT_FLAG} ${BATCH_FLAG} ${SAMPLE_FLAG} \
  ${FEW_SHOT_FLAG} ${EXTRACT_FLAG} \
  --temperature "${TEMPERATURE:-0.0}"

echo "============================================================"
echo "Evaluation completed: $(date)"
echo "============================================================"
