#!/bin/bash
#SBATCH --job-name=distill_gen
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:h100-80:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wxu2@andrew.cmu.edu

# Two-model distillation trajectory generation: 30B teacher reasons, 14B student acts.
set -euo pipefail
echo "Job $SLURM_JOB_NAME $SLURM_JOB_ID  start $(date)"

module load cuda/12.6.1
module load anaconda3/2024.10-1
conda activate my_env

PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
DESIGNBENCH_DIR="/ocean/projects/mch250030p/wxu7/DesignBench"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$DESIGNBENCH_DIR:${PYTHONPATH:-}"
export HF_HOME="/ocean/projects/mch250030p/wxu7/hf_models"
export TRANSFORMERS_CACHE="$HF_HOME"; export HF_HUB_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=0; export HF_HUB_DISABLE_XET=1; export CUDA_DEVICE_ORDER=PCI_BUS_ID

TEACHER_CONFIG="${TEACHER_CONFIG:-configs/model/qwen3_30b_a3b_thinking_2507.yaml}"
STUDENT_CKPT="${STUDENT_CKPT:-checkpoints/sft/gold_warmstart_qwen3_14b_fixed_20260521_001258/final}"
OUT="${OUT:-results/distill/traj.jsonl}"
MAX_PROBLEMS="${MAX_PROBLEMS:-6}"
MAX_STEPS="${MAX_STEPS:-12}"

python scripts/distill_gen.py \
  --teacher-config "$TEACHER_CONFIG" \
  --student-config configs/model/qwen3_14b.yaml \
  --student-checkpoint "$STUDENT_CKPT" \
  --problems-dir "${DESIGNBENCH_DIR}/data/problems" \
  --out "$OUT" \
  --max-problems "$MAX_PROBLEMS" \
  --max-steps "$MAX_STEPS"

echo "Done $(date)"
