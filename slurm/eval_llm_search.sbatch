#!/bin/bash
#SBATCH --job-name=eval_llm_search
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

set -euo pipefail
module load cuda/12.6.1
module load anaconda3/2024.10-1
conda activate my_env
PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:/ocean/projects/mch250030p/wxu7/DesignBench:${PYTHONPATH:-}"
export HF_HOME="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_DISABLE_XET=1
export PYTHONWARNINGS=ignore
# trussme (via DesignBench) imports matplotlib.pyplot; Agg skips GUI-backend
# probing, ~90s of Lustre stat()s per process on this cluster.
export MPLBACKEND=Agg


python scripts/eval_llm_search.py \
  --checkpoint "${CHECKPOINT}" \
  --model-config configs/model/qwen3_14b.yaml \
  --problems-dir "${PROBLEMS_DIR:-/ocean/projects/mch250030p/wxu7/DesignBench/data/problems}" \
  --split-file "${SPLIT_FILE:-data/splits/truss_v1_auto.json}" --split "${SPLIT:-eval}" \
  --max-problems "${MAX_PROBLEMS:-0}" \
  --n-candidates "${K:-8}" \
  --max-steps "${MAX_STEPS:-20}" \
  --temperature "${TEMPERATURE:-0.8}" \
  ${EXTRA:-} \
  --out "${OUT}"
