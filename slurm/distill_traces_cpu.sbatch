#!/bin/bash
#SBATCH --job-name=distill_traces
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Search-policy distillation traces. CPU only: costs nothing against the GPU allocation.
set -euo pipefail
module load anaconda3/2024.10-1
conda activate my_env
PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:/ocean/projects/mch250030p/wxu7/DesignBench:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONWARNINGS=ignore

python scripts/distill_search_traces.py \
  --split-file data/splits/truss_v1.json \
  --split "${SPLIT:-train}" \
  --max-steps "${MAX_STEPS:-20}" \
  --workers 16 \
  --out "${OUT:-results/distill/search_policy_train.jsonl}"
