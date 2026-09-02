#!/bin/bash
#SBATCH --job-name=distill_prefs
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
module load anaconda3/2024.10-1
conda activate my_env
PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:/ocean/projects/mch250030p/wxu7/DesignBench:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore

python scripts/distill_preference_pairs.py \
  --split-file data/splits/truss_v1_auto.json --split "${SPLIT:-train}" \
  --max-steps "${MAX_STEPS:-20}" --pairs-per-state "${PAIRS:-3}" \
  --workers 16 --out "${OUT:-results/distill/search_preferences.jsonl}"
