#!/bin/bash
#SBATCH --job-name=pref_margin
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

set -euo pipefail
module load cuda/12.6.1
module load anaconda3/2024.10-1
conda activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
export PYTHONPATH="/ocean/projects/mch250030p/wxu7/llm_finetune:/ocean/projects/mch250030p/wxu7/DesignBench:${PYTHONPATH:-}"
export HF_HOME="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_DISABLE_XET=1 PYTHONWARNINGS=ignore

python scripts/eval_preference_margin.py \
  --checkpoint "${CHECKPOINT}" \
  --pairs "${PAIRS:-results/distill/prefs_eval_split.jsonl}" \
  --batch-pairs "${BATCH_PAIRS:-4}" \
  --out "results/regret/${RUN_NAME}.json"
