#!/bin/bash
#SBATCH --job-name=dpo_search
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

# Preference training on the search's rankings. Single GPU: LoRA on a 14B in
# bf16 fits, and the reference policy is the same weights with the adapter
# disabled, so there is no second model to hold.
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

python scripts/train_dpo_search.py \
  --epochs "${EPOCHS:-2}" --beta "${BETA:-0.1}" --lr "${LR:-5e-6}" \
  --batch-pairs "${BATCH_PAIRS:-2}" --grad-accum "${GRAD_ACCUM:-8}" \
  --out "${OUT:-checkpoints/dpo/search_prefs}"
