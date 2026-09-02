#!/bin/bash
#SBATCH --job-name=pm_all
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

# All arms scored against ONE base-model load. The load (~5 min) dominates the
# scoring (~2.6 min per arm at 363 pairs), so nine separate jobs would spend most
# of their time loading Qwen3-14B nine times and queueing nine times.
set -euo pipefail
module load cuda/12.6.1
module load anaconda3/2024.10-1
conda activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
export PYTHONPATH="/ocean/projects/mch250030p/wxu7/llm_finetune:/ocean/projects/mch250030p/wxu7/DesignBench:${PYTHONPATH:-}"
export HF_HOME="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_DISABLE_XET=1 PYTHONWARNINGS=ignore

W=checkpoints/sft/gold_warmstart_qwen3_14b_fixed_20260521_001258/final
python scripts/eval_preference_margin.py \
  --pairs "${PAIRS:-results/distill/prefs_eval_split.jsonl}" \
  --batch-pairs "${BATCH_PAIRS:-4}" \
  --out results/regret \
  --checkpoint "$W"                                                        --name warmstart \
  --checkpoint checkpoints/grpo/mt_t3_01b_critfeedback_20260616_0900/final --name champion \
  --checkpoint checkpoints/grpo/t5a_phi1_0823f/checkpoint-100              --name t5a_phi1 \
  --checkpoint checkpoints/grpo/t5b_phi2_0823f/checkpoint-100              --name t5b_phi2 \
  --checkpoint checkpoints/grpo/t5ao_phionly_v1_0823f/checkpoint-100       --name t5ao_phionly_v1 \
  --checkpoint checkpoints/grpo/t5bo_phionly_v2_0823f/checkpoint-100       --name t5bo_phionly_v2 \
  --checkpoint checkpoints/sft/t6_distill_search_0823c/final               --name t6_distill \
  --checkpoint checkpoints/dpo/search_prefs_v4/final                       --name dpo_prefs
