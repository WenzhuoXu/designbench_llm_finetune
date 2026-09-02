#!/bin/bash
#SBATCH --job-name=sft_distill
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:h100-80:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

# Distil the potential-guided search into the policy.
# The dataset is small (59 traces / ~1.2k assistant turns), so 2 GPUs is ample --
# the 8-GPU sft_h100.sbatch would just queue longer for no gain.
set -euo pipefail
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Start: $(date)"

module load cuda/12.6.1
module load anaconda3/2024.10-1
conda activate my_env

PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:/ocean/projects/mch250030p/wxu7/DesignBench:${PYTHONPATH:-}"
export HF_HOME="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_DISABLE_XET=1

# Unique rendezvous port per job: a hardcoded port made any two jobs landing
# on the same node fail instantly with EADDRINUSE (killed T5b, T5c and the
# first distillation run 11-33 s after launch).
MASTER_PORT_DERIVED=$(( 20000 + (SLURM_JOB_ID % 20000) ))
echo "rendezvous port: $MASTER_PORT_DERIVED"

MODEL="${MODEL:-qwen3_14b}"
SFT_CONFIG="${SFT_CONFIG:-warmstart}"
DATA_CONFIG="${DATA_CONFIG:-distill_search}"
PARALLELISM="${PARALLELISM:-deepspeed_zero2}"
RUN_NAME="${RUN_NAME:-sft_distill_${SLURM_JOB_ID}}"
# Start from the warmstart checkpoint so its perfect grammar compliance is kept
# and only the POLICY is changed.
CHECKPOINT="${CHECKPOINT:-checkpoints/sft/gold_warmstart_qwen3_14b_fixed_20260521_001258/final}"

torchrun --nproc_per_node=2 --master_port="$MASTER_PORT_DERIVED" --master_addr=localhost \
  scripts/train_sft.py \
    model="$MODEL" \
    sft="$SFT_CONFIG" \
    data="$DATA_CONFIG" \
    parallelism="$PARALLELISM" \
    run_name="$RUN_NAME" \
    model.model_name_or_path="$CHECKPOINT" \
    model.use_lora=true model.lora.r=32 model.lora.lora_alpha=64 model.lora.lora_dropout=0.0 \
    model.device_map=null \
    logging.wandb_project=designbench-training \
    ${OVERRIDE_ARGS:-}

echo "Done: $(date)"
