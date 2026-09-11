#!/bin/bash
# Launch one point on the distillation scaling curve.
#
#   bash scripts/launch_curve.sh 1k          # smoke
#   bash scripts/launch_curve.sh 10k 100k all
#
# Everything is held fixed across tiers except which tier file is read, so a
# difference between two points is data quantity and nothing else. ZeRO-3 shards
# the 53 GB of frozen base weights across the eight GPUs; with LoRA the
# optimizer state is negligible, so the room goes to activations instead.
set -euo pipefail
cd /ocean/projects/mch250030p/wxu7/llm_finetune
TIERS_DIR=/ocean/projects/mch250030p/wxu7/llm_finetune/data/tiers

for TIER in "$@"; do
  F="$TIERS_DIR/train_${TIER}.jsonl"
  if [ ! -s "$F" ]; then
    echo "missing tier file: $F" >&2
    exit 1
  fi
  echo "submitting tier $TIER ($(wc -l < "$F") trajectories)"
  MODEL=qwen38_27b \
  SFT_CONFIG=distill_curve \
  DATA_CONFIG=distill_corpus \
  PARALLELISM=deepspeed_zero3 \
  RUN_NAME="distill27b_${TIER}" \
  OVERRIDE_ARGS="data.train_jsonl=$F" \
    sbatch --job-name="d27_${TIER}" --time=16:00:00 slurm/sft_h100.sbatch
done
