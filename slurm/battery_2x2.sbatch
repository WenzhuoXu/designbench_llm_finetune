#!/bin/bash
#SBATCH --job-name=battery_2x2
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
# RM partitions reject --qos=gpu; `low` is the valid RM qos for this account
# (see slurm/battery_ladder_full.sbatch, which runs on RM-shared).
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=13
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"
# PyBaMM lives in its own env, not my_env.
BATTPY=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/scripts:/ocean/projects/mch250030p/wxu7/DesignBench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export AWS_REGION=us-west-2
# Credential read from a chmod-600 file outside the repo; never echoed or logged.
export AWS_BEARER_TOKEN_BEDROCK="$(cat "$HOME/.bedrock_token")"

echo "battery 2x2 starting: $(date)"
echo "PROBLEMS_DIR=${PROBLEMS_DIR:-data/battery_problems} N=${N:-24} K=${K:-8} MAX_STEPS=${MAX_STEPS:-12} WORKERS=${WORKERS:-12}"

$BATTPY -u scripts/api_battery_2x2.py \
  --problems-dir "${PROBLEMS_DIR:-data/battery_problems}" \
  --n "${N:-24}" \
  --k "${K:-8}" \
  --max-steps "${MAX_STEPS:-12}" \
  --workers "${WORKERS:-12}" \
  --out "${OUT:-results/api_guidance/battery_2x2.jsonl}"

echo "battery 2x2 done: $(date)"
