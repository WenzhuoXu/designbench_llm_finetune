#!/bin/bash
#SBATCH --job-name=battery_d2
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=13
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -uo pipefail
cd /ocean/projects/mch250030p/wxu7/llm_finetune
BATTPY=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
export PYTHONPATH="$PWD:$PWD/scripts:/ocean/projects/mch250030p/wxu7/DesignBench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export AWS_REGION=us-west-2
export AWS_BEARER_TOKEN_BEDROCK="$(cat "$HOME/.bedrock_token")"
$BATTPY -u scripts/api_battery_d2.py \
  --problems-dir "${PROBLEMS_DIR:-data/battery_grid_q015}" \
  --n "${N:-40}" --k "${K:-8}" --m "${M:-3}" --k2 "${K2:-8}" \
  --max-steps "${MAX_STEPS:-12}" --workers "${WORKERS:-12}" \
  --out "${OUT:-results/api_guidance/battery_d2.jsonl}"
echo "done: $(date)"
