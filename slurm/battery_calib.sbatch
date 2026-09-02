#!/bin/bash
#SBATCH --job-name=batt_calib
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=13
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"
BATTPY=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/scripts:/ocean/projects/mch250030p/wxu7/DesignBench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export AWS_REGION=us-west-2
export AWS_BEARER_TOKEN_BEDROCK="$(cat "$HOME/.bedrock_token")"

# Difficulty calibration, not the experiment. Runs the STATUS-QUO cell only
# (aggregate observation, single action) at three tightening factors, to find the
# one whose feasibility lands near 0.35. The untightened set measured 0.875 --
# only 12.5pp of headroom, which is why the first battery 2x2 could not have
# detected a ~17pp effect (Turn 19). Sweeping first is the cheap way to avoid
# repeating that.
for T in 010 020 030; do
  echo "=== tightening factor 0.$T ==="
  $BATTPY -u scripts/api_battery_2x2.py \
    --problems-dir "data/battery_problems_t${T}" \
    --n 24 --k 8 --max-steps 12 --workers 12 \
    --cells "aggregate:single" \
    --out "results/api_guidance/battery_calib_t${T}.jsonl" || echo "factor 0.$T FAILED"
done
echo "calibration done: $(date)"
