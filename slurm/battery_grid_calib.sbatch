#!/bin/bash
#SBATCH --job-name=bgrid_calib
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=13
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -uo pipefail
PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"
BATTPY=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/scripts:/ocean/projects/mch250030p/wxu7/DesignBench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export AWS_REGION=us-west-2
export AWS_BEARER_TOKEN_BEDROCK="$(cat "$HOME/.bedrock_token")"
# Scan-neighbourhood reachability predicts LLM difficulty poorly (the shipped set has
# median 0 one-step-feasible and the LLM solved 87.5% of it), so calibrate empirically:
# status-quo cell only, one run per q, pick the q landing near 0.45.
for Q in 015 025 040; do
  echo "=== q=0.$Q ==="
  $BATTPY -u scripts/api_battery_2x2.py \
    --problems-dir "data/battery_grid_q${Q}" \
    --n 40 --k 8 --max-steps 12 --workers 12 \
    --cells "aggregate:single" \
    --out "results/api_guidance/bgrid_calib_q${Q}.jsonl" || echo "q=0.$Q FAILED"
done
echo "done: $(date)"
