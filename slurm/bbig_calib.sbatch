#!/bin/bash
#SBATCH --job-name=bbig_calib
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=13
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -uo pipefail
cd /ocean/projects/mch250030p/wxu7/llm_finetune
BATTPY=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
export PYTHONPATH="$PWD:$PWD/scripts:/ocean/projects/mch250030p/wxu7/DesignBench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export AWS_REGION=us-west-2
export AWS_BEARER_TOKEN_BEDROCK="$(cat "$HOME/.bedrock_token")"
for Q in 00 003 006; do
  echo "=== q=0.$Q ==="
  $BATTPY -u scripts/api_battery_2x2.py \
    --problems-dir "data/battery_big_q${Q}" \
    --n 116 --k 8 --max-steps 12 --workers 12 \
    --cells "full:single" \
    --out "results/api_guidance/bbig_calib_q${Q}.jsonl" || echo "q=0.$Q FAILED"
done
echo "done: $(date)"
