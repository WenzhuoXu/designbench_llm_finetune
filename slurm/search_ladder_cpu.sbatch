#!/bin/bash
#SBATCH --job-name=search_ladder
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --qos=low
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# CPU-only search ladder over the full DesignBench problem set.
# No GPU: this costs nothing against the GPU allocation.
set -euo pipefail
module load anaconda3/2024.10-1
conda activate my_env

PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
DESIGNBENCH_DIR="/ocean/projects/mch250030p/wxu7/DesignBench"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$DESIGNBENCH_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# POLICIES uses '+' as separator: sbatch --export splits on commas.
POLICIES_ARG="$(echo "${POLICIES:-greedy_critical+random+lookahead_v1_d1+lookahead_v2_d1+lookahead_v2_d2}" | tr '+' ',')"
echo "policies: $POLICIES_ARG"

python scripts/search_ladder.py \
  --workers 16 \
  --max-steps 20 \
  --tau "${TAU:-0.05}" \
  --feasibility-offset "${OFFSET:-0.0}" \
  --policies "$POLICIES_ARG" \
  --out "${OUT:-results/search_ladder/full}"
