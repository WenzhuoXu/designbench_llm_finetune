#!/bin/bash
#SBATCH --job-name=comp_order
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --qos=low
#SBATCH --time=04:00:00
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

python scripts/composite_ordering_divergence.py \
  --workers 16 \
  --group-size "${GROUP_SIZE:-8}" \
  --max-steps "${MAX_STEPS:-5}" \
  --seeds "${SEEDS:-3}" \
  --split "${SPLIT:-train}" \
  --gamma "${GAMMA:-0.99}" \
  --sampler "${SAMPLER:-v2}" \
  --out "${OUT:-results/search_ladder/composite_ordering.jsonl}"
