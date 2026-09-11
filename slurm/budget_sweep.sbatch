#!/bin/bash
#SBATCH -J budsweep
#SBATCH -A mch250030p
#SBATCH -p RM-small
#SBATCH -q low
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH -t 2:00:00
#SBATCH -o /ocean/projects/mch250030p/wxu7/llm_finetune/logs/budget_sweep_%j.out

set -euo pipefail
module load anaconda3
source activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1

# The teacher's cost-quality frontier. "Comparable feasibility in a handful of
# calls" is only a claim worth making if a search held to the same handful of
# simulations does NOT already get there -- otherwise the distilled model is
# competing with an expensive teacher that nobody would have run that way.
# At 8 analyses the budgeted search is indistinguishable from fully-stressed
# design (0-0 discordant on 20 problems); this finds where it separates.
for D in truss catalogue; do
  for B in 8 16 32 64 128 256; do
    echo "################ $D  budget=$B"
    python scripts/eval_distilled.py \
      --domain "$D" \
      --n 60 \
      --calls 8 \
      --analysis-budget "$B" \
      --arms search_k \
      --workers 46
  done
done
