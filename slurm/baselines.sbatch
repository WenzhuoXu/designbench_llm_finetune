#!/bin/bash
#SBATCH -J baselines
#SBATCH -A mch250030p
#SBATCH -p RM-small
#SBATCH -q low
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH -t 2:00:00
#SBATCH -o /ocean/projects/mch250030p/wxu7/llm_finetune/logs/baselines_%j.out

set -euo pipefail
module load anaconda3
source activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1

# The teacher and the heuristic, on every domain, through the same episode loop
# the distilled model will run. Truss already has this pair measured; the other
# four do not, and catalogue matters most -- it is held out of every training
# tier, so its baseline is the reference for the transfer claim. Measuring all
# five here means step 4 never has to compare against a remembered number.
for D in truss synth pipe catalogue cases; do
  echo "################ $D"
  python scripts/eval_distilled.py \
    --domain "$D" \
    --n 60 \
    --calls 8 \
    --arms fsd,search \
    --workers 46
done
