#!/bin/bash
#SBATCH -J corpus_v2
#SBATCH -A mch250030p
#SBATCH -p RM
#SBATCH -q low
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 4:00:00
#SBATCH -o /ocean/projects/mch250030p/wxu7/llm_finetune/logs/corpus_v2_%x_%j.out

set -euo pipefail
module load anaconda3
source activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune

export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# One domain per job so all five build concurrently and each fits the RM-small
# 2 h wall. 20,000 turns per domain is truss's ceiling at 10,580 instances
# (~0.9 solve rate, ~2.4 turns each), so every domain reaches the same quota
# and the balance is exact by construction rather than by trimming afterwards.
DOMAIN=${DOMAIN:?set DOMAIN}
python scripts/gen_corpus.py \
  --domains "$DOMAIN" \
  --turns-per-domain 80000 \
  --max-instances 60000 \
  --workers 128 \
  --out /ocean/projects/mch250030p/wxu7/llm_finetune/data/corpus_v2/$DOMAIN
