#!/bin/bash
#SBATCH -J cv2c
#SBATCH -A mch250030p
#SBATCH -p RM-small
#SBATCH -q low
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH -t 2:00:00
#SBATCH -o /ocean/projects/mch250030p/wxu7/llm_finetune/logs/corpus_v2_cheap_%j.out

set -euo pipefail
module load anaconda3
source activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune

export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# The four procedurally generated domains cost ~10 s per 12,000 instances, so the
# instance cap -- not the turn quota -- was binding. Over-generate them to 150,000
# turns each and let make_tiers.py trim down to whatever truss (FEA-bound, and the
# only domain with finite supply) actually yields. That keeps the balance exact
# without the cheap domains dictating the corpus size. All four run in one job
# because the QOS allows only five submissions at a time.
for DOMAIN in synth pipe catalogue cases; do
  python scripts/gen_corpus.py \
    --domains "$DOMAIN" \
    --turns-per-domain 80000 \
    --max-instances 400000 \
    --workers 48 \
    --out /ocean/projects/mch250030p/wxu7/llm_finetune/data/corpus_v2/$DOMAIN
done
