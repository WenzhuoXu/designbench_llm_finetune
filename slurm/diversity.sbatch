#!/bin/bash
#SBATCH -J diversity
#SBATCH -A mch250030p
#SBATCH -p RM-small
#SBATCH -q low
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH -t 2:00:00
#SBATCH -o /ocean/projects/mch250030p/wxu7/llm_finetune/logs/diversity_%j.out

set -euo pipefail
module load anaconda3
source activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune

export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1

python scripts/diversity_report.py --workers 46 --sample-nn 3000 "$@"
