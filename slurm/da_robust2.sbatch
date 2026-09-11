#!/bin/bash
#SBATCH -J da_robust2
#SBATCH -A mch250030p
#SBATCH -p RM-small
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH -t 02:00:00
#SBATCH -o /ocean/projects/mch250030p/wxu7/llm_finetune/results/api_guidance/robust2_%j.log
cd /ocean/projects/mch250030p/wxu7/llm_finetune
source ~/.bashrc
module load anaconda3
conda activate my_env
python scripts/da_robust.py --n 430 --start 150 \
  --only best,size_h12,size_p32,best_notopo,h12,full \
  --procs 46 --no-llm \
  --out results/api_guidance/robust2_heldout.jsonl
