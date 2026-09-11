#!/bin/bash
#SBATCH -J da_cost
#SBATCH -A mch250030p
#SBATCH -p RM-small
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH -t 02:00:00
#SBATCH -o /ocean/projects/mch250030p/wxu7/llm_finetune/results/api_guidance/cost_%j.log
cd /ocean/projects/mch250030p/wxu7/llm_finetune
source ~/.bashrc
module load anaconda3
conda activate my_env
python scripts/da_cost.py --n 430 --start 150 \
  \
  --procs 46 \
  --out results/api_guidance/cost_heldout.jsonl
