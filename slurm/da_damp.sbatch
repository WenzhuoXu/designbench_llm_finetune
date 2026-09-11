#!/bin/bash
#SBATCH -J da_damp
#SBATCH -A mch250030p
#SBATCH -p RM-small
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH -t 03:00:00
#SBATCH -o /ocean/projects/mch250030p/wxu7/llm_finetune/results/da_damp_%j.log
cd /ocean/projects/mch250030p/wxu7/llm_finetune
source ~/.bashrc
module load anaconda3
conda activate my_env
python design_agent/da_damp.py 200 150
