#!/bin/bash
#SBATCH -J da_mod2
#SBATCH -A mch250030p
#SBATCH -p RM-small
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH -t 02:00:00
#SBATCH -o /ocean/projects/mch250030p/wxu7/llm_finetune/results/da_mod2_%j.log
cd /ocean/projects/mch250030p/wxu7/llm_finetune
source ~/.bashrc
module load anaconda3
conda activate my_env
python design_agent/da_mod2.py
