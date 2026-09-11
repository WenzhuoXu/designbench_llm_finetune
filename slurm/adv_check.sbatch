#!/bin/bash
#SBATCH -J advchk
#SBATCH -p RM-small
#SBATCH --qos=low
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH -t 00:40:00
#SBATCH -o /ocean/projects/mch250030p/wxu7/llm_finetune/logs/adv_check_%j.out
source ~/.bashrc
module load anaconda3
conda activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
python scripts/adv_check_expert.py
echo DONE
