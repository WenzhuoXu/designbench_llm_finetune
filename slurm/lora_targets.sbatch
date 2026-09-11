#!/bin/bash
#SBATCH -J loratgt
#SBATCH -A mch250030p
#SBATCH -p RM-small
#SBATCH -q low
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -t 0:30:00
#SBATCH -o /ocean/projects/mch250030p/wxu7/llm_finetune/logs/lora_targets_%j.out
set -euo pipefail
module load anaconda3
source activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
export PYTHONWARNINGS=ignore
export HF_HOME=/ocean/projects/mch250030p/wxu7/hf_models
export HF_HUB_OFFLINE=1
python scripts/_lora_targets.py Qwen/Qwen3.8-27B
