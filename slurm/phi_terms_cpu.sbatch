#!/bin/bash
#SBATCH --job-name=phi_terms
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/slurm/%x_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/slurm/%x_%j.err
set -euo pipefail
module load anaconda3/2024.10-1
conda activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
export PYTHONPATH=/ocean/projects/mch250030p/wxu7/llm_finetune:/ocean/projects/mch250030p/wxu7/DesignBench
export OMP_NUM_THREADS=1
python scratch/phi_terms.py
