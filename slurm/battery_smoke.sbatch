#!/bin/bash
#SBATCH --job-name=bat_smoke
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/smoke_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/smoke_%j.out

export OMP_NUM_THREADS=1
export PYTHONPATH=/ocean/projects/mch250030p/wxu7/llm_finetune:/ocean/projects/mch250030p/wxu7/DesignBench
/ocean/projects/mch250030p/wxu7/envs/battery/bin/python \
  /ocean/projects/mch250030p/wxu7/llm_finetune/scripts/battery_smoke.py \
  /ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/smoke.json
