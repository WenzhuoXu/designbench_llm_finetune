#!/bin/bash
#SBATCH --job-name=bat_lad_t
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/ladtest_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/ladtest_%j.out
export OMP_NUM_THREADS=1
export PYTHONPATH=/ocean/projects/mch250030p/wxu7/llm_finetune:/ocean/projects/mch250030p/wxu7/DesignBench
/ocean/projects/mch250030p/wxu7/envs/battery/bin/python \
  /ocean/projects/mch250030p/wxu7/llm_finetune/scripts/battery_ladder.py \
  --limit 2 --max-steps 2 --beam 2 --workers 2 --tag SMOKE
