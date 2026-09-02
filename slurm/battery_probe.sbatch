#!/bin/bash
#SBATCH --job-name=bat_probe
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/probe_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/probe_%j.out
export OMP_NUM_THREADS=1
export PYTHONPATH=/ocean/projects/mch250030p/wxu7/llm_finetune:/ocean/projects/mch250030p/wxu7/DesignBench
BP=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
$BP /ocean/projects/mch250030p/wxu7/llm_finetune/scripts/battery_probe.py spme_lean &
$BP /ocean/projects/mch250030p/wxu7/llm_finetune/scripts/battery_probe.py dfn_full &
wait
