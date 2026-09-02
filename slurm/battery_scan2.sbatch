#!/bin/bash
#SBATCH --job-name=bat_scan2
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/scan2_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/scan2_%j.out
export OMP_NUM_THREADS=1
export PYTHONPATH=/ocean/projects/mch250030p/wxu7/llm_finetune:/ocean/projects/mch250030p/wxu7/DesignBench
BP=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
D=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder
S=/ocean/projects/mch250030p/wxu7/llm_finetune/scripts/battery_scan.py
for V in std dense thin coarse; do
  for I in 8 12 16; do
    $BP $S $I $V > $D/scan_${V}_${I}A.log 2>&1 &
  done
done
wait
echo ALLDONE
