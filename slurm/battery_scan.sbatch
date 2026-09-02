#!/bin/bash
#SBATCH --job-name=bat_scan
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/scan_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/scan_%j.out
export OMP_NUM_THREADS=1
export PYTHONPATH=/ocean/projects/mch250030p/wxu7/llm_finetune:/ocean/projects/mch250030p/wxu7/DesignBench
BP=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
for I in 8 12 16 20; do
  $BP /ocean/projects/mch250030p/wxu7/llm_finetune/scripts/battery_scan.py $I > /ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/scan_${I}A.log 2>&1 &
done
wait
echo ALLDONE
