#!/bin/bash
#SBATCH --job-name=bat_lad_f
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=14:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/ladderfull_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/ladderfull_%j.out
export OMP_NUM_THREADS=1
export PYTHONPATH=/ocean/projects/mch250030p/wxu7/llm_finetune:/ocean/projects/mch250030p/wxu7/DesignBench
BP=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
L=/ocean/projects/mch250030p/wxu7/llm_finetune/scripts/battery_ladder.py
D=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder
$BP $L --workers 24 --max-steps 8 --beam 4 --fidelity dfn_full --scales none \
      --tag full_scales-none    > $D/full_scales-none.log 2>&1 &
$BP $L --workers 24 --max-steps 8 --beam 4 --fidelity dfn_full --scales problem \
      --tag full_scales-problem > $D/full_scales-problem.log 2>&1 &
wait
echo ALLDONE
