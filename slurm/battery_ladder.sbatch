#!/bin/bash
#SBATCH --job-name=bat_ladder
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=12:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/ladder_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/ladder_%j.out
export OMP_NUM_THREADS=1
export PYTHONPATH=/ocean/projects/mch250030p/wxu7/llm_finetune:/ocean/projects/mch250030p/wxu7/DesignBench
BP=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
L=/ocean/projects/mch250030p/wxu7/llm_finetune/scripts/battery_ladder.py

# A: the shipped adapter verbatim -- limit-relative constraint normalisers
$BP $L --workers 12 --max-steps 8 --beam 4 --fidelity dfn_full --scales none \
      --tag dfn_scales-none > /ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/run_scales-none.log 2>&1 &
# B: identical program, but with the per-constraint scales the problem states
$BP $L --workers 12 --max-steps 8 --beam 4 --fidelity dfn_full --scales problem \
      --tag dfn_scales-problem > /ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/run_scales-problem.log 2>&1 &
wait
echo ALLDONE
