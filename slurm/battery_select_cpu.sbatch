#!/bin/bash
#SBATCH --job-name=bat_select
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/slurm/%x_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/slurm/%x_%j.err
# Battery domain uses its own venv: pybamm is NOT in my_env.
export OMP_NUM_THREADS=1
export PYTHONPATH=/ocean/projects/mch250030p/wxu7/llm_finetune:/ocean/projects/mch250030p/wxu7/DesignBench
BP=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
cd /ocean/projects/mch250030p/wxu7/llm_finetune

$BP scripts/battery_group_selection.py \
  --workers 12 \
  --group-size "${GROUP_SIZE:-8}" \
  --max-steps "${MAX_STEPS:-8}" \
  --seeds "${SEEDS:-3}" \
  --fidelity "${FIDELITY:-spme_lean}" \
  --out "${OUT:-results/battery_ladder/group_selection.jsonl}"
