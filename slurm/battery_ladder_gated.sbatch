#!/bin/bash
#SBATCH --job-name=bat_gated
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Does the feasibility offset close the one cell where the potential LOSES to
# undirected enumeration? On thermal_hard the plain Lagrangian scored 1/3 against
# blind enumeration's 3/3, because a 2%-over-temperature cell that charges twice
# as fast outranks a feasible one (alpha*V = 0.10 vs an objective gap of 0.33).
# tau is also dropped to battery scale: at 0.05 the hinge never saturates on
# violations of ~0.02, so satisfied constraints keep paying.
set -euo pipefail
BP=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
L=/ocean/projects/mch250030p/wxu7/llm_finetune/scripts/battery_ladder.py
D=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder
cd /ocean/projects/mch250030p/wxu7/llm_finetune
export PYTHONPATH="/ocean/projects/mch250030p/wxu7/llm_finetune:/ocean/projects/mch250030p/wxu7/DesignBench:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore

$BP $L --workers 24 --max-steps 8 --beam 4 --fidelity dfn_full --scales none \
      --tau 0.005 --feasibility-offset 5.0 \
      --policies "lookahead_v2_d1,lookahead_v2_d2" \
      --tag gated_offset5 > $D/gated_offset5.log 2>&1
echo ALLDONE
