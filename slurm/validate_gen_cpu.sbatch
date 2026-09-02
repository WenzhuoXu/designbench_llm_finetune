#!/bin/bash
#SBATCH --job-name=val_gen
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/slurm/%x_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/slurm/%x_%j.err
set -euo pipefail
module load anaconda3/2024.10-1
conda activate my_env
PD=/ocean/projects/mch250030p/wxu7/llm_finetune
DB=/ocean/projects/mch250030p/wxu7/DesignBench
cd $PD; export PYTHONPATH="$PD:$DB"; export OMP_NUM_THREADS=1

echo "########## VALIDITY: generated ##########"
python scripts/validate_generated_problems.py --dir $DB/data/problems_gen --workers 16
echo "########## VALIDITY: original ##########"
python scripts/validate_generated_problems.py --dir $DB/data/problems --pattern 'auto_problem_*.json' --workers 16

# Solvability through IDENTICAL policies on both sets.
echo "########## SOLVABILITY: generated ##########"
python scripts/search_ladder.py --problems $DB/data/problems_gen --workers 16 --max-steps 20 \
  --policies greedy_critical,lookahead_v2_d1 --out $PD/results/search_ladder/gen_validate
echo "########## SOLVABILITY: original ##########"
python scripts/search_ladder.py --problems $DB/data/problems --workers 16 --max-steps 20 \
  --policies greedy_critical,lookahead_v2_d1 --out $PD/results/search_ladder/orig_validate
