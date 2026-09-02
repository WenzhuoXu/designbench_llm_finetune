#!/bin/bash
#SBATCH --job-name=refs_ladder
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/slurm/%x_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/slurm/%x_%j.err
# Reference solve rates for the ORIGINAL and PREVIOUS-GENERATED sets under the
# identical three policies the new batch is reported on.
set -euo pipefail
module load anaconda3/2024.10-1
conda activate my_env
PD=/ocean/projects/mch250030p/wxu7/llm_finetune
DB=/ocean/projects/mch250030p/wxu7/DesignBench
cd "$PD"; export PYTHONPATH="$PD:$DB"; export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1
POL="greedy_critical,lookahead_v2_d1,lookahead_v2_d2"
echo "########## ORIGINAL (131) ##########"
python scripts/search_ladder.py --problems $DB/data/problems --workers 32 --max-steps 20 \
  --policies "$POL" --out $PD/results/search_ladder/ref3_orig
echo "########## PREVIOUS GENERATED (250) ##########"
python scripts/search_ladder.py --problems $DB/data/problems_gen --workers 32 --max-steps 20 \
  --policies "$POL" --out $PD/results/search_ladder/ref3_gen
echo DONE
