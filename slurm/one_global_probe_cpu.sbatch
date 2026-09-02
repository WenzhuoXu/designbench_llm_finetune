#!/bin/bash
#SBATCH --job-name=one_global
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/slurm/%x_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/slurm/%x_%j.err
set -euo pipefail
module load anaconda3/2024.10-1
conda activate my_env
PD=/ocean/projects/mch250030p/wxu7/llm_finetune
DB=/ocean/projects/mch250030p/wxu7/DesignBench
cd "$PD"; export PYTHONPATH="$PD:$DB"; export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1
DIRS="${DIRS:-$DB/data/problems $DB/data/problems_gen}"
python scripts/one_global_action_probe.py --dirs $DIRS --workers 32 \
  --bounds "${BOUNDS:-declared}" \
  --out "${OUT:-$PD/results/search_ladder/one_global.jsonl}"
echo DONE
