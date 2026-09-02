#!/bin/bash
#SBATCH --job-name=batt_scan_grid
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -uo pipefail
PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"
BATTPY=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/scripts:/ocean/projects/mch250030p/wxu7/DesignBench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

# The battery 2x2 has 24 problems drawn from only 10 (variant, current) scans, so the
# honest clustering unit gives ~10-24 units and an MDE of 20.2pp -- larger than the
# 18.3pp effect it is meant to test. More problems from the SAME scans would be
# pseudo-replication (Turn 18). Each (variant, current) pair is a genuinely different
# design task -- different starting cell, different load case -- so scanning more load
# cases is the honest way to add clusters.
#
# A scan is only ~73 evaluations (~4 min), so this grid is cheap.
CURRENTS="4 6 10 14 18 22 24"
VARIANTS="std dense thin coarse"

n=0
for v in $VARIANTS; do
  for c in $CURRENTS; do
    f="results/battery_ladder/scan_${v}_${c}A.jsonl"
    if [ -s "$f" ]; then echo "skip $v ${c}A (exists)"; continue; fi
    ( $BATTPY -u scripts/battery_scan.py "$c" "$v" > "logs/slurm/scan_${v}_${c}A.log" 2>&1 \
      && echo "done $v ${c}A" || echo "FAILED $v ${c}A" ) &
    n=$((n+1))
    while [ "$(jobs -rp | wc -l)" -ge 13 ]; do wait -n; done
  done
done
wait
echo "launched $n scans; complete: $(date)"
ls -1 results/battery_ladder/scan_*.jsonl | wc -l | sed 's/^/total scan files now: /'
