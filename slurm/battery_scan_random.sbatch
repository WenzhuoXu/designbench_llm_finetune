#!/bin/bash
#SBATCH --job-name=batt_rnd_scan
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -uo pipefail
cd /ocean/projects/mch250030p/wxu7/llm_finetune
BATTPY=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
export PYTHONPATH="$PWD:$PWD/scripts:/ocean/projects/mch250030p/wxu7/DesignBench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

# 45 random starting cells x 2 currents = 90 scans, one problem each, on top of the
# 37 existing. Target n~120: at the observed discordance rate that gives an MDE near
# 12pp, enough to resolve the +11pp depth effect measured on truss (n=37 gives 18-25pp).
# Each scan is ~73 evaluations (~4 min); parallel across 13 workers.
N=${N_VARIANTS:-45}
CUR=${CURRENTS:-8,14}
i=0
for v in $(seq 0 $((N-1))); do
  for c in $(echo "$CUR" | tr ',' ' '); do
    ( $BATTPY -u scripts/battery_scan_random.py --n-variants "$N" --seed 0 \
        --only "rnd$(printf %03d "$v"):$c" > "logs/slurm/rndscan_${v}_${c}.log" 2>&1 ) &
    i=$((i+1))
    while [ "$(jobs -rp | wc -l)" -ge 13 ]; do wait -n; done
  done
done
wait
echo "launched $i scans; done $(date)"
ls -1 results/battery_ladder/scan_rnd*.jsonl 2>/dev/null | wc -l | sed 's/^/random scan files: /'
