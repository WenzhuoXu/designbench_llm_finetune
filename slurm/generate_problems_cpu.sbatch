#!/bin/bash
#SBATCH --job-name=gen_problems
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --qos=low
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Procedural DesignBench truss problem generation, CPU only.
#
# RM-shared + qos=low does NOT draw on the GPU allocation, and this job needs no
# GPU: every step is a linear program (HiGHS) plus small dense FEA solves.
#
# Environment overrides (all optional):
#   N               number of ACCEPTED problems to write            (default 250)
#   OUT             output directory                                (default DesignBench/data/problems_gen)
#   SEED            master seed                                     (default 20260827)
#   ATTEMPTS        cap on candidates tried                         (default 4*N)
#   GATE            none|any  -- require a search policy to solve it (default any)
#   BUDGET_RATIO    goals.maximum_mass / _metadata.optimal_mass     (default 1.1)
#   RUN_LADDER      1 to also run scripts/search_ladder.py on the
#                   generated set AND on the original set, so the
#                   solve-rate comparison is measured, not assumed  (default 1)
#
# Example:
#   N=250 sbatch slurm/generate_problems_cpu.sbatch

set -euo pipefail
module load anaconda3/2024.10-1
conda activate my_env

PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
DESIGNBENCH_DIR="/ocean/projects/mch250030p/wxu7/DesignBench"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$DESIGNBENCH_DIR:${PYTHONPATH:-}"
# The FEA and LP solves are tiny; threading them fights the process pool.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs/slurm

N="${N:-250}"
OUT="${OUT:-${DESIGNBENCH_DIR}/data/problems_gen}"
SEED="${SEED:-20260827}"
ATTEMPTS="${ATTEMPTS:-$(( 4 * N ))}"
GATE="${GATE:-any}"
BUDGET_RATIO="${BUDGET_RATIO:-1.1}"
RUN_LADDER="${RUN_LADDER:-1}"
WORKERS="${SLURM_CPUS_PER_TASK:-32}"

echo "n=$N out=$OUT seed=$SEED attempts=$ATTEMPTS gate=$GATE ratio=$BUDGET_RATIO workers=$WORKERS"

python scripts/generate_truss_problems.py \
  --n "$N" \
  --out "$OUT" \
  --seed "$SEED" \
  --workers "$WORKERS" \
  --max-attempts "$ATTEMPTS" \
  --require-policy-solve "$GATE" \
  --budget-ratio "$BUDGET_RATIO"

if [ "$RUN_LADDER" = "1" ]; then
  # Same policies, same step budget, same guards on both sets: the only honest
  # way to say whether the generated problems are as hard as the originals.
  POLICIES="greedy_critical,greedy_fsd,random,lookahead_v1_d1_minmass_legal,lookahead_v2_d1_minmass_legal,lookahead_v2_d2_minmass_legal"

  echo "=== search ladder on GENERATED set ==="
  python scripts/search_ladder.py \
    --problems "$OUT" \
    --workers "$WORKERS" \
    --max-steps 20 \
    --policies "$POLICIES" \
    --no-diagnostics \
    --out "results/problem_gen/ladder_generated"

  echo "=== search ladder on ORIGINAL set ==="
  python scripts/search_ladder.py \
    --problems "${DESIGNBENCH_DIR}/data/problems" \
    --workers "$WORKERS" \
    --max-steps 20 \
    --policies "$POLICIES" \
    --no-diagnostics \
    --out "results/problem_gen/ladder_original"
fi

echo "done"
