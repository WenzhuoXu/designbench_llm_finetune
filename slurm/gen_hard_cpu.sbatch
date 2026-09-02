#!/bin/bash
#SBATCH --job-name=gen_hard
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/slurm/%x_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/slurm/%x_%j.err
#
# Generate a batch of HARD truss problems, validate it, and measure the search
# ladder on it.  CPU only, free queue -- costs nothing against the GPU allocation.
#
# Env knobs (sbatch --export=ALL,N=40,OUTDIR=...,STRATA=...,TAG=pilot1):
#   N       number of accepted problems to write        (default 40)
#   OUTDIR  where to write them                          (default .../problems_hard)
#   STRATA  weighted strata mix, '+'-separated: "s1:1+s3:1+s5:1"
#   ASPECT  t/r floor for the certified mass reference   (default 0.04 = 1/25)
#   SEED    generator seed                               (default 20260828)
#   TAG     results subdirectory name                    (default hard)
#   POLICIES  '+'-separated policy list                  (default the three we report)
set -euo pipefail
module load anaconda3/2024.10-1
conda activate my_env

PD=/ocean/projects/mch250030p/wxu7/llm_finetune
DB=/ocean/projects/mch250030p/wxu7/DesignBench
cd "$PD"
export PYTHONPATH="$PD:$DB"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

N="${N:-40}"
OUTDIR="${OUTDIR:-$DB/data/problems_hard}"
# sbatch --export splits on commas, so STRATA uses '+' as its separator.
STRATA="$(echo "${STRATA:-s1:1+s2:1+s3:1+s4:1+s5:1}" | tr '+' ',')"
ASPECT="${ASPECT:-0.04}"
SEED="${SEED:-20260828}"
TAG="${TAG:-hard}"
POLICIES_ARG="$(echo "${POLICIES:-greedy_critical+lookahead_v2_d1+lookahead_v2_d2}" | tr '+' ',')"
STEPS="${STEPS:-20}"

echo "N=$N OUTDIR=$OUTDIR STRATA=$STRATA ASPECT=$ASPECT SEED=$SEED TAG=$TAG"
echo "policies: $POLICIES_ARG"

if [ "${SKIP_GEN:-0}" != "1" ]; then
  echo "########## GENERATE ##########"
  python scripts/generate_truss_problems.py \
    --mode hard --n "$N" --out "$OUTDIR" --prefix "${PREFIX:-hard_problem}" \
    --strata "$STRATA" --aspect-min "$ASPECT" --seed "$SEED" \
    --workers 32 --policy-steps 0 --max-attempts "${ATTEMPTS:-0}"
fi

echo "########## VALIDITY ##########"
python scripts/validate_generated_problems.py --dir "$OUTDIR" --workers 32

echo "########## SOLVABILITY: new batch ##########"
python scripts/search_ladder.py --problems "$OUTDIR" --workers 32 --max-steps "$STEPS" \
  --policies "$POLICIES_ARG" --out "$PD/results/search_ladder/${TAG}"

if [ "${RUN_REFS:-0}" = "1" ]; then
  echo "########## SOLVABILITY: originals ##########"
  python scripts/search_ladder.py --problems $DB/data/problems --workers 32 --max-steps "$STEPS" \
    --policies "$POLICIES_ARG" --out "$PD/results/search_ladder/${TAG}_orig"
  echo "########## SOLVABILITY: previous generated ##########"
  python scripts/search_ladder.py --problems $DB/data/problems_gen --workers 32 --max-steps "$STEPS" \
    --policies "$POLICIES_ARG" --out "$PD/results/search_ladder/${TAG}_gen"
fi

echo "########## PER-STRATUM SOLVE RATES / DIFFICULTY LABELS ##########"
# WRITE_LABELS=1 stamps _metadata.difficulty into the specs; otherwise report only.
LBL_ARGS="--report-only"
if [ "${WRITE_LABELS:-0}" = "1" ]; then LBL_ARGS=""; fi
python scripts/label_difficulty.py --problems "$OUTDIR" \
  --runs "$PD/results/search_ladder/${TAG}/runs.jsonl" $LBL_ARGS

if [ "${ONE_GLOBAL:-0}" = "1" ]; then
  echo "########## ONE-GLOBAL-ACTION REACHABILITY ##########"
  python scripts/one_global_action_probe.py --dirs "$OUTDIR" --workers 32 \
    --bounds declared --out "$PD/results/search_ladder/${TAG}/one_global_declared.jsonl"
  python scripts/one_global_action_probe.py --dirs "$OUTDIR" --workers 32 \
    --bounds physical --out "$PD/results/search_ladder/${TAG}/one_global_physical.jsonl"
fi
echo DONE
