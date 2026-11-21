#!/bin/bash
# train_eval.sh   inside job 173954's allocation (srun --jobid=173954 --overlap), after its vLLM servers are gone:
# the round's training with the repo's own launcher (scripts/rl/orchard_sft_rows.sh, as slurm/orchard_sft_rows.sbatch
# runs it), then, if the final adapter is there, the evaluation with the repo's launcher
# (scripts/rl/orchard_dataxp_eval.sh, as slurm/orchard_dataxp_eval.sbatch runs it) for as long as the allocation
# lasts; its episodes resume by key, so a later job with the same OUT finishes what is left.
set -u
RUN_DIR="${RUN_DIR:?}"; ROWS="${ROWS:?}"; DEV="${DEV:?}"; EVAL_OUT="${EVAL_OUT:?}"; EVAL_LABEL="${EVAL_LABEL:?}"
REPO=$HOME/rl/repo
FINAL="$RUN_DIR/final"
L=$HOME/rl/logs
echo "=== $(date '+%F %T %Z') train_eval on $(hostname): run $RUN_DIR"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -n 1)
[ "$USED" -lt 2000 ] || { echo "ABORT: a GPU still holds $USED MiB"; exit 2; }
if [ ! -e "$FINAL/adapter_model.safetensors" ]; then
  export RUN_DIR ROWS DEV CONFIG=configs/rl/dataxp_sft_scr1.yaml OVERRIDES="sft.temperature=1.0"
  bash "$REPO/scripts/rl/orchard_sft_rows.sh" >> "$L/dx_sft_inalloc_${SLURM_JOB_ID}.out" 2>&1
  rc=$?
  echo "=== $(date '+%F %T %Z') training rc=$rc"
  [ "$rc" = 0 ] || exit 3
fi
[ -e "$FINAL/adapter_model.safetensors" ] || { echo "ABORT: no final adapter"; exit 3; }
sleep 20
unset RUN_DIR ROWS DEV CONFIG OVERRIDES
export OUT="$EVAL_OUT" LABEL="$EVAL_LABEL" THINK=1 WORKERS=128 ADAPTER="$FINAL"
export PROBLEMS="/home/wxu2/rl/harness/problems_v2.jsonl /home/wxu2/rl/eval/problems_v3_native.jsonl"
bash "$REPO/scripts/rl/orchard_dataxp_eval.sh" >> "$L/dx_eval_inalloc_${SLURM_JOB_ID}.out" 2>&1
echo "=== $(date '+%F %T %Z') evaluation rc=$?"
