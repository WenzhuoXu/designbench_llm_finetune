#!/bin/bash
# A supervised round on rows (llm_finetune/rl_dataxp/sft_rows.py) on Orchard: one torchrun rank per
# GPU of the job. Restart-safe: the trainer resumes from the newest complete checkpoint of RUN_DIR.
#
#   RUN_DIR=/project/community/wxu2/rl/runs/<name> CONFIG=configs/rl/<cfg>.yaml ROWS=/path/train.jsonl \
#   [DEV=/path/dev.jsonl] [NTRAIN=<GPUs of the job>] [OVERRIDES="sft.lr=5e-5 ..."] bash scripts/rl/orchard_sft_rows.sh
#
# Exit status: the trainer's. The final adapter is RUN_DIR/final.
set -uo pipefail
B="${BASE:-/project/community/wxu2}"
REPO="${REPO:-$HOME/rl/repo}"
RUN_DIR="${RUN_DIR:?run directory}"
CONFIG="${CONFIG:?config yaml (relative to REPO or absolute)}"
ROWS="${ROWS:?training rows jsonl}"
[ "${CONFIG:0:1}" = / ] || CONFIG="$REPO/$CONFIG"
TRAIN_PY="${TRAIN_PY:-$B/rl/envs/train/bin/python}"
C="${CACHE_DIR:-$B/rl/cache}"
for f in "$CONFIG" "$TRAIN_PY" "$ROWS" ${DEV:+"$DEV"}; do
  [ -e "$f" ] || { echo "ABORT: missing $f"; exit 2; }
done
export HF_HOME="${HF_HOME:-$B/rl/hf}"; export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor CUDA_CACHE_PATH=$C/nv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
mkdir -p "$RUN_DIR/logs" "$TMPDIR" "$C/xdg"
cd "$REPO"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -r -a DEVS <<< "$CUDA_VISIBLE_DEVICES"
else NG=$(nvidia-smi -L | wc -l); DEVS=(); for i in $(seq 0 $((NG-1))); do DEVS+=("$i"); done; fi
NTRAIN=${NTRAIN:-${#DEVS[@]}}
[ "$NTRAIN" -le ${#DEVS[@]} ] || { echo "ABORT: NTRAIN=$NTRAIN > ${#DEVS[@]} GPUs"; exit 2; }
TRAIN_DEVS=$(IFS=,; echo "${DEVS[*]:0:$NTRAIN}")
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none} restarts=${SLURM_RESTART_COUNT:-0}"
echo "=== repo=$REPO config=$CONFIG run=$RUN_DIR rows=$ROWS dev=${DEV:-none} ranks=$NTRAIN overrides='${OVERRIDES:-}'"
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=$TRAIN_DEVS PYTHONPATH="$REPO" "$TRAIN_PY" -m torch.distributed.run --standalone \
  --nproc_per_node "$NTRAIN" -m llm_finetune.rl_dataxp.sft_rows --config "$CONFIG" \
  run_dir="$RUN_DIR" "sft.rows_file=$ROWS" ${DEV:+"sft.dev_file=$DEV"} ${OVERRIDES:-} \
  2>&1 | { nl=; while IFS= read -r line || [ -n "$line" ]; do
    # not tee: tee drops an output file for good after one failed write (a pool that was full for a minute)
    printf '%s\n' "$line"
    if { printf '%s%s\n' "$nl" "$line" >> "$RUN_DIR/logs/trainer.log"; } 2>/dev/null; then nl=; else nl=$'\n'; fi
  done; }
rc=${PIPESTATUS[0]}
ELAPSED=$(( $(date +%s) - T0 ))
echo "=== trainer rc=$rc; wall ${ELAPSED} s, GPU-hours $(awk "BEGIN{printf \"%.2f\", $ELAPSED*$NTRAIN/3600}")"
echo "DX_SFT_ROWS_DONE $(date '+%F %T %Z') rc=$rc"
exit $rc
