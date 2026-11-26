#!/bin/bash
# con1 stage-1 continuation on the full consequence row set (docs/con1_full_rows.md): one 24 h `advanced` job of a
# chain on one Orchard node with 4 H100s.
#  1. submits its successor (afterany), unless the run is DONE, STOPped or the chain is at CHAIN_MAX;
#  2. builds the rows on the node's /tmp from the gz states on shared storage (build_full_rows.py; the same bytes every
#     time: the sha1 must equal the first build's, $F/stats_q$Q.json, else the job stops before training);
#  3. trains (scripts/rl/orchard_sft_rows.sh, configs/rl/con1_sft.yaml) from the stage-1 adapter INIT, or, when an
#     exported checkpoint exists in $M/export, from that one (adapter + step; optimizer restarted, lr re-warmed);
#     full checkpoints stay on the node's /tmp;
#  4. exports the newest checkpoint's adapter (~1.9 GB) + state.json + metrics.jsonl to $M/export every EXPORT_EVERY s,
#     replacing the previous export only after the new one is complete;
#  5. 30 min before the time limit (SIGUSR1): waits for the next checkpoint, stops the trainer, exports, exits 0; the
#     successor resumes from the export. At the end of the epoch: final adapter to $M/final, $M/DONE, successor cancelled.
#
#   sbatch -p advanced --qos=adv_4gpu_qos \
#     --export=ALL,INIT=/project/community/wxu2/con1/main/run2/stage1/final,M=/project/community/wxu2/con1/full/run1 \
#     /project/community/wxu2/con1/repo2m/slurm/con1_full.sbatch
# Stop the chain: touch $M/STOP (the running job finishes its day; no successor trains).
# Monitor: bash $REPO/scripts/con1/mon_full.sh <slurm job id>
set -uo pipefail
B=/project/community/wxu2
C1=$B/con1
REPO="${REPO:-$C1/repo2m}"
INIT="${INIT:?the stage-1 final adapter (e.g. $C1/main/run2/stage1/final)}"
M="${M:?run dir on shared storage (e.g. $C1/full/run1)}"
Q="${Q:-36}"
PER="${PER:-28}"
MAX_TOKENS="${MAX_TOKENS:-24000}"
F="${F:-$C1/full}"
DEV="${DEV:-$C1/main/rows/dev_facet.jsonl}"      # stage 1's dev rows: the dev curve continues stage 1's
EXPORT_EVERY="${EXPORT_EVERY:-7200}"
CHAIN_MAX="${CHAIN_MAX:-8}"
CHAIN_I="${CHAIN_I:-1}"
JOB="${SLURM_JOB_ID:-x}"
NTRAIN="${NTRAIN:-4}"
NEXT_ARGS="${NEXT_ARGS:--p advanced --qos=adv_4gpu_qos}"   # the successor's partition (a test chain: preempt)
ROWS_FILE="${ROWS_FILE:-}"                                    # test only: train on this file, no build
T=/tmp/con1_full_$JOB
RUN=$T/run
EXPORT=$M/export
TRAIN_PY=$B/rl/envs/train/bin/python
MODEL=$B/rl/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
G=$C1/gen
D=$C1/data
mkdir -p "$M/logs" "$M/chain" "$T/logs" "$RUN"
log() { echo "=== $(date '+%F %T') $*"; }
log "host=$(hostname) job=$JOB chain=$CHAIN_I/$CHAIN_MAX repo=$REPO q=$Q rows/step=$PER init=$INIT m=$M"
[ -e "$M/DONE" ] && { log "run DONE already"; exit 0; }
[ -e "$M/STOP" ] && { log "STOP file: not training"; exit 0; }
for f in "$( [ "$INIT" = none ] && echo "$TRAIN_PY" || echo "$INIT/adapter_config.json" )" "$DEV" "${ROWS_FILE:-$F/stats_q$Q.json}" "$TRAIN_PY"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 2; }
done
exported_step() { "$TRAIN_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$EXPORT/state.json" 2>/dev/null || echo 0; }
# two failed jobs in a row without progress: no more successors
NFAIL=$(ls "$M/chain" 2>/dev/null | grep -c "^fail_.*_step$(exported_step)$" || true)
if [ "${NFAIL:-0}" -ge 2 ]; then
  log "ABORT: $NFAIL jobs failed at export step $(exported_step); see $M/chain and $M/logs"; touch "$M/STOP_failures"; exit 4
fi
fail() { log "FAILED: $*"; touch "$M/chain/fail_${JOB}_step$(exported_step)"; }

# ---- 1. successor
NEXT=
if [ "$CHAIN_I" -lt "$CHAIN_MAX" ]; then
  NEXT=$(sbatch --parsable $NEXT_ARGS --dependency=afterany:$JOB \
    --export=ALL,INIT="$INIT",M="$M",Q="$Q",PER="$PER",REPO="$REPO",CHAIN_I=$((CHAIN_I+1)),CHAIN_MAX="$CHAIN_MAX",DEV="$DEV",MAX_TOKENS="$MAX_TOKENS" \
    "${SBATCH_FILE:-$REPO/slurm/con1_full.sbatch}") && echo "$NEXT" > "$M/chain/next_of_$JOB" && log "successor $NEXT submitted" \
    || log "!!! successor not submitted"
fi

# ---- 2. rows on node /tmp
ROWS=$T/train_q$Q.jsonl
if [ -n "$ROWS_FILE" ]; then
  ROWS=$ROWS_FILE; log "TEST: rows $ROWS (no build)"
else
WANT=$("$TRAIN_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_sha1"])' "$F/stats_q$Q.json")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
log "building the rows (Q=$Q, cap $MAX_TOKENS tokens)"
T0=$(date +%s)
"$TRAIN_PY" "$REPO/scripts/con1/build_full_rows.py" \
  --states "$C1/main/states_native_offmenu.jsonl.gz" "$F/states_native_offmenu_rest.jsonl.gz" \
           "$G/out/states_native_orch.jsonl.gz" "$G/out/states_scripted.jsonl.gz" \
  --exclude-rows "$C1/main/rows/train_facet.jsonl" "$C1/main/rows/dev_facet.jsonl" \
  --forbid-problems "$D/queries_v2.jsonl" "$G/bench_problems.jsonl" --tokenizer "$MODEL" --target facet \
  --q "$Q" --max-tokens "$MAX_TOKENS" --seed 0 --workers "${BUILD_WORKERS:-30}" \
  --out-train "$ROWS" --out-dev "$T/dev_full_q$Q.jsonl" --stats "$T/stats.json" > "$T/logs/build.log" 2>&1
tail -3 "$T/logs/build.log"
GOT=$("$TRAIN_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_sha1"])' "$T/stats.json" 2>/dev/null)
log "rows built in $(( $(date +%s) - T0 )) s; sha1 $GOT (expected $WANT)"
cp "$T/logs/build.log" "$M/logs/build_$JOB.log" 2>/dev/null
if [ "$GOT" != "$WANT" ]; then fail "rows differ from the first build"; exit 3; fi
fi

# ---- 3. trainer
if [ ! -e "$EXPORT/state.json" ] && ls "$M/chain" | grep -q "^exported_step_"; then
  fail "the run exported before ($(ls "$M/chain" | grep "^exported_step_" | sort -t_ -k3 -n | tail -1)) but $EXPORT/state.json is missing: not restarting from step 0"
  exit 6
fi
if [ -e "$EXPORT/state.json" ]; then
  S=$(exported_step)
  log "resuming from the export of step $S"
  # the run's metrics up to the exported step (steps after it are trained again)
  "$TRAIN_PY" - "$EXPORT/metrics.jsonl" "$RUN/metrics.jsonl" "$S" <<'EOF'
import json, sys
src, dst, s = sys.argv[1], sys.argv[2], int(sys.argv[3])
keep = {}
try:
    for l in open(src):
        if l.strip():
            r = json.loads(l)
            if r["step"] < s:
                keep[r["step"]] = l if l.endswith("\n") else l + "\n"
except FileNotFoundError:
    pass
open(dst, "w").writelines(keep[k] for k in sorted(keep))
print("metrics restored: %d steps" % len(keep))
EOF
  [ -e "$EXPORT/dev_start.json" ] && cp "$EXPORT/dev_start.json" "$RUN/"
fi
if [ "$INIT" = none ]; then IOV="policy.init_adapter=null policy.require_init_adapter=false"; else IOV="policy.init_adapter=$INIT"; fi
OVR="$IOV sft.rows_per_step=$PER sft.save_every=${SAVE_EVERY:-5} sft.keep_last=1 sft.dev_every=${DEV_EVERY:-50}"
OVR="$OVR sft.keep_adapter_every=100000 sft.lazy_rows=true sft.check_rows=false sft.resume_dir=$EXPORT sft.rewarm_steps=${REWARM:-10} ${TEST_OVR:-}"
log "trainer: $OVR"
setsid env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" NTRAIN="$NTRAIN" REPO="$REPO" RUN_DIR="$RUN" \
  CONFIG=configs/rl/con1_sft.yaml ROWS="$ROWS" DEV="$DEV" OVERRIDES="$OVR" \
  bash "$REPO/scripts/rl/orchard_sft_rows.sh" > "$T/logs/train.log" 2>&1 &
TP=$!

# ---- 4. export
newest_ckpt() {   # newest complete checkpoint step on the node, 0 if none
  local best=0 d s
  for d in "$RUN"/ckpt/step_*; do
    [ -e "$d/COMPLETE" ] && [ -e "$d/adapter/adapter_model.safetensors" ] || continue
    s=$((10#${d##*step_})); [ "$s" -gt "$best" ] && best=$s
  done
  echo $best
}
free_mb() { df -B1M --output=avail "$B" | tail -1 | tr -dc 0-9; }
export_newest() {
  # The newest node checkpoint's adapter + state.json + metrics to $EXPORT. With room for two copies: written beside
  # the old export, then swapped in (never a moment without a whole export). Without room (the shared volume is
  # often within a GB of full): the old export is replaced in place, state.json removed first and written last, so
  # a half-copied export has no state.json and is never resumed from (the job refuses to restart from scratch then).
  local s cur st i need mode
  s=$(newest_ckpt); cur=$(exported_step)
  [ "$s" -gt "$cur" ] || { log "export: nothing newer than step $cur"; return 0; }
  st=$T/stage_$s; rm -rf "$st"
  for i in 1 2 3; do               # hard links on /tmp: the trainer may prune the checkpoint meanwhile
    cp -al "$RUN/ckpt/step_$(printf %05d $s)" "$st" 2>/dev/null && break
    rm -rf "$st"; sleep 5; s=$(newest_ckpt); st=$T/stage_$s
  done
  [ -d "$st" ] || { log "!!! export: could not stage a checkpoint"; return 1; }
  cp "$RUN/metrics.jsonl" "$st/metrics.jsonl" 2>/dev/null || touch "$st/metrics.jsonl"
  [ -e "$RUN/dev_start.json" ] && cp "$RUN/dev_start.json" "$st/"
  need=$(( $(du -sm "$st/adapter" | cut -f1) + 600 ))
  rm -rf "$M/export.new"
  if [ -e "$EXPORT/state.json" ] && [ "$(free_mb)" -ge "$need" ]; then mode=swap; else mode=inplace; fi
  if [ "$mode" = swap ]; then
    mkdir -p "$M/export.new" && cp -r "$st/adapter" "$M/export.new/" && cp "$st/metrics.jsonl" "$M/export.new/" \
      && { [ ! -e "$st/dev_start.json" ] || cp "$st/dev_start.json" "$M/export.new/"; } \
      && cmp -s "$st/adapter/adapter_model.safetensors" "$M/export.new/adapter/adapter_model.safetensors" \
      && cp "$st/state.json" "$M/export.new/" \
      && { rm -rf "$M/export.old"; [ ! -e "$EXPORT" ] || mv "$EXPORT" "$M/export.old"; } \
      && mv "$M/export.new" "$EXPORT" && rm -rf "$M/export.old"
  else
    rm -f "$EXPORT/state.json" && rm -rf "$EXPORT/adapter" && mkdir -p "$EXPORT" && cp -r "$st/adapter" "$EXPORT/" \
      && cp "$st/metrics.jsonl" "$EXPORT/" && { [ ! -e "$st/dev_start.json" ] || cp "$st/dev_start.json" "$EXPORT/"; } \
      && cmp -s "$st/adapter/adapter_model.safetensors" "$EXPORT/adapter/adapter_model.safetensors" \
      && cp "$st/state.json" "$EXPORT/state.json"
  fi
  if [ "$(exported_step)" = "$s" ]; then
    log "exported step $s ($mode) to $EXPORT; shared free $(free_mb) MB"
    touch "$M/chain/exported_step_$s"
    rm -rf "$st"
    cp "$T/logs/train.log" "$M/logs/train_$JOB.log.part" 2>/dev/null && mv "$M/logs/train_$JOB.log.part" "$M/logs/train_$JOB.log"
    return 0
  fi
  rm -rf "$M/export.new" "$st"
  log "!!! export of step $s failed ($mode; shared free $(free_mb) MB, need $need MB); export now: step $(exported_step)"
  return 1
}
left_s() {   # seconds before the job's time limit
  local e; e=$(squeue -h -j "$JOB" -o %e 2>/dev/null)
  [ -n "$e" ] && [ "$e" != N/A ] && echo $(( $(date -d "$e" +%s) - $(date +%s) )) || echo 99999
}
STOPPING=0
trap 'STOPPING=1; log "SIGUSR1: time limit near"' USR1
stop_trainer() {
  kill -TERM -- "-$TP" 2>/dev/null || kill -TERM "$TP" 2>/dev/null
  local i; for i in $(seq 1 60); do kill -0 "$TP" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$TP" 2>/dev/null
}
LAST_EXPORT=$(date +%s); EXPORT_OK=1
while kill -0 "$TP" 2>/dev/null; do
  sleep 60
  if [ "$STOPPING" = 1 ]; then
    S0=$(newest_ckpt); W0=$(date +%s)
    log "waiting (at most 15 min) for a checkpoint after step $S0"
    while kill -0 "$TP" 2>/dev/null && [ "$(newest_ckpt)" -le "$S0" ] && [ $(( $(date +%s) - W0 )) -lt 900 ]; do sleep 20; done
    stop_trainer
    until export_newest; do
      [ "$(left_s)" -gt 300 ] || { log "!!! no export of the last checkpoint before the time limit"; break; }
      sleep 60
    done
    cp "$T/logs/train.log" "$M/logs/train_$JOB.log" 2>/dev/null
    log "stopped for the time limit at checkpoint $(newest_ckpt); the successor ${NEXT:-(none)} resumes"
    exit 0
  fi
  if [ $(( $(date +%s) - LAST_EXPORT )) -ge "$EXPORT_EVERY" ] || { [ "$EXPORT_OK" = 0 ] && [ $(( $(date +%s) - LAST_EXPORT )) -ge 600 ]; }; then
    if export_newest; then EXPORT_OK=1; else EXPORT_OK=0; fi
    LAST_EXPORT=$(date +%s)
  fi
done
wait "$TP"; rc=$?
cp "$T/logs/train.log" "$M/logs/train_$JOB.log" 2>/dev/null
grep "DX_SFT_ROWS_DONE\|trainer rc" "$T/logs/train.log" | tail -2

# ---- 5. end
if [ "$rc" = 0 ] && [ -e "$RUN/final/adapter_config.json" ]; then
  for i in $(seq 1 10000); do
    rm -rf "$M/final.new"
    # after 30 min without room, the export (at most EXPORT_EVERY older than the final) makes room
    if [ "$i" -gt 60 ] && [ "$(free_mb)" -lt $(( $(du -sm "$RUN/final" | cut -f1) + 600 )) ] && [ -e "$EXPORT" ]; then
      log "shared volume short for 30 min: the export is dropped to make room for the final adapter"; rm -rf "$EXPORT"
    fi
    mkdir -p "$M/final.new" && cp -r "$RUN/final/." "$M/final.new/"       && cmp -s "$RUN/final/adapter_model.safetensors" "$M/final.new/adapter_model.safetensors"       && mv "$M/final.new" "$M/final" && break
    rm -rf "$M/final.new"; log "!!! final adapter copy failed (shared free $(free_mb) MB), retrying"
    [ "$(left_s)" -gt 300 ] || break
    sleep 30
  done
  [ -e "$M/final/adapter_config.json" ] || { fail "final adapter not copied; it is on $(hostname):$RUN/final"; exit 5; }
  cp "$RUN/metrics.jsonl" "$M/metrics.jsonl" 2>/dev/null
  date '+%F %T' > "$M/DONE"
  [ -n "$NEXT" ] && scancel "$NEXT" && log "successor $NEXT cancelled"
  log "CON1_FULL_DONE: final adapter $M/final"
  exit 0
fi
export_newest
fail "trainer rc=$rc"
tail -30 "$T/logs/train.log"
exit "$rc"
