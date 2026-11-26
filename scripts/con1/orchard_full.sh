#!/bin/bash
# con1 stage-1 continuation on the full consequence row set (docs/con1_full_rows.md): one 24 h `advanced` link of a
# chain on one Orchard node with 4 H100s. The node layout is orchard_main.sh's (/tmp/con1_main_<job>/<stage>/
# metrics.jsonl, ckpt/step_N/COMPLETE, final/; stage = "full"), so the parent's tools work on it unchanged:
# parent_check/mon.py (monitor) and guard_cycle.sh (checkpoint / final copies off the node into con1_reserve;
# scripts/con1/guard_cycle.sh is that script with env RES / STAGES / FORCE, defaults identical).
#  1. submits its successor (afterany), unless the run is DONE, STOPped or the chain is at CHAIN_MAX;
#  2. builds the rows on node /tmp from the gz states (build_full_rows.py; the sha1 must equal the first build's,
#     $F/stats_q$Q.json, else the job stops before training);
#  3. trains (scripts/rl/orchard_sft_rows.sh, configs/rl/con1_sft.yaml) from INIT, or from the newest checkpoint copy:
#     $RES/full_ckpt_step_N (adapter, optimizer, scheduler, rng: an exact resume) or, when newer, the adapter-only
#     copy $RES/full_adapter_step_N the guard makes when there is no room for a second whole copy (fresh optimizer,
#     lr re-warmed over REWARM = 10 steps, schedule position from its state.json);
#  4. runs the guard every 10 min (the final adapter at once, from reserved f-chunks; the newest checkpoint every
#     ~2 h, whole or adapter-only as the volume allows);
#  5. at SIGUSR1 (30 min before the limit): waits for the next checkpoint, stops the trainer, runs the guard with
#     FORCE=1 (copies that checkpoint now), exits 0; the successor resumes from it.
#  At the end of the epoch: the guard copies the final adapter to $RES/full_final; $M/DONE; successor cancelled.
#
#   sbatch -p advanced --qos=adv_4gpu_qos \
#     --export=ALL,INIT=/project/community/wxu2/con1_reserve/stage1_final,M=/project/community/wxu2/con1/full/run1 \
#     /project/community/wxu2/con1/repo2m/slurm/con1_full.sbatch
# Stop the chain: touch $M/STOP (the running link finishes its day; no successor trains).
# Monitor: srun --jobid=<id> --overlap -N1 -n1 python3 /project/community/wxu2/parent_check/mon.py /tmp/con1_main_<id>/full
set -uo pipefail
B=/project/community/wxu2
C1=$B/con1
REPO="${REPO:-$C1/repo2m}"
INIT="${INIT:?the stage-1 final adapter (e.g. $B/con1_reserve/stage1_final)}"
M="${M:?run dir on shared storage (e.g. $C1/full/run1)}"
Q="${Q:-36}"
PER="${PER:-28}"
MAX_TOKENS="${MAX_TOKENS:-24000}"
F="${F:-$C1/full}"
DEV="${DEV:-$C1/main/rows/dev_facet.jsonl}"      # stage 1's dev rows: the dev curve continues stage 1's
STAGE="${STAGE:-full}"
RES="${RES:-$B/con1_reserve}"                     # where the guard copies checkpoints / the final adapter
GUARD="$REPO/scripts/con1/guard_cycle.sh"
GUARD_EVERY="${GUARD_EVERY:-600}"
CHAIN_MAX="${CHAIN_MAX:-8}"
CHAIN_I="${CHAIN_I:-1}"
JOB="${SLURM_JOB_ID:-x}"
NTRAIN="${NTRAIN:-4}"
NEXT_ARGS="${NEXT_ARGS:--p advanced --qos=adv_4gpu_qos}"   # the successor's partition (a test chain: general)
ROWS_FILE="${ROWS_FILE:-}"                                    # test only: train on this file, no build
T=/tmp/con1_main_$JOB
RUN=$T/$STAGE
TRAIN_PY=$B/rl/envs/train/bin/python
MODEL=$B/rl/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
G=$C1/gen
D=$C1/data
mkdir -p "$M/logs" "$M/chain" "$T/logs" "$RUN"
log() { echo "=== $(date '+%F %T') $*"; }
log "host=$(hostname) job=$JOB chain=$CHAIN_I/$CHAIN_MAX repo=$REPO q=$Q rows/step=$PER init=$INIT m=$M res=$RES stage=$STAGE"
[ -e "$M/DONE" ] && { log "run DONE already"; exit 0; }
[ -e "$M/STOP" ] && { log "STOP file: not training"; exit 0; }
for f in "$( [ "$INIT" = none ] && echo "$TRAIN_PY" || echo "$INIT/adapter_config.json" )" "$DEV" \
         "${ROWS_FILE:-$F/stats_q$Q.json}" "$TRAIN_PY" "$GUARD"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 2; }
done
newest_copy() {   # the checkpoint copy to resume from: the guard keeps one whole copy (<stage>_ckpt_step_N: adapter,
  # optimizer, scheduler, rng -> exact resume) and one adapter-only copy (<stage>_adapter_step_N, made when the volume
  # had no room for a second whole copy -> fresh optimizer, lr re-warmed); the higher step wins, a whole copy on a tie
  local d f=0 fd= a=0 ad= s
  for d in $(ls -d "$RES/${STAGE}_ckpt_step_"* 2>/dev/null | sort -r); do
    if [ -e "$d/COMPLETE" ] && [ -e "$d/state.json" ] && [ -e "$d/optimizer.pt" ] && [ -e "$d/adapter/adapter_model.safetensors" ]; then
      fd=$d; f=$((10#${d##*step_})); break
    fi
  done
  for d in $(ls -d "$RES/${STAGE}_adapter_step_"* 2>/dev/null | sort -r); do
    if [ -e "$d/state.json" ] && [ -e "$d/adapter/adapter_model.safetensors" ]; then
      ad=$d; a=$((10#${d##*step_})); break
    fi
  done
  if [ -n "$ad" ] && [ "$a" -gt "$f" ]; then echo "$ad"; elif [ -n "$fd" ]; then echo "$fd"; fi
}
copy_step() { local d; d=$(newest_copy); [ -n "$d" ] && echo $((10#${d##*step_})) || echo 0; }
# two failed links in a row without progress: no more successors
NFAIL=$(ls "$M/chain" 2>/dev/null | grep -c "^fail_.*_step$(copy_step)$" || true)
if [ "${NFAIL:-0}" -ge 2 ]; then
  log "ABORT: $NFAIL links failed at copied step $(copy_step); see $M/chain and $M/logs"; touch "$M/STOP_failures"; exit 4
fi
fail() { log "FAILED: $*"; touch "$M/chain/fail_${JOB}_step$(copy_step)"; }

# ---- 1. successor
NEXT=
if [ "$CHAIN_I" -lt "$CHAIN_MAX" ]; then
  NEXT=$(sbatch --parsable $NEXT_ARGS --dependency=afterany:$JOB \
    --export=ALL,INIT="$INIT",M="$M",Q="$Q",PER="$PER",REPO="$REPO",CHAIN_I=$((CHAIN_I+1)),CHAIN_MAX="$CHAIN_MAX",DEV="$DEV",MAX_TOKENS="$MAX_TOKENS",RES="$RES",STAGE="$STAGE" \
    "${SBATCH_FILE:-$REPO/slurm/con1_full.sbatch}") && echo "$NEXT" > "$M/chain/next_of_$JOB" && log "successor $NEXT submitted" \
    || log "!!! successor not submitted"
fi

# ---- 2. rows on node /tmp
ROWS=$T/train_q$Q.jsonl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
if [ -n "$ROWS_FILE" ]; then
  ROWS=$ROWS_FILE; log "TEST: rows $ROWS (no build)"
else
  WANT=$("$TRAIN_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_sha1"])' "$F/stats_q$Q.json")
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
RD=$(newest_copy)
if [ -z "$RD" ] && ls "$M/chain" | grep -q "^copied_step_"; then
  fail "checkpoints of this run were copied before ($(ls "$M/chain" | grep "^copied_step_" | sort -t_ -k3 -n | tail -1)) but none is left in $RES: not restarting from step 0"
  exit 6
fi
if [ -n "$RD" ]; then
  S=$(copy_step)
  log "resuming from the checkpoint copy $RD (step $S)"
  # the run's metrics before that step (the metrics of every earlier link, kept in $M), for the monitor's history
  "$TRAIN_PY" - "$RUN/metrics.jsonl" "$S" "$M"/metrics_*.jsonl <<'EOF'
import json, sys
dst, s = sys.argv[1], int(sys.argv[2])
keep = {}
for src in sys.argv[3:]:
    try:
        for l in open(src):
            if l.strip():
                r = json.loads(l)
                if r["step"] < s:
                    keep[r["step"]] = l if l.endswith("\n") else l + "\n"
    except (OSError, ValueError):
        pass
open(dst, "w").writelines(keep[k] for k in sorted(keep))
print("metrics restored: %d steps" % len(keep))
EOF
  [ -e "$M/dev_start.json" ] && cp "$M/dev_start.json" "$RUN/"
fi
if [ "$INIT" = none ]; then IOV="policy.init_adapter=null policy.require_init_adapter=false"; else IOV="policy.init_adapter=$INIT"; fi
OVR="$IOV sft.rows_per_step=$PER sft.save_every=${SAVE_EVERY:-5} sft.keep_last=1 sft.dev_every=${DEV_EVERY:-50}"
OVR="$OVR sft.keep_adapter_every=100000 sft.lazy_rows=true sft.check_rows=false sft.resume_dir=${RD:-null}"
OVR="$OVR sft.rewarm_steps=${REWARM:-10} ${TEST_OVR:-}"   # the re-warm applies only to an adapter-only resume
log "trainer: $OVR"
setsid env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" NTRAIN="$NTRAIN" REPO="$REPO" RUN_DIR="$RUN" \
  CONFIG=configs/rl/con1_sft.yaml ROWS="$ROWS" DEV="$DEV" OVERRIDES="$OVR" \
  bash "$REPO/scripts/rl/orchard_sft_rows.sh" > "$T/logs/train.log" 2>&1 &
TP=$!

# ---- 4. guard + logs to shared storage
newest_ckpt() {   # newest complete checkpoint step on the node, 0 if none
  local best=0 d s
  for d in "$RUN"/ckpt/step_*; do
    [ -e "$d/COMPLETE" ] || continue
    s=$((10#${d##*step_})); [ "$s" -gt "$best" ] && best=$s
  done
  echo $best
}
guard() {   # one guard cycle for this job's stage (FORCE=1: copy the newest checkpoint whatever the last copy's age)
  RES="$RES" STAGES="$STAGE" FORCE="${1:-}" bash "$GUARD" "$JOB" 2>&1 | sed "s/^/guard: /"
  local s; s=$(copy_step); [ "$s" -gt 0 ] && touch "$M/chain/copied_step_$s"
}
keep_logs() {   # small files, written as new files by this node only
  cp "$RUN/metrics.jsonl" "$M/metrics_$JOB.jsonl.part" 2>/dev/null && mv "$M/metrics_$JOB.jsonl.part" "$M/metrics_$JOB.jsonl"
  [ -e "$RUN/dev_start.json" ] && [ ! -e "$M/dev_start.json" ] && cp "$RUN/dev_start.json" "$M/"
  cp "$T/logs/train.log" "$M/logs/train_$JOB.log.part" 2>/dev/null && mv "$M/logs/train_$JOB.log.part" "$M/logs/train_$JOB.log"
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
LAST_GUARD=$(date +%s)
while kill -0 "$TP" 2>/dev/null; do
  sleep 60
  if [ "$STOPPING" = 1 ]; then
    S0=$(newest_ckpt); W0=$(date +%s)
    log "waiting (at most 15 min) for a checkpoint after step $S0"
    while kill -0 "$TP" 2>/dev/null && [ "$(newest_ckpt)" -le "$S0" ] && [ $(( $(date +%s) - W0 )) -lt 900 ]; do sleep 20; done
    stop_trainer
    S1=$(newest_ckpt)
    until guard 1; [ "$(copy_step)" -ge "$S1" ]; do
      [ "$(left_s)" -gt 300 ] || { log "!!! checkpoint $S1 not copied before the time limit (copy: step $(copy_step))"; break; }
      sleep 60
    done
    keep_logs
    log "stopped for the time limit; checkpoint $S1, copy in $RES: step $(copy_step); the successor ${NEXT:-(none)} resumes"
    exit 0
  fi
  if [ $(( $(date +%s) - LAST_GUARD )) -ge "$GUARD_EVERY" ]; then
    guard; keep_logs; LAST_GUARD=$(date +%s)
  fi
done
wait "$TP"; rc=$?

# ---- 5. end (the final adapter first: a requeued preempt test link, 176044, was ended by Slurm seconds after its
# trainer exited, before its end steps; the checkpoint copy stays the fallback)
if [ "$rc" = 0 ] && [ -e "$RUN/final/adapter_config.json" ]; then
  until guard; [ -e "$RES/${STAGE}_final/adapter_config.json" ]; do
    [ "$(left_s)" -gt 300 ] || break
    log "!!! final adapter not copied yet, retrying"; sleep 60
  done
  [ -e "$RES/${STAGE}_final/adapter_config.json" ] || { fail "final adapter not copied; it is on $(hostname):$RUN/final"; keep_logs; exit 5; }
  echo "$RES/${STAGE}_final $(date '+%F %T')" > "$M/DONE"
  keep_logs
  grep "DX_SFT_ROWS_DONE\|trainer rc" "$T/logs/train.log" | tail -2
  [ -n "$NEXT" ] && scancel "$NEXT" && log "successor $NEXT cancelled"
  log "CON1_FULL_DONE: final adapter $RES/${STAGE}_final"
  exit 0
fi
guard 1
keep_logs
fail "trainer rc=$rc"
tail -30 "$T/logs/train.log"
exit "$rc"
