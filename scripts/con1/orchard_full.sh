#!/bin/bash
# con1 stage-1 continuation on the full consequence row set (docs/con1_full_rows.md): one 24 h `advanced` link of a
# chain on one Orchard node with 4 H100s. Node layout as orchard_main.sh's (/tmp/con1_main_<job>/<stage>/
# metrics.jsonl, ckpt/step_N/COMPLETE, final/; stage "full"), so parent_check/mon.py reads it unchanged.
# Copies off the node use the parent's in-place slot method (parent_check/guard_v3.sh; scripts/con1/guard_v3.sh is
# that script with env RES / STAGES / FORCE / NO_FINAL): no space is ever released on /project/community; each
# rolling copy is a tar of <ckpt>/adapter + state.json written into con1_reserve/slots/A.bin or B.bin (alternating;
# A.meta / B.meta = "stage step length sha1 time", written after a sha1 read-back). Optimizer state does not fit:
# every resume (24 h limit or crash) restarts AdamW with the lr re-warmed over REWARM = 10 steps and the schedule
# position from state.json.
#  1. submits its successor (afterany), unless the run is DONE, STOPped or the chain is at CHAIN_MAX;
#  2. builds the rows on node /tmp (build_full_rows.py: facetm, later questions short, Q 36, cap 22,000 tokens; the
#     sha1 must equal the first build's in $F/stats_facetm_short_q36.json, else it stops);
#  3. the start adapter, always at the fixed node path /tmp/con1_init (the trainer's fingerprint holds the path):
#     - a link that resumes: the newest "full" slot (A or B), restored and sha1-checked into /tmp/con1_resume; its
#       adapter is also /tmp/con1_init (only its LoRA config matters there: the resume overwrites every weight);
#     - the first link: INIT=F (default) restores the stage-1 final from slots F1-F3 per F.meta (must read
#       "stage1 final ..."), sha1-checked; INIT=<dir> copies an adapter directory; INIT=none (tests) = a fresh LoRA;
#  4. trains (scripts/rl/orchard_sft_rows.sh, configs/rl/con1_sft.yaml); full checkpoints every 5 steps on node /tmp;
#  5. runs the guard every 10 min from inside the job (srun --overlap): the newest checkpoint into the older slot
#     every ~2 h (NO_FINAL: the continuation never writes a final; REUSE_SLOT: when stage 2's final holds one of A/B,
#     the other slot is rewritten each time, so the only copy is invalid during a ~1 min write);
#  6. at SIGUSR1 (30 min before the limit): waits for the next checkpoint, stops the trainer, FORCE-copies it into a
#     slot, exits 0. At the end of the epoch the last checkpoint (= the final adapter) is FORCE-copied the same way;
#     $M/DONE names the slot that holds it; successor cancelled.
#
#   sbatch -p advanced --qos=adv_4gpu_qos --export=ALL,M=/project/community/wxu2/con1/full/run1 \
#     /project/community/wxu2/con1/repo2m/slurm/con1_full.sbatch
# Stop the chain: touch $M/STOP (the running link finishes its day; no successor trains).
# Monitor: srun --jobid=<id> --overlap -N1 -n1 python3 /project/community/wxu2/parent_check/mon.py /tmp/con1_main_<id>/full
set -uo pipefail
B=/project/community/wxu2
C1=$B/con1
REPO="${REPO:-$C1/repo2m}"
INIT="${INIT:-F}"
M="${M:?run dir on shared storage (e.g. $C1/full/run1)}"
Q="${Q:-36}"
PER="${PER:-28}"
MAX_TOKENS="${MAX_TOKENS:-22000}"
TARGET="${TARGET:-facetm}"                       # answer format (targets.py; facetm chosen 11-26 from the drc arm)
FOLLOW="${FOLLOW:-short}"                        # later questions of a row: "Same answer form." (full text in the 1st)
TAG=$TARGET; [ "$FOLLOW" = short ] && TAG=${TARGET}_short
PREFIX_WEIGHT="${PREFIX_WEIGHT-0}"              # loss weight of each answer's think-closing piece (owner 11-26: 0)
PWARG=; [ -n "$PREFIX_WEIGHT" ] && PWARG="--prefix-weight $PREFIX_WEIGHT" && TAG=${TAG}_pw$PREFIX_WEIGHT
TAG_SUFFIX="${TAG_SUFFIX:-}"                     # _nc: rows without the move request (a repo with con1 eff023a)
TAG=${TAG}${TAG_SUFFIX}
F="${F:-$C1/full}"
DEV="${DEV:-$C1/main/rows/dev_$TARGET.jsonl}"    # stage 1's dev states and moves, in the training format
STAGE="${STAGE:-full}"
RES="${RES:-$B/con1_reserve}"
SL=$RES/slots
GUARD="$REPO/scripts/con1/guard_v3.sh"
GUARD_EVERY="${GUARD_EVERY:-600}"
CHAIN_MAX="${CHAIN_MAX:-10}"
CHAIN_I="${CHAIN_I:-1}"
JOB="${SLURM_JOB_ID:-x}"
NTRAIN="${NTRAIN:-4}"
NEXT_ARGS="${NEXT_ARGS:--p advanced --qos=adv_4gpu_qos}"   # the successor's partition (a test chain: general)
ROWS_FILE="${ROWS_FILE:-}"                                    # test only: train on this file, no build
T=/tmp/con1_main_$JOB
RUN=$T/$STAGE
INIT_DIR=/tmp/con1_init
RESUME_DIR=/tmp/con1_resume
TRAIN_PY=$B/rl/envs/train/bin/python
MODEL=$B/rl/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
G=$C1/gen
D=$C1/data
mkdir -p "$M/logs" "$M/chain" "$T/logs" "$RUN"
log() { echo "=== $(date '+%F %T') $*"; }
log "host=$(hostname) job=$JOB chain=$CHAIN_I/$CHAIN_MAX repo=$REPO target=$TARGET follow=$FOLLOW prefix_weight=${PREFIX_WEIGHT:-default} q=$Q cap=$MAX_TOKENS rows/step=$PER init=$INIT m=$M slots=$SL stage=$STAGE"
[ -e "$M/DONE" ] && { log "run DONE already"; exit 0; }
[ -e "$M/STOP" ] && { log "STOP file: not training"; exit 0; }
for f in "$DEV" "${ROWS_FILE:-$F/stats_${TAG}_q$Q.json}" "$TRAIN_PY" "$GUARD"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 2; }
done
best_slot() {   # "X step length sha1" of the slot holding this stage's newest copy, or nothing
  local x line best= bs=-1 s
  for x in A B; do
    line=$(cat "$SL/$x.meta" 2>/dev/null) || continue
    set -- $line
    [ "${1:-}" = "$STAGE" ] && [ -n "${4:-}" ] || continue
    s=$((10#${2#step_}))
    [ "$s" -gt "$bs" ] && { bs=$s; best="$x $2 $3 $4"; }
  done
  echo "$best"
}
copy_step() { local b; b=$(best_slot); [ -n "$b" ] && set -- $b && echo $((10#${2#step_})) || echo 0; }
restore() {   # restore FILES-LIST LENGTH SHA1 DIR: the tar in the files (in order) -> DIR, after a sha1 check
  # (cat's status is dropped: it gets EPIPE once head -c has its bytes, which pipefail would count as a failure)
  local files=$1 len=$2 sha=$3 dir=$4 got
  got=$({ cat $files 2>/dev/null || true; } | head -c "$len" | sha1sum | cut -d' ' -f1)
  [ "$got" = "$sha" ] || { log "!!! sha1 of $files ($len bytes) is $got, meta says $sha"; return 1; }
  rm -rf "$dir" && mkdir -p "$dir" && { cat $files 2>/dev/null || true; } | head -c "$len" | tar -x -C "$dir"
}
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
    --export=ALL,INIT="$INIT",M="$M",Q="$Q",PER="$PER",REPO="$REPO",CHAIN_I=$((CHAIN_I+1)),CHAIN_MAX="$CHAIN_MAX",DEV="$DEV",MAX_TOKENS="$MAX_TOKENS",TARGET="$TARGET",FOLLOW="$FOLLOW",PREFIX_WEIGHT="$PREFIX_WEIGHT",RES="$RES",STAGE="$STAGE",TAG_SUFFIX="$TAG_SUFFIX" \
    "${SBATCH_FILE:-$REPO/slurm/con1_full.sbatch}") && echo "$NEXT" > "$M/chain/next_of_$JOB" && log "successor $NEXT submitted" \
    || log "!!! successor not submitted"
fi

# ---- 2. start adapter (before the rows: a missing start point stops the link early)
RD=
BS=$(best_slot)
if [ -n "$BS" ]; then
  set -- $BS
  log "resuming from slot $1 ($STAGE $2, $3 bytes)"
  restore "$SL/$1.bin" "$3" "$4" "$RESUME_DIR" || { fail "slot $1 does not restore"; exit 6; }
  [ -e "$RESUME_DIR/state.json" ] && [ -e "$RESUME_DIR/adapter/adapter_model.safetensors" ] || { fail "slot $1 lacks adapter/state.json"; exit 6; }
  RD=$RESUME_DIR
  if [ "$INIT" != none ]; then rm -rf "$INIT_DIR" && cp -al "$RESUME_DIR/adapter" "$INIT_DIR"; fi
elif ls "$M/chain" | grep -q "^copied_step_"; then
  fail "this run copied checkpoints before ($(ls "$M/chain" | grep "^copied_step_" | sort -t_ -k3 -n | tail -1)) but no $STAGE slot is left in $SL: not restarting from step 0"
  exit 6
elif [ "$INIT" = F ]; then
  line=$(cat "$SL/F.meta" 2>/dev/null); set -- $line
  [ "${1:-} ${2:-}" = "stage1 final" ] || { fail "F.meta does not hold the stage-1 final: '${line}'"; exit 2; }
  log "restoring the stage-1 final from F1-F3 ($3 bytes, written $5)"
  restore "$SL/F1.bin $SL/F2.bin $SL/F3.bin" "$3" "$4" "$INIT_DIR" || { fail "F slots do not restore"; exit 2; }
  [ -e "$INIT_DIR/adapter_config.json" ] || { fail "the F tar has no adapter_config.json"; exit 2; }
elif [ "$INIT" != none ]; then
  [ -e "$INIT/adapter_config.json" ] || { log "ABORT: missing $INIT/adapter_config.json"; exit 2; }
  rm -rf "$INIT_DIR" && cp -r "$INIT" "$INIT_DIR"
fi

# ---- 3. rows on node /tmp
ROWS=$T/train_q$Q.jsonl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
if [ -n "$ROWS_FILE" ]; then
  ROWS=$ROWS_FILE; log "TEST: rows $ROWS (no build)"
else
  WANT=$("$TRAIN_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_sha1"])' "$F/stats_${TAG}_q$Q.json")
  log "building the rows ($TARGET, later questions $FOLLOW, Q=$Q, cap $MAX_TOKENS tokens)"
  T0=$(date +%s)
  "$TRAIN_PY" "$REPO/scripts/con1/build_full_rows.py" \
    --states "$C1/main/states_native_offmenu.jsonl.gz" "$F/states_native_offmenu_rest.jsonl.gz" \
             "$G/out/states_native_orch.jsonl.gz" "$G/out/states_scripted.jsonl.gz" \
    --exclude-rows "$C1/main/rows/train_facet.jsonl" "$C1/main/rows/dev_facet.jsonl" "$C1/main/rows/dev_facetm.jsonl" \
    --forbid-problems "$D/queries_v2.jsonl" "$G/bench_problems.jsonl" --tokenizer "$MODEL" --target "$TARGET" --follow "$FOLLOW" $PWARG \
    --q "$Q" --max-tokens "$MAX_TOKENS" --seed 0 --workers "${BUILD_WORKERS:-30}" \
    --out-train "$ROWS" --out-dev "$T/dev_full_q$Q.jsonl" --stats "$T/stats.json" > "$T/logs/build.log" 2>&1
  tail -3 "$T/logs/build.log"
  GOT=$("$TRAIN_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_sha1"])' "$T/stats.json" 2>/dev/null)
  log "rows built in $(( $(date +%s) - T0 )) s; sha1 $GOT (expected $WANT)"
  cp "$T/logs/build.log" "$M/logs/build_$JOB.log" 2>/dev/null
  if [ "$GOT" != "$WANT" ]; then fail "rows differ from the first build"; exit 3; fi
fi

# ---- 4. trainer
if [ -n "$RD" ]; then
  S=$(copy_step)
  # the run's metrics before that step (every earlier link's, kept in $M), for the monitor's history
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
DEV_USED=$DEV
if [ "$FOLLOW" = short ] || [ -n "$PREFIX_WEIGHT" ]; then
  # the dev rows in the training layout: later questions and think-closing weight as build_full_rows.py writes them
  # (states, moves and answer texts unchanged); on node /tmp only
  DEV_USED=$T/dev_train_layout.jsonl
  PYTHONPATH="$REPO/scripts/con1:$REPO" FOLLOW="$FOLLOW" PREFIX_WEIGHT="$PREFIX_WEIGHT" "$TRAIN_PY" - "$DEV" "$DEV_USED" <<'EOF' || { fail "dev rows not converted"; exit 2; }
import json, os, sys
from targets import FOLLOW_HEAD, question
short, pw = os.environ["FOLLOW"] == "short", os.environ["PREFIX_WEIGHT"]
n = na = 0
with open(sys.argv[2], "w", encoding="utf-8") as fo:
    for line in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(line)
        if r.get("supervise") == "every_assistant":
            msgs, fmt = r["messages"], r["format"]
            for k, mv in zip(range(3, len(msgs), 2), r["con1"]["moves"][1:]) if short else ():
                assert msgs[k]["role"] == "user" and msgs[k]["content"] == question(fmt, mv, False)
                msgs[k]["content"] = FOLLOW_HEAD.format(move=mv) + "Same answer form."
                n += 1
            for m in msgs[2::2] if pw != "" else ():
                assert m["role"] == "assistant" and m.get("supervise_spans") is not None
                m["prefix_weight"] = float(pw)
                na += 1
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print("dev rows: %d later questions shortened, %d answers given prefix weight %s" % (n, na, pw or "default"))
EOF
fi
if [ "$INIT" = none ]; then IOV="policy.init_adapter=null policy.require_init_adapter=false"; else IOV="policy.init_adapter=$INIT_DIR"; fi
OVR="$IOV sft.rows_per_step=$PER sft.save_every=${SAVE_EVERY:-5} sft.keep_last=1 sft.dev_every=${DEV_EVERY:-50}"
OVR="$OVR sft.keep_adapter_every=100000 sft.lazy_rows=true sft.check_rows=false sft.resume_dir=${RD:-null}"
OVR="$OVR sft.rewarm_steps=${REWARM:-10} ${TEST_OVR:-}"
log "trainer: $OVR"
setsid env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" NTRAIN="$NTRAIN" REPO="$REPO" RUN_DIR="$RUN" \
  CONFIG=configs/rl/con1_sft.yaml ROWS="$ROWS" DEV="$DEV_USED" OVERRIDES="$OVR" \
  bash "$REPO/scripts/rl/orchard_sft_rows.sh" > "$T/logs/train.log" 2>&1 &
TP=$!

# ---- 5. guard + logs to shared storage
newest_ckpt() {   # newest complete checkpoint step on the node, 0 if none
  local best=0 d s
  for d in "$RUN"/ckpt/step_*; do
    [ -e "$d/COMPLETE" ] || continue
    s=$((10#${d##*step_})); [ "$s" -gt "$best" ] && best=$s
  done
  echo $best
}
guard() {   # one guard cycle for this job's stage (FORCE=1: copy the newest checkpoint whatever the last copy's age)
  RES="$RES" STAGES="$STAGE" FORCE="${1:-}" NO_FINAL=1 REUSE_SLOT=1 bash "$GUARD" "$JOB" 2>&1 | sed "s/^/guard: /"
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
copy_now() {   # FORCE-copy checkpoint >= $1 into a slot, retrying until 5 min before the time limit
  until guard 1; [ "$(copy_step)" -ge "$1" ]; do
    [ "$(left_s)" -gt 300 ] || { log "!!! checkpoint $1 not copied before the time limit (slot copy: step $(copy_step))"; return 1; }
    sleep 60
  done
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
    stop_trainer; sleep 10
    S1=$(newest_ckpt)
    copy_now "$S1"
    keep_logs
    log "stopped for the time limit; checkpoint $S1, slot copy: step $(copy_step) ($(best_slot | cut -d' ' -f1)); the successor ${NEXT:-(none)} resumes"
    exit 0
  fi
  if [ $(( $(date +%s) - LAST_GUARD )) -ge "$GUARD_EVERY" ]; then
    guard; keep_logs; LAST_GUARD=$(date +%s)
  fi
done
wait "$TP"; rc=$?

# ---- 6. end: the last checkpoint (step = all steps) holds the final adapter; it goes into a slot first
if [ "$rc" = 0 ] && [ -e "$RUN/final/adapter_config.json" ]; then
  SN=$(newest_ckpt)
  if copy_now "$SN"; then
    echo "final adapter = slot $(best_slot) (tar of adapter/ + state.json in $SL/$(best_slot | cut -d' ' -f1).bin) $(date '+%F %T')" > "$M/DONE"
  else
    fail "final checkpoint $SN not copied; it is on $(hostname):$RUN/final"; keep_logs; exit 5
  fi
  keep_logs
  grep "DX_SFT_ROWS_DONE\|trainer rc" "$T/logs/train.log" | tail -2
  [ -n "$NEXT" ] && scancel "$NEXT" && log "successor $NEXT cancelled"
  log "CON1_FULL_DONE: $(cat "$M/DONE")"
  exit 0
fi
guard 1
keep_logs
fail "trainer rc=$rc"
tail -30 "$T/logs/train.log"
exit "$rc"
