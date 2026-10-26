#!/bin/bash
# b005 copy of sample_job.sh (2026-10-24): identical except PORT0 (job-derived, not 8766).
# Own-action row sampling on one node: N vLLM servers (one per allocated GPU, ports 8766+i) and
# N sampler shards per domain (states split by line index mod N; each shard's output is
# append-only and dataxp.ownrows.sample resumes on it), then a merge into <rows>/<d>/samples.jsonl.
# Before each stage, reshard.py seeds every shard file with the complete records any other
# sample file of the domain already holds (smoke/, another job's OWN_OUTSUB, an earlier
# sharding), so a requeue, a resubmission with another GPU count, or a second job on the same
# domain resumes rather than re-drawing; the merge dedupes by (sid, sample) in any case.
# Stage 1 (OWN_SMOKE=1) draws --per-variant OWN_SMOKE_PER_VARIANT states per shard into
# <rows>/<d>/smoke/ and gates on the reply statistics before the full stage.
#
# Knobs are OWN_-prefixed because the cluster's login environment exports names like TEMP=/tmp:
# OWN_ROWS (rows dir, default $ROOT/rows) OWN_DOMAINS OWN_OUTSUB OWN_K OWN_MAXTOK OWN_TEMP OWN_CONC
# OWN_SMOKE OWN_SMOKE_PER_VARIANT OWN_SMOKE_ONLY OWN_FORCE VLLM_ENV VLLM_EXTRA; DATAXP_SPELLING is
# passed through to every dataxp process (unset = multitool).
set -u
shopt -s nullglob
ROOT=/project/community/wxu2/corpus
CODE=$ROOT/code
SCR=$ROOT/scripts
PY=/project/community/wxu2/envs/ownrows_o/bin/python
ROWS=${OWN_ROWS:-$ROOT/rows}
K=${OWN_K:-4}; MAXTOK=${OWN_MAXTOK:-2000}; TEMPERATURE=${OWN_TEMP:-0.6}; CONC=${OWN_CONC:-64}
SMOKE=${OWN_SMOKE:-1}; SMOKE_PER_VARIANT=${OWN_SMOKE_PER_VARIANT:-2}; SMOKE_ONLY=${OWN_SMOKE_ONLY:-0}; FORCE=${OWN_FORCE:-0}
DOMAINS=${OWN_DOMAINS:-"grid2op v2verb"}
OUTSUB=${OWN_OUTSUB:-}
PORT0=${OWN_PORT0:-$((20000 + (${SLURM_JOB_ID:-0} % 1000) * 10))}   # b005 copy: job-derived ports, so two sampling jobs on one node never share 8766
JOB=${SLURM_JOB_ID:-local}
LOG=$ROOT/logs/$JOB
mkdir -p "$LOG"
export PYTHONPATH=$CODE MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$CODE" || exit 1

echo "== job $JOB restart ${SLURM_RESTART_COUNT:-0} node $(hostname) start $(date -u +%FT%TZ) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ROWS=$ROWS K=$K MAXTOK=$MAXTOK TEMPERATURE=$TEMPERATURE CONC=$CONC DOMAINS='$DOMAINS' OUTSUB='$OUTSUB' SMOKE=$SMOKE/$SMOKE_PER_VARIANT DATAXP_SPELLING=${DATAXP_SPELLING:-<unset: multitool>}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
N=${#GPUS[@]}
echo "servers: $N on gpus ${GPUS[*]}"
for d in $DOMAINS; do [ -f "$ROWS/$d/states.jsonl" ] || { echo "missing $ROWS/$d/states.jsonl"; exit 1; }; done

# ---- servers, one per GPU
PIDS=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} bash "$SCR/serve_one.sh" $((PORT0+i)) > "$LOG/server$i.log" 2>&1 &
  PIDS+=($!)
done
cleanup() { echo "stopping servers $(date -u +%FT%TZ)"; kill "${PIDS[@]}" 2>/dev/null; sleep 8; kill -9 "${PIDS[@]}" 2>/dev/null; }
trap cleanup EXIT
t0=$(date +%s)
for i in $(seq 0 $((N-1))); do
  port=$((PORT0+i))
  until curl -sf -m 5 "http://127.0.0.1:$port/v1/models" > /dev/null; do
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then echo "server $i died; tail of its log:"; tail -40 "$LOG/server$i.log"; exit 2; fi
    if [ $(( $(date +%s) - t0 )) -gt 2400 ]; then echo "server $i not up after 40 min"; tail -20 "$LOG/server$i.log"; exit 2; fi
    sleep 15
  done
  echo "server $i up on port $port after $(( $(date +%s) - t0 )) s"
done

# ---- shards (deterministic)
for d in $DOMAINS; do "$PY" "$SCR/shard.py" "$ROWS/$d/states.jsonl" "$N" "$ROWS/$d" || exit 1; done

count_records() {  # $1 = subdir ('' = the row dir itself)
  local tot=0 d f
  for d in $DOMAINS; do for f in "$ROWS/$d/$1"/samples.shard*.jsonl; do tot=$((tot + $(wc -l < "$f"))); done; done
  echo $tot
}

run_stage() {  # $1 = stage name, $2 = extra sampler args, $3 = output subdir ('' = <rows>/<d>)
  local stage=$1 extra=$2 sub=$3 d i
  for d in $DOMAINS; do mkdir -p "$ROWS/$d/$sub"; "$PY" "$SCR/reshard.py" "$ROWS/$d" "$N" "$sub" || exit 1; done
  local before ts
  before=$(count_records "$sub"); ts=$(date +%s)
  echo "== stage $stage start $(date -u +%FT%TZ): $before records already on disk"
  local WP=()
  for i in $(seq 0 $((N-1))); do
    ( for d in $DOMAINS; do
        echo "--- $d shard $i $(date -u +%FT%TZ)"
        "$PY" -m dataxp.ownrows.sample --states "$ROWS/$d/states.shard$i.jsonl" \
            --out "$ROWS/$d/$sub/samples.shard$i.jsonl" \
            --k "$K" --max-tokens "$MAXTOK" --temperature "$TEMPERATURE" --concurrency "$CONC" --prompt probe \
            --endpoint "http://127.0.0.1:$((PORT0+i))/v1" --model Qwen/Qwen3.8-27B $extra
        echo "--- $d shard $i rc $? $(date -u +%FT%TZ)"
      done ) > "$LOG/${stage}_worker$i.log" 2>&1 &
    WP+=($!)
  done
  local rc=0 p
  for p in "${WP[@]}"; do wait "$p" || rc=1; done
  local after dt
  after=$(count_records "$sub"); dt=$(( $(date +%s) - ts ))
  echo "== stage $stage end $(date -u +%FT%TZ): $((after - before)) new records in $dt s = $("$PY" -c "print(round(($after-$before)*60/max($dt,1),1), 'replies/min over $N servers,', round(($after-$before)*60/max($dt,1)/$N,1), 'per server')"), worker rc $rc"
  return $rc
}

if [ "$SMOKE" = "1" ]; then
  run_stage smoke "--per-variant $SMOKE_PER_VARIANT" smoke
  for d in $DOMAINS; do
    echo "-- smoke check $d"
    "$PY" "$SCR/check_samples.py" --gate "$ROWS/$d/smoke"/samples.shard*.jsonl
    gr=$?
    if [ $gr -ne 0 ] && [ "$FORCE" != "1" ]; then echo "smoke gate failed for $d (rc $gr); stopping (OWN_FORCE=1 overrides)"; exit 3; fi
  done
  if [ "$SMOKE_ONLY" = "1" ]; then echo "SMOKE_ONLY: done"; exit 0; fi
fi

run_stage full "" "$OUTSUB"
for d in $DOMAINS; do
  echo "-- merge $d (every sample file under $ROWS/$d)"
  "$PY" "$SCR/merge_shards.py" --out "$ROWS/$d/samples.jsonl" "$ROWS/$d"/samples.shard*.jsonl "$ROWS/$d"/*/samples.shard*.jsonl
  "$PY" "$SCR/check_samples.py" "$ROWS/$d/samples.jsonl"
done
echo "== done $(date -u +%FT%TZ)"
