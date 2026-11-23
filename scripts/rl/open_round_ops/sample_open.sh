#!/bin/bash
# sample_open.sh: drc GPU 4's share of the open-rules round (the steps of scripts/rl/orchard_screen_data.sh with
# OPEN_EPISODES and SCORE_ALONGSIDE): states of EPISODES (open_states.py, eval problems excluded) unless OUT/states.jsonl
# exists (STATES_IN: a states file to use instead), 8 replies per state at T=0.8 (KV budget), scoring alongside, last
# scoring pass, gain rows (--min-pct 0 --max-words 1000). Resumable: replies by (sid, sample), scoring by state.
set -uo pipefail
W=/storage/wxu/rl_open
OUT=${OUT:-$W/round}; mkdir -p "$OUT/logs"
EPISODES=${EPISODES:-$W/collect/dx38mtR.episodes.jsonl}
VENV=/storage/wenwenw/envs/vllm_drc
SIM_PY=/home/wxu/corpus_env/bin/python
SNAP=/storage/wxu/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
ADAPTER=/storage/wxu/rl_fc/inputs/dx38mtR_final
CODE=/storage/wxu/rl_fc/harness/dataxp_code_v3
S=$W/screen
EXCLUDE="$W/exclude/problems_v3_native.jsonl $W/exclude/problems_v2.jsonl"
PORT=${PORT:-8793}
B=/storage/wxu/cache/vllm_serve
mkdir -p $B/tmp $B/xdg $B/vllm $B/triton $B/inductor $B/nv $B/flashinfer
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HOME=/storage/wxu/huggingface HF_HUB_CACHE=/storage/wxu/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TMPDIR=$B/tmp XDG_CACHE_HOME=$B/xdg VLLM_CACHE_ROOT=$B/vllm
export TRITON_CACHE_DIR=$B/triton TORCHINDUCTOR_CACHE_DIR=$B/inductor
export CUDA_CACHE_PATH=$B/nv FLASHINFER_WORKSPACE_BASE=$B/flashinfer MPLBACKEND=Agg
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0 DATAXP_SPELLING=multitool
for f in "$VENV/bin/vllm" "$SIM_PY" "$SNAP/config.json" "$ADAPTER/adapter_model.safetensors" "$CODE/dataxp/domain.py" $EXCLUDE \
         "$S/open_states.py" "$S/sample_states.py" "$S/score_states.py" "$S/build_rows.py" "$S/open_rules.py" \
         "$S/complete_states.py" "$W/screen_repo/scripts/rl_screen/score_loop.sh"; do
  [ -e "$f" ] || { echo "ABORT: missing $f"; exit 2; }
done
sim() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$SIM_PY" "$@"; }
echo "=== $(date '+%F %T %Z') host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} out=$OUT"
GOT=$($VENV/bin/python -c "import torch;print(torch.cuda.get_device_name(0))" 2>/dev/null | tail -1)
echo "device 0: $GOT"
case "$GOT" in *Blackwell*) ;; *) echo "ABORT: not the Blackwell"; exit 4 ;; esac

if [ ! -s "$OUT/states.jsonl" ]; then
  if [ -n "${STATES_IN:-}" ]; then cp "$STATES_IN" "$OUT/states.jsonl"
  else
    sim "$S/open_states.py" --episodes $EPISODES --out "$OUT/states.jsonl" --exclude-specs $EXCLUDE 2>&1 | tee "$OUT/logs/open_states.log"
    grep -q OPEN_STATES_DONE "$OUT/logs/open_states.log" || { echo "ABORT: open_states failed"; exit 3; }
  fi
fi
echo "=== states: $(wc -l < "$OUT/states.jsonl")"

PID=; SPID=
cleanup() {
  [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null && { kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID"; }
  sleep 10
  [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null && kill -KILL -- "-$PID" 2>/dev/null
  PID=
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT

if [ ! -e "$OUT/samples.done" ]; then
  setsid "$VENV/bin/vllm" serve "$SNAP" --served-model-name base --host 127.0.0.1 --port "$PORT" \
    --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-262144}" --gpu-memory-utilization "${GPU_UTIL:-0.95}" \
    --max-num-seqs "${MAX_SEQS:-64}" --limit-mm-per-prompt '{"image":0,"video":0}' --enable-lora --max-lora-rank 64 \
    --lora-modules "dx38mtR=$ADAPTER" > "$OUT/logs/vllm_server.log" 2>&1 &
  PID=$!
  URL=http://127.0.0.1:$PORT/v1
  T0=$(date +%s)
  while :; do
    kill -0 "$PID" 2>/dev/null || { echo "ABORT: vLLM server exited"; tail -40 "$OUT/logs/vllm_server.log"; exit 7; }
    curl -sf "$URL/models" >/dev/null 2>&1 && break
    [ $(( $(date +%s) - T0 )) -lt 3600 ] || { echo "ABORT: server not ready"; exit 7; }
    sleep 10
  done
  echo "=== vLLM server ready after $(( $(date +%s) - T0 )) s"
  rm -f "$OUT/logs/score.stop"
  SIM_PY="$SIM_PY" REPO="$W/screen_repo" H="$W/harness_link" bash "$W/screen_repo/scripts/rl_screen/score_loop.sh" \
      "$OUT" 8 900 "${SCORE_WORKERS:-12}" < /dev/null &
  SPID=$!
  for pass in 1 2; do
    "$SIM_PY" "$S/sample_states.py" --states "$OUT/states.jsonl" --out "$OUT/samples.jsonl" --servers "$URL" --model dx38mtR \
        --k 8 --temperature 0.8 --workers "${SAMPLE_WORKERS:-48}" --kv-budget "${KV_BUDGET:-450000}" --prompt-weight 1.8 \
        --reply-tokens 700 --first engibench grid2op photonics 2>&1 | tee -a "$OUT/logs/samples.log"
    rc=${PIPESTATUS[0]}
    [ "$rc" = 3 ] || break
  done
  [ "$rc" = 0 ] || [ "$rc" = 3 ] || { echo "ABORT: sampling failed rc=$rc"; exit 5; }
  date '+%F %T %Z' > "$OUT/samples.done"
  cleanup
  touch "$OUT/logs/score.stop"; echo "=== waiting for the scoring loop"; wait "$SPID"
fi
sim "$S/score_states.py" --code "$CODE" --states "$OUT/states.jsonl" --samples "$OUT/samples.jsonl" --out "$OUT/scored.jsonl" \
    --workers "${SCORE_WORKERS_LAST:-20}" 2>&1 | grep -v -i "warn\|numba\|^[[:space:]]*$" | tee "$OUT/logs/scored.log"
grep -q SCORE_STATES_DONE "$OUT/logs/scored.log" || { echo "ABORT: scoring failed"; exit 6; }
"$SIM_PY" "$S/build_rows.py" --states "$OUT/states.jsonl" --samples "$OUT/samples.jsonl" --scored "$OUT/scored.jsonl" \
    --out "$OUT/rows_gain.jsonl" --min-pct 0 --max-words 1000 --tier open1_gain > "$OUT/rows_gain.report.txt" || { echo "ABORT: rows failed"; exit 8; }
cat "$OUT/rows_gain.report.txt"
echo "DRC_OPEN_ROUND_DONE $(date '+%F %T %Z')"
