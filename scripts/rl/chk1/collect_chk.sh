#!/bin/bash
# collect_chk.sh: chk1 data collection on drc GPU 4. dx38mtR on the training pool (photonics, grid2op, engibench of
# train_p2_datav4, no benchmark problem), harness v3rc --checks 4 --check-actions (each new move dx38 proposes is
# checked in the simulator before it commits; naming a checked move again commits it), open rules, T=0, 20 turns,
# thinking on, every prompt and whole reply recorded (--record-prompts). POOL (default chk1/pool_fwd.jsonl), OUT.
# Resumable by episode key.
set -uo pipefail
W=/storage/wxu/rl_open
OUT=${OUT:-$W/chk1/collect_fwd}; mkdir -p "$OUT/logs"
POOL=${POOL:-$W/chk1/pool_fwd.jsonl}
LABEL=dx38mtR
ADAPTER=/storage/wxu/rl_fc/inputs/dx38mtR_final
VENV=/storage/wenwenw/envs/vllm_drc
SIM_PY=/home/wxu/corpus_env/bin/python
SNAP=/storage/wxu/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
H=$W/rl/harness
CODE=/storage/wxu/rl_fc/harness/dataxp_code_v3
PORT=${PORT:-8795}
B=/storage/wxu/cache/vllm_serve
mkdir -p $B/tmp $B/xdg $B/vllm $B/triton $B/inductor $B/nv $B/flashinfer
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HOME=/storage/wxu/huggingface HF_HUB_CACHE=/storage/wxu/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TMPDIR=$B/tmp XDG_CACHE_HOME=$B/xdg VLLM_CACHE_ROOT=$B/vllm
export TRITON_CACHE_DIR=$B/triton TORCHINDUCTOR_CACHE_DIR=$B/inductor
export CUDA_CACHE_PATH=$B/nv FLASHINFER_WORKSPACE_BASE=$B/flashinfer MPLBACKEND=Agg
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0 DATAXP_SPELLING=multitool DX_ORIG_SCRIPTS=$H/scripts
for f in "$VENV/bin/vllm" "$SIM_PY" "$SNAP/config.json" "$ADAPTER/adapter_model.safetensors" "$H/scripts/dataxp_policy_eval_v3rc.py" \
         "$H/configs/task_instruction_V2.txt" "$CODE/dataxp/domain.py" "$POOL" "$W/rl/repo/llm_finetune/rl_dataxp/rollout.py"; do
  [ -e "$f" ] || { echo "ABORT: missing $f"; exit 2; }
done
echo "=== $(date '+%F %T %Z') host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} pool=$POOL out=$OUT harness sha1 $(sha1sum $H/scripts/dataxp_policy_eval_v3rc.py | cut -c1-12)"
GOT=$($VENV/bin/python -c "import torch;print(torch.cuda.get_device_name(0))" 2>/dev/null | tail -1)
echo "device 0: $GOT"
case "$GOT" in *Blackwell*) ;; *) echo "ABORT: not the Blackwell"; exit 4 ;; esac

PID=
cleanup() {
  [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null && { kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID"; }
  sleep 10
  [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null && kill -KILL -- "-$PID" 2>/dev/null
  PID=
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT

setsid "$VENV/bin/vllm" serve "$SNAP" --served-model-name base --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-262144}" --gpu-memory-utilization "${GPU_UTIL:-0.95}" \
  --max-num-seqs "${MAX_SEQS:-64}" --limit-mm-per-prompt '{"image":0,"video":0}' --enable-lora --max-lora-rank 64 \
  --lora-modules "$LABEL=$ADAPTER" > "$OUT/logs/vllm_server.log" 2>&1 &
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

cd "$W/rl/repo"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$W/rl/repo" "$SIM_PY" -m llm_finetune.rl_dataxp.rollout eval \
  --script "$H/scripts/dataxp_policy_eval_v3rc.py" --problems "$POOL" --variants photonics grid2op engibench \
  --servers "$URL" --model "$LABEL" --label "$LABEL" --runs 1 --turns 20 --workers "${WORKERS:-24}" \
  --out "$OUT/$LABEL.episodes.jsonl" -- \
  --code "$CODE" --native-prompt served --instruction-file "$H/configs/task_instruction_V2.txt" --open-rules --record-prompts \
  --checks 4 --check-actions \
  2>&1 | grep -v -i "warn\|numba" | tee -a "$OUT/logs/collect.log"
rc=${PIPESTATUS[0]}
echo "COLLECT_CHK_DONE rc=$rc $(date '+%F %T %Z')"
exit $rc
