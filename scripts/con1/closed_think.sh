#!/bin/bash
# closed_think.sh : on drc GPU 4, one adapter's zero-shot answers with the think block closed in the prompt
# (con1-2m-rows, 11-27). The prompt is the usual one (con1's skill_client layout, ending '<think>\n') plus the trained
# think-closing piece '\n</think>\n\n' (think_client.py --budget 0: no think tokens, the answer generated right after
# the closing piece). For checkpoints trained with the think-closing piece at weight 0 (the full-set continuation since
# 11-26), which no longer close the think block themselves; the content of the answer is what is measured.
#   RUNS="label:queries_file ..." ADAPTER=dir ADAPTER_NAME=name OUT=dir bash closed_think.sh
#   e.g. RUNS="c2m_full_00850_closed_v2:/storage/wxu/con1/arm/main/queries_v2.jsonl c2m_full_00850_closed_v3:..._v3.jsonl"
# Every question of the file (both groups), one line per answer in the preds schema (predictor = label). GPU 4 under
# arm/gpu4.lock as con1's drc_arm.sh; resumable.
set -uo pipefail
C=/storage/wxu/con1; A=$C/arm
PY=/storage/wenwenw/envs/vllm_drc/bin/python
MODEL=/storage/wxu/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
ADAPTER=${ADAPTER:?adapter dir}; ADAPTER_NAME=${ADAPTER_NAME:?served name}; OUT=${OUT:?output dir}; RUNS=${RUNS:?label:queries ...}
CLIENT=${CLIENT:-$C/full_eval/think_diag/think_client.py}
export PYTHONPATH=$C/pylib:$C/repo HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-4} OMP_NUM_THREADS=4 TMPDIR=$C/tmp
export HF_HOME=/storage/wxu/huggingface HF_HUB_CACHE=/storage/wxu/huggingface/hub HF_HUB_DISABLE_XET=1
export VLLM_CACHE_ROOT=$C/cache/vllm CUDA_CACHE_PATH=$C/cache/nv VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$OUT/logs"
[ -e "$ADAPTER/adapter_config.json" ] || { echo "ABORT: no adapter in $ADAPTER"; exit 2; }
echo "=== $(date '+%F %T') closed_think $ADAPTER_NAME ($ADAPTER): $RUNS"
exec 9>"$A/gpu4.lock"
flock 9; echo "=== $(date '+%F %T') holding $A/gpu4.lock"
read -r TOT USED <<< "$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | tr -d ',')"
while [ "${USED:-99999}" -ge 5000 ]; do sleep 120; read -r TOT USED <<< "$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | tr -d ',')"; done
PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
setsid $PY /storage/wenwenw/envs/vllm_drc/bin/vllm serve $MODEL --served-model-name base --host 127.0.0.1 --port $PORT \
  --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.92 --max-num-seqs 48 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' --enable-lora --max-lora-rank 64 --max-loras 1 \
  --lora-modules "$ADAPTER_NAME=$ADAPTER" > "$OUT/logs/vllm_$ADAPTER_NAME.log" 2>&1 &
SP=$!
trap 'kill -TERM -- -$SP 2>/dev/null; sleep 10; kill -KILL -- -$SP 2>/dev/null' EXIT
T0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  kill -0 $SP 2>/dev/null || { echo "ABORT: vLLM exited"; tail -30 "$OUT/logs/vllm_$ADAPTER_NAME.log"; exit 7; }
  [ $(( $(date +%s) - T0 )) -lt 3600 ] || { echo "ABORT: vLLM not ready"; exit 7; }
  sleep 10
done
echo "=== $(date '+%F %T') server ready"
for run in $RUNS; do
  IFS=: read -r L QF <<< "$run"
  $PY $CLIENT --queries $QF --tokenizer $MODEL --base-url "http://127.0.0.1:$PORT/v1" --model $ADAPTER_NAME --label $L \
    --budget 0 --answer-tokens 4096 --workers 48 --out $OUT/preds_$L.jsonl > "$OUT/logs/client_$L.log" 2>&1
  echo "=== $(date '+%F %T') $L rc=$?: $(tail -1 "$OUT/logs/client_$L.log")"
done
kill -TERM -- -$SP 2>/dev/null; sleep 15
flock -u 9; echo "=== $(date '+%F %T') GPU work done, lock released"
echo "CLOSED_THINK_DONE $(date '+%F %T')"
