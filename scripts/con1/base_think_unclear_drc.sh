#!/bin/bash
# base_think_unclear_drc.sh : on drc GPU 4, arm D-unclear of the 11-28 base-model read (parent's order): arm D exactly
# (base Qwen3.8-27B, no adapter, thinking, temperature 0, think budget 8,000, answer cap 1,024, max-model-len 32,768,
# metrics named, reasoning effort at the template's default, --no-defined-terms, subset_300.tsv) with ONE change: the
# answer form 'dirunc' (targets.py; 'dir' with "..., or 'unclear' if its direction cannot be determined from the
# information shown." in the CHANGES sentence). think_client.py from full_eval/lib_v9. form_render_check.py first (arm
# D's prompt rendered with lib_v6 has arm D's recorded prompt_tokens; the new prompt is arm D's with that clause inserted
# once and nothing else changed), abort on any failure. GPU 4 under arm/gpu4.lock; resumable per question.
#   -> full_eval/base_think2/preds_base_v3_dirunc_8k_sub300.jsonl
set -uo pipefail
C=/storage/wxu/con1; A=$C/arm; O=$C/full_eval/base_think2; L9=$C/full_eval/lib_v9; L6=$C/full_eval/lib_v6
PY=/storage/wenwenw/envs/vllm_drc/bin/python
MODEL=/storage/wxu/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
Q=$A/main/queries_v3.jsonl
LABEL=base_v3_dirunc_8k_sub300
INSERT=", or 'unclear' if its direction cannot be determined from the information shown"
export CON1_DIR=$L9
export PYTHONPATH=$C/pylib:$C/repo HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-4} OMP_NUM_THREADS=4 TMPDIR=$C/tmp
export HF_HOME=/storage/wxu/huggingface HF_HUB_CACHE=/storage/wxu/huggingface/hub HF_HUB_DISABLE_XET=1
export VLLM_CACHE_ROOT=$C/cache/vllm CUDA_CACHE_PATH=$C/cache/nv VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$O/logs"
echo "=== $(date '+%F %T') $LABEL queries=$Q ($(sha1sum < $Q | cut -c1-12)) subset $(sha1sum < $O/subset_300.tsv | cut -c1-12) lib_v9"
$PY $L9/form_render_check.py --queries $Q --tokenizer $MODEL --subset $O/subset_300.tsv --form dirunc --ref-targets $L6 \
  --ref-form dir --ref-preds $O/preds_base_v3_dir_8k.jsonl --insert "$INSERT" > "$O/logs/form_render_check_$LABEL.txt" 2>&1
RC=$?; tail -4 "$O/logs/form_render_check_$LABEL.txt"
[ $RC -eq 0 ] || { echo "ABORT: form_render_check failed"; exit 6; }
exec 9>"$A/gpu4.lock"
flock 9; echo "=== $(date '+%F %T') holding $A/gpu4.lock"
read -r TOT USED <<< "$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | tr -d ',')"
while [ "${USED:-99999}" -ge 5000 ]; do sleep 120; read -r TOT USED <<< "$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | tr -d ',')"; done
PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
setsid $PY /storage/wenwenw/envs/vllm_drc/bin/vllm serve $MODEL --served-model-name base --host 127.0.0.1 --port $PORT \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92 --max-num-seqs 48 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' > "$O/logs/vllm_$LABEL.log" 2>&1 &
SP=$!
trap 'kill -TERM -- -$SP 2>/dev/null; sleep 10; kill -KILL -- -$SP 2>/dev/null' EXIT
T0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  kill -0 $SP 2>/dev/null || { echo "ABORT: vLLM exited"; tail -30 "$O/logs/vllm_$LABEL.log"; exit 7; }
  [ $(( $(date +%s) - T0 )) -lt 3600 ] || { echo "ABORT: vLLM not ready"; exit 7; }
  sleep 10
done
echo "=== $(date '+%F %T') server ready"
$PY $L9/think_client.py --queries $Q --tokenizer $MODEL --base-url "http://127.0.0.1:$PORT/v1" --model base \
  --label $LABEL --no-defined-terms --target dirunc --named-metrics --budget 8000 --answer-tokens 1024 --workers 48 \
  --subset $O/subset_300.tsv --out $O/preds_$LABEL.jsonl > "$O/logs/client_$LABEL.log" 2>&1
echo "=== $(date '+%F %T') rc=$?: $(tail -1 "$O/logs/client_$LABEL.log")"
kill -TERM -- -$SP 2>/dev/null; sleep 15
flock -u 9; echo "=== $(date '+%F %T') GPU work done, lock released"
echo "BASE_THINK_UNCLEAR_DRC_DONE $(date '+%F %T')"
