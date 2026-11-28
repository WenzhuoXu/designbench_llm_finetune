#!/bin/bash
# base_think_rep_drc.sh : on drc GPU 4, arm D of the 11-28 base-model read once more on subset_300.tsv, identical
# settings (base Qwen3.8-27B, no adapter, thinking, temperature 0, answer form 'dir' with the metrics named, think
# budget 8,000, answer cap 1,024, max-model-len 32768, think_client.py from full_eval/lib_v4 = commit cfa930e), to
# measure run-to-run variation at a fixed budget (the parent's order). Not the same hardware as Orchard's H100s (drc's
# 96 GB card), so the variation includes the hardware; Orchard's own repeat (arm base_v3_dir_8k_rep) is on the same
# hardware. GPU 4 under arm/gpu4.lock; resumable.
#   -> full_eval/base_think2/preds_base_v3_dir_8k_rep_drc.jsonl
set -uo pipefail
C=/storage/wxu/con1; A=$C/arm; O=$C/full_eval/base_think2; L4=$C/full_eval/lib_v4
PY=/storage/wenwenw/envs/vllm_drc/bin/python
MODEL=/storage/wxu/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
Q=$A/main/queries_v3.jsonl
export CON1_DIR=$L4
export PYTHONPATH=$C/pylib:$C/repo HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-4} OMP_NUM_THREADS=4 TMPDIR=$C/tmp
export HF_HOME=/storage/wxu/huggingface HF_HUB_CACHE=/storage/wxu/huggingface/hub HF_HUB_DISABLE_XET=1
export VLLM_CACHE_ROOT=$C/cache/vllm CUDA_CACHE_PATH=$C/cache/nv VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$O/logs"
[ -s "$O/subset_300.tsv" ] || cp $C/full_eval/base_think/subset_300.tsv "$O/"
echo "=== $(date '+%F %T') base_think_rep_drc queries=$Q ($(sha1sum < $Q | cut -c1-12)) subset $(sha1sum < $O/subset_300.tsv | cut -c1-12)"
exec 9>"$A/gpu4.lock"
flock 9; echo "=== $(date '+%F %T') holding $A/gpu4.lock"
read -r TOT USED <<< "$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | tr -d ',')"
while [ "${USED:-99999}" -ge 5000 ]; do sleep 120; read -r TOT USED <<< "$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | tr -d ',')"; done
PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
setsid $PY /storage/wenwenw/envs/vllm_drc/bin/vllm serve $MODEL --served-model-name base --host 127.0.0.1 --port $PORT \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92 --max-num-seqs 48 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' > "$O/logs/vllm_rep_drc.log" 2>&1 &
SP=$!
trap 'kill -TERM -- -$SP 2>/dev/null; sleep 10; kill -KILL -- -$SP 2>/dev/null' EXIT
T0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  kill -0 $SP 2>/dev/null || { echo "ABORT: vLLM exited"; tail -30 "$O/logs/vllm_rep_drc.log"; exit 7; }
  [ $(( $(date +%s) - T0 )) -lt 3600 ] || { echo "ABORT: vLLM not ready"; exit 7; }
  sleep 10
done
echo "=== $(date '+%F %T') server ready"
$PY $L4/think_client.py --queries $Q --tokenizer $MODEL --base-url "http://127.0.0.1:$PORT/v1" --model base \
  --label base_v3_dir_8k_rep_drc --target dir --named-metrics --budget 8000 --answer-tokens 1024 --workers 48 \
  --subset $O/subset_300.tsv --out $O/preds_base_v3_dir_8k_rep_drc.jsonl > "$O/logs/client_rep_drc.log" 2>&1
echo "=== $(date '+%F %T') rc=$?: $(tail -1 "$O/logs/client_rep_drc.log")"
kill -TERM -- -$SP 2>/dev/null; sleep 15
flock -u 9; echo "=== $(date '+%F %T') GPU work done, lock released"
echo "BASE_THINK_REP_DRC_DONE $(date '+%F %T')"
