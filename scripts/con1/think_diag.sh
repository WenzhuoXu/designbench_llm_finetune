#!/bin/bash
# think_diag.sh : the thinking diagnostic on drc GPU 4 (con1-2m-rows, 11-27; the parent session's request). Held-out
# family thermal2, PLACE_BLOCK and MOVE_SINK questions (309), zero-shot, facetm question as in the existing runs, room
# to think (budget 1,000 tokens) before the answer (think_client.py):
#   T-base       Qwen3.8-27B, no adapter, thinks as it likes up to the budget
#   T-s1         the stage-1 final adapter (arm/run_c2m_stage1_final/final), same
#   T-s1-min256  the stage-1 final adapter with '</think>' / end-of-turn banned for its first 256 think tokens
# ARMS="label:model:min_think[:budget] ..." (budget default 1000); e.g. ARMS="T-base-4k:base:0:4000" (the base model
# ran into the 1,000-token budget on ~97% of questions)
# GPU 4 under arm/gpu4.lock as con1's drc_arm.sh (the per-link eval waits on the same lock); every client stops by
# STOP_AT (default 14:45) so GPU 4 is free for the 15:10 link eval; resumable. Then think_score.py and the parent's
# review_tmp/ea/ea2.py per arm. Outputs: full_eval/think_diag/
set -uo pipefail
C=/storage/wxu/con1; A=$C/arm; D=$C/full_eval/think_diag
PY=/storage/wenwenw/envs/vllm_drc/bin/python
MODEL=/storage/wxu/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
ADAPTER=$A/run_c2m_stage1_final/final
QUERIES=$A/main/queries_v2.jsonl
STOP_AT=$(date -d "today ${STOP_AT:-14:45}" +%s)
ARMS=${ARMS:-"T-base:base:0 T-s1:c2m_stage1_final:0 T-s1-min256:c2m_stage1_final:256"}
export PYTHONPATH=$C/pylib:$C/repo HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-4} OMP_NUM_THREADS=4 TMPDIR=$C/tmp
export HF_HOME=/storage/wxu/huggingface HF_HUB_CACHE=/storage/wxu/huggingface/hub HF_HUB_DISABLE_XET=1
export VLLM_CACHE_ROOT=$C/cache/vllm CUDA_CACHE_PATH=$C/cache/nv VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$D/logs"
echo "=== $(date '+%F %T') think_diag arms: $ARMS; stop by $(date -d @$STOP_AT '+%T')"
exec 9>"$A/gpu4.lock"
flock 9; echo "=== $(date '+%F %T') holding $A/gpu4.lock"
read -r TOT USED <<< "$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | tr -d ',')"
[ "${TOT:-0}" -gt 90000 ] && [ "${USED:-99999}" -lt 5000 ] || { echo "ABORT: GPU $CUDA_VISIBLE_DEVICES ${TOT} MiB, ${USED} in use"; exit 4; }
PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
setsid $PY /storage/wenwenw/envs/vllm_drc/bin/vllm serve $MODEL --served-model-name base --host 127.0.0.1 --port $PORT \
  --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.92 --max-num-seqs 48 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' --enable-lora --max-lora-rank 64 --max-loras 1 \
  --lora-modules "c2m_stage1_final=$ADAPTER" > "$D/logs/vllm.log" 2>&1 &
SP=$!
trap 'kill -TERM -- -$SP 2>/dev/null; sleep 10; kill -KILL -- -$SP 2>/dev/null' EXIT
T0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  kill -0 $SP 2>/dev/null || { echo "ABORT: vLLM exited"; tail -30 "$D/logs/vllm.log"; exit 7; }
  [ $(( $(date +%s) - T0 )) -lt 3600 ] || { echo "ABORT: vLLM not ready"; exit 7; }
  sleep 10
done
echo "=== $(date '+%F %T') server ready"
for arm in $ARMS; do
  IFS=: read -r L M K B <<< "$arm"
  left=$(( STOP_AT - $(date +%s) ))
  [ $left -gt 60 ] || { echo "=== $(date '+%F %T') $L not run: past the stop time"; continue; }
  timeout $left $PY $D/think_client.py --queries $QUERIES --tokenizer $MODEL --base-url "http://127.0.0.1:$PORT/v1" \
    --model $M --label $L --min-think $K --budget ${B:-1000} --answer-tokens 1024 --workers 48 --out $D/preds_$L.jsonl \
    > "$D/logs/client_$L.log" 2>&1
  echo "=== $(date '+%F %T') $L rc=$?: $(tail -1 "$D/logs/client_$L.log")"
done
kill -TERM -- -$SP 2>/dev/null; sleep 15
flock -u 9; echo "=== $(date '+%F %T') GPU work done, lock released"
for arm in $ARMS; do
  IFS=: read -r L M K B <<< "$arm"
  [ -s $D/preds_$L.jsonl ] || continue
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /opt/anaconda3/bin/python $D/think_score.py $D/preds_$L.jsonl > $D/score_$L.txt 2>&1
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /opt/anaconda3/bin/python $C/review_tmp/ea/ea2.py $D/preds_$L.jsonl zero > $D/ea2_$L.txt 2>&1
  echo "##### $L"; cat $D/score_$L.txt $D/ea2_$L.txt
done
echo "THINK_DIAG_DONE $(date '+%F %T')"
