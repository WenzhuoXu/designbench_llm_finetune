#!/bin/bash
# con1 facetc, after drc_arm.sh LAB=Fc: the link scores (CPU) and the dependence test (GPU 4 under arm/gpu4.lock).
#   LAB=Fc setsid nohup bash scripts/con1/drc_chain_post.sh > arm/chain_post_Fc.out 2>&1 &
set -uo pipefail
C=/storage/wxu/con1
A=$C/arm
REPO=$C/repo
LAB="${LAB:-Fc}"
RUN=$A/run_$LAB
PY=/storage/wenwenw/envs/vllm_drc/bin/python
MODEL=/storage/wxu/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
cd $REPO
echo "=== $(date '+%F %T') waiting for the arm $LAB to finish"
until grep -q "DRC_ARM_DONE $LAB" $A/con1_arm_$LAB.out 2>/dev/null; do
  grep -q "ABORT" $A/con1_arm_$LAB.out 2>/dev/null && { echo "the arm aborted"; exit 3; }
  sleep 120
done
echo "=== $(date '+%F %T') link scores"
/opt/anaconda3/bin/python scripts/con1/chain_score.py --queries $A/queries.jsonl --chain $A/chain/chain_r1.jsonl \
  --preds $A/preds_$LAB.jsonl --label $LAB --out $A/chain_scores_$LAB.json
echo "=== $(date '+%F %T') dependence test: waiting for GPU 4"
export PYTHONPATH=$C/pylib:$REPO HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 TMPDIR=$C/tmp HF_HOME=/storage/wxu/huggingface HF_HUB_CACHE=/storage/wxu/huggingface/hub
export VLLM_CACHE_ROOT=$C/cache/vllm CUDA_CACHE_PATH=$C/cache/nv VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
exec 9>"$A/gpu4.lock"; flock 9
until [ "$(nvidia-smi -i 4 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 5000 ]; do sleep 60; done
PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
setsid $PY /storage/wenwenw/envs/vllm_drc/bin/vllm serve $MODEL --served-model-name base --host 127.0.0.1 --port $PORT \
  --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.92 --max-num-seqs 48 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' --enable-lora --max-lora-rank 64 --max-loras 1 \
  --lora-modules "$LAB=$RUN/final" > "$RUN/logs/vllm_dep.log" 2>&1 &
SP=$!
trap 'kill -TERM -- -$SP 2>/dev/null; sleep 10; kill -KILL -- -$SP 2>/dev/null' EXIT
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  kill -0 $SP 2>/dev/null || { echo "vLLM exited"; tail -20 "$RUN/logs/vllm_dep.log"; exit 7; }
  sleep 10
done
$PY scripts/con1/chain_dep.py --queries $A/queries.jsonl --preds $A/preds_$LAB.jsonl --label $LAB --tokenizer $MODEL \
  --base-url "http://127.0.0.1:$PORT/v1" --model $LAB --out $A/chain_dep_$LAB.jsonl --n 300
kill -TERM -- -$SP 2>/dev/null; sleep 15
echo "CHAIN_POST_DONE $(date '+%F %T')"
