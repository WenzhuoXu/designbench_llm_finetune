#!/bin/bash
# con1 one-GPU arm on drc GPU 4: train (restart-safe: the trainer resumes from RUN_DIR's newest checkpoint), serve the
# adapter, ask the round-1 held-out-problem questions in the arm's format, score with facets.py next to round 1's arms.
#   LAB=Fm FMT=facetm ROWS=/storage/wxu/con1/arm/rows_facetm.jsonl bash scripts/con1/drc_arm.sh
set -uo pipefail
C=/storage/wxu/con1
A=$C/arm
REPO=$C/repo
LAB="${LAB:?label}"; FMT="${FMT:?answer format}"; ROWS="${ROWS:?rows}"
RUN=$A/run_$LAB
PY=/storage/wenwenw/envs/vllm_drc/bin/python
MODEL=/storage/wxu/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
FAM="engibench grid2op photonics circuit2 frame3d2"
export PYTHONPATH=$C/pylib:$REPO HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-4}} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export TMPDIR=$C/tmp TRITON_CACHE_DIR=$C/cache/triton TORCHINDUCTOR_CACHE_DIR=$C/cache/inductor
export HF_HOME=/storage/wxu/huggingface HF_HUB_CACHE=/storage/wxu/huggingface/hub HF_HUB_DISABLE_XET=1
export VLLM_CACHE_ROOT=$C/cache/vllm CUDA_CACHE_PATH=$C/cache/nv VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$RUN/logs" "$TMPDIR"
cd $REPO
echo "=== $(date '+%F %T') host=$(hostname) arm=$LAB fmt=$FMT rows=$ROWS gpu=$CUDA_VISIBLE_DEVICES"
# only the 96 GB card; wait (WAIT=1) until nothing but its small resident service (legal_helper, ~3 GB) holds it
gpu_mem() { nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | tr -d ','; }
read -r TOT USED <<< "$(gpu_mem)"
[ "${TOT:-0}" -gt 90000 ] || { echo "ABORT: GPU $CUDA_VISIBLE_DEVICES has ${TOT} MiB"; exit 4; }
while [ "${USED:-99999}" -ge 5000 ]; do
  [ "${WAIT:-0}" = 1 ] || { echo "ABORT: GPU $CUDA_VISIBLE_DEVICES has ${USED} MiB in use"; exit 4; }
  sleep 120; read -r TOT USED <<< "$(gpu_mem)"
done
echo "=== $(date '+%F %T') GPU $CUDA_VISIBLE_DEVICES free enough (${USED} MiB in use)"
free -g | head -2
if [ ! -e "$RUN/final/adapter_config.json" ]; then
  $PY -m torch.distributed.run --standalone --nproc_per_node 1 -m llm_finetune.rl_dataxp.sft_rows \
    --config configs/rl/con1_arm_drc.yaml run_dir="$RUN" "sft.rows_file=$ROWS" > "$RUN/logs/trainer.log" 2>&1
  echo "trainer rc=$? $(date '+%F %T')"; grep "^step\|dev loss\|peak" "$RUN/logs/trainer.log" | tail -3
fi
[ -e "$RUN/final/adapter_config.json" ] || { echo "ABORT: no final adapter"; tail -30 "$RUN/logs/trainer.log"; exit 3; }
PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
setsid $PY /storage/wenwenw/envs/vllm_drc/bin/vllm serve $MODEL --served-model-name base --host 127.0.0.1 --port $PORT \
  --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.92 --max-num-seqs 48 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' --enable-lora --max-lora-rank 64 --max-loras 1 \
  --lora-modules "$LAB=$RUN/final" > "$RUN/logs/vllm.log" 2>&1 &
SP=$!
trap 'kill -TERM -- -$SP 2>/dev/null; sleep 10; kill -KILL -- -$SP 2>/dev/null' EXIT
T0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  kill -0 $SP 2>/dev/null || { echo "ABORT: vLLM exited"; tail -30 "$RUN/logs/vllm.log"; exit 7; }
  [ $(( $(date +%s) - T0 )) -lt 3600 ] || { echo "ABORT: vLLM not ready"; exit 7; }
  sleep 10
done
echo "=== $(date '+%F %T') server ready"
$PY scripts/con1/skill_client.py --queries $A/queries.jsonl --tokenizer $MODEL --base-url "http://127.0.0.1:$PORT/v1" \
  --model $LAB --label $LAB --target $FMT --out $A/preds_$LAB.jsonl --max-tokens 2048 --workers 48 --passes zero \
  --groups heldout_problem --variants $FAM > "$RUN/logs/client.log" 2>&1
tail -2 "$RUN/logs/client.log"
kill -TERM -- -$SP 2>/dev/null; sleep 15
echo "=== $(date '+%F %T') facets"
export OMP_NUM_THREADS=1 DATAXP_SPELLING=multitool PYTHONWARNINGS=ignore
P="$A/preds_$LAB.jsonl"; for x in F M trees; do [ -s $A/r1/preds_$x.jsonl ] && P="$P $A/r1/preds_$x.jsonl"; done
CUDA_VISIBLE_DEVICES= $PY scripts/con1/facets.py --queries $A/queries.jsonl --preds $P --groups heldout_problem --variants $FAM \
  --problems $C/gen/in/bench_problems.jsonl --code ${CODE:-$A/code_v3} \
  --harness $C/gen/h/dataxp_policy_eval_v3rc.py --boot 1000 --out $A/facets_$LAB.txt --json $A/facets_$LAB.json > /dev/null
grep -A7 "group means over all" $A/facets_$LAB.txt
echo "DRC_ARM_DONE $LAB $(date '+%F %T')"
