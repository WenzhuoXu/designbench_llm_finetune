#!/bin/bash
# (con1-2m-rows copy of con1/repo/scripts/con1/drc_arm.sh, 11-26: env QGROUPS for the question group; the sequential
#  pass is scored with --kshot-baselines (copy_last, copy_mode); unset QGROUPS = the original)
# con1 one-GPU arm / eval on drc GPU 4: train (restart-safe: the trainer resumes from RUN_DIR's newest checkpoint; skipped
# when RUN_DIR/final exists), serve the adapter, ask the held-out-problem questions in the arm's format, score with
# facets.py (no harness: drc has no engibench env; the harness only serves the Phi verdict, out of the rules).
#   LAB=Fm FMT=facetm ROWS=/storage/wxu/con1/arm/rows_facetm.jsonl bash scripts/con1/drc_arm.sh
# Options (env):
#   PASSES   "zero" (default) or "zero seq"
#   QUERIES  questions file (default arm/queries.jsonl = round 1's skill questions)
#   FAM      families (default round 1's five; "all" = every family in QUERIES)
#   CMP      comparison predictions scored next to the zero pass (default round 1's F, M, trees)
#   MAXTOK   answer cap (default 2048)
#   OUTDIR   where preds and facets go (default arm/)
#   WAIT=1   wait for GPU 4 instead of aborting
# GPU 4 is taken under an flock (arm/gpu4.lock): a second instance waits until the first has finished its GPU work.
set -uo pipefail
C=/storage/wxu/con1
A=$C/arm
REPO=$C/repo
LAB="${LAB:?label}"; FMT="${FMT:?answer format}"; ROWS="${ROWS:?rows}"
RUN=$A/run_$LAB
PY=/storage/wenwenw/envs/vllm_drc/bin/python
MODEL=/storage/wxu/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
QUERIES="${QUERIES:-$A/queries.jsonl}"
FAM="${FAM:-engibench grid2op photonics circuit2 frame3d2}"
VARG="--variants $FAM"; [ "$FAM" = all ] && VARG=""
CMP="${CMP:-$A/r1/preds_F.jsonl $A/r1/preds_M.jsonl $A/r1/preds_trees.jsonl}"
MAXTOK="${MAXTOK:-2048}"
OUTDIR="${OUTDIR:-$A}"
QGROUPS="${QGROUPS:-heldout_problem}"   # c2m copy: the question group (heldout_problem / heldout_family)
# pylib_fla: flash-linear-attention 0.5.2 (Orchard's train env has the same); without it transformers runs the
# reference gated-delta kernel (~760 s per 16-row step here). Checked on GPU 4 11-26: outputs and gradients match the
# reference (cos >= 0.9999, rel err <= 0.6%).
export PYTHONPATH=$C/pylib_fla:$C/pylib:$REPO HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export DATAXP_OFFLOAD_PAGEABLE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-4}} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export TMPDIR=$C/tmp TRITON_CACHE_DIR=$C/cache/triton TORCHINDUCTOR_CACHE_DIR=$C/cache/inductor
export HF_HOME=/storage/wxu/huggingface HF_HUB_CACHE=/storage/wxu/huggingface/hub HF_HUB_DISABLE_XET=1
export VLLM_CACHE_ROOT=$C/cache/vllm CUDA_CACHE_PATH=$C/cache/nv VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$RUN/logs" "$TMPDIR" "$OUTDIR"
cd $REPO
echo "=== $(date '+%F %T') host=$(hostname) arm=$LAB fmt=$FMT rows=$ROWS gpu=$CUDA_VISIBLE_DEVICES queries=$QUERIES fam=$FAM out=$OUTDIR"
exec 9>"$A/gpu4.lock"
flock 9; echo "=== $(date '+%F %T') holding $A/gpu4.lock"
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
  ( while sleep 20; do echo "$(date '+%T') $(free -m | sed -n 2p) | gpu $(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv,noheader)"; done ) > "$RUN/logs/mem.log" 2>&1 &
  MEMLOG=$!
  $PY -m torch.distributed.run --standalone --nproc_per_node 1 -m llm_finetune.rl_dataxp.sft_rows \
    --config configs/rl/con1_arm_drc.yaml run_dir="$RUN" "sft.rows_file=$ROWS" > "$RUN/logs/trainer.log" 2>&1
  echo "trainer rc=$? $(date '+%F %T')"; kill $MEMLOG 2>/dev/null; grep "^step\|dev loss\|peak" "$RUN/logs/trainer.log" | tail -3
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
$PY scripts/con1/skill_client.py --queries $QUERIES --tokenizer $MODEL --base-url "http://127.0.0.1:$PORT/v1" \
  --model $LAB --label $LAB --target $FMT --out $OUTDIR/preds_$LAB.jsonl --max-tokens $MAXTOK --workers 48 --passes ${PASSES:-zero} \
  --groups $QGROUPS $VARG > "$RUN/logs/client.log" 2>&1
tail -2 "$RUN/logs/client.log"
kill -TERM -- -$SP 2>/dev/null; sleep 15
flock -u 9; echo "=== $(date '+%F %T') GPU work done, lock released"
echo "=== $(date '+%F %T') facets"
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
for PS in ${PASSES:-zero}; do
  P="$OUTDIR/preds_$LAB.jsonl"
  [ "$PS" = zero ] && for x in $CMP; do [ -s $x ] && P="$P $x"; done
  KB=; [ "$PS" = seq ] && KB=--kshot-baselines   # c2m copy: copy_last / copy_mode next to the sequential pass
  CUDA_VISIBLE_DEVICES= /opt/anaconda3/bin/python scripts/con1/facets.py --queries $QUERIES --preds $P --pass $PS $KB \
    --groups $QGROUPS $VARG --boot 1000 --out $OUTDIR/facets_${LAB}_$PS.txt --json $OUTDIR/facets_${LAB}_$PS.json > /dev/null
  echo "== pass $PS"; grep -A7 "group means over all" $OUTDIR/facets_${LAB}_$PS.txt
done
if [[ " ${PASSES:-zero} " == *" seq "* ]]; then    # the same questions zero-shot (positions >= 1): the k-shot gain
  CUDA_VISIBLE_DEVICES= /opt/anaconda3/bin/python scripts/con1/facets.py --queries $QUERIES --preds $OUTDIR/preds_$LAB.jsonl \
    --pass zero --min-qi 1 --groups $QGROUPS $VARG --boot 1000 \
    --out $OUTDIR/facets_${LAB}_zero_q1.txt --json $OUTDIR/facets_${LAB}_zero_q1.json > /dev/null
  echo "== pass zero, positions >= 1"; grep -A7 "group means over all" $OUTDIR/facets_${LAB}_zero_q1.txt
fi
echo "DRC_ARM_DONE $LAB $(date '+%F %T')"
