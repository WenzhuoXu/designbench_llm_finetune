#!/bin/bash
# con1 policy-rows training on drc GPU 4 (single GPU, restart-safe: resumes from the newest checkpoint of RUN_DIR).
#   RUN_DIR=/storage/wxu/con1/stage2/runs/<name> ROWS=... DEV=... [INIT=<adapter dir>] bash scripts/con1/drc_policy_train.sh
set -uo pipefail
C=/storage/wxu/con1
REPO=$C/repo
RUN_DIR="${RUN_DIR:?run dir}"; ROWS="${ROWS:?rows}"; DEV="${DEV:-}"
PY=/storage/wenwenw/envs/vllm_drc/bin/python
export PYTHONPATH=$C/pylib:$REPO HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${GPU:-4} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export TMPDIR=$C/tmp TRITON_CACHE_DIR=$C/cache/triton TORCHINDUCTOR_CACHE_DIR=$C/cache/inductor
mkdir -p "$RUN_DIR/logs" "$TMPDIR"
cd $REPO
echo "=== $(date '+%F %T') host=$(hostname) run=$RUN_DIR rows=$ROWS init=${INIT:-config}"
$PY -m torch.distributed.run --standalone --nproc_per_node 1 -m llm_finetune.rl_dataxp.sft_rows \
  --config configs/rl/con1_policy_drc.yaml run_dir="$RUN_DIR" "sft.rows_file=$ROWS" ${DEV:+"sft.dev_file=$DEV"} \
  ${INIT:+"policy.init_adapter=$INIT"} 2>&1 | tee -a "$RUN_DIR/logs/trainer.log"
echo "DRC_POLICY_TRAIN_DONE rc=${PIPESTATUS[0]} $(date '+%F %T')"
