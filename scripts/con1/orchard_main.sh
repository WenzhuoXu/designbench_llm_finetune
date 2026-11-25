#!/bin/bash
# con1 main run on one Orchard node (docs/consequence_round_con1.md section 11): the two stages in the published order.
#   stage 1  pref1 -> consequence rows (format FMT, PER_FAMILY training states per family, 8 families)
#            then the skill questions (queries_v2, held-out problems, zero-shot + sequential) in format FMT
#   stage 2  stage-1 adapter -> the policy rows (ownrows_own, rationale then action), one epoch
#            then the skill questions again (zero-shot): does the consequence skill survive the policy stage?
# Checkpoints on the node's /tmp (advanced is not preempted; /project/community has ~10 GB free); final adapters to
# $M/<stage>/final.   FMT=facet PER_FAMILY=1000 M=/project/community/wxu2/con1/main/run1 bash scripts/con1/orchard_main.sh
set -uo pipefail
B=/project/community/wxu2
REPO="${REPO:-$B/con1/repo}"
FMT="${FMT:?answer format}"
PER_FAMILY="${PER_FAMILY:-1000}"
M="${M:?run dir on /project/community}"
ROWS_SRC="${ROWS_SRC:-$B/con1/main/rows}"
TMPRUN="/tmp/con1_main_${SLURM_JOB_ID:-x}"
PREF1=$B/rl/runs/pref1_dx38mtR_20261123/final
VENV=$B/rl/envs/vllm
TRAIN_PY=$B/rl/envs/train/bin/python
MODEL_PATH=$B/rl/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
C=$B/rl/cache
Q=$B/con1/data/queries_v2.jsonl
mkdir -p "$M/logs" "$TMPRUN"
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none} fmt=$FMT per_family=$PER_FAMILY"
# rows: the first PER_FAMILY rows of each family (the prep rows are already in a seeded random order)
if [ ! -s "$M/train_stage1.jsonl" ]; then
  "$TRAIN_PY" - "$ROWS_SRC/train_$FMT.jsonl" "$M/train_stage1.jsonl" "$PER_FAMILY" <<'EOF'
import collections, json, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = collections.Counter()
with open(dst + ".tmp", "w", encoding="utf-8") as fo:
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        if c[r["domain"]] < n:
            c[r["domain"]] += 1
            fo.write(line)
import os
os.replace(dst + ".tmp", dst)
print("stage-1 rows", sum(c.values()), dict(c))
EOF
fi
cp -n "$ROWS_SRC/dev_$FMT.jsonl" "$M/dev_stage1.jsonl"
IFS=, read -r -a DEVS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NG=${#DEVS[@]}

train() {   # name rows dev init
  local name=$1 rows=$2 dev=$3 init=$4
  if [ -e "$M/$name/final/adapter_config.json" ]; then echo "$name: final adapter exists"; FINAL=$M/$name/final; return 0; fi
  CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${DEVS[*]}") NTRAIN=$NG REPO="$REPO" RUN_DIR="$TMPRUN/$name" \
    CONFIG=configs/rl/con1_sft.yaml ROWS="$rows" DEV="$dev" \
    OVERRIDES="sft.save_every=10 sft.keep_last=1 sft.dev_every=20 sft.keep_adapter_every=100000 policy.init_adapter=$init" \
    bash "$REPO/scripts/rl/orchard_sft_rows.sh" > "$M/logs/train_$name.log" 2>&1
  [ -e "$TMPRUN/$name/final/adapter_config.json" ] || { echo "ABORT: $name has no final adapter"; tail -30 "$M/logs/train_$name.log"; exit 3; }
  FINAL=$TMPRUN/$name/final                      # what the rest of the job uses; the shared copy is best effort
  cp "$TMPRUN/$name/metrics.jsonl" "$M/${name}_metrics.jsonl" 2>/dev/null
  if [ "$(df -BG --output=avail "$B" | tail -1 | tr -dc 0-9)" -gt 3 ]; then
    mkdir -p "$M/$name/final" && cp -r "$TMPRUN/$name/final/." "$M/$name/final/" && echo "$name adapter kept in $M/$name/final"       || { rm -rf "$M/$name/final"; echo "!!! $name adapter NOT kept (copy failed)"; }
  else
    echo "!!! $name adapter NOT kept: /project/community has under 3 GB free"
  fi
  echo "=== $(date '+%F %T') $name trained"
}

export HF_HOME=$B/rl/hf HF_HUB_CACHE=$B/rl/hf/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
SP=()
stop_servers() {
  for p in "${SP[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"; }; done
  sleep 20
  for p in "${SP[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && kill -KILL -- "-$p" 2>/dev/null; done
  SP=()
}
trap stop_servers EXIT
ask() {   # label adapter passes out
  local lab=$1 ad=$2 passes=$3 out=$4
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false XDG_CACHE_HOME=$C/xdg \
    TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor CUDA_CACHE_PATH=$C/nv VLLM_CACHE_ROOT=$C/vllm \
    FLASHINFER_WORKSPACE_BASE=$C/flashinfer VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0
  mapfile -t PORTS < <("$TRAIN_PY" -c 'import socket, sys
ss = [socket.socket() for _ in range(int(sys.argv[1]))]
for s in ss: s.bind(("127.0.0.1", 0))
for s in ss: print(s.getsockname()[1])' "$NG")
  local URLS=() k
  for k in $(seq 0 $((NG-1))); do
    CUDA_VISIBLE_DEVICES=${DEVS[k]} PYTHONNOUSERSITE=1 VLLM_NO_USAGE_STATS=1 VLLM_CACHE_ROOT=$C/vllm TMPDIR=$C/tmp \
      setsid "$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name base --host 127.0.0.1 --port "${PORTS[k]}" \
      --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.95 --max-num-seqs 64 --enable-prefix-caching \
      --limit-mm-per-prompt '{"image":0,"video":0}' --enable-lora --max-lora-rank 64 --max-loras 1 \
      --lora-modules "$lab=$ad" > "$M/logs/vllm_${lab}_$k.log" 2>&1 &
    SP[k]=$!
    URLS+=("http://127.0.0.1:${PORTS[k]}/v1")
  done
  local T0=$(date +%s) UP
  while :; do
    UP=0
    for k in $(seq 0 $((NG-1))); do
      kill -0 "${SP[k]}" 2>/dev/null || { echo "ABORT: vLLM server $k exited"; tail -30 "$M/logs/vllm_${lab}_$k.log"; exit 7; }
      curl -sf "${URLS[k]}/models" >/dev/null 2>&1 && UP=$((UP+1))
    done
    [ "$UP" -eq "$NG" ] && break
    [ $(( $(date +%s) - T0 )) -lt 3600 ] || { echo "ABORT: servers not ready"; exit 7; }
    sleep 10
  done
  ( cd "$REPO" && PYTHONPATH="$REPO" "$TRAIN_PY" scripts/con1/skill_client.py --queries "$Q" --tokenizer "$MODEL_PATH" \
      --base-url "${URLS[@]}" --model "$lab" --label "$lab" --target "$FMT" --out "$out" --max-tokens 4096 \
      --workers 128 --passes $passes --groups heldout_problem >> "$M/logs/client_$lab.log" 2>&1 )
  stop_servers
  echo "=== $(date '+%F %T') $lab answered ($passes)"
}

train stage1 "$M/train_stage1.jsonl" "$M/dev_stage1.jsonl" "$PREF1"
S1=$FINAL
ask stage1 "$S1" "zero seq" "$M/preds_stage1.jsonl"
train stage2 "$B/con1/stage2/policy_train.jsonl" "$B/con1/stage2/policy_dev.jsonl" "$S1"
ask stage2 "$FINAL" "zero" "$M/preds_stage2.jsonl"
echo "CON1_MAIN_DONE $(date '+%F %T %Z')"
