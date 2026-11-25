#!/bin/bash
# con1 skill test on Orchard (docs/consequence_round_con1.md §3): NSERVERS vLLM servers (one GPU each) serving the
# base model with LoRA adapters (ADAPTERS="label=dir label=dir"), then skill_client.py for each label in turn.
#   OUT=<dir> QUERIES=<queries.jsonl> ADAPTERS="con1=/.../final pref1=/.../final" [LABELS="con1 pref1"]
#   [MAXTOK_con1=2048] [MAXTOK_pref1=6144] bash scripts/con1/orchard_skill.sh
# Restart-safe: skill_client resumes by (state, pass, question).
set -uo pipefail
B="${BASE:-/project/community/wxu2}"
REPO="${REPO:-$HOME/con1/repo}"
OUT="${OUT:?output dir}"
QUERIES="${QUERIES:?queries.jsonl}"
ADAPTERS="${ADAPTERS:?label=adapter_dir ...}"
NSERVERS=${NSERVERS:-4}
VENV="${VLLM_VENV:-$B/rl/envs/vllm}"
TRAIN_PY="${TRAIN_PY:-$B/rl/envs/train/bin/python}"
MODEL_PATH="${MODEL_PATH:-$B/rl/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
C="${CACHE_DIR:-$B/rl/cache}"
export HF_HOME="${HF_HOME:-$B/rl/hf}"; export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor
export CUDA_CACHE_PATH=$C/nv VLLM_CACHE_ROOT=$C/vllm FLASHINFER_WORKSPACE_BASE=$C/flashinfer
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$OUT/logs" "$TMPDIR" "$C/xdg"
LORA=(); LABS=()
for kv in $ADAPTERS; do
  lab=${kv%%=*}; dir=${kv#*=}
  [ -e "$dir/adapter_config.json" ] || { echo "ABORT: no adapter at $dir"; exit 2; }
  LORA+=("$lab=$dir"); LABS+=("$lab")
done
LABELS=${LABELS:-${LABS[*]}}
for f in "$VENV/bin/vllm" "$TRAIN_PY" "$MODEL_PATH/config.json" "$QUERIES"; do [ -e "$f" ] || { echo "ABORT: missing $f"; exit 2; }; done
cd "$REPO"
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none} adapters=$ADAPTERS labels=$LABELS"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -r -a DEVS <<< "$CUDA_VISIBLE_DEVICES"
else NG=$(nvidia-smi -L | wc -l); DEVS=(); for i in $(seq 0 $((NG-1))); do DEVS+=("$i"); done; fi
PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"; }; done
  sleep 15
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && kill -KILL -- "-$p" 2>/dev/null; done
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
mapfile -t PORTS < <("$TRAIN_PY" -c 'import socket, sys
ss = [socket.socket() for _ in range(int(sys.argv[1]))]
for s in ss: s.bind(("127.0.0.1", 0))
for s in ss: print(s.getsockname()[1])' "$NSERVERS")
URLS=()
for k in $(seq 0 $((NSERVERS-1))); do
  CUDA_VISIBLE_DEVICES=${DEVS[k]} setsid "$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name base \
    --host 127.0.0.1 --port "${PORTS[k]}" --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-65536}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.95}" --max-num-seqs "${MAX_SEQS:-64}" --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' --enable-lora --max-lora-rank 64 --max-loras ${#LORA[@]} \
    --lora-modules "${LORA[@]}" > "$OUT/logs/vllm_${k}_job${SLURM_JOB_ID:-x}.log" 2>&1 &
  PIDS[k]=$!
  URLS+=("http://127.0.0.1:${PORTS[k]}/v1")
done
T0=$(date +%s)
while :; do
  UP=0
  for k in $(seq 0 $((NSERVERS-1))); do
    kill -0 "${PIDS[k]}" 2>/dev/null || { echo "ABORT: vLLM server $k exited"; tail -40 "$OUT/logs/vllm_${k}_job${SLURM_JOB_ID:-x}.log"; exit 7; }
    curl -sf "${URLS[k]}/models" >/dev/null 2>&1 && UP=$((UP+1))
  done
  [ "$UP" -eq "$NSERVERS" ] && break
  [ $(( $(date +%s) - T0 )) -lt 3600 ] || { echo "ABORT: $UP/$NSERVERS servers ready"; exit 7; }
  sleep 10
done
echo "=== servers ready after $(( $(date +%s) - T0 )) s"
rc=0
for lab in $LABELS; do
  mt=MAXTOK_$lab; mt=${!mt:-2048}
  PYTHONPATH="$REPO" "$TRAIN_PY" scripts/con1/skill_client.py --queries "$QUERIES" --tokenizer "$MODEL_PATH" \
    --base-url "${URLS[@]}" --model "$lab" --label "$lab" --out "$OUT/preds_$lab.jsonl" --max-tokens "$mt" \
    --workers "${WORKERS:-96}" ${PASSES:+--passes $PASSES} 2>&1 | tee -a "$OUT/logs/client_$lab.log"
  r=${PIPESTATUS[0]}; [ $r -ne 0 ] && rc=$r
done
echo "CON1_SKILL_DONE $(date '+%F %T %Z') rc=$rc wall $(( $(date +%s) - T0 )) s"
exit $rc
