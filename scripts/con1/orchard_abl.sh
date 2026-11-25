#!/bin/bash
# con1 ablation round on one Orchard node (docs/consequence_round_con1.md §9): one 1-GPU LoRA run per arm in parallel
# (from pref1, consequence rows only, checkpoints on the node's /tmp), then vLLM servers with every arm's adapter and
# the zero-shot held-out-problem questions per arm in that arm's answer format.
#   ARMS="B:verbatim:/path/rows_B.jsonl M:masked:/path/rows_M.jsonl ..." OUT=/project/community/wxu2/con1/abl1 \
#   QUERIES=... VARIANTS="engibench grid2op ..." [OVERRIDES="sft.rows_per_step=16 ..."] bash scripts/con1/orchard_abl.sh
# Restart: training restarts in /tmp of the new node (ablations are short); the client resumes by question.
set -uo pipefail
B="${BASE:-/project/community/wxu2}"
REPO="${REPO:-$B/con1/repo}"
OUT="${OUT:?output dir}"
ARMS="${ARMS:?label:format:rows ...}"
# a pending job can be re-pointed: $OUT/ARMS.override (one line, same syntax) replaces the exported ARMS
[ -s "$OUT/ARMS.override" ] && { ARMS="$(head -1 "$OUT/ARMS.override")"; echo "ARMS from $OUT/ARMS.override"; }
QUERIES="${QUERIES:?queries.jsonl}"
VARIANTS="${VARIANTS:?families}"
TMPRUN="${TMPRUN:-/tmp/con1_abl_${SLURM_JOB_ID:-x}}"
VENV="${VLLM_VENV:-$B/rl/envs/vllm}"
TRAIN_PY="${TRAIN_PY:-$B/rl/envs/train/bin/python}"
MODEL_PATH="${MODEL_PATH:-$B/rl/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
C="${CACHE_DIR:-$B/rl/cache}"
mkdir -p "$OUT/logs" "$TMPRUN"
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none} arms=$ARMS tmp=$TMPRUN"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -r -a DEVS <<< "$CUDA_VISIBLE_DEVICES"
else NG=$(nvidia-smi -L | wc -l); DEVS=(); for i in $(seq 0 $((NG-1))); do DEVS+=("$i"); done; fi
read -r -a ARR <<< "$ARMS"
[ ${#ARR[@]} -le ${#DEVS[@]} ] || { echo "ABORT: ${#ARR[@]} arms > ${#DEVS[@]} GPUs"; exit 2; }
# ---- phase 1: training, one GPU per arm, starts staggered (each process loads the 27B model)
TPIDS=(); k=0
for arm in "${ARR[@]}"; do
  IFS=: read -r lab fmt rows <<< "$arm"
  [ -s "$rows" ] || { echo "ABORT: no rows $rows"; exit 2; }
  if [ -e "$TMPRUN/$lab/final/adapter_config.json" ]; then echo "arm $lab: adapter exists"; k=$((k+1)); continue; fi
  ( CUDA_VISIBLE_DEVICES=${DEVS[k]} NTRAIN=1 REPO="$REPO" RUN_DIR="$TMPRUN/$lab" CONFIG=configs/rl/con1_sft.yaml ROWS="$rows" \
    OVERRIDES="${OVERRIDES:-sft.rows_per_step=16 sft.save_every=100000 sft.keep_last=1 sft.dev_every=0}" \
    bash "$REPO/scripts/rl/orchard_sft_rows.sh" > "$OUT/logs/train_$lab.log" 2>&1 ) &
  TPIDS+=($!)
  echo "arm $lab ($fmt) training on GPU ${DEVS[k]}: $rows"
  k=$((k+1)); sleep "${STAGGER_S:-150}"
done
rc=0
for p in "${TPIDS[@]:-}"; do [ -n "$p" ] && { wait "$p" || rc=$?; }; done
echo "=== $(date '+%F %T') training done (last rc $rc)"
for arm in "${ARR[@]}"; do
  IFS=: read -r lab fmt rows <<< "$arm"
  [ -e "$TMPRUN/$lab/final/adapter_config.json" ] || { echo "ABORT: arm $lab has no final adapter"; tail -30 "$OUT/logs/train_$lab.log"; exit 3; }
  cp "$TMPRUN/$lab/metrics.jsonl" "$OUT/metrics_$lab.jsonl" 2>/dev/null
done
# ---- phase 2: serve every adapter, ask each arm its questions
LORA=()
for arm in "${ARR[@]}"; do IFS=: read -r lab fmt rows <<< "$arm"; LORA+=("$lab=$TMPRUN/$lab/final"); done
export HF_HOME="${HF_HOME:-$B/rl/hf}"; export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor
export CUDA_CACHE_PATH=$C/nv VLLM_CACHE_ROOT=$C/vllm FLASHINFER_WORKSPACE_BASE=$C/flashinfer
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0
PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"; }; done
  sleep 15
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && kill -KILL -- "-$p" 2>/dev/null; done
}
trap cleanup EXIT
mapfile -t PORTS < <("$TRAIN_PY" -c 'import socket, sys
ss = [socket.socket() for _ in range(int(sys.argv[1]))]
for s in ss: s.bind(("127.0.0.1", 0))
for s in ss: print(s.getsockname()[1])' "${#DEVS[@]}")
URLS=()
for k in $(seq 0 $((${#DEVS[@]}-1))); do
  CUDA_VISIBLE_DEVICES=${DEVS[k]} setsid "$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name base \
    --host 127.0.0.1 --port "${PORTS[k]}" --dtype bfloat16 --max-model-len 65536 \
    --gpu-memory-utilization 0.95 --max-num-seqs 64 --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' --enable-lora --max-lora-rank 64 --max-loras ${#LORA[@]} \
    --lora-modules "${LORA[@]}" > "$OUT/logs/vllm_${k}.log" 2>&1 &
  PIDS[k]=$!
  URLS+=("http://127.0.0.1:${PORTS[k]}/v1")
done
T0=$(date +%s)
while :; do
  UP=0
  for k in $(seq 0 $((${#DEVS[@]}-1))); do
    kill -0 "${PIDS[k]}" 2>/dev/null || { echo "ABORT: vLLM server $k exited"; tail -40 "$OUT/logs/vllm_${k}.log"; exit 7; }
    curl -sf "${URLS[k]}/models" >/dev/null 2>&1 && UP=$((UP+1))
  done
  [ "$UP" -eq "${#DEVS[@]}" ] && break
  [ $(( $(date +%s) - T0 )) -lt 3600 ] || { echo "ABORT: servers not ready"; exit 7; }
  sleep 10
done
echo "=== $(date '+%F %T') servers ready"
cd "$REPO"
for arm in "${ARR[@]}"; do
  IFS=: read -r lab fmt rows <<< "$arm"
  PYTHONPATH="$REPO" "$TRAIN_PY" scripts/con1/skill_client.py --queries "$QUERIES" --tokenizer "$MODEL_PATH" \
    --base-url "${URLS[@]}" --model "$lab" --label "$lab" --target "$fmt" --out "$OUT/preds_$lab.jsonl" \
    --max-tokens 2048 --workers 96 --passes zero --groups heldout_problem --variants $VARIANTS \
    >> "$OUT/logs/client_$lab.log" 2>&1 &
done
wait
# P-form test (round 2): each arm answers the policy prompt of every test state; p_score.py (CPU) simulates the chosen
# action and pairs it with the reply's EXPECTED RESULT block
if [ -n "${P_TEST:-}" ]; then
  for arm in "${ARR[@]}"; do
    IFS=: read -r lab fmt rows <<< "$arm"
    PYTHONPATH="$REPO" "$TRAIN_PY" scripts/con1/p_client.py --queries "$QUERIES" --base-url "${URLS[@]}" --model "$lab"       --label "$lab" --out "$OUT/predsP_$lab.jsonl" --groups heldout_problem --variants $VARIANTS --max-tokens 8192       --workers 64 >> "$OUT/logs/pclient_$lab.log" 2>&1 &
  done
  wait
fi
if [ "$(df -BG --output=avail "$B" | tail -1 | tr -dc 0-9)" -gt "${KEEP_IF_FREE_GB:-40}" ]; then
  for arm in "${ARR[@]}"; do IFS=: read -r lab fmt rows <<< "$arm"; mkdir -p "$OUT/adapters/$lab"; cp -r "$TMPRUN/$lab/final/." "$OUT/adapters/$lab/"; done
  echo "adapters copied to $OUT/adapters"
else
  echo "adapters NOT kept (free space under ${KEEP_IF_FREE_GB:-40} GB); they rebuild from the rows"
fi
echo "CON1_ABL_DONE $(date '+%F %T %Z')"
