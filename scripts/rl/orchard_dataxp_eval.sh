#!/bin/bash
# Uncapped dataxp evaluation through the training harness path (python -m llm_finetune.rl_dataxp.rollout
# eval): T=0, --turns 20, no completion cap (max_tokens null), the harness flags of the RL config, so
# an evaluation episode is a training rollout at T=0. One node: NSERVERS vLLM servers (one GPU each),
# the episodes spread over them. Restart-safe: episodes resume by key, so a requeued (preempted) job
# redoes only the unfinished ones.
#
#   OUT=/home/wxu2/rl/eval/<label> LABEL=base PROBLEMS="/home/wxu2/rl/harness/problems_v2.jsonl ..." \
#   [ADAPTER=/path/to/peft_dir] [THINK=1] [VARIANTS="circuit2 ..."] [RUNS=2] [TURNS=20] [NSERVERS=4] \
#   bash scripts/rl/orchard_dataxp_eval.sh
#
# ADAPTER: serve a LoRA adapter under LABEL and evaluate it (default: the un-tuned base, served as
# 'base'). THINK=1: thinking on (no --no-think), for a policy trained with its rationale inside the
# think block (dataxp_grpo_p2_datav4_dx38mtR.yaml); default thinking off, as P2. Knobs as orchard_dataxp_grpo.sh: BASE, REPO, VLLM_VENV, SIM_PY, MODEL_PATH, CACHE_DIR,
# HF_HOME, MAX_MODEL_LEN (the model's own max_position_embeddings), GPU_UTIL (0.95), MAX_SEQS (64),
# MAX_LORA_RANK (64), WORKERS (32), SERVER_WAIT_S (3600), HARNESS (/home/wxu2/rl/harness).
# HARNESS_SCRIPT: the harness script under $HARNESS/scripts (default dataxp_policy_eval_v3.py);
# HARNESS_ARGS: extra harness flags. The refusal-note arm of an A/B is
#   HARNESS_SCRIPT=dataxp_policy_eval_v3n.py HARNESS_ARGS=--reject-note
set -uo pipefail
B="${BASE:-/project/community/wxu2}"
REPO="${REPO:-$HOME/rl/repo}"
OUT="${OUT:?output dir}"
LABEL="${LABEL:?label (the served model name for an adapter)}"
# no braces in the message: a '}' inside it ends the expansion early and the rest is appended to the value
PROBLEMS="${PROBLEMS:?problem files, one variant + spec per line}"
NSERVERS=${NSERVERS:-4}
VENV="${VLLM_VENV:-$B/rl/envs/vllm}"
SIM_PY="${SIM_PY:-$B/rl/envs/sim/bin/python}"
MODEL_PATH="${MODEL_PATH:-$B/rl/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
H="${HARNESS:-$HOME/rl/harness}"
HS="${HARNESS_SCRIPT:-dataxp_policy_eval_v3.py}"
C="${CACHE_DIR:-$B/rl/cache}"
for f in "$VENV/bin/vllm" "$SIM_PY" "$MODEL_PATH/config.json" "$H/scripts/$HS" \
         "$H/configs/task_instruction_V2.txt" $PROBLEMS ${ADAPTER:+"$ADAPTER/adapter_config.json"}; do
  [ -e "$f" ] || { echo "ABORT: missing $f"; exit 2; }
done
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$("$SIM_PY" -c 'import json, sys
c = json.load(open(sys.argv[1])); t = c.get("text_config", c)
print(t.get("max_position_embeddings") or c.get("max_position_embeddings"))' "$MODEL_PATH/config.json")}"
export HF_HOME="${HF_HOME:-$B/rl/hf}"; export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor
export CUDA_CACHE_PATH=$C/nv VLLM_CACHE_ROOT=$C/vllm FLASHINFER_WORKSPACE_BASE=$C/flashinfer MPLBACKEND=Agg
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0
export DX_ORIG_SCRIPTS=$H/scripts DATAXP_SPELLING=multitool
# molopt's PyTDC model files (built once on the login node, which has internet): $H/molopt_oracle/oracle/
export DATAXP_MOLOPT_ORACLE_DIR="${DATAXP_MOLOPT_ORACLE_DIR:-$H/molopt_oracle}"
mkdir -p "$OUT/logs" "$TMPDIR" "$C/xdg"
cd "$REPO"
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none} restarts=${SLURM_RESTART_COUNT:-0}"
echo "=== repo=$REPO ($(cat "$REPO/GIT_COMMIT" 2>/dev/null || git -C "$REPO" rev-parse --short HEAD 2>/dev/null)) label=$LABEL adapter=${ADAPTER:-none} think=${THINK:-0} out=$OUT harness=$HS ${HARNESS_ARGS:-}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -r -a DEVS <<< "$CUDA_VISIBLE_DEVICES"
else NG=$(nvidia-smi -L | wc -l); DEVS=(); for i in $(seq 0 $((NG-1))); do DEVS+=("$i"); done; fi
[ "$NSERVERS" -le ${#DEVS[@]} ] || { echo "ABORT: NSERVERS=$NSERVERS > ${#DEVS[@]} GPUs"; exit 2; }

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"; }; done
  sleep 15
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && kill -KILL -- "-$p" 2>/dev/null; done
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT

NOTHINK=(--no-think); [ -n "${THINK:-}" ] && NOTHINK=()
LORA_ARGS=(); SERVED=base
if [ -n "${ADAPTER:-}" ]; then
  LORA_ARGS=(--enable-lora --max-lora-rank "${MAX_LORA_RANK:-64}" --lora-modules "$LABEL=$ADAPTER"); SERVED=$LABEL
fi
mapfile -t PORTS < <("$SIM_PY" -c 'import socket, sys
ss = [socket.socket() for _ in range(int(sys.argv[1]))]
for s in ss: s.bind(("127.0.0.1", 0))
for s in ss: print(s.getsockname()[1])' "$NSERVERS")
URLS=()
for k in $(seq 0 $((NSERVERS-1))); do
  CUDA_VISIBLE_DEVICES=${DEVS[k]} setsid "$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name base \
    --host 127.0.0.1 --port "${PORTS[k]}" --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "${GPU_UTIL:-0.95}" --max-num-seqs "${MAX_SEQS:-64}" \
    --limit-mm-per-prompt '{"image":0,"video":0}' "${LORA_ARGS[@]}" \
    > "$OUT/logs/vllm_server_${k}_job${SLURM_JOB_ID:-x}.log" 2>&1 &
  PIDS[k]=$!
  URLS+=("http://127.0.0.1:${PORTS[k]}/v1")
done
T0=$(date +%s)
while :; do
  UP=0
  for k in $(seq 0 $((NSERVERS-1))); do
    kill -0 "${PIDS[k]}" 2>/dev/null || { echo "ABORT: vLLM server $k exited"; tail -40 "$OUT/logs/vllm_server_${k}_job${SLURM_JOB_ID:-x}.log"; exit 7; }
    curl -sf "${URLS[k]}/models" >/dev/null 2>&1 && UP=$((UP+1))
  done
  [ "$UP" -eq "$NSERVERS" ] && break
  [ $(( $(date +%s) - T0 )) -lt "${SERVER_WAIT_S:-3600}" ] || { echo "ABORT: $UP/$NSERVERS servers ready"; exit 7; }
  sleep 10
done
echo "=== $NSERVERS vLLM server(s) ready after $(( $(date +%s) - T0 )) s; evaluating $SERVED"

PYTHONPATH="$REPO" "$SIM_PY" -m llm_finetune.rl_dataxp.rollout eval \
  --script "$H/scripts/$HS" --problems $PROBLEMS ${VARIANTS:+--variants $VARIANTS} \
  --servers "${URLS[@]}" --model "$SERVED" --label "$LABEL" --runs "${RUNS:-2}" --turns "${TURNS:-20}" \
  --workers "${WORKERS:-32}" --out "$OUT/$LABEL.episodes.jsonl" -- \
  --code "$H/dataxp_code_v3" "${NOTHINK[@]}" --native-prompt served --instruction-file "$H/configs/task_instruction_V2.txt" ${HARNESS_ARGS:-} \
  2>&1 | tee -a "$OUT/logs/eval.log"
rc=${PIPESTATUS[0]}
ELAPSED=$(( $(date +%s) - T0 ))
echo "=== eval rc=$rc; wall ${ELAPSED} s, GPU-hours $(awk "BEGIN{printf \"%.2f\", $ELAPSED*$NSERVERS/3600}")"
echo "DX_EVAL_DONE $(date '+%F %T %Z') rc=$rc"
exit $rc
