#!/bin/bash
# Multi-turn GRPO on dataxp, one node (Orchard by default; Bridges-2 via the knobs below): NVLLM vLLM servers (one GPU each, runtime LoRA
# loading) + a torchrun trainer on the next NTRAIN GPUs. Restart-safe: the trainer resumes from
# its newest complete checkpoint and the step's rollouts resume by episode key, so a requeued or
# resubmitted job continues where the last one stopped.
#
#   RUN_DIR=/project/community/wxu2/rl/runs/<name> CONFIG=configs/rl/dataxp_grpo_p2_datav4.yaml \
#   [NVLLM=2] [NTRAIN=2] [MAX_LORA_RANK=64] [SMOKE=1] [OVERRIDES="grpo.lr=3e-5 ..."] \
#   bash scripts/rl/orchard_dataxp_grpo.sh
#
# SMOKE=1 is the canary: 3 steps of 2 problems x 4 rollouts, 3 turns (overrides appended last).
# Knobs: BASE (site root for the defaults, Orchard /project/community/wxu2), REPO (this checkout; default
# $HOME/rl/repo: persistent files stay out of Orchard's periodically wiped /project/community), VLLM_VENV,
# TRAIN_PY, MODEL_PATH, CACHE_DIR, HF_HOME, MAX_MODEL_LEN (40960, as dx_eval_orchard.sh),
# GPU_UTIL (0.90), MAX_SEQS (64), SERVER_WAIT_S (3600).
# Exit status: the trainer's (0 done, 4 stopped by a guard, other = failure).
set -uo pipefail
B="${BASE:-/project/community/wxu2}"
REPO="${REPO:-$HOME/rl/repo}"
RUN_DIR="${RUN_DIR:?run directory}"
CONFIG="${CONFIG:?config yaml (relative to REPO or absolute)}"
[ "${CONFIG:0:1}" = / ] || CONFIG="$REPO/$CONFIG"
NVLLM=${NVLLM:-2}; NTRAIN=${NTRAIN:-2}
VENV="${VLLM_VENV:-$B/envs/vllm029}"
TRAIN_PY="${TRAIN_PY:-$B/envs/d27o/bin/python}"
MODEL_PATH="${MODEL_PATH:-$B/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
C="${CACHE_DIR:-$B/rl/cache}"
for f in "$CONFIG" "$TRAIN_PY" "$VENV/bin/vllm" "$MODEL_PATH/config.json"; do
  [ -e "$f" ] || { echo "ABORT: missing $f"; exit 2; }
done

export HF_HOME="${HF_HOME:-$B/hf}"; export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor
export CUDA_CACHE_PATH=$C/nv VLLM_CACHE_ROOT=$C/vllm FLASHINFER_WORKSPACE_BASE=$C/flashinfer MPLBACKEND=Agg
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
mkdir -p "$RUN_DIR/logs" "$TMPDIR" "$C/xdg"
cd "$REPO"
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none} restarts=${SLURM_RESTART_COUNT:-0}"
echo "=== repo=$REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo no-git)) config=$CONFIG run=$RUN_DIR"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -r -a DEVS <<< "$CUDA_VISIBLE_DEVICES"
else NG=$(nvidia-smi -L | wc -l); DEVS=(); for i in $(seq 0 $((NG-1))); do DEVS+=("$i"); done; fi
[ $((NVLLM + NTRAIN)) -le ${#DEVS[@]} ] || { echo "ABORT: NVLLM=$NVLLM + NTRAIN=$NTRAIN > ${#DEVS[@]} GPUs"; exit 2; }
TRAIN_DEVS=$(IFS=,; echo "${DEVS[*]:$NVLLM:$NTRAIN}")

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"; }; done
  sleep 15
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && kill -KILL -- "-$p" 2>/dev/null; done
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT

mapfile -t PORTS < <("$TRAIN_PY" -c 'import socket, sys
ss = [socket.socket() for _ in range(int(sys.argv[1]))]
for s in ss: s.bind(("127.0.0.1", 0))
for s in ss: print(s.getsockname()[1])' "$NVLLM")
URLS=()
for k in $(seq 0 $((NVLLM-1))); do
  CUDA_VISIBLE_DEVICES=${DEVS[k]} setsid "$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name base \
    --host 127.0.0.1 --port "${PORTS[k]}" --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-40960}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.90}" --max-num-seqs "${MAX_SEQS:-64}" \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --enable-lora --max-lora-rank "${MAX_LORA_RANK:-64}" --max-loras 2 --max-cpu-loras 2 \
    > "$RUN_DIR/logs/vllm_server_${k}_job${SLURM_JOB_ID:-x}.log" 2>&1 &
  PIDS[k]=$!
  URLS+=("\"http://127.0.0.1:${PORTS[k]}/v1\"")
done
SERVERS="[$(IFS=,; echo "${URLS[*]}")]"
T0=$(date +%s)
while :; do
  UP=0
  for k in $(seq 0 $((NVLLM-1))); do
    kill -0 "${PIDS[k]}" 2>/dev/null || { echo "ABORT: vLLM server $k exited"; tail -40 "$RUN_DIR/logs/vllm_server_${k}_job${SLURM_JOB_ID:-x}.log"; exit 7; }
    curl -sf "http://127.0.0.1:${PORTS[k]}/v1/models" >/dev/null 2>&1 && UP=$((UP+1))
  done
  [ "$UP" -eq "$NVLLM" ] && break
  [ $(( $(date +%s) - T0 )) -lt "${SERVER_WAIT_S:-3600}" ] || { echo "ABORT: $UP/$NVLLM servers ready"; exit 7; }
  sleep 10
done
echo "=== $NVLLM vLLM server(s) ready after $(( $(date +%s) - T0 )) s on GPUs ${DEVS[*]:0:$NVLLM}: $SERVERS"

SMOKE_OV=()
[ -n "${SMOKE:-}" ] && SMOKE_OV=(grpo.steps=3 rollout.problems_per_step=2 rollout.group_size=4 rollout.horizon=3 \
  rollout.keep_full_every=1 guards.enabled=false)
echo "=== trainer on GPUs $TRAIN_DEVS ($NTRAIN rank(s)) $(date '+%T')"
CUDA_VISIBLE_DEVICES=$TRAIN_DEVS PYTHONPATH="$REPO" "$TRAIN_PY" -m torch.distributed.run --standalone \
  --nproc_per_node "$NTRAIN" -m llm_finetune.rl_dataxp.trainer --config "$CONFIG" \
  run_dir="$RUN_DIR" "rollout.servers=$SERVERS" ${OVERRIDES:-} "${SMOKE_OV[@]}" \
  2>&1 | tee -a "$RUN_DIR/logs/trainer.log"
rc=${PIPESTATUS[0]}
ELAPSED=$(( $(date +%s) - T0 ))
echo "=== trainer rc=$rc; wall ${ELAPSED} s, GPU-hours $(awk "BEGIN{printf \"%.2f\", $ELAPSED*($NVLLM+$NTRAIN)/3600}")"
echo "DX_GRPO_DONE $(date '+%F %T %Z') rc=$rc"
exit $rc
