#!/bin/bash
# con1 stage 2 + control on one Orchard node (docs/consequence_round_con1.md section 11, note of 11-25 16:55):
#   control  pref1's final adapter      -> the policy rows (ownrows_own, 8 families), one epoch
#   stage2   the round-1 winner adapter -> the same policy rows, same steps, same lr
# Two 2-GPU trainings in parallel (checkpoints on the node's /tmp; advanced is not preempted), final adapters copied to
# $S2/<arm>/final; then vLLM servers with both adapters: the skill questions (zero-shot, the winner's answer format) and
# the real task (pref1's benchmark settings: v3r --open-rules, thinking on, T = 0, 20 turns, 2 runs, the 8 training
# families' benchmark problems).
#   S2=/project/community/wxu2/con1/stage2 bash scripts/con1/orchard_stage2.sh
# $S2/WINNER: line 1 the winner's adapter dir, line 2 its answer format (written after round 1 is scored).
set -uo pipefail
B=/project/community/wxu2
REPO="${REPO:-$B/con1/repo}"
S2="${S2:-$B/con1/stage2}"
TMPRUN="/tmp/con1_s2_${SLURM_JOB_ID:-x}"
PREF1=$B/rl/runs/pref1_dx38mtR_20261123/final
VENV=$B/rl/envs/vllm
TRAIN_PY=$B/rl/envs/train/bin/python
SIM_PY=$B/rl/envs/sim/bin/python
MODEL_PATH=$B/rl/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
C=$B/rl/cache
H=$HOME/rl/harness
mkdir -p "$S2/logs" "$TMPRUN"
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none}"
T0=$(date +%s)
while [ ! -s "$S2/WINNER" ]; do
  [ $(( $(date +%s) - T0 )) -lt 10800 ] || { echo "ABORT: no $S2/WINNER after 3 h"; exit 2; }
  sleep 60
done
WIN="$(sed -n 1p "$S2/WINNER")"; WFMT="$(sed -n 2p "$S2/WINNER")"
[ -e "$WIN/adapter_config.json" ] || { echo "ABORT: winner adapter $WIN missing"; exit 2; }
echo "winner adapter $WIN (format $WFMT)"
# ---- phase 1: two trainings, 2 GPUs each
IFS=, read -r -a DEVS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
OVR="sft.save_every=10 sft.keep_last=1 sft.dev_every=20 sft.keep_adapter_every=100000"
PIDS=()
k=0
for spec in "control:$PREF1" "stage2:$WIN"; do
  lab=${spec%%:*}; init=${spec#*:}
  if [ -e "$S2/$lab/final/adapter_config.json" ]; then echo "$lab: final adapter exists"; k=$((k+2)); continue; fi
  ( CUDA_VISIBLE_DEVICES=${DEVS[k]},${DEVS[k+1]} NTRAIN=2 REPO="$REPO" RUN_DIR="$TMPRUN/$lab" CONFIG=configs/rl/con1_sft.yaml \
    ROWS="$S2/policy_train.jsonl" DEV="$S2/policy_dev.jsonl" OVERRIDES="$OVR policy.init_adapter=$init" \
    bash "$REPO/scripts/rl/orchard_sft_rows.sh" > "$S2/logs/train_$lab.log" 2>&1 ) &
  PIDS+=($!)
  echo "$lab training on GPUs ${DEVS[k]},${DEVS[k+1]} from $init"
  k=$((k+2)); sleep 120
done
for p in "${PIDS[@]:-}"; do [ -n "$p" ] && wait "$p"; done
for lab in control stage2; do
  if [ ! -e "$S2/$lab/final/adapter_config.json" ]; then
    [ -e "$TMPRUN/$lab/final/adapter_config.json" ] || { echo "ABORT: $lab has no final adapter"; tail -30 "$S2/logs/train_$lab.log"; exit 3; }
    mkdir -p "$S2/$lab/final" && cp -r "$TMPRUN/$lab/final/." "$S2/$lab/final/"
    cp "$TMPRUN/$lab/metrics.jsonl" "$S2/$lab/metrics.jsonl" 2>/dev/null
  fi
done
echo "=== $(date '+%F %T') training done"
# ---- phase 2: servers with both adapters
export HF_HOME=$B/rl/hf HF_HUB_CACHE=$B/rl/hf/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor
export CUDA_CACHE_PATH=$C/nv VLLM_CACHE_ROOT=$C/vllm FLASHINFER_WORKSPACE_BASE=$C/flashinfer
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0
SP=()
cleanup() {
  for p in "${SP[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"; }; done
  sleep 15
  for p in "${SP[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && kill -KILL -- "-$p" 2>/dev/null; done
}
trap cleanup EXIT
mapfile -t PORTS < <("$TRAIN_PY" -c 'import socket, sys
ss = [socket.socket() for _ in range(int(sys.argv[1]))]
for s in ss: s.bind(("127.0.0.1", 0))
for s in ss: print(s.getsockname()[1])' "${#DEVS[@]}")
URLS=()
for k in $(seq 0 $((${#DEVS[@]}-1))); do
  CUDA_VISIBLE_DEVICES=${DEVS[k]} setsid "$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name base \
    --host 127.0.0.1 --port "${PORTS[k]}" --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.95 \
    --max-num-seqs 64 --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}' --enable-lora \
    --max-lora-rank 64 --max-loras 2 --lora-modules "control=$S2/control/final" "stage2=$S2/stage2/final" \
    > "$S2/logs/vllm_$k.log" 2>&1 &
  SP[k]=$!
  URLS+=("http://127.0.0.1:${PORTS[k]}/v1")
done
T0=$(date +%s)
while :; do
  UP=0
  for k in $(seq 0 $((${#DEVS[@]}-1))); do
    kill -0 "${SP[k]}" 2>/dev/null || { echo "ABORT: vLLM server $k exited"; tail -30 "$S2/logs/vllm_$k.log"; exit 7; }
    curl -sf "${URLS[k]}/models" >/dev/null 2>&1 && UP=$((UP+1))
  done
  [ "$UP" -eq "${#DEVS[@]}" ] && break
  [ $(( $(date +%s) - T0 )) -lt 3600 ] || { echo "ABORT: servers not ready"; exit 7; }
  sleep 10
done
echo "=== $(date '+%F %T') servers ready"
cd "$REPO"
for lab in control stage2; do
  PYTHONPATH="$REPO" "$TRAIN_PY" scripts/con1/skill_client.py --queries $B/con1/data/queries_v2.jsonl --tokenizer "$MODEL_PATH" \
    --base-url "${URLS[@]}" --model "$lab" --label "$lab" --target "$WFMT" --out "$S2/preds_$lab.jsonl" \
    --max-tokens 4096 --workers 64 --passes zero --groups heldout_problem >> "$S2/logs/client_$lab.log" 2>&1
done
echo "=== $(date '+%F %T') skill questions done; real task"
export DX_ORIG_SCRIPTS=$H/scripts DATAXP_SPELLING=multitool DATAXP_MOLOPT_ORACLE_DIR=$H/molopt_oracle MPLBACKEND=Agg
mkdir -p "$S2/bench"
for lab in control stage2; do
  PYTHONPATH="$REPO" "$SIM_PY" -m llm_finetune.rl_dataxp.rollout eval \
    --script "$H/scripts/dataxp_policy_eval_v3r.py" --problems "$H/problems_v2.jsonl" "$HOME/rl/eval/problems_v3_native.jsonl" \
    --variants engibench grid2op photonics circuit2 frame3d2 grid2 seir2 traffic2 --servers "${URLS[@]}" \
    --model "$lab" --label "$lab" --runs 2 --turns 20 --workers 64 --out "$S2/bench/$lab.episodes.jsonl" -- \
    --code "$H/dataxp_code_v3" --native-prompt served --instruction-file "$H/configs/task_instruction_V2.txt" --open-rules \
    >> "$S2/logs/bench_$lab.log" 2>&1 &
done
wait
echo "CON1_STAGE2_DONE $(date '+%F %T %Z')"
