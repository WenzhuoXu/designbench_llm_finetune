#!/bin/bash
# Data for a screened supervised round (scripts/rl_screen/): the designs the policy reaches, k of its
# own replies at each, every reply checked in the simulator, one training row per state whose best
# reply passes the screen. One node: NSERVERS vLLM servers (one GPU each) serve the adapter.
#
#   OUT=/home/wxu2/rl/screen/<name> LABEL=dx38mtR ADAPTER=/path/to/peft_dir \
#   ROLLOUTS=/project/community/wxu2/rl/runs/<run>/rollouts POOL=/home/wxu2/rl/problems/train_p2_datav4.jsonl \
#   [VARIANTS="photonics grid2op engibench"] [COLLECT_VARIANTS="photonics grid2op engibench"] [K=8] \
#   [TEMPERATURE=0.8] [FIRST="photonics grid2op engibench"] bash scripts/rl/orchard_screen_data.sh
# VARIANTS: only the stored rollouts of these families give states (default: every family).
#
# Stages (each is skipped when its output is complete, so a restarted job goes on where it stopped):
#   states_run      the states of the stored rollouts under ROLLOUTS (policy_states.py)
#   collect         new evaluation-style episodes (T=0, 20 turns, no completion cap) of the adapter on
#                   every POOL problem of COLLECT_VARIANTS, with the harness's refusal note
#                   (dataxp_policy_eval_v3n.py --reject-note): the note only moves the episode on;
#                   the states' prompts are rebuilt without it. COLLECT_VARIANTS="" skips the stage.
#   states          both state files merged, one line per distinct state
#   samples         K replies per state (sample_states.py), families in FIRST first
#   scored          every reply applied and scored in the simulator (score_states.py); the servers
#                   are stopped before this stage
#   rows            rows_gain.jsonl (the best of K that raises the potential) and rows_pct90.jsonl
#                   (the dx38 screen: at or above the 90th percentile of the legal moves, and a gain)
# EXCLUDE (default: the two evaluation problem files) is checked against every episode's problem.
# Knobs as orchard_dataxp_eval.sh: BASE, REPO, VLLM_VENV, SIM_PY, MODEL_PATH, CACHE_DIR, MAX_MODEL_LEN,
# GPU_UTIL, MAX_SEQS, MAX_LORA_RANK, NSERVERS (4), HARNESS; WORKERS (160, collection episodes in
# flight), SAMPLE_WORKERS (40, states in flight, up to K - 1 requests each), CPU_WORKERS (the job's
# CPUs minus 8). A smoke run:
# STATES_LIMIT (episodes per rollout file), SAMPLE_LIMIT (states sampled), a short POOL.
set -uo pipefail
B="${BASE:-/project/community/wxu2}"
REPO="${REPO:-$HOME/rl/repo}"
OUT="${OUT:?output dir}"
LABEL="${LABEL:?the served name of the adapter}"
ADAPTER="${ADAPTER:?adapter dir}"
ROLLOUTS="${ROLLOUTS:?rollouts dir of the run whose states are used}"
POOL="${POOL:?training problem pool, one variant + spec per line}"
NSERVERS=${NSERVERS:-4}
VENV="${VLLM_VENV:-$B/rl/envs/vllm}"
SIM_PY="${SIM_PY:-$B/rl/envs/sim/bin/python}"
MODEL_PATH="${MODEL_PATH:-$B/rl/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
H="${HARNESS:-$HOME/rl/harness}"
C="${CACHE_DIR:-$B/rl/cache}"
EXCLUDE="${EXCLUDE:-$HOME/rl/eval/problems_v3_native.jsonl $H/problems_v2.jsonl}"
COLLECT_VARIANTS="${COLLECT_VARIANTS-photonics grid2op engibench}"
FIRST="${FIRST-photonics grid2op engibench}"
K=${K:-8}
CPU_WORKERS=${CPU_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-32} > 16 ? ${SLURM_CPUS_PER_TASK:-32} - 8 : 8 ))}
S="$REPO/scripts/rl_screen"
for f in "$VENV/bin/vllm" "$SIM_PY" "$MODEL_PATH/config.json" "$H/scripts/dataxp_policy_eval_v3.py" \
         "$H/scripts/dataxp_policy_eval_v3n.py" "$H/configs/task_instruction_V2.txt" "$POOL" $EXCLUDE \
         "$ADAPTER/adapter_config.json" "$ROLLOUTS" "$S/policy_states.py"; do
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
mkdir -p "$OUT/logs" "$OUT/collect" "$TMPDIR" "$C/xdg"
cd "$REPO"
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none} restarts=${SLURM_RESTART_COUNT:-0}"
echo "=== repo=$REPO label=$LABEL adapter=$ADAPTER rollouts=$ROLLOUTS out=$OUT k=$K collect='$COLLECT_VARIANTS' first='$FIRST' cpu_workers=$CPU_WORKERS"

# one thread per simulator, as the rollouts ran
sim() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$REPO" "$SIM_PY" "$@"; }
stage_done() { [ -e "$OUT/$1.done" ]; }
mark() { date '+%F %T %Z' > "$OUT/$1.done"; }

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"; }; done
  sleep 15
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && kill -KILL -- "-$p" 2>/dev/null; done
  PIDS=()
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT

URLS=()
start_servers() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -r -a DEVS <<< "$CUDA_VISIBLE_DEVICES"
  else NG=$(nvidia-smi -L | wc -l); DEVS=(); for i in $(seq 0 $((NG-1))); do DEVS+=("$i"); done; fi
  [ "$NSERVERS" -le ${#DEVS[@]} ] || { echo "ABORT: NSERVERS=$NSERVERS > ${#DEVS[@]} GPUs"; exit 2; }
  mapfile -t PORTS < <("$SIM_PY" -c 'import socket, sys
ss = [socket.socket() for _ in range(int(sys.argv[1]))]
for s in ss: s.bind(("127.0.0.1", 0))
for s in ss: print(s.getsockname()[1])' "$NSERVERS")
  for k in $(seq 0 $((NSERVERS-1))); do
    CUDA_VISIBLE_DEVICES=${DEVS[k]} setsid "$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name base \
      --host 127.0.0.1 --port "${PORTS[k]}" --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "${GPU_UTIL:-0.95}" --max-num-seqs "${MAX_SEQS:-64}" \
      --limit-mm-per-prompt '{"image":0,"video":0}' --enable-lora --max-lora-rank "${MAX_LORA_RANK:-64}" \
      --lora-modules "$LABEL=$ADAPTER" > "$OUT/logs/vllm_server_${k}_job${SLURM_JOB_ID:-x}.log" 2>&1 &
    PIDS[k]=$!
    URLS+=("http://127.0.0.1:${PORTS[k]}/v1")
  done
}
wait_servers() {
  local T0=$(date +%s) UP k
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
  echo "=== $NSERVERS vLLM server(s) ready after $(( $(date +%s) - T0 )) s"
}

NEED_GPU=0
{ [ -n "$COLLECT_VARIANTS" ] && ! stage_done collect; } && NEED_GPU=1
stage_done samples || NEED_GPU=1
[ "$NEED_GPU" = 1 ] && start_servers

# ---- states of the stored rollouts (CPU, while the servers load)
if ! stage_done states_run; then
  mapfile -t FILES < <(ls "$ROLLOUTS"/step_*.jsonl | "$SIM_PY" -c 'import os, sys
best = {}
for p in sys.stdin.read().split():
    step = os.path.basename(p)[5:10]
    if step not in best or not p.endswith(".compact.jsonl"):
        best[step] = p
print("\n".join(best[s] for s in sorted(best)))')
  echo "=== states_run: ${#FILES[@]} rollout files"
  sim "$S/policy_states.py" --episodes "${FILES[@]}" --script "$H/scripts/dataxp_policy_eval_v3.py" \
      --out "$OUT/states_run.jsonl" --exclude-specs $EXCLUDE --workers "$CPU_WORKERS" --limit "${STATES_LIMIT:-0}" \
      ${VARIANTS:+--variants $VARIANTS} 2>&1 \
      | grep -v -i "warn\|numba\|^[[:space:]]*$" | tee "$OUT/logs/states_run.log"
  grep -q POLICY_STATES_DONE "$OUT/logs/states_run.log" || { echo "ABORT: states_run failed"; exit 3; }
  mark states_run
fi

# ---- new episodes with the refusal note, on every pool problem of COLLECT_VARIANTS
if [ -n "$COLLECT_VARIANTS" ] && ! stage_done collect; then
  wait_servers
  sim -m llm_finetune.rl_dataxp.rollout eval \
    --script "$H/scripts/dataxp_policy_eval_v3n.py" --problems "$POOL" --variants $COLLECT_VARIANTS \
    --servers "${URLS[@]}" --model "$LABEL" --label "$LABEL" --runs 1 --turns "${TURNS:-20}" \
    --workers "${WORKERS:-160}" --out "$OUT/collect/$LABEL.episodes.jsonl" -- \
    --code "$H/dataxp_code_v3" --native-prompt served --instruction-file "$H/configs/task_instruction_V2.txt" --reject-note \
    2>&1 | tee -a "$OUT/logs/collect.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "ABORT: collect failed"; exit 4; }
  mark collect
fi
if [ -n "$COLLECT_VARIANTS" ] && ! stage_done states_collect; then
  sim "$S/policy_states.py" --episodes "$OUT/collect/$LABEL.episodes.jsonl" --script "$H/scripts/dataxp_policy_eval_v3.py" \
      --out "$OUT/states_collect.jsonl" --exclude-specs $EXCLUDE --workers "$CPU_WORKERS" 2>&1 \
      | grep -v -i "warn\|numba\|^[[:space:]]*$" | tee "$OUT/logs/states_collect.log"
  grep -q POLICY_STATES_DONE "$OUT/logs/states_collect.log" || { echo "ABORT: states_collect failed"; exit 3; }
  mark states_collect
fi
if ! stage_done states; then
  "$SIM_PY" - "$OUT/states.jsonl" "$OUT/states_run.jsonl" $([ -n "$COLLECT_VARIANTS" ] && echo "$OUT/states_collect.jsonl") <<'EOF' || { echo "ABORT: merge failed"; exit 3; }
import collections, json, os, sys
out, files = sys.argv[1], sys.argv[2:]
seen, n = set(), collections.Counter()
with open(out + ".tmp", "w", encoding="utf-8") as fo:
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["sid"] in seen:
                n["already in an earlier file"] += 1
                continue
            seen.add(r["sid"])
            n[r["variant"]] += 1
            fo.write(line if line.endswith("\n") else line + "\n")
os.replace(out + ".tmp", out)
print("=== states: %d distinct; %s" % (len(seen), dict(sorted(n.items()))))
EOF
  mark states
fi

# ---- K replies per state
if ! stage_done samples; then
  wait_servers
  "$SIM_PY" "$S/sample_states.py" --states "$OUT/states.jsonl" --out "$OUT/samples.jsonl" --servers "${URLS[@]}" \
      --model "$LABEL" --k "$K" --temperature "${TEMPERATURE:-0.8}" --workers "${SAMPLE_WORKERS:-40}" --limit "${SAMPLE_LIMIT:-0}" --first $FIRST \
      2>&1 | tee -a "$OUT/logs/samples.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" = 3 ]; then          # some replies failed: one more pass over what is missing
    wait_servers                  # aborts the job if a server is gone (the stage is not marked done)
    "$SIM_PY" "$S/sample_states.py" --states "$OUT/states.jsonl" --out "$OUT/samples.jsonl" --servers "${URLS[@]}" \
        --model "$LABEL" --k "$K" --temperature "${TEMPERATURE:-0.8}" --workers "${SAMPLE_WORKERS:-40}" --limit "${SAMPLE_LIMIT:-0}" --first $FIRST \
        2>&1 | tee -a "$OUT/logs/samples.log"
    rc=${PIPESTATUS[0]}
  fi
  [ "$rc" = 0 ] || [ "$rc" = 3 ] || { echo "ABORT: sampling failed rc=$rc"; exit 5; }
  [ "$rc" = 3 ] && echo "!!! some replies are still missing after two passes (see samples.log); their states keep the replies they have"
  mark samples
fi
cleanup                              # the GPUs are not needed from here on

# ---- every reply in the simulator
if ! stage_done scored; then
  sim "$S/score_states.py" --code "$H/dataxp_code_v3" --states "$OUT/states.jsonl" --samples "$OUT/samples.jsonl" \
      --out "$OUT/scored.jsonl" --workers "$CPU_WORKERS" 2>&1 | grep -v -i "warn\|numba\|^[[:space:]]*$" | tee "$OUT/logs/scored.log"
  grep -q SCORE_STATES_DONE "$OUT/logs/scored.log" || { echo "ABORT: scoring failed"; exit 6; }
  mark scored
fi

# ---- rows under the two screens
"$SIM_PY" "$S/build_rows.py" --states "$OUT/states.jsonl" --samples "$OUT/samples.jsonl" --scored "$OUT/scored.jsonl" \
    --out "$OUT/rows_gain.jsonl" --min-pct 0 --tier "${TIER:-scr1}_gain" > "$OUT/rows_gain.report.txt" || { echo "ABORT: rows failed"; exit 8; }
"$SIM_PY" "$S/build_rows.py" --states "$OUT/states.jsonl" --samples "$OUT/samples.jsonl" --scored "$OUT/scored.jsonl" \
    --out "$OUT/rows_pct90.jsonl" --min-pct 90 --tier "${TIER:-scr1}_pct90" > "$OUT/rows_pct90.report.txt" || { echo "ABORT: rows failed"; exit 8; }
cat "$OUT/rows_gain.report.txt"
tail -n 1 "$OUT/rows_pct90.report.txt"
mark rows
echo "DX_SCREEN_DATA_DONE $(date '+%F %T %Z')"
