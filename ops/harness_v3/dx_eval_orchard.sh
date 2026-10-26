#!/bin/bash
# dx_eval_orchard.sh (2026-10-23): the dataxp policy-evaluation driver (dx_eval.sh -> dx_eval_v3.sh on
# drc) ported to Orchard (SLURM, H100 nodes). Same client script and code copy as drc's v3 harness
# (/storage/wxu/corpus/harness, copied verbatim to $H), same ARMS / EP_* semantics; the differences:
#   * paths under /project/community/wxu2; the model is served from its local HF snapshot (offline);
#   * one vLLM server per allocated GPU (NSERVERS, default every GPU SLURM gave the job); the
#     (arm, variant) pairs are distributed round-robin over the servers and run in parallel, so there
#     is one episodes file per arm AND variant: $OUT/<label>_<variant>_episodes.episodes.jsonl, with
#     the client's stdout in $OUT/<label>_<variant>.client.log and each server's log in
#     $OUT/vllm_server_<k>.log;
#   * only the episodes part (the states / fresh / probe parts need DX tiers that are not on Orchard);
#   * routing as in dx_eval_v3.sh (dataxp.domain.speaks_verbs asked of the code copy): native-verb
#     variants -> problems_v3_native.jsonl, verb-surface v2 variants -> problems_v2.jsonl; a mixed
#     EP_VARIANTS aborts (two problem files); plain pre-verb variants are not served here;
#   * PYTHONHASHSEED=0 always (metrics_after key order is set-derived; drc's acceptance runs use it),
#     VLLM_USE_FLASHINFER_SAMPLER=0 from the start (no nvcc on the nodes; T=0 is argmax either way);
#   * SKIP_DONE=1 (default): an (arm, variant) whose client log already ends in the harness summary
#     ("elapsed ... s | crashed ...") is skipped, so a requeued job redoes only the unfinished pairs
#     (the harness rewrites an episodes file from scratch, open('w')).
#
#   OUT=<dir> ARMS="base_nothink:base:nothink" EP_VARIANTS="grid2op photonics molopt engibench" \
#   [ADAPTERS="cot=/path/final ..."] [NSERVERS=4] bash /project/community/wxu2/corpus/harness/scripts/dx_eval_orchard.sh
# An ARM is label:served-model:think|nothink; served-model 'base' means the un-tuned Qwen3.8-27B.
# Knobs (env; defaults as dx_eval.sh / dx_eval_v3.sh unless noted): EP_N (60), EP_TURNS (20), EP_ARMS,
# EP_OBS, EP_PROBLEMS, EP_PARAMLINE, EP_GUARD, EP_RECORD, EP_SHARD, EP_START, EP_NATIVE_PROMPT,
# INSTR_FILE, EVAL_SCRIPT, CODE_OVERRIDE, CLIENT_PY_OVERRIDE, V3PY, VLLM_VENV (first of envs/vllm_o,
# envs/vllm_eval, envs/vllm029 that has bin/vllm), MODEL_PATH, MAX_MODEL_LEN (40960, the README's
# native-run value), GPU_UTIL (0.90), MAX_SEQS (32), TP_FLAG, WORKERS (16 per (arm, variant) pair;
# dx_eval ran 32 per arm over all variants), THINK_WORKERS (8), THINK_MAXT (2048), NOTHINK_MAXT (1024),
# BASE_THINK_MAXT (20000), SERVER_WAIT_S (3600), SKIP_DONE (1).
# Exits non-zero if a server dies or any client step fails; prints DX_EVAL_DONE ... failures=N and the
# GPU-hours (elapsed x NSERVERS) at the end.
set -uo pipefail
B=/project/community/wxu2
H="${H:-$B/corpus/harness}"
C="${CACHE_DIR:-$B/corpus/cache}"
OUT="${OUT:?output dir}"
ARMS="${ARMS:?label:model:think ...}"
EP_VARIANTS="${EP_VARIANTS:?variants (one kind: native-verb or verb-surface)}"
VENV="${VLLM_VENV:-}"
if [ -z "$VENV" ]; then
  for cand in $B/envs/vllm_o $B/envs/vllm_eval $B/envs/vllm029; do
    [ -x "$cand/bin/vllm" ] && { VENV=$cand; break; }
  done
fi
[ -n "$VENV" ] && [ -x "$VENV/bin/vllm" ] || { echo "ABORT: no vllm under VLLM_VENV=${VENV:-<none>}"; exit 2; }
MODEL_PATH="${MODEL_PATH:-$B/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
[ -f "$MODEL_PATH/config.json" ] || { echo "ABORT: no model snapshot at $MODEL_PATH"; exit 2; }
V3PY="${V3PY:-$B/envs/corpus_o/bin/python}"
CLIENT_PY="${CLIENT_PY_OVERRIDE:-$B/envs/corpus_o/bin/python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$H/scripts/dataxp_policy_eval_v3.py}"
CODE="${CODE_OVERRIDE:-$H/dataxp_code_v3}"
INSTR="${INSTR_FILE:-$H/configs/task_instruction_V2.txt}"   # passed through; ignored for verb-spelled variants
[ -f "$EVAL_SCRIPT" ] || { echo "ABORT: no evaluation script at $EVAL_SCRIPT"; exit 2; }
[ -d "$CODE/dataxp" ] || { echo "ABORT: no dataxp package under $CODE"; exit 2; }
[ -x "$CLIENT_PY" ] || { echo "ABORT: no client python at $CLIENT_PY"; exit 2; }
[ -f "$INSTR" ] || { echo "ABORT: no instruction file at $INSTR"; exit 2; }

# routing: NATIVE=1 when EP_VARIANTS names a native-verb variant, SURFACE=1 a verb-surface v2 variant
KIND=$("$V3PY" -c 'import sys; sys.path.insert(0, sys.argv[1])
from dataxp.domains import get_domain
from dataxp.domain import speaks_verbs
ds = [get_domain(v) for v in sys.argv[2:]]
nat = [bool(getattr(d, "native_moves", False)) for d in ds]
print(int(any(nat)), int(any(speaks_verbs(d) and not n for d, n in zip(ds, nat))))' "$CODE" $EP_VARIANTS) \
  || { echo "ABORT: could not classify EP_VARIANTS='$EP_VARIANTS' with $V3PY and $CODE"; exit 2; }
NATIVE=${KIND% *}; SURFACE=${KIND#* }
if [ "$NATIVE" = 1 ] && [ "$SURFACE" = 1 ]; then
  echo "ABORT: EP_VARIANTS='$EP_VARIANTS' mixes native-verb and verb-surface variants (two problem files); run them separately"; exit 2
fi
if [ "$NATIVE" = 1 ]; then EP_PROBLEMS="${EP_PROBLEMS:-$H/problems_v3_native.jsonl}"
elif [ "$SURFACE" = 1 ]; then EP_PROBLEMS="${EP_PROBLEMS:-$H/problems_v2.jsonl}"
else echo "ABORT: EP_VARIANTS='$EP_VARIANTS' names no verb-spelled variant; the plain SET_PARAM harness is not on Orchard"; exit 2; fi
[ -f "$EP_PROBLEMS" ] || { echo "ABORT: no problems file at $EP_PROBLEMS"; exit 2; }

export HF_HOME=$B/hf HF_HUB_CACHE=$B/hf/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg VLLM_CACHE_ROOT=$C/vllm TRITON_CACHE_DIR=$C/triton_vllm
export TORCHINDUCTOR_CACHE_DIR=$C/inductor_vllm FLASHINFER_WORKSPACE_BASE=$C/flashinfer CUDA_CACHE_PATH=$C/nv
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg VLLM_USE_FLASHINFER_SAMPLER=0
export DX_ORIG_SCRIPTS=$H/scripts DATAXP_MOLOPT_ORACLE_DIR="${DATAXP_MOLOPT_ORACLE_DIR:-$H/molopt_oracle}"
mkdir -p "$OUT" "$TMPDIR" "$C/xdg"
cd "$H"
T_START=$(date +%s)
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none} partition=${SLURM_JOB_PARTITION:-none}"
echo "=== script=$EVAL_SCRIPT client=$CLIENT_PY native=$NATIVE verb_surface=$SURFACE"
echo "=== code=$CODE instr=$INSTR obs=${EP_OBS:-v1} problems=$EP_PROBLEMS max_model_len=${MAX_MODEL_LEN:-40960} param_line=${EP_PARAMLINE:-no}"
echo "=== vllm=$VENV ($("$VENV/bin/python" -c 'import vllm, torch; print(vllm.__version__, "torch", torch.__version__)' 2>/dev/null)) model=$MODEL_PATH"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
"$CLIENT_PY" -c "import sys, numpy, scipy; print('client python', sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__)" \
  || { echo "ABORT: client env lacks numpy/scipy"; exit 3; }

# GPUs: SLURM renumbers the allocated devices 0..n-1 for the job (CUDA_VISIBLE_DEVICES, or nvidia-smi's
# list when the variable is unset); server k takes the k-th of them
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -r -a DEVS <<< "$CUDA_VISIBLE_DEVICES"
else NG=$(nvidia-smi -L 2>/dev/null | wc -l); DEVS=(); for i in $(seq 0 $((NG-1))); do DEVS+=("$i"); done; fi
NSERVERS=${NSERVERS:-${#DEVS[@]}}
[ "$NSERVERS" -ge 1 ] && [ "$NSERVERS" -le "${#DEVS[@]}" ] || { echo "ABORT: NSERVERS=$NSERVERS but ${#DEVS[@]} GPU(s) visible (${DEVS[*]:-none})"; exit 2; }

LORA=(); NL=0
for kv in ${ADAPTERS:-}; do
  [ -f "${kv#*=}/adapter_config.json" ] || { echo "ABORT: no adapter at ${kv#*=}"; exit 5; }
  LORA+=("$kv"); NL=$((NL+1))
done
EXTRA=(); [ $NL -gt 0 ] && EXTRA=(--enable-lora --max-lora-rank 64 --max-loras $NL --lora-modules "${LORA[@]}")

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; }; done
  sleep 20
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && kill -KILL -- "-$p" 2>/dev/null; done
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT

# NSERVERS distinct free ports, chosen while every socket is still open (so no two coincide)
mapfile -t PORTS < <("$CLIENT_PY" -c 'import socket, sys
n = int(sys.argv[1]); ss = []
for _ in range(n):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); ss.append(s)
for s in ss: print(s.getsockname()[1])' "$NSERVERS")
echo "=== $NSERVERS server(s): GPUs ${DEVS[*]:0:$NSERVERS}, ports ${PORTS[*]}"
for k in $(seq 0 $((NSERVERS-1))); do
  CUDA_VISIBLE_DEVICES=${DEVS[k]} setsid "$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name base --host 127.0.0.1 --port "${PORTS[k]}" \
    --dtype bfloat16 --max-model-len ${MAX_MODEL_LEN:-40960} --gpu-memory-utilization ${GPU_UTIL:-0.90} --max-num-seqs ${MAX_SEQS:-32} ${TP_FLAG:-} \
    --limit-mm-per-prompt '{"image":0,"video":0}' "${EXTRA[@]}" > "$OUT/vllm_server_$k.log" 2>&1 &
  PIDS[k]=$!
done
alive() { kill -0 "${PIDS[$1]}" 2>/dev/null; }
ready() { curl -sf "http://127.0.0.1:${PORTS[$1]}/v1/models" >/dev/null 2>&1; }
T0=$(date +%s); UP=0
while :; do
  UP=0
  for k in $(seq 0 $((NSERVERS-1))); do
    alive $k || { echo "ABORT: server $k (GPU ${DEVS[k]}) exited"; tail -40 "$OUT/vllm_server_$k.log"; exit 7; }
    ready $k && UP=$((UP+1))
  done
  [ "$UP" -eq "$NSERVERS" ] && break
  [ $(( $(date +%s) - T0 )) -lt ${SERVER_WAIT_S:-3600} ] || { echo "ABORT: only $UP/$NSERVERS servers ready after ${SERVER_WAIT_S:-3600} s"; for k in $(seq 0 $((NSERVERS-1))); do tail -5 "$OUT/vllm_server_$k.log"; done; exit 7; }
  sleep 10
done
for k in $(seq 0 $((NSERVERS-1))); do
  for kv in ${ADAPTERS:-}; do
    curl -s "http://127.0.0.1:${PORTS[k]}/v1/models" | grep -q "\"id\":\"${kv%%=*}\"" || { echo "ABORT: adapter ${kv%%=*} not served by server $k"; exit 7; }
  done
done
echo "=== all servers ready after $(( $(date +%s) - T0 )) s: $(curl -s "http://127.0.0.1:${PORTS[0]}/v1/models" | tr -d '\n' | head -c 200)"

# (arm, variant) tasks, round-robin over the servers
TASKS=()
for arm in $ARMS; do
  IFS=: read -r label model think <<< "$arm"
  [[ " ${EP_ARMS:-$label} " == *" $label "* ]] || continue
  for v in $EP_VARIANTS; do TASKS+=("$label:$model:$think:$v"); done
done
echo "=== ${#TASKS[@]} (arm, variant) task(s): ${TASKS[*]}"
rm -f "$OUT/failures.txt"
run_task() {   # label model think variant server-index
  local label=$1 model=$2 think=$3 variant=$4 k=$5
  local url="http://127.0.0.1:${PORTS[k]}/v1"
  local TFLAG=() MAXT=${THINK_MAXT:-2048} WK=${WORKERS:-16}
  # NOTHINK_MAXT: DX-23's v2 observations make the no-think reply longer; 1024 cut 30/643 turns (DRC j158)
  if [ "$think" = nothink ]; then TFLAG=(--no-think); MAXT=${NOTHINK_MAXT:-1024}; fi
  # the un-tuned model thinks long (DX-18: ~40% of replies past 8k tokens); few such requests fit the KV cache
  [ "$model" = base ] && [ "$think" = think ] && { MAXT=${BASE_THINK_MAXT:-20000}; WK=${THINK_WORKERS:-8}; }
  local out="$OUT/${label}_${variant}_episodes" log="$OUT/${label}_${variant}.client.log"
  if [ "${SKIP_DONE:-1}" = 1 ] && [ -s "$out.episodes.jsonl" ] && grep -q '^elapsed ' "$log" 2>/dev/null; then
    echo "[$label $variant] already complete ($(wc -l < "$out.episodes.jsonl") episodes) -- skipped"; return 0
  fi
  alive $k || { echo "!!! server $k is gone before $label $variant"; echo "$label $variant server_gone" >> "$OUT/failures.txt"; return 7; }
  echo "################ $label $variant episodes (n=${EP_N:-60}, ${EP_TURNS:-20} turns, T=0, max_tokens $MAXT, workers $WK, server $k)  $(date +%T)"
  "$CLIENT_PY" "$EVAL_SCRIPT" episodes --code "$CODE" --model "$model" --base-url "$url" --max-tokens $MAXT "${TFLAG[@]}" \
      --policy llm --policy-label "$label" --n ${EP_N:-60} --turns ${EP_TURNS:-20} --variants $variant \
      ${EP_START:+--start-index $EP_START} ${EP_SHARD:+--shard $EP_SHARD} ${EP_RECORD:+--record-prompts} \
      ${EP_OBS:+--obs $EP_OBS} --problems-file "$EP_PROBLEMS" ${EP_PARAMLINE:+--param-line} ${EP_GUARD:+--loop-guard} \
      ${EP_NATIVE_PROMPT:+--native-prompt $EP_NATIVE_PROMPT} \
      --temperature 0 --instruction-file "$INSTR" --workers $WK --out "$out" > "$log" 2>&1
  local rc=$?
  grep -v -i warning "$log" | tail -8 | sed "s/^/[$label $variant] /"
  [ "$rc" -eq 0 ] || { echo "!!! client step failed rc=$rc: $label $variant ($(date +%T))"; echo "$label $variant rc=$rc" >> "$OUT/failures.txt"; }
  return $rc
}
RUNNERS=()
for k in $(seq 0 $((NSERVERS-1))); do
  (
    i=$k
    while [ $i -lt ${#TASKS[@]} ]; do
      IFS=: read -r label model think v <<< "${TASKS[i]}"
      run_task "$label" "$model" "$think" "$v" "$k"
      i=$((i+NSERVERS))
    done
  ) &
  RUNNERS+=($!)
done
wait "${RUNNERS[@]}"
FAIL=0; [ -s "$OUT/failures.txt" ] && { FAIL=1; echo "=== failures:"; cat "$OUT/failures.txt"; }
ELAPSED=$(( $(date +%s) - T_START ))
echo "=== wall $ELAPSED s, GPU-hours $(awk "BEGIN{printf \"%.2f\", $ELAPSED*$NSERVERS/3600}") ($NSERVERS GPU(s))"
echo; echo "DX_EVAL_DONE $(date '+%F %T %Z') failures=$FAIL"
exit $FAIL
