#!/bin/bash
# dx_eval_v3.sh (2026-10-23): dx_eval.sh with the evaluation script, CODE_OVERRIDE, INSTR_FILE,
# EP_PROBLEMS and the client python as parameters. When EP_VARIANTS names a verb-spelled variant --
# a native-verb one (grid2op photonics molopt engibench) or a v2 one with a verb surface (the seven
# v2 domains since 2026-10-23, docs/dataxp_verb_surface.md); dataxp.domain.speaks_verbs decides,
# asked of the v3 code copy, so the routing follows the domain flag and not a list here -- the run
# takes the v3 harness under /storage/wxu/corpus/harness (dataxp_policy_eval_v3.py, dataxp_code_v3).
# Native variants add problems_v3_native.jsonl and the corpus env (rdkit / grid2op / ceviche);
# verb-surface v2 variants keep the v2 evaluation problems (/storage/wxu/sftprobe/problems_v2.jsonl,
# unchanged) and the client python of the original run. Otherwise the original paths, so an
# old-variant run is the run dx_eval.sh makes.
#   EVAL_SCRIPT=<path>  CODE_OVERRIDE=<dataxp package dir>  INSTR_FILE=<txt>  EP_PROBLEMS=<jsonl>
#   CLIENT_PY_OVERRIDE=<python>  EP_NATIVE_PROMPT=served|probe (v3 script only)  REPO=<collaborator repo>
#   V3PY=<python that imports the v3 code copy for the routing question>
# The verb-spelled task instruction is built by the v3 script itself (dataxp.ownrows.common.instruction_for);
# INSTR_FILE and EP_PARAMLINE are passed through and ignored for verb-spelled variants.
#
# DX-20 evaluation driver: one vLLM process (base + any LoRA adapters), then for each ARM:
#   states   one move at every dev_in and held-out thermal teacher state (T=0, and T=0.7 for the
#            parse-rate criterion), in the tier's exact training prompt
#   episodes DX-14's 60 fresh problems per domain x 7 domains, up to 20 moves, T=0
# Regret is computed afterwards on CPU from the states' .actions.jsonl (dataxp_regret.py).
#
#   SITE=drc|b2  OUT=<dir>  ADAPTERS="cot=/path/final nocot=/path/final"  \
#   ARMS="base_nothink:base:nothink base_think:base:think cot:cot:think"  bash scripts/dx_eval.sh
# An ARM is label:served-model:think|nothink; served-model 'base' means the un-tuned Qwen3.8-27B.
# PARTS (default "states fresh probe episodes"), EP_ARMS (arms that also get episodes; default every
# arm), EP_N / EP_TURNS / EP_VARIANTS (episodes per domain / turns / domains; 60 / 20 / all seven),
# T07_REPS (T=0.7 samples, 4), WORKERS (32) and THINK_WORKERS (8, the un-tuned model thinking),
# EP_START / EP_SHARD / EP_RECORD (training-data draws for DX-21: first index, k/m shard, keep prompts),
# BASE_THINK_MAXT (20000). Exits non-zero if the server dies or any client step fails.
# probe = scripts/dx_perturb_probe.py run on $TIERS/probe_set_v0.jsonl (criterion 4; scored on CPU later).
set -uo pipefail
SITE="${SITE:?drc | b2}"
# v3 lives outside the collaborator's repo: name it explicitly (the states/probe parts and the F3
# instruction default are resolved against it)
REPO="${REPO:-/storage/wxu/sftprobe/repo}"
if [ "$SITE" = drc ]; then
  B=/storage/wenwenw; P=/storage/wxu/sftprobe
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPUS:-4}
  export HF_HOME=/storage/wxu/huggingface HF_HUB_CACHE=/storage/wxu/huggingface/hub
  VENV=$B/envs/vllm_drc; CLIENT_PY=$B/envs/d27_drc/bin/python
  CODE=$B/dataxp_run/code; RUN1_EVAL=$B/dbft/data/dx_v0/run1_eval; RUN1_FULL=/storage/wxu/dataxp/run1; TIERS=$B/dbft/data/dx_v0
  C=$P/cache
else
  B=/ocean/projects/mch250030p/$USER
  module load anaconda3/2024.10-1 >/dev/null 2>&1
  # vLLM's FlashInfer top-k/top-p sampler JIT-compiles at warm-up and needs nvcc (j79 died on
  # "Could not find nvcc"); the env's torch is cu13, so the cu13 toolkit
  module load cuda-h100/13.3.1 >/dev/null 2>&1 || module load cuda/12.6.1 >/dev/null 2>&1
  which nvcc >/dev/null 2>&1 && echo "nvcc: $(which nvcc)" || echo "WARNING: no nvcc on PATH"
  # ...and then ninja and a C++17 host compiler (j81 fell back after "No such file: 'ninja'"); use
  # vLLM's own top-k/top-p sampler here from the start. T=0 is argmax either way; only the T=0.7
  # random stream differs from DRC's.
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export HF_HOME=$B/hf_models HF_HUB_CACHE=$B/hf_models
  VENV=$B/envs/vllm_env; CLIENT_PY=$B/envs/d27_env/bin/python
  CODE=$B/dx/dataxp_code; RUN1_EVAL=$B/dx/run1_eval; RUN1_FULL=$B/dx/run1; TIERS=$B/dx/tiers_v0
  C=$B/tmp
fi
# DX-23 overrides (unset = DX-20 behaviour): CODE_OVERRIDE (another dataxp package, e.g. run2's v2 code),
# DEDUP_DIR (problems to avoid; run1 by default), INSTR_FILE, MAX_MODEL_LEN (v2 prompts reach ~15k tokens),
# EP_OBS (v1|v2), EP_PROBLEMS (a problems file shared by every arm), EP_PARAMLINE (non-empty: --param-line, DX-24),
# EP_GUARD (non-empty: --loop-guard, DX-25b)
RUN1_FULL="${DEDUP_DIR:-$RUN1_FULL}"
# v3 parameters (see the header). NATIVE=1 when EP_VARIANTS names a native-verb variant (native_moves),
# SURFACE=1 when it names a v2 variant with a verb surface (speaks_verbs and not native_moves); the
# v3 code copy answers, through V3PY. The two kinds take different problem files, so one run holds one kind.
V3="${V3:-/storage/wxu/corpus/harness}"
V3PY="${V3PY:-/storage/wxu/corpus/env/bin/python}"
NATIVE=0; SURFACE=0
if [ -n "${EP_VARIANTS:-}" ]; then
  KIND=$("$V3PY" -c 'import sys; sys.path.insert(0, sys.argv[1])
from dataxp.domains import get_domain
from dataxp.domain import speaks_verbs
ds = [get_domain(v) for v in sys.argv[2:]]
nat = [bool(getattr(d, "native_moves", False)) for d in ds]
print(int(any(nat)), int(any(speaks_verbs(d) and not n for d, n in zip(ds, nat))))' "$V3/dataxp_code_v3" $EP_VARIANTS) \
    || { echo "ABORT: could not classify EP_VARIANTS='$EP_VARIANTS' with $V3PY and $V3/dataxp_code_v3"; exit 2; }
  NATIVE=${KIND% *}; SURFACE=${KIND#* }
fi
if [ "$NATIVE" = 1 ] && [ "$SURFACE" = 1 ]; then
  echo "ABORT: EP_VARIANTS='$EP_VARIANTS' mixes native-verb and verb-surface variants, which take different problem files; run them separately"; exit 2
fi
if [ "$NATIVE" = 1 ] || [ "$SURFACE" = 1 ]; then
  EVAL_SCRIPT="${EVAL_SCRIPT:-$V3/scripts/dataxp_policy_eval_v3.py}"
  CODE="${CODE_OVERRIDE:-$V3/dataxp_code_v3}"
  INSTR_DEFAULT="$REPO/configs/dataxp/task_instruction_V2.txt"   # passed through; ignored for verb-spelled variants
  if [ "$NATIVE" = 1 ]; then
    EP_PROBLEMS="${EP_PROBLEMS:-$V3/problems_v3_native.jsonl}"
    CLIENT_PY="${CLIENT_PY_OVERRIDE:-/storage/wxu/corpus/env/bin/python}"
  else
    # verb-surface v2 variants: the v2 evaluation problems, unchanged, and the original client python
    EP_PROBLEMS="${EP_PROBLEMS:-/storage/wxu/sftprobe/problems_v2.jsonl}"
    CLIENT_PY="${CLIENT_PY_OVERRIDE:-$CLIENT_PY}"
  fi
else
  EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO/scripts/dataxp_policy_eval.py}"
  CODE="${CODE_OVERRIDE:-$CODE}"
  CLIENT_PY="${CLIENT_PY_OVERRIDE:-$CLIENT_PY}"
  INSTR_DEFAULT="$REPO/configs/dataxp/task_instruction_F3.txt"
fi
[ -f "$EVAL_SCRIPT" ] || { echo "ABORT: no evaluation script at $EVAL_SCRIPT"; exit 2; }
[ -d "$CODE/dataxp" ] || { echo "ABORT: no dataxp package under $CODE"; exit 2; }
OUT="${OUT:?output dir}"
# a FREE port, never a fixed one: GPU-shared puts several jobs on one node, and with a fixed port a
# later server failed to bind while its health check reached another job's server (09-30, w006: four
# jobs' clients on one server -- think-mode timeouts, and a LoRA arm would have hit a server without
# its adapter)
PORT="${PORT:-}"
ARMS="${ARMS:?label:model:think ...}"
PARTS="${PARTS:-states fresh probe episodes}"
INSTR="${INSTR_FILE:-$INSTR_DEFAULT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TMPDIR=$C XDG_CACHE_HOME=$C/xdg VLLM_CACHE_ROOT=$C/vllm TRITON_CACHE_DIR=$C/triton_vllm
export TORCHINDUCTOR_CACHE_DIR=$C/inductor_vllm FLASHINFER_WORKSPACE_BASE=$C/flashinfer
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
mkdir -p "$OUT" "$TMPDIR"
cd "$REPO"
echo "=== $(date '+%F %T %Z') site=$SITE host=$(hostname)  $(cat DEPLOYED 2>/dev/null)"
echo "=== script=$EVAL_SCRIPT client=$CLIENT_PY native=$NATIVE verb_surface=$SURFACE"
echo "=== code=$CODE dedup=$RUN1_FULL instr=$INSTR obs=${EP_OBS:-v1} problems=${EP_PROBLEMS:-generated} max_model_len=${MAX_MODEL_LEN:-32768} param_line=${EP_PARAMLINE:-no}"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
"$CLIENT_PY" -c "import numpy, scipy; print('client numpy', numpy.__version__, 'scipy', scipy.__version__)" || { echo "ABORT: client env lacks numpy/scipy"; exit 3; }

LORA=(); NL=0
for kv in ${ADAPTERS:-}; do
  [ -f "${kv#*=}/adapter_config.json" ] || { echo "ABORT: no adapter at ${kv#*=}"; exit 5; }
  LORA+=("$kv"); NL=$((NL+1))
done
EXTRA=(); [ $NL -gt 0 ] && EXTRA=(--enable-lora --max-lora-rank 64 --max-loras $NL --lora-modules "${LORA[@]}")
SERVER_PID=""
cleanup() { if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID"; sleep 20; kill -KILL -- "-$SERVER_PID" 2>/dev/null; fi; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT
[ -n "$PORT" ] || PORT=$("$CLIENT_PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
echo "=== port $PORT"
ours() {   # is the process listening on $PORT our server? (no ss on the node: trust the free port)
  command -v ss >/dev/null 2>&1 || return 0
  ss -ltnp 2>/dev/null | grep ":$PORT " | grep -q "pid=$SERVER_PID,"
}
start_server() {
  setsid "$VENV/bin/vllm" serve Qwen/Qwen3.8-27B --served-model-name base --host 127.0.0.1 --port $PORT \
    --dtype bfloat16 --max-model-len ${MAX_MODEL_LEN:-32768} --gpu-memory-utilization ${GPU_UTIL:-0.90} --max-num-seqs ${MAX_SEQS:-32} ${TP_FLAG:-} \
    --limit-mm-per-prompt '{"image":0,"video":0}' "${EXTRA[@]}" > "$OUT/vllm_server.log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 360); do
    kill -0 $SERVER_PID 2>/dev/null || return 1
    curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && ours && return 0; sleep 10
  done
  return 1
}
if ! start_server; then
  if grep -q -i "flashinfer" "$OUT/vllm_server.log"; then
    # last resort, logged: vLLM's own top-k/top-p sampler instead of FlashInfer's (T=0 is argmax
    # either way; only the T=0.7 random stream differs from the other site)
    echo "=== server failed in FlashInfer; retrying once with VLLM_USE_FLASHINFER_SAMPLER=0"
    mv "$OUT/vllm_server.log" "$OUT/vllm_server.failed1.log"; cleanup; SERVER_PID=""
    export VLLM_USE_FLASHINFER_SAMPLER=0
    start_server || { echo "ABORT: server exited"; tail -40 "$OUT/vllm_server.log"; exit 7; }
  else
    echo "ABORT: server exited"; tail -40 "$OUT/vllm_server.log"; exit 7
  fi
fi
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "ABORT: server never answered"; exit 7; }
ours || { echo "ABORT: the server on port $PORT is not ours"; exit 7; }
for kv in ${ADAPTERS:-}; do
  curl -s http://127.0.0.1:$PORT/v1/models | grep -q "\"id\":\"${kv%%=*}\"" || { echo "ABORT: adapter ${kv%%=*} not served"; exit 7; }
done
echo "=== server ready after ~$((i*10)) s: $(curl -s http://127.0.0.1:$PORT/v1/models | tr -d '\n' | head -c 300)"
URL=http://127.0.0.1:$PORT/v1

FAIL=0
run() {   # run one client step; the server must be alive before and the client must exit 0 after
  kill -0 "$SERVER_PID" 2>/dev/null && ours || { echo "ABORT: vLLM server is gone or not ours"; tail -30 "$OUT/vllm_server.log"; exit 7; }
  "$CLIENT_PY" "$@" 2>&1 | grep --line-buffered -v -i warning
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { echo "!!! client step failed rc=$rc: $*"; FAIL=1; }
}
for arm in $ARMS; do
  IFS=: read -r label model think <<< "$arm"
  TFLAG=(); MAXT=${THINK_MAXT:-2048}
  # NOTHINK_MAXT: DX-23's v2 observations make the no-think reply longer; 1024 cut 30/643 turns (DRC j158)
  if [ "$think" = nothink ]; then TFLAG=(--no-think); MAXT=${NOTHINK_MAXT:-1024}; fi
  # the un-tuned model thinks long: DX-18 saw ~40% of its replies pass 8k tokens; 20k + the longest
  # prompt (9.3k) fits max-model-len 32768. A reply cut off mid-think is counted as think_unclosed.
  WK=${WORKERS:-32}
  # ...and at most ~8 such requests fit the KV cache at once (275,797 tokens on an H100 at 0.90):
  # 32 in flight thrash (j95: 128/318 requests timed out)
  [ "$model" = base ] && [ "$think" = think ] && { MAXT=${BASE_THINK_MAXT:-20000}; WK=${THINK_WORKERS:-8}; }
  COMMON=(--code "$CODE" --model "$model" --base-url "$URL" --max-tokens $MAXT "${TFLAG[@]}" --policy llm --policy-label "$label")
  if [[ " $PARTS " == *" states "* ]]; then
    for t in dev_in heldout_thermal; do
      echo; echo "################ $label states $t T=0  $(date +%T)"
      run "$EVAL_SCRIPT" states "${COMMON[@]}" --run1 "$RUN1_EVAL" --tier "$TIERS/${t}_cot.jsonl" \
        --temperature 0 --workers $WK --out "$OUT/${label}_states_${t}_T0"
      # the plan pre-registered the T=0.7 parse rate over 4 samples
      for r in $(seq 1 ${T07_REPS:-4}); do
        suf=T07; [ "$r" -gt 1 ] && suf=T07r$r
        echo "################ $label states $t T=0.7 sample $r  $(date +%T)"
        run "$EVAL_SCRIPT" states "${COMMON[@]}" --run1 "$RUN1_EVAL" --tier "$TIERS/${t}_cot.jsonl" \
          --temperature 0.7 --workers $WK --out "$OUT/${label}_states_${t}_${suf}"
      done
    done
  fi
  if [[ " $PARTS " == *" fresh "* ]] && [ -s "$TIERS/fresh_states_v0.jsonl" ]; then
    # teacher states on DX-14's fresh problems (phi_select replayed, same renderer): ~6x the dev_in
    # problems for criterion 1
    echo; echo "################ $label states fresh T=0  $(date +%T)"
    run "$EVAL_SCRIPT" states "${COMMON[@]}" --run1 "$TIERS/fresh_run1" --tier "$TIERS/fresh_states_v0.jsonl" \
      --temperature 0 --workers $WK --out "$OUT/${label}_states_fresh_T0"
  fi
  if [[ " $PARTS " == *" probe "* ]] && [ -s "$TIERS/probe_set_v0.jsonl" ]; then
    echo; echo "################ $label probe (perturbation pairs, T=0)  $(date +%T)"
    run scripts/dx_perturb_probe.py run --set "$TIERS/probe_set_v0.jsonl" --model "$model" --base-url "$URL" \
      --max-tokens $MAXT "${TFLAG[@]}" --workers $WK --out "$OUT/${label}_probe"
    if [ "$think" = think ] && [ -s "$OUT/${label}_probe.replies.jsonl" ]; then
      echo "################ $label probe: reasoning swap (causal test)  $(date +%T)"
      run scripts/dx_perturb_probe.py swap --set "$TIERS/probe_set_v0.jsonl" --replies "$OUT/${label}_probe.replies.jsonl" \
        --model "$model" --base-url "$URL" --max-tokens 256 --workers $WK --out "$OUT/${label}_probe"
    fi
  fi
  if [[ " $PARTS " == *" episodes "* ]] && [[ " ${EP_ARMS:-$label} " == *" $label "* ]]; then
    echo; echo "################ $label episodes (7 x ${EP_N:-60}, ${EP_TURNS:-20} turns, T=0)  $(date +%T)"
    run "$EVAL_SCRIPT" episodes "${COMMON[@]}" --run1 "$RUN1_FULL" --n ${EP_N:-60} --turns ${EP_TURNS:-20} ${EP_VARIANTS:+--variants $EP_VARIANTS} \
      ${EP_START:+--start-index $EP_START} ${EP_SHARD:+--shard $EP_SHARD} ${EP_RECORD:+--record-prompts} \
      ${EP_OBS:+--obs $EP_OBS} ${EP_PROBLEMS:+--problems-file $EP_PROBLEMS} ${EP_PARAMLINE:+--param-line} ${EP_GUARD:+--loop-guard} ${EP_NATIVE_PROMPT:+--native-prompt $EP_NATIVE_PROMPT} \
      --temperature 0 --instruction-file "$INSTR" --workers $WK --out "$OUT/${label}_episodes"
  fi
done
echo; echo "DX_EVAL_DONE $(date '+%F %T %Z') failures=$FAIL"
exit $FAIL
