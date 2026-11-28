#!/bin/bash
# base_think.sh : the untrained base model with thinking on every family, clean questions (con1-2m-rows, 11-27; the
# parent session's request). Orchard 4 x H100, one vLLM server per GPU (the RL session's orchard_dataxp_eval.sh
# pattern), think_client.py over all four (a state's questions stay on one server):
#   pass A  queries_v3.jsonl (no move request), zero-shot, all groups (3,687 held-out-problem questions of the 8 trained
#           families + 1,400 unseen-family questions), think budget 4,000 -> $O/preds_base_v3_4k.jsonl
#   pass B  a fixed random 300-question subset (27 or 28 per family, all 11 families; random.Random('base-think-sub|0'),
#           written once to $O/subset_300.tsv), think budget 8,000 -> $O/preds_base_v3_8k_sub300.jsonl
# Base Qwen3.8-27B, no adapter, temperature 0, think to </think> or the budget, then '\n</think>\n\n' and the answer
# (think_client.py, as the checked T-base-4k-q run). Every answer is appended and flushed as it finishes, so a re-run
# (or a requeue) resumes where the last one stopped. No training, no adapter, nothing under con1_reserve or repo2m.
#   sbatch -p advanced --qos=adv_4gpu_qos -A ccm --gres=gpu:4 -c 32 --mem=256G -t 8:00:00 \
#     -o $HOME/con1_base_think/logs/base_think_%j.out \
#     --wrap "bash /project/community/wxu2/con1/repo2m_v3/scripts/con1/base_think.sh"
set -uo pipefail
B=/project/community/wxu2
R=${REPO:-$B/con1/repo2m_v3}
# outputs in $HOME: /project/community ran out of space (0 bytes free, 11-27 18:40, job 176529 failed on it); caches on
# the node's own /tmp (vLLM writes compile caches at start)
O=${OUT:-$HOME/con1_base_think}
Q=${QUERIES:-$B/con1/noconf/queries_v3.jsonl}
VENV=$B/rl/envs/vllm
MODEL=$B/rl/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
C=/tmp/base_think_cache_${SLURM_JOB_ID:-x}
mkdir -p $C/tmp $C/xdg $C/triton $C/inductor $C/nv $C/vllm $C/flashinfer
NSERVERS=${NSERVERS:-4}
mkdir -p "$O/logs"
export HF_HOME=$B/rl/hf HF_HUB_CACHE=$B/rl/hf/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$R
export TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor
export CUDA_CACHE_PATH=$C/nv VLLM_CACHE_ROOT=$C/vllm FLASHINFER_WORKSPACE_BASE=$C/flashinfer
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0
echo "=== $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-none} restarts=${SLURM_RESTART_COUNT:-0} repo=$R ($(cat $R/GIT_COMMIT)) queries=$Q ($(sha1sum < $Q | cut -c1-12)) out=$O arms=${ARMS:-default}"
[ -s "$Q" ] && [ -e "$MODEL/config.json" ] && [ -e "$VENV/bin/vllm" ] || { echo "ABORT: missing input"; exit 2; }
grep -q "YOUR MOVE" "$Q" && { echo "ABORT: $Q carries the move request"; exit 2; }
# the subset: once, deterministic
if [ ! -s "$O/subset_300.tsv" ]; then
  "$VENV/bin/python" - "$Q" "$O/subset_300.tsv" <<'EOF'
import collections, json, random, sys
by = collections.defaultdict(list)
for l in open(sys.argv[1]):
    q = json.loads(l)
    for j in range(len(q["moves"])):
        by[q["variant"]].append((q["sid"], j))
fams = sorted(by)
rng = random.Random("base-think-sub|0")
n = {f: 300 // len(fams) + (1 if k < 300 % len(fams) else 0) for k, f in enumerate(fams)}
out = []
for f in fams:
    out += rng.sample(sorted(by[f]), n[f])
open(sys.argv[2] + ".part", "w").write("".join("%s\t%d\n" % x for x in out))
import os; os.replace(sys.argv[2] + ".part", sys.argv[2])
print("subset: %d questions, per family %s" % (len(out), n))
EOF
fi
NG=$(nvidia-smi -L | wc -l); [ "$NSERVERS" -le "$NG" ] || { echo "ABORT: $NSERVERS servers, $NG GPUs"; exit 2; }
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -TERM -- "-$p" 2>/dev/null; done; sleep 15
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -KILL -- "-$p" 2>/dev/null; done; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
mapfile -t PORTS < <("$VENV/bin/python" -c 'import socket, sys
ss = [socket.socket() for _ in range(int(sys.argv[1]))]
for s in ss: s.bind(("127.0.0.1", 0))
for s in ss: print(s.getsockname()[1])' "$NSERVERS")
URLS=()
for k in $(seq 0 $((NSERVERS-1))); do
  CUDA_VISIBLE_DEVICES=$k setsid "$VENV/bin/vllm" serve "$MODEL" --served-model-name base --host 127.0.0.1 --port "${PORTS[k]}" \
    --dtype bfloat16 --max-model-len ${MAX_MODEL_LEN:-32768} --gpu-memory-utilization 0.92 --max-num-seqs ${MAX_SEQS:-48} \
    --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}' > "$O/logs/vllm_${k}_job${SLURM_JOB_ID:-x}.log" 2>&1 &
  PIDS[k]=$!
  URLS+=("http://127.0.0.1:${PORTS[k]}/v1")
done
T0=$(date +%s)
while :; do
  UP=0
  for k in $(seq 0 $((NSERVERS-1))); do
    kill -0 "${PIDS[k]}" 2>/dev/null || { echo "ABORT: vLLM server $k exited"; tail -30 "$O/logs/vllm_${k}_job${SLURM_JOB_ID:-x}.log"; exit 7; }
    curl -sf "${URLS[k]}/models" >/dev/null 2>&1 && UP=$((UP+1))
  done
  [ "$UP" -eq "$NSERVERS" ] && break
  [ $(( $(date +%s) - T0 )) -lt 3600 ] || { echo "ABORT: $UP/$NSERVERS servers ready"; exit 7; }
  sleep 10
done
echo "=== $(date '+%F %T') $NSERVERS servers ready after $(( $(date +%s) - T0 )) s"
W=$(( ${WORKERS_PER_SERVER:-48} * NSERVERS ))
run() {   # label budget out [extra args]
  local L=$1 BUD=$2 OUTF=$3; shift 3
  "$VENV/bin/python" "$R/scripts/con1/think_client.py" --no-defined-terms --queries "$Q" --tokenizer "$MODEL" --base-url "${URLS[@]}" \
    --model base --label "$L" --budget "$BUD" --answer-tokens 1024 --workers "$W" --out "$OUTF" "$@" \
    > "$O/logs/client_${L}_job${SLURM_JOB_ID:-x}.log" 2>&1
  echo "=== $(date '+%F %T') $L rc=$?: $(tail -1 "$O/logs/client_${L}_job${SLURM_JOB_ID:-x}.log")"
}
# ARMS="label:target:named:budget:all|sub ..." (11-28): answer form (targets.FORMATS), named metrics (1/0), think
# budget, every question or subset_300.tsv; default = the 11-27 run (facetm, metrics not named, 4,000 + 8,000 subset)
ARMS=${ARMS:-"base_v3_4k:facetm:0:4000:all base_v3_8k_sub300:facetm:0:8000:sub"}
FILES=()
for arm in $ARMS; do
  IFS=: read -r L TG NM BUD SUB <<< "$arm"
  EXTRA=(--target "$TG"); [ "$NM" = 1 ] && EXTRA+=(--named-metrics); [ "$SUB" = sub ] && EXTRA+=(--subset "$O/subset_300.tsv")
  run "$L" "$BUD" "$O/preds_$L.jsonl" "${EXTRA[@]}"
  FILES+=("$O/preds_$L.jsonl")
done
for f in "${FILES[@]}"; do
  "$VENV/bin/python" - "$f" <<'EOF'
import collections, json, sys
c = collections.Counter(); g = collections.Counter()
for l in open(sys.argv[1]):
    try:
        r = json.loads(l)
    except ValueError:
        c["torn line"] += 1; continue
    c["lines"] += 1; c["error" if str(r.get("finish", "")).startswith("error") else "ok"] += 1
    c["think_end " + str(r.get("think_end"))] += 1; g[(r.get("group"), r["variant"])] += 1
print(sys.argv[1], dict(c)); print("  per group/family:", dict(sorted(g.items())))
EOF
done
echo "BASE_THINK_DONE $(date '+%F %T')"
