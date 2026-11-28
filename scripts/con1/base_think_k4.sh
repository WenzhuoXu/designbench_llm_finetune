#!/bin/bash
# base_think_k4.sh : arm K4 of the 11-28 base-model read (parent's order, after round 3): when the untrained base is
# wrong at temperature 0, is a right answer among its own samples? Orchard 4 x H100, one vLLM server per GPU (as
# base_think.sh). Arm D's text and settings unchanged (dir form, metrics named, think budget 8,000, answer cap 1,024, no
# adapter, think_client --no-defined-terms, reasoning effort at the template's default) except: temperature 0.8 on both
# calls (think and answer) and 4 samples per question, seeds 1-4 (sample k: seed k on both calls). Questions: k4_subset.tsv
# (every unflagged question of inventory2, molopt, thermal2 + the subset_300 questions of photonics, frame3d2, seir2,
# engibench, grid2op: 1,428 questions, 5,712 blocks). Before the servers start: effort_render_check.py on these questions
# (unset == xhigh token for token; question text == arm D's code, con1_repo_v4 targets.question; prompt tokens == arm D's
# recorded prompt_tokens), abort on any failure. The vLLM servers' default sampling parameters (the model's
# generation_config) are printed from their logs. Resumable per (question, sample).
#   sbatch -p advanced --qos=adv_4gpu_qos -A ccm --gres=gpu:4 -c 32 --mem=256G -t 8:00:00 --dependency=afterany:176605 \
#     -o $HOME/con1_k4/logs/k4_%j.out --wrap "bash $HOME/con1_repo_k4/scripts/con1/base_think_k4.sh"
set -uo pipefail
B=/project/community/wxu2
R=${REPO:-$HOME/con1_repo_k4}
O=${OUT:-$HOME/con1_k4}
Q=${QUERIES:-$B/con1/noconf/queries_v3.jsonl}
REF_LIB=${REF_LIB:-$HOME/con1_repo_v4/scripts/con1}
REF_PREDS=${REF_PREDS:-$HOME/con1_base_think2/preds_base_v3_dir_8k.jsonl}
VENV=$B/rl/envs/vllm
MODEL=$B/rl/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
C=/tmp/base_think_k4_cache_${SLURM_JOB_ID:-x}
mkdir -p $C/tmp $C/xdg $C/triton $C/inductor $C/nv $C/vllm $C/flashinfer "$O/logs"
NSERVERS=${NSERVERS:-4}
LABEL=base_v3_dir_8k_k4_t08
export HF_HOME=$B/rl/hf HF_HUB_CACHE=$B/rl/hf/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$R CON1_DIR=$R/scripts/con1
export TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor
export CUDA_CACHE_PATH=$C/nv VLLM_CACHE_ROOT=$C/vllm FLASHINFER_WORKSPACE_BASE=$C/flashinfer
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0
echo "=== $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-none} restarts=${SLURM_RESTART_COUNT:-0} repo=$R ($(cat $R/GIT_COMMIT 2>/dev/null)) queries=$Q ($(sha1sum < $Q | cut -c1-12)) subset $(sha1sum < $O/k4_subset.tsv | cut -c1-12) out=$O"
[ -s "$Q" ] && [ -s "$O/k4_subset.tsv" ] && [ -e "$MODEL/config.json" ] && [ -e "$VENV/bin/vllm" ] && [ -s "$REF_PREDS" ] || { echo "ABORT: missing input"; exit 2; }
grep -q "YOUR MOVE" "$Q" && { echo "ABORT: $Q carries the move request"; exit 2; }
"$VENV/bin/python" "$R/scripts/con1/effort_render_check.py" --queries "$Q" --tokenizer "$MODEL" --subset "$O/k4_subset.tsv" \
  --no-defined-terms --ref-targets "$REF_LIB" --ref-preds "$REF_PREDS" > "$O/logs/render_check_job${SLURM_JOB_ID:-x}.txt" 2>&1
RC=$?; tail -6 "$O/logs/render_check_job${SLURM_JOB_ID:-x}.txt"
[ $RC -eq 0 ] || { echo "ABORT: render check failed"; exit 6; }
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
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92 --max-num-seqs 48 \
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
grep -h -i "sampling param\|generation config\|generation_config" "$O/logs/vllm_0_job${SLURM_JOB_ID:-x}.log" | head -3
W=$(( ${WORKERS_PER_SERVER:-48} * NSERVERS ))
"$VENV/bin/python" "$R/scripts/con1/think_client.py" --no-defined-terms --queries "$Q" --tokenizer "$MODEL" --base-url "${URLS[@]}" \
  --model base --label "$LABEL" --target dir --named-metrics --budget 8000 --answer-tokens 1024 --workers "$W" \
  --temperature 0.8 --samples 4 --seed-base 1 --subset "$O/k4_subset.tsv" --out "$O/preds_$LABEL.jsonl" \
  > "$O/logs/client_${LABEL}_job${SLURM_JOB_ID:-x}.log" 2>&1
echo "=== $(date '+%F %T') $LABEL rc=$?: $(tail -1 "$O/logs/client_${LABEL}_job${SLURM_JOB_ID:-x}.log")"
"$VENV/bin/python" - "$O/preds_$LABEL.jsonl" <<'EOF'
import collections, json, sys
c = collections.Counter(); g = collections.Counter()
for l in open(sys.argv[1]):
    try:
        r = json.loads(l)
    except ValueError:
        c["torn line"] += 1; continue
    c["lines"] += 1; c["error" if str(r.get("finish", "")).startswith("error") else "ok"] += 1
    c["think_end " + str(r.get("think_end"))] += 1; g[(r["variant"], r.get("sample"))] += 1
print(sys.argv[1], dict(c)); print("  per family/sample:", dict(sorted(g.items())))
EOF
echo "BASE_THINK_K4_DONE $(date '+%F %T')"
