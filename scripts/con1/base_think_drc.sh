#!/bin/bash
# base_think_drc.sh : part of base_think.sh's run on drc GPU 4 while the Orchard job waits for its slot (con1-2m-rows,
# 11-27). The same client, model, settings and labels as base_think.sh (think_client.py; base Qwen3.8-27B, no adapter,
# temperature 0, think to </think> or the budget, then the answer), so the answer files can be merged by (sid, qi):
#   pass A, unseen families only (heldout_family, 1,400 questions), budget 4,000 -> full_eval/base_think/preds_base_v3_4k.jsonl
#   pass B, the fixed 300-question subset (the same random.Random('base-think-sub|0') draw), budget 8,000
#           -> full_eval/base_think/preds_base_v3_8k_sub300.jsonl
# Copied to Orchard's con1/full/base_think/ before the Orchard job starts, these questions count as done there (the
# client resumes by (sid, qi)). GPU 4 under arm/gpu4.lock; resumable.
set -uo pipefail
C=/storage/wxu/con1; A=$C/arm; O=$C/full_eval/base_think
PY=/storage/wenwenw/envs/vllm_drc/bin/python
MODEL=/storage/wxu/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
Q=$A/main/queries_v3.jsonl
CLIENT=$C/full_eval/think_diag/think_client.py
export PYTHONPATH=$C/pylib:$C/repo HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-4} OMP_NUM_THREADS=4 TMPDIR=$C/tmp
export HF_HOME=/storage/wxu/huggingface HF_HUB_CACHE=/storage/wxu/huggingface/hub HF_HUB_DISABLE_XET=1
export VLLM_CACHE_ROOT=$C/cache/vllm CUDA_CACHE_PATH=$C/cache/nv VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$O/logs"
grep -q "YOUR MOVE" "$Q" && { echo "ABORT: $Q carries the move request"; exit 2; }
echo "=== $(date '+%F %T') base_think_drc queries=$Q ($(sha1sum < $Q | cut -c1-12))"
if [ ! -s "$O/subset_300.tsv" ]; then
  $PY - "$Q" "$O/subset_300.tsv" <<'EOF'
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
exec 9>"$A/gpu4.lock"
flock 9; echo "=== $(date '+%F %T') holding $A/gpu4.lock"
read -r TOT USED <<< "$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | tr -d ',')"
while [ "${USED:-99999}" -ge 5000 ]; do sleep 120; read -r TOT USED <<< "$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | tr -d ',')"; done
PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
setsid $PY /storage/wenwenw/envs/vllm_drc/bin/vllm serve $MODEL --served-model-name base --host 127.0.0.1 --port $PORT \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92 --max-num-seqs 48 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' > "$O/logs/vllm_drc.log" 2>&1 &
SP=$!
trap 'kill -TERM -- -$SP 2>/dev/null; sleep 10; kill -KILL -- -$SP 2>/dev/null' EXIT
T0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  kill -0 $SP 2>/dev/null || { echo "ABORT: vLLM exited"; tail -30 "$O/logs/vllm_drc.log"; exit 7; }
  [ $(( $(date +%s) - T0 )) -lt 3600 ] || { echo "ABORT: vLLM not ready"; exit 7; }
  sleep 10
done
echo "=== $(date '+%F %T') server ready"
$PY $CLIENT --queries $Q --tokenizer $MODEL --base-url "http://127.0.0.1:$PORT/v1" --model base --label base_v3_4k \
  --budget 4000 --answer-tokens 1024 --workers 48 --groups heldout_family --out $O/preds_base_v3_4k.jsonl > "$O/logs/client_drc_4k.log" 2>&1
echo "=== $(date '+%F %T') pass A (unseen families) rc=$?: $(tail -1 "$O/logs/client_drc_4k.log")"
$PY $CLIENT --queries $Q --tokenizer $MODEL --base-url "http://127.0.0.1:$PORT/v1" --model base --label base_v3_8k_sub300 \
  --budget 8000 --answer-tokens 1024 --workers 48 --subset $O/subset_300.tsv --out $O/preds_base_v3_8k_sub300.jsonl > "$O/logs/client_drc_8k.log" 2>&1
echo "=== $(date '+%F %T') pass B (subset, 8,000) rc=$?: $(tail -1 "$O/logs/client_drc_8k.log")"
kill -TERM -- -$SP 2>/dev/null; sleep 15
flock -u 9; echo "=== $(date '+%F %T') GPU work done, lock released"
echo "BASE_THINK_DRC_DONE $(date '+%F %T')"
