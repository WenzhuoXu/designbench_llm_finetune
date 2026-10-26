#!/bin/bash
# Own-action rows for engibench and photonics (training problems = index <= 69; the harness's
# evaluation problems are the last 60 by index). Waits for the photonics corpus and for bw_rows2
# (which holds the GPU server) to finish, then: select_states -> cand_phi -> sample -> score -> tiers.
set -uo pipefail
P=/storage/wxu/corpus
export PYTHONPATH=$P/code_v3
PY=$P/env/bin/python
W=$P/rows
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p $W/logs
while [ ! -s $P/out/photonics/photonics/accepted/problems.jsonl ]; do sleep 300; done
echo "=== $(date '+%F %T') photonics corpus present"
for v in engibench photonics; do
  mkdir -p $W/$v
  [ -s $W/$v/states.jsonl ] && continue
  $PY -m dataxp.ownrows.select_states --data $P/out/$v --variants $v --out $W/$v/all --per-domain 0 --dev-per-domain 12 --seed 0 > $W/logs/select_$v.log 2>&1 || { echo "select $v failed"; tail -5 $W/logs/select_$v.log; }
  $PY - "$v" <<'EOF'
import json, os, sys
v = sys.argv[1]; W = "/storage/wxu/corpus/rows/" + v
n = k = 0
with open(os.path.join(W, "states.jsonl"), "w") as out:
    for l in open(os.path.join(W, "all", "states.jsonl")):
        o = json.loads(l); n += 1
        pid = o.get("instance") or o.get("problem_id") or ""
        try: idx = int(str(pid).rsplit("_", 1)[-1])
        except ValueError: idx = -1
        if 0 <= idx <= 69:
            out.write(l); k += 1
print("%s states: %d of %d kept (problem index <= 69)" % (v, k, n))
EOF
done
wc -l $W/engibench/states.jsonl $W/photonics/states.jsonl
echo "=== $(date '+%F %T') cand_phi"
for v in engibench photonics; do
  [ -s $W/$v/states_phi.jsonl ] || $PY -m dataxp.ownrows.cand_phi --states $W/$v/states.jsonl --out $W/$v/states_phi.jsonl --workers 4 > $W/logs/cand_$v.log 2>&1
  tail -1 $W/logs/cand_$v.log
done
while pgrep -f "bash /storage/wxu/corpus/bw_rows2.sh" >/dev/null; do sleep 120; done
pkill -f "port 8766" || true; sleep 20
echo "=== $(date '+%F %T') server"
GPU_UTIL=0.84 bash /storage/wxu/sftprobe/serve_bw2.sh > $W/logs/serve3.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8766/v1/models >/dev/null && break; sleep 10; done
if ! curl -sf http://127.0.0.1:8766/v1/models >/dev/null; then echo "server not up"; tail -20 $W/logs/serve3.log; kill $SP; exit 3; fi
echo "=== $(date '+%F %T') sample K=4"
for v in engibench photonics; do
  $PY -m dataxp.ownrows.sample --states $W/$v/states.jsonl --out $W/$v/samples.jsonl --k 4 --endpoint http://127.0.0.1:8766/v1 \
      --model Qwen/Qwen3.8-27B --max-tokens 2000 --temperature 0.6 --concurrency 64 --prompt probe >> $W/logs/sample_$v.log 2>&1
  echo "$(date '+%F %T') sample $v rc=$? lines=$(wc -l < $W/$v/samples.jsonl)"
done
kill $SP 2>/dev/null; sleep 5; pkill -f "port 8766" || true; sleep 20
echo "=== $(date '+%F %T') score + tiers"
for v in engibench photonics; do
  $PY -m dataxp.ownrows.score --states $W/$v/states_phi.jsonl --samples $W/$v/samples.jsonl --out $W/$v/scored.jsonl --workers 4 > $W/logs/score_$v.log 2>&1
  $PY -m dataxp.ownrows.build_tiers --states $W/$v/states_phi.jsonl --samples $W/$v/samples.jsonl --scored $W/$v/scored.jsonl \
      --out $W/$v/tiers --min-pct 90 --max-words 320 --tag own > $W/logs/tiers_$v.log 2>&1
  tail -3 $W/logs/score_$v.log; tail -3 $W/logs/tiers_$v.log
done
echo "BW_ROWS3_DONE $(date '+%F %T')"
