#!/bin/bash
# Own-action rows, relaunch of bw_rows.sh (2026-10-23 09:10): states and cand_phi caches already
# exist; the sampler is restarted with 64 streams, max_tokens 2000 and K=4 (the first launch ran
# at ~11 replies/min with 32 streams and ~2,400-token replies). Append-only outputs resume.
set -uo pipefail
P=/storage/wxu/corpus
export PYTHONPATH=$P/code_v3
PY=$P/env/bin/python
W=$P/rows
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export DATAXP_MOLOPT_ORACLE_DIR=/home/wxu/.dataxp/molopt_oracle/oracle
pkill -f "port 8766" || true; sleep 15
for d in v2 grid2op; do [ -s $W/$d/states_phi.jsonl ] || { echo "missing $W/$d/states_phi.jsonl"; exit 2; }; done
echo "=== $(date '+%F %T') server"
GPU_UTIL=0.84 bash /storage/wxu/sftprobe/serve_bw2.sh > $W/logs/serve2.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8766/v1/models >/dev/null && break; sleep 10; done
if ! curl -sf http://127.0.0.1:8766/v1/models >/dev/null; then echo "server not up"; tail -20 $W/logs/serve2.log; kill $SP; exit 3; fi
echo "=== $(date '+%F %T') sample K=4, 64 streams, max 2000 tokens"
for d in v2 grid2op; do
  $PY -m dataxp.ownrows.sample --states $W/$d/states.jsonl --out $W/$d/samples.jsonl --k 4 --endpoint http://127.0.0.1:8766/v1 \
      --model Qwen/Qwen3.8-27B --max-tokens 2000 --temperature 0.6 --concurrency 64 --prompt probe >> $W/logs/sample_$d.log 2>&1
  echo "$(date '+%F %T') sample $d rc=$? lines=$(wc -l < $W/$d/samples.jsonl)"
done
kill $SP 2>/dev/null; sleep 5; pkill -f "port 8766" || true; sleep 20
echo "=== $(date '+%F %T') score + tiers"
for d in v2 grid2op; do
  $PY -m dataxp.ownrows.score --states $W/$d/states_phi.jsonl --samples $W/$d/samples.jsonl --out $W/$d/scored.jsonl --workers 4 > $W/logs/score_$d.log 2>&1
  $PY -m dataxp.ownrows.build_tiers --states $W/$d/states_phi.jsonl --samples $W/$d/samples.jsonl --scored $W/$d/scored.jsonl \
      --out $W/$d/tiers --min-pct 90 --tag own > $W/logs/tiers_$d.log 2>&1
  tail -3 $W/logs/score_$d.log; tail -3 $W/logs/tiers_$d.log
done
echo "BW_ROWS_DONE $(date '+%F %T')"
