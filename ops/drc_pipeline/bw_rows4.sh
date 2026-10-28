#!/bin/bash
# Own-action rows for the five v2 training families in VERB spelling (owner decision 2026-10-23):
# re-select states with the verb-surface code (prompt parts change; Φ caches are reused by sid),
# sample K=4 from the un-tuned model, score, build tiers. Waits for bw_rows3 (engibench/photonics)
# to release the GPU server. grid2op rows: sampled here too (its earlier run was cut short).
set -uo pipefail
P=/storage/wxu/corpus
export PYTHONPATH=$P/code_v3
PY=$P/env/bin/python
W=$P/rows
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
SUPP=/storage/wxu/sftprobe/src/dataxp/run2_supp/batch1_idx0130-0259
mkdir -p $W/logs $W/v2verb
echo "=== $(date '+%F %T') select states (verb surface)"
$PY -m dataxp.ownrows.select_states --data $SUPP --variants circuit2 frame3d2 grid2 seir2 traffic2 \
    --out $W/v2verb --per-domain 600 --dev-per-domain 12 --seed 0 > $W/logs/select_v2verb.log 2>&1 || { echo "select failed"; tail -5 $W/logs/select_v2verb.log; exit 2; }
grep -c '"spelling": "verb_surface"' $W/v2verb/states.jsonl || true
# reuse the Φ cache when the state ids are the same as the first run
$PY - <<'EOF'
import json, os
W = "/storage/wxu/corpus/rows"
old = {json.loads(l)["sid"]: l for l in open(os.path.join(W, "v2", "states_phi.jsonl"))} if os.path.exists(os.path.join(W, "v2", "states_phi.jsonl")) else {}
new_sids = [json.loads(l)["sid"] for l in open(os.path.join(W, "v2verb", "states.jsonl"))]
hit = sum(s in old for s in new_sids)
print("phi cache: %d of %d states reusable" % (hit, len(new_sids)))
if hit == len(new_sids):
    with open(os.path.join(W, "v2verb", "states_phi.jsonl"), "w") as f:
        for s in new_sids: f.write(old[s])
EOF
[ -s $W/v2verb/states_phi.jsonl ] || $PY -m dataxp.ownrows.cand_phi --states $W/v2verb/states.jsonl --out $W/v2verb/states_phi.jsonl --workers 4 > $W/logs/cand_v2verb.log 2>&1
while pgrep -f "bash /storage/wxu/corpus/bw_rows3.sh" >/dev/null; do sleep 120; done
pkill -f "[v]llm serve" || true; sleep 20
echo "=== $(date '+%F %T') server"
GPU_UTIL=0.84 bash /storage/wxu/sftprobe/serve_bw2.sh > $W/logs/serve4.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8766/v1/models >/dev/null && break; sleep 10; done
if ! curl -sf http://127.0.0.1:8766/v1/models >/dev/null; then echo "server not up"; tail -20 $W/logs/serve4.log; kill $SP; exit 3; fi
echo "=== $(date '+%F %T') sample K=4"
for d in v2verb grid2op; do
  $PY -m dataxp.ownrows.sample --states $W/$d/states.jsonl --out $W/$d/samples.jsonl --k 4 --endpoint http://127.0.0.1:8766/v1 \
      --model Qwen/Qwen3.8-27B --max-tokens 2000 --temperature 0.6 --concurrency 64 --prompt probe >> $W/logs/sample_$d.log 2>&1
  echo "$(date '+%F %T') sample $d rc=$? lines=$(wc -l < $W/$d/samples.jsonl)"
done
kill $SP 2>/dev/null; sleep 5; pkill -f "[v]llm serve" || true; sleep 20
echo "=== $(date '+%F %T') score + tiers"
for d in v2verb grid2op; do
  $PY -m dataxp.ownrows.score --states $W/$d/states_phi.jsonl --samples $W/$d/samples.jsonl --out $W/$d/scored.jsonl --workers 4 > $W/logs/score_$d.log 2>&1
  $PY -m dataxp.ownrows.build_tiers --states $W/$d/states_phi.jsonl --samples $W/$d/samples.jsonl --scored $W/$d/scored.jsonl \
      --out $W/$d/tiers --min-pct 90 --tag own > $W/logs/tiers_$d.log 2>&1
  tail -3 $W/logs/score_$d.log; tail -3 $W/logs/tiers_$d.log
done
echo "BW_ROWS4_DONE $(date '+%F %T')"
