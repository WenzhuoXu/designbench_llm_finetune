#!/bin/bash
# Own-action rows (dataxp/ownrows) for the slope test: the five v2 training families (states from
# the collaborator's run2_supp batch, disjoint from the evaluation problems) and grid2op (problems
# 0-68; 69-129 are the evaluation problems). engibench / photonics follow when their corpora land;
# molopt is held out and gets no rows. Steps: select_states -> cand_phi (CPU) -> sample (K=6 from
# the un-tuned Qwen3.8-27B on GPU 4, thinking off, append-only/resumable) -> score -> build_tiers.
set -uo pipefail
P=/storage/wxu/corpus
export PYTHONPATH=$P/code_v3
PY=$P/env/bin/python
W=$P/rows
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export DATAXP_MOLOPT_ORACLE_DIR=/home/wxu/.dataxp/molopt_oracle/oracle
mkdir -p $W/v2 $W/grid2op $W/logs
SUPP=/storage/wxu/sftprobe/src/dataxp/run2_supp/batch1_idx0130-0259
echo "=== $(date '+%F %T') select states"
[ -s $W/v2/states.jsonl ] || $PY -m dataxp.ownrows.select_states --data $SUPP --variants circuit2 frame3d2 grid2 seir2 traffic2 \
    --out $W/v2 --per-domain 600 --dev-per-domain 12 --seed 0 > $W/logs/select_v2.log 2>&1 || { echo "select v2 failed"; tail -5 $W/logs/select_v2.log; }
if [ ! -s $W/grid2op/states.jsonl ]; then
  $PY -m dataxp.ownrows.select_states --data $P/out/grid2op --variants grid2op --out $W/grid2op/all --per-domain 0 --dev-per-domain 12 --seed 0 > $W/logs/select_grid2op.log 2>&1 || { echo "select grid2op failed"; tail -5 $W/logs/select_grid2op.log; }
  # keep only problems 0..68 (69..129 are the harness's evaluation problems)
  $PY - <<'EOF'
import json, os
W = "/storage/wxu/corpus/rows/grid2op"
n = k = 0
with open(os.path.join(W, "states.jsonl"), "w") as out:
    for l in open(os.path.join(W, "all", "states.jsonl")):
        o = json.loads(l); n += 1
        pid = o.get("instance") or o.get("problem_id") or ""
        try: idx = int(str(pid).rsplit("_", 1)[-1])
        except ValueError: idx = -1
        if 0 <= idx <= 68:
            out.write(l); k += 1
print("grid2op states: %d of %d kept (problem index <= 68)" % (k, n))
EOF
fi
wc -l $W/v2/states.jsonl $W/grid2op/states.jsonl
echo "=== $(date '+%F %T') cand_phi"
[ -s $W/v2/states_phi.jsonl ] || $PY -m dataxp.ownrows.cand_phi --states $W/v2/states.jsonl --out $W/v2/states_phi.jsonl --workers 4 > $W/logs/cand_v2.log 2>&1
[ -s $W/grid2op/states_phi.jsonl ] || $PY -m dataxp.ownrows.cand_phi --states $W/grid2op/states.jsonl --out $W/grid2op/states_phi.jsonl --workers 4 > $W/logs/cand_grid2op.log 2>&1
tail -2 $W/logs/cand_v2.log $W/logs/cand_grid2op.log
echo "=== $(date '+%F %T') server"
GPU_UTIL=0.84 bash /storage/wxu/sftprobe/serve_bw2.sh > $W/logs/serve.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8766/v1/models >/dev/null && break; sleep 10; done
if ! curl -sf http://127.0.0.1:8766/v1/models >/dev/null; then echo "server not up"; tail -20 $W/logs/serve.log; kill $SP; exit 3; fi
echo "=== $(date '+%F %T') sample K=6"
$PY -m dataxp.ownrows.sample --states $W/v2/states.jsonl --out $W/v2/samples.jsonl --k 6 --endpoint http://127.0.0.1:8766/v1 \
    --model Qwen/Qwen3.8-27B --max-tokens 4000 --temperature 0.6 --concurrency 32 --prompt probe > $W/logs/sample_v2.log 2>&1
$PY -m dataxp.ownrows.sample --states $W/grid2op/states.jsonl --out $W/grid2op/samples.jsonl --k 6 --endpoint http://127.0.0.1:8766/v1 \
    --model Qwen/Qwen3.8-27B --max-tokens 4000 --temperature 0.6 --concurrency 32 --prompt probe > $W/logs/sample_grid2op.log 2>&1
tail -2 $W/logs/sample_v2.log $W/logs/sample_grid2op.log
kill $SP 2>/dev/null; sleep 5; pkill -f "port 8766" || true; sleep 20
echo "=== $(date '+%F %T') score + tiers"
for d in v2 grid2op; do
  $PY -m dataxp.ownrows.score --states $W/$d/states_phi.jsonl --samples $W/$d/samples.jsonl --out $W/$d/scored.jsonl --workers 4 > $W/logs/score_$d.log 2>&1
  $PY -m dataxp.ownrows.build_tiers --states $W/$d/states_phi.jsonl --samples $W/$d/samples.jsonl --scored $W/$d/scored.jsonl \
      --out $W/$d/tiers --min-pct 90 --max-words 320 --tag own > $W/logs/tiers_$d.log 2>&1
  tail -3 $W/logs/score_$d.log; tail -3 $W/logs/tiers_$d.log
done
echo "BW_ROWS_DONE $(date '+%F %T')"
