#!/bin/bash
# Score sampled replies and build own/own_act tiers.  Usage:
#   ROOT=/storage/wxu/corpus/rows    SPELLING=multitool     GRPS="v2verb grid2op"             bash bw_score.sh
#   ROOT=/storage/wxu/corpus/rows_sp SPELLING=set_parameter GRPS="v2 grid2op engibench photonics" bash bw_score.sh
# (GROUPS is a reserved bash variable; GRPS is used instead.)
set -uo pipefail
P=/storage/wxu/corpus
export PYTHONPATH=$P/code_v3 DATAXP_SPELLING=${SPELLING:-multitool}
PY=$P/env/bin/python
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export DATAXP_MOLOPT_ORACLE_DIR=/home/wxu/.dataxp/molopt_oracle/oracle
ROOT=${ROOT:?}; mkdir -p $ROOT/logs
for g in ${GRPS:?}; do
  W=$ROOT/$g
  [ -s $W/samples.jsonl ] || { echo "no samples for $g"; continue; }
  echo "=== $(date '+%F %T') score $g ($DATAXP_SPELLING)"
  $PY -m dataxp.ownrows.score --states $W/states_phi.jsonl --samples $W/samples.jsonl --out $W/scored.jsonl --workers 6 > $ROOT/logs/score_$g.log 2>&1; echo "score rc=$?"
  $PY -m dataxp.ownrows.build_tiers --states $W/states_phi.jsonl --samples $W/samples.jsonl --scored $W/scored.jsonl \
      --out $W/tiers --min-pct 90 --max-words 320 --tag own > $ROOT/logs/tiers_$g.log 2>&1; echo "tiers rc=$?"
  tail -4 $ROOT/logs/score_$g.log; cat $W/tiers/manifest_own.json 2>/dev/null | head -c 1500; echo
done
echo "BW_SCORE_DONE $(date '+%F %T')"
