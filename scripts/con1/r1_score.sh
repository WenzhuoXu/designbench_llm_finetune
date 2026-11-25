#!/bin/bash
# con1 round-1 scoring (CPU, Orchard): facet trees on round 1's training states (with the observation's first-order
# estimates), then facets.py over the four arms + trees on the held-out-problem questions of round 1's families.
set -uo pipefail
B=/project/community/wxu2/con1
R=$B/repo/scripts/con1
G=$B/gen
D=$B/data
O=$B/abl1/score
PY=/project/community/wxu2/rl/envs/sim/bin/python
CODE=$HOME/rl/harness/dataxp_code_v3
HV=$G/h/dataxp_policy_eval_v3rc.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATAXP_SPELLING=multitool PYTHONWARNINGS=ignore TREES_THREADS=2
mkdir -p $O
Q=$B/skill/queries.jsonl
ST="$D/states_native.jsonl.gz $D/states_native_drc_snap.jsonl.gz $G/out/states_scripted.jsonl.gz"
FAM="engibench grid2op photonics circuit2 frame3d2"
echo "=== $(date) observation estimates"
[ -s $O/obs_test.jsonl ] || $PY $R/obs_features.py --code $CODE --harness $HV --problems $G/bench_problems.jsonl --states $Q \
  --out $O/obs_test.jsonl --workers 30 | tail -2
[ -s $O/obs_train.done ] || { $PY $R/obs_features.py --code $CODE --harness $HV --problems $G/train_p2_datav4.jsonl --states $ST \
  --rows $B/abl1/rows/rows_masked.jsonl --out $O/obs_train.jsonl --workers 30 | tail -2 && touch $O/obs_train.done; }
echo "=== $(date) trees"
[ -s $O/preds_trees.jsonl ] || $PY $R/trees_facets.py --train $ST --rows $B/abl1/rows/rows_masked.jsonl --queries $Q \
  --out $O/preds_trees.jsonl --families $FAM --obs-feats $O/obs_test.jsonl $O/obs_train.jsonl | tail -8
echo "=== $(date) facets"
P=""
for a in M F D S; do [ -s $B/abl1/out/preds_$a.jsonl ] && P="$P $B/abl1/out/preds_$a.jsonl"; done
$PY $R/facets.py --queries $Q --preds $P $O/preds_trees.jsonl --groups heldout_problem --variants $FAM \
  --problems $G/bench_problems.jsonl --code $CODE --harness $HV --boot 1000 --out $O/facets_r1.txt --json $O/facets_r1.json
echo "R1_SCORE_DONE $(date)"
