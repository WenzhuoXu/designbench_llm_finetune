#!/bin/bash
# con1 main-run scoring (CPU, Orchard): facets of a stage's answers on queries_v2's held-out problems, against the
# facet trees and no change; zero-shot and (when present) the sequential pass.   STAGE=stage1 M=<run dir> bash main_score.sh
set -uo pipefail
B=/project/community/wxu2/con1
R=$B/repo/scripts/con1
G=$B/gen
STAGE="${STAGE:?stage1|stage2}"
M="${M:?run dir}"
PY=/project/community/wxu2/rl/envs/sim/bin/python
export PYTHONPATH=$B/pylib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATAXP_SPELLING=multitool PYTHONWARNINGS=ignore
Q=$B/data/queries_v2.jsonl
for P in zero seq; do
  grep -q "\"pass\": \"$P\"" $M/preds_$STAGE.jsonl 2>/dev/null || continue
  TR=""; [ "$P" = zero ] && TR=$B/main/preds_trees.jsonl
  $PY $R/facets.py --queries $Q --preds $M/preds_$STAGE.jsonl $TR --pass $P --groups heldout_problem \
    --problems $G/bench_problems.jsonl --code $HOME/rl/harness/dataxp_code_v3 --harness $G/h/dataxp_policy_eval_v3rc.py \
    --boot 1000 --out $M/facets_${STAGE}_$P.txt --json $M/facets_${STAGE}_$P.json > /dev/null
  echo "=== $STAGE $P"; grep -A8 "group means over all" $M/facets_${STAGE}_$P.txt
done
echo "MAIN_SCORE_DONE $STAGE $(date)"
