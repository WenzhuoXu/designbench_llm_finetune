#!/bin/bash
# con1 main run, stage-1 data (CPU, Orchard): 2,000 training states per family (8 families), the native ones given
# their off-menu moves, consequence rows in every candidate format (stage 1 = consequence rows only), and the facet
# trees on the same states (the surface-feature control) answering queries_v2's held-out-problem questions.
set -uo pipefail
B=/project/community/wxu2/con1
R=$B/repo/scripts/con1
G=$B/gen
D=$B/data
O=$B/main
PY=/project/community/wxu2/rl/envs/sim/bin/python
CODE=$HOME/rl/harness/dataxp_code_v3
HV=$G/h/dataxp_policy_eval_v3rc.py
export PYTHONPATH=/project/community/wxu2/con1/pylib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATAXP_SPELLING=multitool PYTHONWARNINGS=ignore TREES_THREADS=2
mkdir -p $O/rows
ST="$G/out/states_native_orch.jsonl.gz $D/states_native.jsonl.gz $D/states_native_drc_snap.jsonl.gz $G/out/states_scripted.jsonl.gz"
echo "=== $(date) selection"
[ -e $O/sel.done ] || { $PY $R/build_rows.py --no-defined-terms --states $ST --out-train $O/sel_rows.jsonl --out-dev $O/sel_dev.jsonl \
  --per-family 2000 --target masked --max-tokens 24000 | grep "rows\|available\|DONE" && \
  $PY -c "
import json
s=set()
for p in ['$O/sel_rows.jsonl','$O/sel_dev.jsonl']:
    for l in open(p): r=json.loads(l); s.add(r['sid']) if r.get('domain') in ('engibench','grid2op','photonics') else None
open('$O/sel_native_sids.txt','w').write('\n'.join(sorted(s))+'\n'); print('native selected', len(s))
" && touch $O/sel.done; }
echo "=== $(date) off-menu moves for the selected native states"
[ -e $O/offmenu.done ] || { $PY $R/add_offmenu.py --code $CODE --harness $HV --problems $G/train_p2_datav4.jsonl --states $ST \
  --sids $O/sel_native_sids.txt --out $O/states_native_offmenu.jsonl.gz --workers 46 | tail -2 && touch $O/offmenu.done; }
for F in masked facet delta absolute; do
  echo "=== $(date) rows $F"
  [ -s $O/rows/train_$F.jsonl ] || $PY $R/build_rows.py --no-defined-terms --states $O/states_native_offmenu.jsonl.gz $ST \
    --out-train $O/rows/train_$F.jsonl --out-dev $O/rows/dev_$F.jsonl --per-family 2000 --target $F --max-tokens 24000 \
    > $O/rows/build_$F.log 2>&1
  tail -1 $O/rows/build_$F.log
done
grep "train rows\|est tokens\|verdict" $O/rows/build_masked.log | head -40
echo "=== $(date) trees on the main-run states"
Q=$D/queries_v2.jsonl
[ -s $O/obs_test.jsonl ] || $PY $R/obs_features.py --code $CODE --harness $HV --problems $G/bench_problems.jsonl --states $Q \
  --out $O/obs_test.jsonl --workers 46 | tail -1
[ -e $O/obs_train.done ] || { $PY $R/obs_features.py --code $CODE --harness $HV --problems $G/train_p2_datav4.jsonl \
  --states $O/states_native_offmenu.jsonl.gz $ST --rows $O/rows/train_masked.jsonl --out $O/obs_train.jsonl --workers 46 \
  | tail -1 && touch $O/obs_train.done; }
[ -s $O/preds_trees.jsonl ] || $PY $R/trees_facets.py --train $O/states_native_offmenu.jsonl.gz $ST --rows $O/rows/train_masked.jsonl \
  --queries $Q --out $O/preds_trees.jsonl --obs-feats $O/obs_test.jsonl $O/obs_train.jsonl | tail -10
echo "MAIN_PREP_DONE $(date)"
