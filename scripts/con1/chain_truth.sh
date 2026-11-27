#!/bin/bash
# con1 facetc truth on Orchard CPU: per-element state before/after every move of round 1's rows and of the
# held-out-problem questions, chain families only (scripts/con1/chain_truth.py).
set -uo pipefail
B=/project/community/wxu2/con1
R=$B/repo/scripts/con1
G=$B/gen
D=$B/data
PY=/project/community/wxu2/rl/envs/sim/bin/python
export PYTHONPATH=$B/pylib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATAXP_SPELLING=multitool PYTHONWARNINGS=ignore
mkdir -p $B/chain
ST="$D/states_native.jsonl.gz $D/states_native_drc_snap.jsonl.gz $G/out/states_scripted.jsonl.gz $G/out/states_native_orch.jsonl.gz"
$PY $R/chain_truth.py --code $HOME/rl/harness/dataxp_code_v3 --harness $G/h/dataxp_policy_eval_v3rc.py \
  --problems $G/train_p2_datav4.jsonl $G/bench_problems.jsonl --states $ST --rows $B/abl1/rows/rows_facetm.jsonl \
  --queries $B/skill/queries.jsonl --out $B/chain/chain_r1.jsonl --workers ${WORKERS:-30}
$PY - $B/chain/chain_r1.jsonl <<'PYEOF'
import json, sys, collections
c = collections.Counter()
for l in open(sys.argv[1]):
    r = json.loads(l); k = (r["split"], r["variant"])
    c[k + ("states",)] += 1
    c[k + ("prompt_same",)] += r["checks"]["prompt_same"]
    c[k + ("result_same",)] += r["checks"]["result_same"]; c[k + ("result_checked",)] += r["checks"]["result_checked"]
    c[k + ("elements",)] += len(r["before"]); c[k + ("moves",)] += len(r["after"])
    c[k + ("refused",)] += sum(v is None for v in r["after"].values())
for k in sorted(c): print(k, c[k])
PYEOF
echo "CHAIN_JOB_DONE $(date)"
