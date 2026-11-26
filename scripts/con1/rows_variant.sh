#!/bin/bash
# con1: consequence rows of one more answer format on the main run's states (same selection and seed as main_prep.sh,
# so the same states as the stage-1 run), with two rendered answers printed for a check.   FMT=facetm bash rows_variant.sh
set -uo pipefail
B=/project/community/wxu2/con1
R=$B/repo/scripts/con1
G=$B/gen
D=$B/data
O=$B/main
FMT="${FMT:?answer format}"
PY=/project/community/wxu2/rl/envs/sim/bin/python
export PYTHONPATH=$B/pylib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATAXP_SPELLING=multitool PYTHONWARNINGS=ignore
ST="$G/out/states_native_orch.jsonl.gz $D/states_native.jsonl.gz $D/states_native_drc_snap.jsonl.gz $G/out/states_scripted.jsonl.gz"
echo "=== $(date) rows $FMT"
[ -s $O/rows/train_$FMT.jsonl ] || $PY $R/build_rows.py --states $O/states_native_offmenu.jsonl.gz $ST \
  --out-train $O/rows/train_$FMT.jsonl.tmp --out-dev $O/rows/dev_$FMT.jsonl --per-family 2000 --target $FMT --max-tokens 24000 \
  > $O/rows/build_$FMT.log 2>&1 && mv $O/rows/train_$FMT.jsonl.tmp $O/rows/train_$FMT.jsonl
tail -3 $O/rows/build_$FMT.log
# the same states as the facet rows? (sids in order)
$PY - $O/rows/train_facet.jsonl $O/rows/train_$FMT.jsonl <<'EOF'
import json, sys
a = [json.loads(l)["sid"] for l in open(sys.argv[1])]
b = [json.loads(l)["sid"] for l in open(sys.argv[2])]
print("rows facet %d, %s %d, same sids in the same order: %s" % (len(a), sys.argv[2].rsplit("_", 1)[1], len(b), a == b))
seen = set()
for line in open(sys.argv[2]):
    r = json.loads(line)
    if r.get("domain") in ("circuit2", "photonics", "grid2op") and r["domain"] not in seen:
        seen.add(r["domain"])
        ans = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
        print("--- %s %s" % (r["domain"], r["sid"])); print(ans[0]); print("..."); print(ans[1])
    if len(seen) == 3:
        break
EOF
echo "ROWS_VARIANT_DONE $FMT $(date)"
