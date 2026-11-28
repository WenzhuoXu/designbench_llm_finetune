#!/bin/bash
# con1, 11-26: rebuild the main-run rows with the native off-menu moves (build_rows.pick_moves skipped src 'offmenu'
# until now), in facet and facetm, and a second facet set on the same states with 12 different moves each (facet_b).
# Old files are kept as *_nooffmenu.jsonl until the new ones are checked.
set -uo pipefail
B=/project/community/wxu2/con1
R=$B/repo/scripts/con1
G=$B/gen
D=$B/data
O=$B/main/rows
PY=/project/community/wxu2/rl/envs/sim/bin/python
export PYTHONPATH=$B/pylib OMP_NUM_THREADS=1 DATAXP_SPELLING=multitool PYTHONWARNINGS=ignore
ST="$B/main/states_native_offmenu.jsonl.gz $G/out/states_native_orch.jsonl.gz $D/states_native.jsonl.gz $D/states_native_drc_snap.jsonl.gz $G/out/states_scripted.jsonl.gz"
for F in facet facetm; do
  [ -e $O/train_${F}_nooffmenu.jsonl ] || { mv $O/train_$F.jsonl $O/train_${F}_nooffmenu.jsonl; mv $O/dev_$F.jsonl $O/dev_${F}_nooffmenu.jsonl; }
  echo "=== $(date) rows $F (with off-menu)"
  $PY $R/build_rows.py --no-defined-terms --states $ST --out-train $O/train_$F.jsonl --out-dev $O/dev_$F.jsonl --per-family 2000 --target $F \
    --max-tokens 24000 > $O/build_$F.log 2>&1
  grep "src offmenu\|train rows\|DONE" $O/build_$F.log
done
echo "=== $(date) rows facet_b (same states, 12 other moves)"
$PY $R/build_rows.py --no-defined-terms --states $ST --out-train $O/train_facet_b.jsonl --out-dev $O/dev_facet_b.jsonl --per-family 2000 \
  --target facet --max-tokens 24000 --move-seed 1 --exclude-rows $O/train_facet.jsonl $O/dev_facet.jsonl > $O/build_facet_b.log 2>&1
grep "excluding\|dropped\|src offmenu\|train rows\|DONE" $O/build_facet_b.log
$PY - $O/train_facet.jsonl $O/train_facet_b.jsonl $O/train_facetm.jsonl <<'EOF'
import json, sys
A = {}
for l in open(sys.argv[1]):
    r = json.loads(l); A[r["sid"]] = set(r["con1"]["moves"])
nb = ov = 0; sids_b = set()
for l in open(sys.argv[2]):
    r = json.loads(l); sids_b.add(r["sid"]); nb += 1
    ov += len(A.get(r["sid"], set()) & set(r["con1"]["moves"]))
fm = [json.loads(l)["sid"] for l in open(sys.argv[3])]
print("A %d rows, B %d rows; B states all in A: %s; moves shared A/B: %d; facetm same states as facet: %s" % (
    len(A), nb, sids_b <= set(A), ov, set(fm) == set(A)))
EOF
df -BG --output=avail /project/community | tail -1
echo "ROWS_FIX_DONE $(date)"
