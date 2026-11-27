#!/bin/bash
# hf_score.sh LAB : on drc, the held-out-family sequential pass of LAB against its own zero-shot answers on the same
# questions (positions >= 1) and against the copiers, in one facets.py run so the differences are paired by state:
# the zero-pass answers are scored as a second predictor "<LAB>_zero" (a relabelled copy of the rows; facets.py and
# the answers unchanged). Needs full_eval/hf/preds_LAB.jsonl with both passes (drc_arm_groups.sh, PASSES="zero seq").
#   out: full_eval/hf/facets_LAB_seq_vs_zero.{txt,json}
set -uo pipefail
L=${1:?label}
C=/storage/wxu/con1; A=$C/arm; E=${OUTDIR:-$C/full_eval/hf}
P=$E/preds_$L.jsonl
Z=$E/preds_${L}_zero_as_seq.jsonl
[ -s "$P" ] || { echo "ABORT: no $P"; exit 2; }
/opt/anaconda3/bin/python - "$P" "$Z" "$L" <<'EOF'
import json, sys
src, dst, lab = sys.argv[1:]
n = 0
with open(dst, "w") as fo:
    for line in open(src):
        r = json.loads(line)
        if r.get("pass") == "zero":
            r["pass"], r["predictor"] = "seq", lab + "_zero"
            fo.write(json.dumps(r) + "\n")
            n += 1
print("zero-pass answers relabelled as %s_zero: %d" % (lab, n))
EOF
cd $C/repo
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore /opt/anaconda3/bin/python scripts/con1/facets.py \
  --queries $A/main/queries_v2.jsonl --preds "$P" "$Z" --pass seq --min-qi 1 --kshot-baselines --groups heldout_family \
  --boot 1000 --out "$E/facets_${L}_seq_vs_zero.txt" --json "$E/facets_${L}_seq_vs_zero.json" > "$E/score_${L}_seq_vs_zero.log" 2>&1
echo "facets rc=$?"
grep -A12 "group means over all (primary)" "$E/facets_${L}_seq_vs_zero.txt"
grep -A12 "paired differences over all (primary)" "$E/facets_${L}_seq_vs_zero.txt"
echo "HF_SCORE_DONE $L $(date '+%F %T')"
