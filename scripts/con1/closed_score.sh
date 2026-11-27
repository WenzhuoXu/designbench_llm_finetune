#!/bin/bash
# closed_score.sh LABEL : on drc (CPU), the closed-think answers of a checkpoint (closed_think.sh, full_eval/closed/
# preds_c2m_LABEL_closed_{v2,v3}.jsonl) next to the same checkpoint's standard-protocol answers and the stage-1 final's,
# per prompt version and group (con1-2m-rows, 11-27):
#   1. compare_models.py (facets.py's scoring; reference = the stage-1 final on the same prompt version; paired se over
#      states): readable share, regime_phys, cpl_dir_BA, cpl_size per family and group
#   2. dirlayer.py (control layer), closed-think v2 and v3 next to the stage-1 final v2 and v3
#   3. indep_score.py on each closed-think file and group; the parent's ea1 / ea2 on the unseen-family answers
#   bash /storage/wxu/con1/full_eval/closed_score.sh full_00850   -> full_eval/closed/report_LABEL.txt
set -uo pipefail
L=${1:?label}
C=/storage/wxu/con1; E=$C/full_eval; X=$E/closed; R=$C/repo/scripts/con1; D=$C/dl; O=$C/dirlayer
PY=/opt/anaconda3/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=
OUT=$X/report_$L.txt
{
echo "# $(date '+%F %T') c2m_$L, think block closed in the prompt (closed_think.sh) vs the standard protocol and the stage-1 final"
for V in v2 v3; do
  CL=$X/preds_c2m_${L}_closed_$V.jsonl; [ -s $CL ] || { echo "no $CL"; continue; }
  if [ $V = v2 ]; then SA=$E/preds_c2m_${L}.jsonl; SH=$E/hf/preds_c2m_${L}.jsonl; S1A=$E/preds_c2m_stage1_final.jsonl; S1H=$E/hf/preds_c2m_stage1_final.jsonl; P=; else
    SA=$E/v3/preds_c2m_${L}_v3.jsonl; SH=$E/v3/hf/preds_c2m_${L}_v3.jsonl; S1A=$E/v3/preds_c2m_stage1_final_v3.jsonl; S1H=$E/v3/hf/preds_c2m_stage1_final_v3.jsonl; P=_v3; fi
  for G in heldout_problem heldout_family; do
    S=$SA; S1=$S1A; [ $G = heldout_family ] && S=$SH && S1=$S1H
    echo "## $V $G: stage-1 final (reference), c2m_$L standard protocol, c2m_$L closed think"
    $PY $E/compare_models.py --queries $C/arm/main/queries_$V.jsonl --pred c2m_stage1_final$P=$S1 --pred c2m_$L$P=$S \
      --pred c2m_${L}_closed_$V=$CL --ref c2m_stage1_final$P --group $G --passes zero --out $X/cmp_${L}_${V}_$G.txt > /dev/null
    cat $X/cmp_${L}_${V}_$G.txt
    $PY $E/indep_score.py --queries $C/arm/main/queries_$V.jsonl --preds $CL --predictor c2m_${L}_closed_$V --group $G | grep -v DONE
  done
  echo "## $V unseen families, ea1 / ea2 (closed think)"
  $PY - $CL $X/hf_${L}_closed_$V.jsonl <<'EOF'
import json, sys
with open(sys.argv[2], "w") as fo:
    for l in open(sys.argv[1]):
        if json.loads(l)["group"] == "heldout_family":
            fo.write(l)
EOF
  $PY $C/review_tmp/ea/ea1.py $X/hf_${L}_closed_$V.jsonl zero inventory2,molopt,thermal2 > $X/ea1_${L}_closed_$V.txt 2>&1
  $PY $C/review_tmp/ea/ea2.py $X/hf_${L}_closed_$V.jsonl zero > $X/ea2_${L}_closed_$V.txt 2>&1
  grep "^PLACE_BLOCK  .*_temp_rise" $X/ea2_${L}_closed_$V.txt
  awk '/=+ thermal2 zero/{on=1} on && /missing FEASIBLE AFTER|applied_both/{print "thermal2 " $0}' $X/ea1_${L}_closed_$V.txt | head -2
  $PY $E/think_diag/think_check2.py $X/hf_${L}_closed_$V.jsonl | sed -n 2,3p
done
echo "## control layer (dirlayer.py, zero-shot)"
CMPV="--cmp rule_vs=$O/rules_v2/preds_rule_vs.jsonl --cmp rule_elem=$O/rules_v2/preds_rule_elem.jsonl --cmp rule_cat=$O/rules_v2/preds_rule_cat.jsonl --cmp rule_catelem=$O/rules_v2/preds_rule_catelem.jsonl"
RD="--thermal-reader $C/review_tmp/d3_reader.py --obs $D/obs_test.jsonl"
$PY $R/dirlayer.py --queries $C/arm/main/queries_v2.jsonl --states $D/test_states.jsonl.gz --pass zero $CMPV $RD \
  --pred s1_v2=$E/preds_c2m_stage1_final.jsonl,$E/hf/preds_c2m_stage1_final.jsonl \
  --pred closed_v2=$X/preds_c2m_${L}_closed_v2.jsonl \
  --pred s1_v3=$E/v3/preds_c2m_stage1_final_v3.jsonl,$E/v3/hf/preds_c2m_stage1_final_v3.jsonl \
  --pred closed_v3=$X/preds_c2m_${L}_closed_v3.jsonl --out $X/dirlayer_$L.json > $X/dirlayer_$L.txt 2> $X/dirlayer_$L.err
echo "dirlayer rc $?"; cat $X/dirlayer_$L.txt
} > $OUT 2>&1
cat $OUT
echo "CLOSED_SCORE_DONE $L $(date '+%F %T')"
