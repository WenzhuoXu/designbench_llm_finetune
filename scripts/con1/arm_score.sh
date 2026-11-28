#!/bin/bash
# arm_score.sh LABEL PREDS FORM : on drc (CPU), one Order-2 thinking arm scored against the stage-1 final v3 answers
# (con1-2m-rows, 11-28, parent's order 2). Truth: queries_v3_inv2cs.jsonl (inventory2 on the common-seed truth);
# flags: flags_v3_inv2cs.tsv (no-op, off-range). Reports in full_eval/arms/LABEL/:
#   split_facets.txt, split_indep.txt   dir_split.py with each reader (items 1, 2, 3, 5 of the order)
#   facets_{headline,all,flagged}_G.txt  compare_models.py group means (fixed reader, LABEL read in FORM)
#   indep_{headline,all}_G.txt           indep_score.py group means (second scorer)
#   dirlayer.txt                         the control layer (lib_v5 dirlayer, both models on the same questions)
set -uo pipefail
LAB=$1; P=$2; FM=$3
C=/storage/wxu/con1; E=$C/full_eval; L=$E/lib_v6; D=$C/dl; O=$C/dirlayer; X=$E/arms/$LAB
PY=/opt/anaconda3/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES= CON1_LIB=$L
QN=$C/arm/main/queries_v3_inv2cs.jsonl; FL=$E/flags_v3_inv2cs.tsv; FA=$E/inv2cs/flagged_any.tsv
S1A=$E/v3/preds_c2m_stage1_final_v3.jsonl; S1H=$E/v3/hf/preds_c2m_stage1_final_v3.jsonl
mkdir -p $X
{ echo "arm $LAB  preds $P  sha1 $(sha1sum < $P | cut -c1-12)  lines $(wc -l < $P)  form $FM"
  echo "truth $QN sha1 $(sha1sum < $QN | cut -c1-12) (inventory2 common-seed truth); flags $FL ($(wc -l < $FL) lines)"
  echo "lib $L; stage 1: $S1A, $S1H"; } | tee $X/report.txt
for R in facets indep; do
  $PY $E/dir_split.py --queries $QN --base $P --base-fmt $FM --ref $S1A,$S1H --flags $FL --reader $R \
    --out $X/split_$R.txt > /dev/null 2> $X/split_$R.err; echo "dir_split $R rc $?" | tee -a $X/report.txt
done
for G in heldout_problem heldout_family; do
  S1=$S1A; [ $G = heldout_family ] && S1=$S1H
  for SPLIT in headline all flagged; do
    ARG=; [ $SPLIT = headline ] && ARG="--exclude $FA"; [ $SPLIT = flagged ] && ARG="--only $FA"
    $PY $E/compare_models.py --queries $QN --pred c2m_stage1_final_v3=$S1 --pred $LAB=$P --fmt-of $LAB=$FM \
      --ref c2m_stage1_final_v3 --group $G --passes zero $ARG --out $X/facets_${SPLIT}_$G.txt > /dev/null 2>> $X/cmp.err
    echo "compare_models $G $SPLIT rc $?" | tee -a $X/report.txt
    [ $SPLIT = flagged ] && continue
    { $PY $E/indep_score.py --queries $QN --preds $S1 --predictor c2m_stage1_final_v3 --group $G $ARG | grep -v DONE
      $PY $E/indep_score.py --queries $QN --preds $P --predictor $LAB --group $G --fmt $FM $ARG | grep -v DONE; } \
      > $X/indep_${SPLIT}_$G.txt 2>> $X/indep.err
  done
done
CMPV="--cmp rule_vs=$O/rules_v2/preds_rule_vs.jsonl --cmp rule_elem=$O/rules_v2/preds_rule_elem.jsonl --cmp rule_cat=$O/rules_v2/preds_rule_cat.jsonl --cmp rule_catelem=$O/rules_v2/preds_rule_catelem.jsonl"
RD="--thermal-reader $C/review_tmp/d3_reader.py --obs $D/obs_test.jsonl"
$PY $L/dirlayer.py --queries $QN --states $D/test_states.jsonl.gz --pass zero $CMPV $RD \
  --pred s1_v3=$S1A,$S1H --pred $LAB=$P --both s1_v3,$LAB --out $X/dirlayer.json > $X/dirlayer.txt 2> $X/dirlayer.err
echo "dirlayer rc $?" | tee -a $X/report.txt
echo "ARM_SCORE_DONE $LAB $(date '+%F %T')" | tee -a $X/report.txt
