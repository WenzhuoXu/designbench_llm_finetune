#!/bin/bash
# reader_fix_rescore.sh : on drc (CPU), the 11-27 answers rescored with the fixed reader (facets.py's _change_lines in
# full_eval/lib_v4 = commit cfa930e; indep_score.py's changes_items), parent's order 1a + 1c (con1-2m-rows, 11-28).
# Reads full_eval/base_think/ and full_eval/v3/ (nothing there is written); new reports in full_eval/base_think_fix/:
#   flags        make_flags.py on queries_v3 -> full_eval/flags_v3.tsv (no-op, off-range)
#   stage1_check the stage-1 final's v3 answers, old reader vs fixed reader (must not move)
#   facets_*     base (raw answers, no think_norm) vs the stage-1 final, fixed reader: headline = flagged questions out,
#                then the flagged questions alone, then everything
#   indep_*      the second scorer on the same splits
#   counts       base_think_counts.py with the fixed reader (lines truly absent)
#   dirlayer     the control layer with the fixed reader
set -uo pipefail
C=/storage/wxu/con1; E=$C/full_eval; BT=$E/base_think; X=$E/base_think_fix; L4=$E/lib_v4; D=$C/dl; O=$C/dirlayer
PY=/opt/anaconda3/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=
QV3=$C/arm/main/queries_v3.jsonl
B4=$BT/preds_base_v3_4k.jsonl
S1A=$E/v3/preds_c2m_stage1_final_v3.jsonl; S1H=$E/v3/hf/preds_c2m_stage1_final_v3.jsonl
mkdir -p $X
FL=$E/flags_v3.tsv
$PY $E/make_flags.py $QV3 $FL | tee $X/flags.txt
cut -f1,2 $FL | sort -u > $X/flagged_any.tsv
echo "== stage-1 final v3: old reader vs fixed reader" | tee $X/stage1_check.txt
for G in heldout_problem heldout_family; do
  S1=$S1A; [ $G = heldout_family ] && S1=$S1H
  $PY $E/compare_models.py --queries $QV3 --pred c2m_stage1_final_v3=$S1 --ref c2m_stage1_final_v3 --group $G --passes zero \
    --out $X/s1_old_$G.txt > /dev/null
  CON1_LIB=$L4 $PY $E/compare_models.py --queries $QV3 --pred c2m_stage1_final_v3=$S1 --ref c2m_stage1_final_v3 --group $G \
    --passes zero --out $X/s1_fixed_$G.txt > /dev/null
  if diff -q <(grep -v "^models side" $X/s1_old_$G.txt) <(grep -v "^models side" $X/s1_fixed_$G.txt) > /dev/null; then
    echo "  $G: identical (every family and group line)"; else echo "  $G: DIFFERENT"; diff $X/s1_old_$G.txt $X/s1_fixed_$G.txt | head; fi
done | tee -a $X/stage1_check.txt
for G in heldout_problem heldout_family; do
  S1=$S1A; [ $G = heldout_family ] && S1=$S1H
  for SPLIT in headline flagged all; do
    ARG=; [ $SPLIT = headline ] && ARG="--exclude $X/flagged_any.tsv"; [ $SPLIT = flagged ] && ARG="--only $X/flagged_any.tsv"
    CON1_LIB=$L4 $PY $E/compare_models.py --queries $QV3 --pred c2m_stage1_final_v3=$S1 --pred base_v3_4k=$B4 \
      --ref c2m_stage1_final_v3 --group $G --passes zero $ARG --out $X/facets_${SPLIT}_$G.txt > /dev/null
    { $PY $E/indep_score.py --queries $QV3 --preds $S1 --predictor c2m_stage1_final_v3 --group $G $ARG | grep -v DONE
      $PY $E/indep_score.py --queries $QV3 --preds $B4 --predictor base_v3_4k --group $G $ARG | grep -v DONE; } > $X/indep_${SPLIT}_$G.txt
  done
done
for f in no-op off-range; do awk -F'\t' -v f=$f '$3==f{print $1"\t"$2}' $FL > $X/flag_$f.tsv; done
for G in heldout_problem heldout_family; do
  S1=$S1A; [ $G = heldout_family ] && S1=$S1H
  for f in no-op off-range; do
    [ -s $X/flag_$f.tsv ] || continue
    CON1_LIB=$L4 $PY $E/compare_models.py --queries $QV3 --pred c2m_stage1_final_v3=$S1 --pred base_v3_4k=$B4 \
      --ref c2m_stage1_final_v3 --group $G --passes zero --only $X/flag_$f.tsv --out $X/facets_flag_${f}_$G.txt > /dev/null 2>&1
  done
done
CON1_LIB=$L4 $PY $E/base_think_counts.py $B4 $B4 $QV3 > $X/counts_4k_raw_fixed.txt
CMPV="--cmp rule_vs=$O/rules_v2/preds_rule_vs.jsonl --cmp rule_elem=$O/rules_v2/preds_rule_elem.jsonl --cmp rule_cat=$O/rules_v2/preds_rule_cat.jsonl --cmp rule_catelem=$O/rules_v2/preds_rule_catelem.jsonl"
RD="--thermal-reader $C/review_tmp/d3_reader.py --obs $D/obs_test.jsonl"
$PY $L4/dirlayer.py --queries $C/arm/main/queries_v2.jsonl --states $D/test_states.jsonl.gz --pass zero $CMPV $RD \
  --pred s1_v3=$S1A,$S1H --pred base_4k=$B4 --both s1_v3,base_4k --out $X/dirlayer.json > $X/dirlayer.txt 2> $X/dirlayer.err
echo "dirlayer rc $?"
echo "READER_FIX_RESCORE_DONE $(date '+%F %T')"
