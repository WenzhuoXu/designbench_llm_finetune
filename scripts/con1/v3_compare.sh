#!/bin/bash
# v3_compare.sh LABEL : on drc (CPU), one adapter's zero-shot answers to the same questions with the old prompt
# (queries_v2, move request in) and the clean prompt (queries_v3, con1 eff023a), side by side (con1-2m-rows, 11-27):
#   1. compare_models.py (facets.py's scoring, paired bootstrap over states): readable share, regime_phys, cpl_dir_BA,
#      cpl_size per family and over the group, v3 minus v2; trained families (heldout_problem) and unseen (heldout_family)
#   2. the direction control layer (con1's dirlayer.py, as dirlayer_run.sh runs it): both answer files, every family
#   3. indep_score.py (the second scorer, no facets.py) on each answer file and group
# Needs full_eval/{,hf/}preds_c2m_LABEL.jsonl (v2) and full_eval/v3/{,hf/}preds_c2m_LABEL_v3.jsonl (eval_link.sh QV=v3).
#   bash /storage/wxu/con1/full_eval/v3_compare.sh stage1_final        -> full_eval/v3/compare_LABEL.txt
set -uo pipefail
L=${1:?label}
C=/storage/wxu/con1; E=$C/full_eval; V=$E/v3; R=$C/repo/scripts/con1; D=$C/dl; O=$C/dirlayer
PY=/opt/anaconda3/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=
A2=$E/preds_c2m_$L.jsonl; H2=$E/hf/preds_c2m_$L.jsonl
A3=$V/preds_c2m_${L}_v3.jsonl; H3=$V/hf/preds_c2m_${L}_v3.jsonl
for f in $A2 $H2 $A3 $H3; do [ -s $f ] || { echo "ABORT: no $f"; exit 2; }; done
OUT=$V/compare_$L.txt
{
echo "# $(date '+%F %T') c2m_$L: queries_v2 (move request in the prompt) vs queries_v3 (cut), zero-shot, same adapter"
echo "## 1. facets (compare_models.py), reference = v2"
for G in heldout_problem heldout_family; do
  P2=$A2; P3=$A3; [ $G = heldout_family ] && P2=$H2 && P3=$H3
  $PY $E/compare_models.py --queries $C/arm/main/queries_v2.jsonl --pred c2m_$L=$P2 --pred c2m_${L}_v3=$P3 \
    --ref c2m_$L --group $G --passes zero --out $V/compare_${L}_$G.txt | grep -v "^COMPARE_DONE"
done
echo "## 2. direction control layer (dirlayer.py, zero-shot; model (a) minus the best no-physics comparator)"
CMPV="--cmp rule_vs=$O/rules_v2/preds_rule_vs.jsonl --cmp rule_elem=$O/rules_v2/preds_rule_elem.jsonl --cmp rule_cat=$O/rules_v2/preds_rule_cat.jsonl --cmp rule_catelem=$O/rules_v2/preds_rule_catelem.jsonl"
RD="--thermal-reader $C/review_tmp/d3_reader.py --obs $D/obs_test.jsonl"
$PY $R/dirlayer.py --queries $C/arm/main/queries_v2.jsonl --states $D/test_states.jsonl.gz --pass zero $CMPV $RD \
  --pred v2=$A2,$H2 --pred v3=$A3,$H3 --out $V/dirlayer_$L.json > $V/dirlayer_$L.txt 2> $V/dirlayer_$L.err
echo "dirlayer rc $?"; cat $V/dirlayer_$L.txt
echo "## 3. second scorer (indep_score.py)"
for G in heldout_problem heldout_family; do
  P2=$A2; P3=$A3; [ $G = heldout_family ] && P2=$H2 && P3=$H3
  $PY $E/indep_score.py --queries $C/arm/main/queries_v2.jsonl --preds $P2 --predictor c2m_$L --group $G | grep -v DONE
  $PY $E/indep_score.py --queries $C/arm/main/queries_v2.jsonl --preds $P3 --predictor c2m_${L}_v3 --group $G | grep -v DONE
done
} > $OUT 2>&1
cat $OUT
echo "V3_COMPARE_DONE $L $(date '+%F %T')"
