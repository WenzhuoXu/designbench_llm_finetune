#!/bin/bash
# con1 direction control layer on drc (CPU): the review's rule answer files for both question sets, then dirlayer.py
# on every answer file that exists. Outputs in /storage/wxu/con1/dirlayer/.
set -uo pipefail
C=/storage/wxu/con1
R=$C/repo/scripts/con1
D=$C/dl
O=$C/dirlayer
PY=/opt/anaconda3/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=
mkdir -p $O/rules_v2 $O/rules_r1
cd $O
[ -s $O/rules_v2/preds_rule_vs.jsonl ] || $PY $D/f5_rulepred.py --repo $R --src $D/test_states.jsonl.gz --queries $C/arm/main/queries_v2.jsonl --outdir $O/rules_v2 > $O/rules_v2.log 2>&1
[ -s $O/rules_r1/preds_rule_vs.jsonl ] || $PY $D/f5_rulepred.py --repo $R --src $D/test_states.jsonl.gz --queries $C/arm/queries.jsonl --outdir $O/rules_r1 > $O/rules_r1.log 2>&1
ls $O/rules_v2 $O/rules_r1
CMPV="--cmp rule_vs=$O/rules_v2/preds_rule_vs.jsonl --cmp rule_elem=$O/rules_v2/preds_rule_elem.jsonl --cmp rule_cat=$O/rules_v2/preds_rule_cat.jsonl --cmp rule_catelem=$O/rules_v2/preds_rule_catelem.jsonl"
CMPR="--cmp rule_vs=$O/rules_r1/preds_rule_vs.jsonl --cmp rule_elem=$O/rules_r1/preds_rule_elem.jsonl --cmp rule_cat=$O/rules_r1/preds_rule_cat.jsonl --cmp rule_catelem=$O/rules_r1/preds_rule_catelem.jsonl"
RD="--thermal-reader $C/review_tmp/d3_reader.py --obs $D/obs_test.jsonl"
S1C2M=$C/full_eval/preds_c2m_stage1_final.jsonl,$C/full_eval/hf/preds_c2m_stage1_final.jsonl
PREF1=$C/full_eval/hf/preds_c2m_pref1.jsonl
EXTRA=""; for f in $C/full_eval/hf/preds_c2m_*750*.jsonl $C/full_eval/hf/preds_*Fm*.jsonl; do [ -s "$f" ] && EXTRA="$EXTRA --pred $(basename $f .jsonl | sed 's/^preds_//')=$f"; done
echo "=== zero-shot, queries_v2 (all 11 families)"
$PY $R/dirlayer.py --queries $C/arm/main/queries_v2.jsonl --states $D/test_states.jsonl.gz --pass zero $CMPV $RD \
  --pred s1_facetm=$S1C2M --pred s1_facet=$D/preds_stage1_run3.jsonl --pred trees=$C/arm/main/preds_trees.jsonl \
  --pred pref1=$PREF1 $EXTRA --out $O/zero_v2.json > $O/zero_v2.txt 2> $O/zero_v2.err; echo "rc $?"
echo "=== sequential, queries_v2"
$PY $R/dirlayer.py --queries $C/arm/main/queries_v2.jsonl --states $D/test_states.jsonl.gz --pass seq $CMPV $RD \
  --pred s1_facetm=$S1C2M --pred s1_facet=$D/preds_stage1_run3.jsonl --pred pref1=$PREF1 $EXTRA \
  --both s1_facetm,pref1 --out $O/seq_v2.json > $O/seq_v2.txt 2> $O/seq_v2.err; echo "rc $?"
echo "=== zero-shot, lines both stage 1 and pref1 give a direction"
$PY $R/dirlayer.py --queries $C/arm/main/queries_v2.jsonl --states $D/test_states.jsonl.gz --pass zero $CMPV $RD \
  --families inventory2 molopt thermal2 --pred s1_facetm=$S1C2M --pred pref1=$PREF1 --both s1_facetm,pref1 \
  --boot 200 --out $O/zero_both.json > $O/zero_both.txt 2> $O/zero_both.err; echo "rc $?"
echo "=== round-1 question set (5 families): the 750-row facetm arm, zero and sequential"
for PS in zero seq; do
  $PY $R/dirlayer.py --queries $C/arm/queries.jsonl --states $D/test_states.jsonl.gz --pass $PS $CMPR $RD \
    --families engibench grid2op photonics circuit2 frame3d2 --pred Fm=$C/arm/preds_Fm.jsonl \
    --pred F_r1=$C/arm/r1/preds_F.jsonl --pred trees_r1=$C/arm/r1/preds_trees.jsonl \
    --out $O/${PS}_r1.json > $O/${PS}_r1.txt 2> $O/${PS}_r1.err; echo "rc $?"
done
echo "DIRLAYER_DONE $(date '+%F %T')"
