#!/bin/bash
# base_think_score.sh : on drc (CPU), the base model with thinking (base_think.sh's answers, copied from Orchard into
# full_eval/base_think/) next to the stage-1 final's queries_v3 answers (full_eval/v3/), every family, zero-shot:
#   0. think_norm.py (layout only) on both base files; base_think_counts.py (think_end, unreadable, normalised,
#      omitted lines, per group / family)
#   1. compare_models.py (facets.py's scoring), reference = the stage-1 final, paired se over states; all questions
#      (an unreadable answer is wrong) and the questions the base model answered readably (both models on those)
#   2. the direction control layer (dirlayer.py): base and stage 1, every family
#   3. indep_score.py (second scorer) on each file and group
#   4. the 300-question subset: budget 4,000 vs 8,000 (compare_models, reference 4,000; counts)
# Outputs: full_eval/base_think/report_*.txt
set -uo pipefail
C=/storage/wxu/con1; E=$C/full_eval; BT=$E/base_think; R=$C/repo/scripts/con1; D=$C/dl; O=$C/dirlayer
PY=/opt/anaconda3/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=
QV3=$C/arm/main/queries_v3.jsonl
B4=$BT/preds_base_v3_4k.jsonl; B8=$BT/preds_base_v3_8k_sub300.jsonl
S1A=$E/v3/preds_c2m_stage1_final_v3.jsonl; S1H=$E/v3/hf/preds_c2m_stage1_final_v3.jsonl
for f in $B4 $S1A $S1H; do [ -s $f ] || { echo "ABORT: no $f"; exit 2; }; done
cd $BT
echo "## 0. normalisation and counts"
$PY $E/think_diag/think_norm.py $B4 $BT/preds_base_v3_4k_norm.jsonl
[ -s $B8 ] && $PY $E/think_diag/think_norm.py $B8 $BT/preds_base_v3_8k_sub300_norm.jsonl
$PY $E/base_think_counts.py $B4 $BT/preds_base_v3_4k_norm.jsonl $QV3 | tee $BT/report_counts_4k.txt
[ -s $B8 ] && $PY $E/base_think_counts.py $B8 $BT/preds_base_v3_8k_sub300_norm.jsonl $QV3 | tee $BT/report_counts_8k_sub300.txt
# readable-only copies of the base answers (the second denominator)
$PY - $BT/preds_base_v3_4k_norm.jsonl $BT/preds_base_v3_4k_norm_readable.jsonl $QV3 <<'EOF'
import json, sys
sys.path.insert(0, "/storage/wxu/con1/repo/scripts/con1")
import facets as F
qs = {json.loads(l)["sid"]: json.loads(l) for l in open(sys.argv[3])}
n = k = 0
with open(sys.argv[2], "w") as fo:
    for l in open(sys.argv[1]):
        r = json.loads(l); n += 1
        if F.pred_facets(r.get("text"), "facetm", qs[r["sid"]])["regime"] != "unreadable":
            fo.write(l); k += 1
print("readable base answers: %d of %d" % (k, n))
EOF
echo "## 1. facets: base (thinking, budget 4,000) vs the stage-1 final (no thinking), queries_v3"
for G in heldout_problem heldout_family; do
  S1=$S1A; [ $G = heldout_family ] && S1=$S1H
  $PY $E/compare_models.py --queries $QV3 --pred c2m_stage1_final_v3=$S1 --pred base_v3_4k=$BT/preds_base_v3_4k_norm.jsonl \
    --ref c2m_stage1_final_v3 --group $G --passes zero --out $BT/report_facets_all_$G.txt > /dev/null
  $PY $E/compare_models.py --queries $QV3 --pred c2m_stage1_final_v3=$S1 --pred base_v3_4k=$BT/preds_base_v3_4k_norm_readable.jsonl \
    --ref c2m_stage1_final_v3 --group $G --passes zero --out $BT/report_facets_readable_$G.txt > /dev/null
  echo "### $G, all questions"; cat $BT/report_facets_all_$G.txt
  echo "### $G, questions the base model answered readably"; cat $BT/report_facets_readable_$G.txt
done
echo "## 2. direction control layer (dirlayer.py, zero-shot)"
CMPV="--cmp rule_vs=$O/rules_v2/preds_rule_vs.jsonl --cmp rule_elem=$O/rules_v2/preds_rule_elem.jsonl --cmp rule_cat=$O/rules_v2/preds_rule_cat.jsonl --cmp rule_catelem=$O/rules_v2/preds_rule_catelem.jsonl"
RD="--thermal-reader $C/review_tmp/d3_reader.py --obs $D/obs_test.jsonl"
$PY $R/dirlayer.py --queries $C/arm/main/queries_v2.jsonl --states $D/test_states.jsonl.gz --pass zero $CMPV $RD \
  --pred s1_v3=$S1A,$S1H --pred base_4k=$BT/preds_base_v3_4k_norm.jsonl --both s1_v3,base_4k \
  --out $BT/report_dirlayer.json > $BT/report_dirlayer.txt 2> $BT/report_dirlayer.err
echo "dirlayer rc $?"; cat $BT/report_dirlayer.txt
echo "## 3. second scorer (indep_score.py)"
for G in heldout_problem heldout_family; do
  S1=$S1A; [ $G = heldout_family ] && S1=$S1H
  $PY $E/indep_score.py --queries $QV3 --preds $S1 --predictor c2m_stage1_final_v3 --group $G | grep -v DONE
  $PY $E/indep_score.py --queries $QV3 --preds $BT/preds_base_v3_4k_norm.jsonl --predictor base_v3_4k --group $G | grep -v DONE
done | tee $BT/report_indep.txt
if [ -s $B8 ]; then
  echo "## 4. the 300-question subset: budget 4,000 vs 8,000"
  for G in heldout_problem heldout_family; do
    $PY $E/compare_models.py --queries $QV3 --pred base_v3_4k=$BT/preds_base_v3_4k_norm.jsonl \
      --pred base_v3_8k_sub300=$BT/preds_base_v3_8k_sub300_norm.jsonl --ref base_v3_4k --group $G --passes zero \
      --out $BT/report_budget_$G.txt > /dev/null
    cat $BT/report_budget_$G.txt
    for P in "base_v3_4k $BT/preds_base_v3_4k_norm.jsonl" "base_v3_8k_sub300 $BT/preds_base_v3_8k_sub300_norm.jsonl"; do
      set -- $P
      $PY - $2 $BT/subset_300.tsv $1 $G $QV3 > $BT/sub_$1_$G.jsonl <<'EOF'
import json, sys
sub = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open(sys.argv[2]) if l.strip()}
qs = {json.loads(l)["sid"]: json.loads(l)["group"] for l in open(sys.argv[5])}
for l in open(sys.argv[1]):
    r = json.loads(l)
    if (r["sid"], r["qi"]) in sub and qs[r["sid"]] == sys.argv[4]:
        sys.stdout.write(l)
EOF
      $PY $E/indep_score.py --queries $QV3 --preds $BT/sub_$1_$G.jsonl --predictor $1 --group $G | grep -v DONE
    done
  done | tee $BT/report_budget_indep.txt
fi
echo "BASE_THINK_SCORE_DONE $(date '+%F %T')"
