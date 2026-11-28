#!/bin/bash
# inv2cs_score.sh : on drc (CPU), the common-seed inventory2 truth (queries_v3_inv2cs.jsonl, inv2_common_seed.py)
# checked and used (con1-2m-rows, 11-28, parent's order). New reports in full_eval/inv2cs/; old reports stay.
#   1. prompts: system / user of every record identical to queries_v3 (byte comparison), and noconf_check --pairs
#      queries_v2 queries_v3_inv2cs --every-question (the move-request check of every question's prompt)
#   2. flags on the new truth (make_flags.py: inventory2 moves that are now exact no-ops get the flag)
#   3. the unseen families rescored on the new truth (facets.py fixed reader via compare_models, indep_score.py):
#      the stage-1 final v3, step 850 closed v3 and base 4k; all questions, and with the new flags out; plus dirlayer
set -uo pipefail
C=/storage/wxu/con1; E=$C/full_eval; X=$E/inv2cs; L5=$E/lib_v5; D=$C/dl; O=$C/dirlayer
PY=/opt/anaconda3/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES= PYTHONPATH=$C/pylib:$C/repo
Q3=$C/arm/main/queries_v3.jsonl; QN=$C/arm/main/queries_v3_inv2cs.jsonl
mkdir -p $X
echo "queries_v3_inv2cs.jsonl sha1 $(sha1sum < $QN | cut -c1-12)" | tee $X/report.txt
$PY - $Q3 $QN <<'EOF' | tee -a $X/report.txt
import json, sys, collections
c = collections.Counter()
for l3, ln in zip(open(sys.argv[1]), open(sys.argv[2])):
    a, b = json.loads(l3), json.loads(ln)
    c["records"] += 1
    c["system+user identical"] += a["system"] == b["system"] and a["user"] == b["user"]
    c["line identical (other families)"] += l3 == ln
    c["moves differ"] += a["moves"] != b["moves"]
    c["inventory2 records"] += a["variant"] == "inventory2"
    other = {k for k in set(a) | set(b) if k not in ("moves", "truth") and a.get(k) != b.get(k)}
    c["fields other than moves/truth differ"] += bool(other)
print("prompt check vs queries_v3:", dict(c))
EOF
(cd $L5 && $PY noconf_check.py --pairs $C/arm/main/queries_v2.jsonl $QN --every-question --fmt facetm \
  --out $X/noconf_v2_vs_inv2cs.json > $X/noconf_v2_vs_inv2cs.txt 2>&1); echo "noconf rc $?" | tee -a $X/report.txt
grep "TOTAL\|inventory2" $X/noconf_v2_vs_inv2cs.txt | tee -a $X/report.txt
$PY $E/make_flags.py $QN $E/flags_v3_inv2cs.tsv | tee $X/flags.txt | grep "inventory2\|flags:" | tee -a $X/report.txt
cut -f1,2 $E/flags_v3_inv2cs.tsv | sort -u > $X/flagged_any.tsv
H=heldout_family
P1=$E/v3/hf/preds_c2m_stage1_final_v3.jsonl
P2=$E/closed/preds_c2m_full_00850_closed_v3.jsonl
P3=$E/base_think/preds_base_v3_4k.jsonl
for SPLIT in all headline; do
  ARG=; [ $SPLIT = headline ] && ARG="--exclude $X/flagged_any.tsv"
  for TRUTH in old new; do
    QQ=$Q3; [ $TRUTH = new ] && QQ=$QN
    CON1_LIB=$L5 $PY $E/compare_models.py --queries $QQ --pred c2m_stage1_final_v3=$P1 --pred c2m_full_00850_closed_v3=$P2 \
      --pred base_v3_4k=$P3 --ref c2m_stage1_final_v3 --group $H --passes zero $ARG --out $X/facets_${SPLIT}_${TRUTH}truth.txt > /dev/null
    { for P in "c2m_stage1_final_v3 $P1" "c2m_full_00850_closed_v3 $P2" "base_v3_4k $P3"; do set -- $P
        $PY $E/indep_score.py --queries $QQ --preds $2 --predictor $1 --group $H $ARG | grep -v DONE; done; } > $X/indep_${SPLIT}_${TRUTH}truth.txt
  done
done
CMPV="--cmp rule_vs=$O/rules_v2/preds_rule_vs.jsonl --cmp rule_elem=$O/rules_v2/preds_rule_elem.jsonl --cmp rule_cat=$O/rules_v2/preds_rule_cat.jsonl --cmp rule_catelem=$O/rules_v2/preds_rule_catelem.jsonl"
RD="--thermal-reader $C/review_tmp/d3_reader.py --obs $D/obs_test.jsonl"
$PY $L5/dirlayer.py --queries $QN --states $D/test_states.jsonl.gz --pass zero $CMPV $RD --families inventory2 \
  --pred s1_v3=$P1 --pred closed850_v3=$P2 --pred base_4k=$P3 --out $X/dirlayer_newtruth.json > $X/dirlayer_newtruth.txt 2> $X/dirlayer_newtruth.err
echo "dirlayer rc $?" | tee -a $X/report.txt
echo "INV2CS_SCORE_DONE $(date '+%F %T')"
