#!/bin/bash
# The held-out-family test (queries_v2 heldout_family, 1,400 questions, zero-shot + sequential, facetm questions) for
# two existing adapters, one after the other, with the stage-1 run's script and scorer (no training):
#   c2m_pref1 = pref1 final (/storage/wxu/rl_open/inputs/pref1_final), the model before any consequence training
#   c2m_Fm    = con1's 750-row facetm arm adapter (/storage/wxu/con1/arm/run_Fm/final)
# Each label's run dir holds a symlink to the adapter, so con1's run_Fm logs are left alone.
set -uo pipefail
C=/storage/wxu/con1; A=$C/arm; E=$C/full_eval; H=$E/hf
mkdir -p $A/run_c2m_pref1 $A/run_c2m_Fm $H
[ -e $A/run_c2m_pref1/final ] || ln -s /storage/wxu/rl_open/inputs/pref1_final $A/run_c2m_pref1/final
[ -e $A/run_c2m_Fm/final ] || ln -s $A/run_Fm/final $A/run_c2m_Fm/final
for LAB in c2m_pref1 c2m_Fm; do
  echo "=== $(date '+%F %T') $LAB"
  [ -s $H/facets_${LAB}_seq.txt ] || LAB=$LAB FMT=facetm ROWS=none WAIT=1 QUERIES=$A/main/queries_v2.jsonl FAM=all CMP=" " \
    MAXTOK=4096 OUTDIR=$H PASSES="zero seq" QGROUPS=heldout_family bash $E/drc_arm_groups.sh > $H/arm_$LAB.out 2>&1
  grep "server ready\|SKILL_CLIENT_DONE\|ABORT\|DRC_ARM_DONE" $H/arm_$LAB.out | tail -4
  OUTDIR=$H bash $E/hf_score.sh $LAB > $H/hf_score_$LAB.out 2>&1; tail -1 $H/hf_score_$LAB.out
  bash $E/hf_report.sh $LAB > /dev/null 2>&1; echo "report: $(wc -l < $H/report_$LAB.txt) lines"
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore /opt/anaconda3/bin/python $E/copy_type.py \
    --preds $H/preds_$LAB.jsonl --model $LAB --group heldout_family --sign review --out $H/copy_type_$LAB.txt > /dev/null 2>&1
  echo "copy_type: $(tail -1 $H/copy_type_$LAB.txt | cut -c1-80)"
done
echo "HF_OTHERS_DONE $(date '+%F %T')"
