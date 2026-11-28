#!/bin/bash
# arm_split.sh OUTDIR "ARMSPEC [ARMSPEC ...]" [SUBSET|-] [SELECT ...] : on drc (CPU), the 11-28 scoring (parent's order A)
# of one or more thinking arms against stage 1 v3 and the no-physics table: arm_split.py with each reader, and
# recount_split.py (the attack check's reader, own statistics) on the first arm as the second scorer.
#   ARMSPEC = LABEL=preds.jsonl:FORM ; SELECT = LABEL:closed|budget (think_end of that arm)
# Truth queries_v3_inv2cs.jsonl (inventory2 on the common-seed truth); question flags flags_v3_inv2cs.tsv; line flags
# line_flags_v3_inv2cs.tsv (make_line_flags.py); table: review_tmp/v1128D/attack/comp_{preds,best}.pkl (read only).
set -uo pipefail
X=$1; ARMS=$2; SUB=${3:--}; shift 3 || shift $#
SEL=("$@")
C=/storage/wxu/con1; E=$C/full_eval; AT=$C/review_tmp/v1128D/attack
PY=/opt/anaconda3/bin/python
export CON1_LIB=$E/lib_v6 OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
QN=$C/arm/main/queries_v3_inv2cs.jsonl; QF=$E/flags_v3_inv2cs.tsv; LF=$E/line_flags_v3_inv2cs.tsv
S1=$E/v3/preds_c2m_stage1_final_v3.jsonl,$E/v3/hf/preds_c2m_stage1_final_v3.jsonl
mkdir -p $X
ARGS=(); for A in $ARMS; do ARGS+=(--arm "$A"); done
SARGS=(); [ "$SUB" != - ] && SARGS+=(--subset "$SUB"); for S in "${SEL[@]}"; do SARGS+=(--select "$S"); done
{ echo "arm_split.sh $(date '+%F %T'); arms: $ARMS; subset $SUB; select ${SEL[*]:-none}"
  for A in $ARMS; do P=${A#*=}; P=${P%:*}; echo "  $P sha1 $(sha1sum < $P | cut -c1-12) lines $(wc -l < $P)"; done
  echo "  truth $QN sha1 $(sha1sum < $QN | cut -c1-12); line flags $LF sha1 $(sha1sum < $LF | cut -c1-12)"
  echo "  table $AT/comp_preds.pkl sha1 $(sha1sum < $AT/comp_preds.pkl | cut -c1-12), comp_best.pkl $(sha1sum < $AT/comp_best.pkl | cut -c1-12)"; } > $X/report.txt
for R in facets indep; do
  $PY $E/arms/arm_split.py --queries $QN --qflags $QF --lflags $LF --ref $S1 --comp $AT/comp_preds.pkl \
    --comp-best $AT/comp_best.pkl "${ARGS[@]}" "${SARGS[@]}" --reader $R --keys-out $X/keys_$R.tsv \
    --out $X/split_$R.txt > /dev/null 2> $X/split_$R.err
  echo "arm_split $R rc $?" >> $X/report.txt
done
# second scorer: every arm on the questions arm_split used (its keys file; selection already applied)
for A in $ARMS; do
  L=${A%%=*}; P=${A#*=}; P=${P%:*}
  $PY $E/arms/recount_split.py $QN $QF $LF $AT/comp_preds.pkl $AT/comp_best.pkl $S1 $P $X/keys_facets.tsv > $X/recount_$L.txt 2>&1
  echo "recount $L rc $?" >> $X/report.txt
done
echo "ARM_SPLIT_SH_DONE $(date '+%F %T')" >> $X/report.txt
cat $X/report.txt
