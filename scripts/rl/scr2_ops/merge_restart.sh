#!/bin/bash
# merge_restart.sh <tag> <fraction for drc> <drc replies gz or -> <base states file> <sampler arguments ...>   (login node)
# The main sampler is stopped; drc's replies (the gz) and the extra GPU job's replies so far are appended to the main
# samples file (only what it does not hold); the states still to do are split (the last <fraction> are left to drc,
# which works from the tail); the main sampler is started again on the rest with the given arguments.
set -u
TAG="${1:?tag}"; FRAC="${2:?fraction}"; G="${3:?gz or -}"; BASE="${4:?base states file}"; shift 4
J=173954
P=/project/community/wxu2/rl/screen; OUT=$P/dx38mtR_scr2_fcq; H=$HOME/rl/logs/fcswap; D=$OUT/logs/swap
PY=/project/community/wxu2/rl/envs/sim/bin/python
[ "$G" = - ] || [ -s "$G" ] || { echo "ABORT: no file $G"; exit 2; }
[ -s "$BASE" ] || { echo "ABORT: no file $BASE"; exit 2; }
[ -e "$D/states_main_$TAG.jsonl" ] && { echo "ABORT: tag $TAG was used already"; exit 2; }
echo "=== $(date '+%F %T %Z') merge_restart $TAG"
bash "$H/stop_tuned.sh" | grep -v "^ \|STEPID"
for i in $(seq 1 24); do
  squeue -h -s -j $J -o %j | grep -q -x "fcsampler\|bash" || break
  sleep 5
done
squeue -h -s -j $J -o %j | grep -q -x "fcsampler\|bash" && { echo "ABORT: the sampler step is still there"; exit 3; }
N0=$(wc -l < "$OUT/samples.jsonl")
if [ "$G" != - ]; then
  "$PY" "$H/merge_append.py" "$OUT/samples.jsonl" "$G" || { echo "ABORT: merge of $G failed"; exit 4; }
fi
G1=$P/dx38mtR_scr2_fcq_g1/samples.jsonl
if [ -s "$G1" ]; then
  "$PY" "$H/merge_append.py" "$OUT/samples.jsonl" "$G1" || { echo "ABORT: merge of $G1 failed"; exit 4; }
fi
echo "samples.jsonl: $N0 -> $(wc -l < "$OUT/samples.jsonl") lines"
"$PY" "$H/split_states.py" "$BASE" "$OUT/samples.jsonl" "$FRAC" "$D/states_main_$TAG.jsonl" "$D/sids_drc_$TAG.txt" || { echo "ABORT: split failed"; exit 5; }
bash "$H/start_tuned.sh" 96 "$D/states_main_$TAG.jsonl" "$@" | tail -n 4
sleep 15
tail -n 2 "$OUT/logs/samples_tuned.log" | cut -c1-220
