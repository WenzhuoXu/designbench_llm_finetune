#!/bin/bash
# health.sh   (login node): one status line for job 173954's sampling; exit 0 = fine, 10 = needs a look,
# 11 = the sampler moved on to another family (told once per family), 12 = the sampler finished its list
J=173954
OUT=/project/community/wxu2/rl/screen/dx38mtR_scr2_fcq
H=$HOME/rl/logs/fcswap
BAD=""
ST=$(squeue -h -j $J -o %T 2>/dev/null)
[ "$ST" = RUNNING ] || BAD="$BAD job=${ST:-gone};"
STEPS=$(squeue -h -s -j $J -o %j 2>/dev/null | tr '\n' ' ')
# finished: the last start of the sampler in its log is followed by its closing line
DONE=$(awk '/tuned sampler: \// {s=NR} /^SAMPLE_STATES_DONE/ {d=NR} END {print (d > s && s > 0) ? 1 : 0}' "$OUT/logs/samples_tuned.log" 2>/dev/null)
DONE=${DONE:-0}
case " $STEPS " in *" fcsampler "*) ;; *) [ "$DONE" = 1 ] || BAD="$BAD no sampler step;" ;; esac
case " $STEPS " in *" fcscore "*) ;; *) BAD="$BAD no scoring step;" ;; esac
NOW=$(date +%s); MT=$(stat -c %Y "$OUT/samples.jsonl"); AGE=$(( NOW - MT ))
[ "$DONE" = 1 ] || [ "$AGE" -lt 900 ] || BAD="$BAD samples.jsonl not written for ${AGE}s;"
LAST=$(grep 'states, .* replies' "$OUT/logs/samples_tuned.log" | tail -n 1 | cut -c1-110)
ERRS=$(echo "$LAST" | sed -n 's/.* replies, \([0-9]*\) errors.*/\1/p')
[ "${ERRS:-0}" -lt 50 ] || BAD="$BAD $ERRS reply errors;"
grep -q "ABORT" "$H/step.log" "$H/score_step.log" 2>/dev/null && [ ! -e "$H/abort.seen" ] && BAD="$BAD ABORT in a step log;"
FAM=$(tail -c 400000 "$OUT/samples.jsonl" | grep -o '"sid": "[a-z0-9_]*|' | tail -n 1 | sed 's/"sid": "//; s/|//')
PREV=$(cat "$H/family.last" 2>/dev/null)
N=$(wc -l < "$OUT/samples.jsonl"); NS=$(grep -c '"state": true' "$OUT/scored.jsonl" 2>/dev/null)
SC=$(tail -n 1 "$OUT/logs/scored_early.log" 2>/dev/null | cut -c1-120)
echo "$(date '+%m-%d %H:%M') job=$ST steps=[$STEPS] replies=$N family=$FAM scored_states=$NS | $LAST | $SC"
[ -n "$BAD" ] && { echo "ATTENTION:$BAD"; exit 10; }
[ "$DONE" = 1 ] && { echo "SAMPLER FINISHED"; exit 12; }
if [ -n "$FAM" ] && [ "$FAM" != "$PREV" ]; then
  echo "$FAM" > "$H/family.last"
  [ -n "$PREV" ] && { echo "FAMILY CHANGE: $PREV -> $FAM"; exit 11; }
fi
exit 0
