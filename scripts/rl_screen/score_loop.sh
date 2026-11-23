#!/bin/bash
# score_loop.sh <OUT> <K> <period seconds> <workers>: scoring while the GPUs sample (orchard_screen_data.sh with
# SCORE_ALONGSIDE). Every <period> seconds, the states whose K replies are all in OUT/samples.jsonl and that have no
# state record in OUT/scored.jsonl are copied to a snapshot (complete_states.py) and scored with the launcher's own
# scoring command into OUT/scored.jsonl. A state is scored only with all its replies, so the launcher's last pass over
# the whole samples file adds what is left and repeats nothing. Ends after the pass in progress once
# OUT/logs/score.stop exists. Runs on the sampling job's node (the only writer of scored.jsonl while it runs).
# Environment: SIM_PY, REPO, H (harness dir with dataxp_code_v3).
set -u
OUT="$1"; K="$2"; PERIOD="$3"; W="$4"
S="$REPO/scripts/rl_screen"
L="$OUT/logs/scored_alongside.log"
echo "=== $(date '+%F %T %Z') score loop: $W workers, every $PERIOD s" >> "$L"
while [ ! -e "$OUT/logs/score.stop" ]; do
  T0=$(date +%s)
  "$SIM_PY" "$S/complete_states.py" "$OUT/samples.jsonl" "$OUT/scored.jsonl" "$OUT/logs/score_snapshot.jsonl" "$K" \
      > "$OUT/logs/score_snapshot.txt" 2>&1
  rc=$?
  N=$(awk '/^snapshot states/ {print $3}' "$OUT/logs/score_snapshot.txt")
  echo "=== $(date '+%F %T %Z') rc=$rc $(head -n 1 "$OUT/logs/score_snapshot.txt" | cut -c1-300)" >> "$L"
  if [ "$rc" = 0 ] && [ "${N:-0}" -gt 0 ]; then
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$REPO" \
      "$SIM_PY" "$S/score_states.py" --code "$H/dataxp_code_v3" --states "$OUT/states.jsonl" \
      --samples "$OUT/logs/score_snapshot.jsonl" --out "$OUT/scored.jsonl" --workers "$W" 2>&1 \
      | grep -v -i "warn\|numba\|^[[:space:]]*$" | cut -c1-300 >> "$L"
  fi
  while [ $(( $(date +%s) - T0 )) -lt "$PERIOD" ] && [ ! -e "$OUT/logs/score.stop" ]; do sleep 20; done
done
echo "=== $(date '+%F %T %Z') score loop stopped" >> "$L"
