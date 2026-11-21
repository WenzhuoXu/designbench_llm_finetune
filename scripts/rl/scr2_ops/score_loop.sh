#!/bin/bash
# score_loop.sh [workers=24] [period seconds=900]   inside job 173954's allocation (srun --jobid=173954 --overlap)
# Scoring while the GPUs sample: every <period> seconds, the states whose 8 replies are all in samples.jsonl and that
# have no record in scored.jsonl yet are scored with the launcher's own scoring command (same environment, same code,
# same output file), on a snapshot of those replies. A state is only ever scored with all its replies, so the last
# pass over the whole samples file adds what is left and repeats nothing. Ends when logs/swap/score.stop exists.
set -u
W="${1:-24}"; PERIOD="${2:-900}"
OUT=/project/community/wxu2/rl/screen/dx38mtR_scr2_fcq
HERE=$HOME/rl/logs/fcswap; D=$OUT/logs/swap
REPO=$HOME/rl/repo; S=$REPO/scripts/rl_screen; H=$HOME/rl/harness
[ -s "$D/orig.environ" ] && [ -s "$D/orig.cmdline" ] || { echo "ABORT: no saved environment in $D"; exit 2; }
PY=$(tr '\0' '\n' < "$D/orig.cmdline" | head -n 1)
[ "$(pgrep -u "$USER" -f "score_states.py --code" | wc -l)" = 0 ] || { echo "ABORT: a scoring run is already there"; exit 3; }
echo "=== $(date '+%F %T %Z') score loop: $W workers, every $PERIOD s" >> "$OUT/logs/scored_early.log"
while [ ! -e "$D/score.stop" ]; do
  T0=$(date +%s)
  "$PY" "$HERE/complete_states.py" "$OUT/samples.jsonl" "$OUT/scored.jsonl" "$D/score_snapshot.jsonl" 8 > "$D/snapshot.txt" 2>&1
  rc=$?
  N=$(tail -n 1 "$D/snapshot.txt" | awk '/^snapshot states/ {print $3}')
  echo "=== $(date '+%F %T %Z') $(head -n 1 "$D/snapshot.txt" | cut -c1-300)" >> "$OUT/logs/scored_early.log"
  if [ "$rc" = 0 ] && [ "${N:-0}" -gt 0 ]; then
    "$PY" "$HERE/run_env.py" "$D" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$REPO" -- \
        "$PY" "$S/score_states.py" --code "$H/dataxp_code_v3" --states "$OUT/states.jsonl" --samples "$D/score_snapshot.jsonl" \
        --out "$OUT/scored.jsonl" --workers "$W" 2>&1 | grep -v -i "warn\|numba\|^[[:space:]]*$" \
        | grep "states\|SCORE_STATES_DONE\|!!!\|rror" | cut -c1-300 >> "$OUT/logs/scored_early.log"
  fi
  while [ $(( $(date +%s) - T0 )) -lt "$PERIOD" ] && [ ! -e "$D/score.stop" ]; do sleep 20; done
done
echo "=== $(date '+%F %T %Z') score loop stopped" >> "$OUT/logs/scored_early.log"
