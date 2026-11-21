#!/bin/bash
# node_start.sh <states in flight> [states file|""] [extra sampler arguments ...]
# Runs inside job 173954's allocation (srun --jobid=173954 --overlap). The launcher's own sampler is stopped with
# SIGSTOP (it stays alive, so the job and its four servers stay up) and the same command is run with another number
# of states in flight. Replies go to the same samples.jsonl; the sampler skips what is already there.
set -u
N="${1:?states in flight}"; STATES="${2:-}"; shift; [ $# -gt 0 ] && shift; EXTRA=("$@")
OUT=/project/community/wxu2/rl/screen/dx38mtR_scr2_fcq
HERE=$HOME/rl/logs/fcswap
D=$OUT/logs/swap; mkdir -p "$D"
PAT="sample_states.py --states"
if [ ! -s "$D/orig.pid" ]; then
  PIDS=$(pgrep -u "$USER" -f "$PAT" | tr '\n' ' ')
  set -- $PIDS
  [ $# = 1 ] || { echo "ABORT: expected one sampler, found: $PIDS"; exit 2; }
  { cp "/proc/$1/cmdline" "$D/orig.cmdline" && cp "/proc/$1/environ" "$D/orig.environ" && readlink "/proc/$1/cwd" > "$D/orig.cwd"; } \
      || { echo "ABORT: cannot read /proc/$1"; exit 2; }
  tr '\0' ' ' < "$D/orig.cmdline" | grep -q -- "--out $OUT/samples.jsonl " || { echo "ABORT: pid $1 is not this round's sampler"; exit 2; }
  kill -STOP "$1" || { echo "ABORT: cannot stop $1"; exit 2; }
  echo "$1" > "$D/orig.pid"
  echo "=== $(date '+%F %T %Z') the launcher's sampler (pid $1) is stopped"
fi
P=$(cat "$D/orig.pid")
grep -q '^State:.*stopped' "/proc/$P/status" || { echo "ABORT: pid $P is not stopped"; grep State "/proc/$P/status"; exit 3; }
OTHERS=$(pgrep -u "$USER" -f "$PAT" | grep -v -x "$P" | tr '\n' ' ')
[ -z "$OTHERS" ] || { echo "ABORT: another sampler is running: $OTHERS"; exit 4; }
PY=$(tr '\0' '\n' < "$D/orig.cmdline" | head -n 1)
echo "=== $(date '+%F %T %Z') tuned sampler: $N states in flight ${STATES:+states $STATES} ${EXTRA[*]:-}" | tee -a "$OUT/logs/samples_tuned.log"
exec "$PY" "$HERE/swap_run.py" "$D" "$N" "$STATES" ${EXTRA[@]+"${EXTRA[@]}"} >> "$OUT/logs/samples_tuned.log" 2>&1
