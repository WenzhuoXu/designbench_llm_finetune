#!/bin/bash
# finish_rows.sh   (login node; run with nohup, it takes a while): after the sampler has finished the whole list:
# the scoring loop is stopped, one last scoring pass runs over the whole samples file (the launcher's command, same
# environment), then the launcher's build_rows command of a forecast round, then the stage markers.
set -u
J=173954
P=/project/community/wxu2/rl/screen; OUT=$P/dx38mtR_scr2_fcq; H=$HOME/rl/logs/fcswap; D=$OUT/logs/swap
PY=/project/community/wxu2/rl/envs/sim/bin/python
REPO=$HOME/rl/repo; S=$REPO/scripts/rl_screen; HN=$HOME/rl/harness
echo "=== $(date '+%F %T %Z') finish_rows"
[ "$(squeue -h -j $J -o %T)" = RUNNING ] || { echo "ABORT: job $J is not running"; exit 2; }
squeue -h -s -j $J -o %j | grep -q -x "fcsampler\|bash" && { echo "ABORT: a sampler step is running"; exit 2; }
tail -n 2 "$OUT/logs/samples_tuned.log" | grep -q "^SAMPLE_STATES_DONE" || { echo "ABORT: the sampler's last run did not finish"; exit 2; }
"$PY" "$H/dups.py" "$OUT/samples.jsonl" | head -n 2
touch "$D/score.stop"
for i in $(seq 1 360); do
  squeue -h -s -j $J -o %j | grep -q -x fcscore || break
  sleep 10
done
squeue -h -s -j $J -o %j | grep -q -x fcscore && { echo "ABORT: the scoring loop did not stop"; exit 3; }
echo "=== $(date '+%F %T %Z') scoring loop stopped; states scored so far $(grep -c '"state": true' "$OUT/scored.jsonl")"
srun --jobid=$J --overlap -J fcscorefin -N1 -n1 -c96 "$PY" "$H/run_env.py" "$D" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$REPO" -- \
    "$PY" "$S/score_states.py" --code "$HN/dataxp_code_v3" --states "$OUT/states.jsonl" --samples "$OUT/samples.jsonl" \
    --out "$OUT/scored.jsonl" --workers 92 2>&1 | grep -v -i "warn\|numba\|cgroup\|^[[:space:]]*$" > "$OUT/logs/scored.log"
grep "states\|SCORE_STATES_DONE\|!!!" "$OUT/logs/scored.log" | tail -n 8 | cut -c1-200
grep -q SCORE_STATES_DONE "$OUT/logs/scored.log" || { echo "ABORT: scoring failed"; exit 4; }
echo "=== $(date '+%F %T %Z') scored: $(grep -c '"state": true' "$OUT/scored.jsonl") states"
srun --jobid=$J --overlap -J fcrows -N1 -n1 -c4 "$PY" "$H/run_env.py" "$D" -- \
    "$PY" "$S/build_rows.py" --states "$OUT/states.jsonl" --samples "$OUT/samples.jsonl" --scored "$OUT/scored.jsonl" \
    --out "$OUT/rows_fc.jsonl" --min-pct 0 --max-words 1000 --require-forecasts \
    --sample-prompt "served+forecast_v1+cue" --tier scr2_fc --teacher dx38mtR-own-think-forecast-checked \
    > "$OUT/rows_fc.report.txt" 2> "$D/rows_fc.err" || { echo "ABORT: rows failed"; grep -v cgroup "$D/rows_fc.err" | tail -n 5; exit 5; }
grep -q "^BUILD_ROWS_DONE" "$OUT/rows_fc.report.txt" || { echo "ABORT: rows did not finish"; exit 5; }
for m in samples scored rows; do date '+%F %T %Z' > "$OUT/$m.done"; done
cat "$OUT/rows_fc.report.txt"
echo "FINISH_ROWS_DONE $(date '+%F %T %Z') rows $(wc -l < "$OUT/rows_fc.jsonl")"
