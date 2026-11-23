#!/bin/bash
# stop_at.sh D SAMPLER_PID MAIN_PID (drc, setsid): drc's share of the grid2op states is file index [0, D); Orchard samples
# [D, 4440). Every 2 min: once every state below D has 8 replies (or drc has started more than 60 states at or above D,
# so the ones still short below D are failed replies), the sampler is stopped; sample_open.sh then ends (vLLM stopped by
# its trap), the scoring loop is told to stop, states.jsonl is cut to [0, D) plus the states above D that drc touched
# (full list kept as states_full.jsonl), samples.done is written and sample_open.sh runs again for the last scoring
# pass and the gain rows.
set -u
D=$1; SP=$2; MAIN=$3
W=/storage/wxu/rl_open; OUT=$W/round; L=$OUT/logs/stop_at.log
SIM_PY=/home/wxu/corpus_env/bin/python
echo "$(date '+%F %T') D=$D sampler=$SP main=$MAIN" >> $L
while :; do
  r=$($SIM_PY $W/cover_upto.py $OUT/states.jsonl $OUT/samples.jsonl $D)
  echo "$(date '+%F %T') complete below D / touched above D: $r" >> $L
  set -- $r
  [ "$1" -ge "$D" ] && break
  [ "$2" -gt 60 ] && { echo "$(date '+%F %T') more than 60 states above D started; stopping" >> $L; break; }
  kill -0 $SP 2>/dev/null || { echo "$(date '+%F %T') sampler already gone" >> $L; break; }
  sleep 120
done
kill -TERM $SP 2>/dev/null
for i in $(seq 1 90); do kill -0 $MAIN 2>/dev/null || break; sleep 10; done
kill -0 $MAIN 2>/dev/null && { echo "$(date '+%F %T') ABORT: sample_open.sh still running" >> $L; exit 3; }
touch $OUT/logs/score.stop
for i in $(seq 1 360); do pgrep -f "score_loop.sh $OUT " >/dev/null || break; sleep 10; done
pgrep -f "score_loop.sh $OUT " >/dev/null && { echo "$(date '+%F %T') ABORT: scoring loop still running" >> $L; exit 4; }
[ -e $OUT/states_full.jsonl ] || cp $OUT/states.jsonl $OUT/states_full.jsonl
$SIM_PY $W/cover_upto.py $OUT/states_full.jsonl $OUT/samples.jsonl $D --cut $OUT/states.jsonl >> $L 2>&1 || { echo "ABORT: cut failed" >> $L; exit 5; }
date '+%F %T %Z' > $OUT/samples.done
echo "$(date '+%F %T') last pass" >> $L
CUDA_VISIBLE_DEVICES=4 OUT=$OUT EPISODES=$W/collect/dx38mtR.episodes.jsonl bash $W/sample_open.sh >> $OUT/logs/run.out 2>&1 < /dev/null
echo "$(date '+%F %T') STOP_AT_DONE rc=$?" >> $L
