#!/bin/bash
# start_score.sh [workers=24] [period seconds=900]   (login node): the scoring loop as a step of job 173954
set -u
J=173954; H=$HOME/rl/logs/fcswap
W="${1:-24}"; P="${2:-900}"
[ "$(squeue -h -j $J -o %T)" = RUNNING ] || { echo "job $J is not running"; exit 2; }
rm -f /project/community/wxu2/rl/screen/dx38mtR_scr2_fcq/logs/swap/score.stop
echo "=== $(date '+%F %T %Z') start_score $W $P" >> "$H/score_step.log"
nohup setsid srun --jobid=$J --overlap -J fcscore -N1 -n1 -c$(( W + 4 )) bash "$H/score_loop.sh" "$W" "$P" >> "$H/score_step.log" 2>&1 < /dev/null &
echo "srun pid $!"
sleep 15
tail -n 3 "$H/score_step.log"
squeue -s -j $J -o '%.14i %.12j %.10M %N'
