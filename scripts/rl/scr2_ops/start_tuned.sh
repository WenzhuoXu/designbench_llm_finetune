#!/bin/bash
# start_tuned.sh <states in flight> [states file|""] [extra sampler arguments ...]   (login node): the tuned sampler as a step of job 173954
set -u
J=173954; H=$HOME/rl/logs/fcswap
[ "$(squeue -h -j $J -o %T)" = RUNNING ] || { echo "job $J is not running"; exit 2; }
echo "=== $(date '+%F %T %Z') start_tuned $*" >> "$H/step.log"
nohup setsid srun --jobid=$J --overlap -J fcsampler -N1 -n1 -c8 bash "$H/node_start.sh" "$@" >> "$H/step.log" 2>&1 < /dev/null &
echo "srun pid $!"
sleep 20
tail -n 5 "$H/step.log"
squeue -s -j $J -o '%.14i %.12j %.10M %N'
