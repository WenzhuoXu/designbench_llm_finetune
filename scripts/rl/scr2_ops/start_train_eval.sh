#!/bin/bash
# start_train_eval.sh <run dir> <train rows> <dev rows> <eval out> <eval label>   (login node)
# Ends the data job's vLLM servers and starts the training, then the evaluation, as a step of job 173954.
set -u
J=173954; H=$HOME/rl/logs/fcswap
[ $# = 5 ] || { echo "usage: start_train_eval.sh <run dir> <train rows> <dev rows> <eval out> <eval label>"; exit 2; }
[ "$(squeue -h -j $J -o %T)" = RUNNING ] || { echo "job $J is not running"; exit 2; }
squeue -h -s -j $J -o %j | grep -q -x "fcsampler\|bash\|fcscore\|fcscorefin\|fcrows\|fctrain" && { echo "ABORT: a step is still running"; squeue -h -s -j $J -o '%i %j'; exit 3; }
[ -s "$2" ] && [ -s "$3" ] || { echo "ABORT: rows files missing"; exit 2; }
srun --jobid=$J --overlap -J fckill -N1 -n1 bash "$H/kill_servers.sh" 2>&1 | grep -v cgroup
echo "=== $(date '+%F %T %Z') start_train_eval $*" >> "$H/train_eval.log"
RUN_DIR="$1" ROWS="$2" DEV="$3" EVAL_OUT="$4" EVAL_LABEL="$5" nohup setsid srun --jobid=$J --overlap -J fctrain -N1 -n1 -c64 \
    bash "$H/train_eval.sh" >> "$H/train_eval.log" 2>&1 < /dev/null &
echo "srun pid $!"
sleep 45
grep -v cgroup "$H/train_eval.log" | tail -n 8
squeue -s -j $J -o '%.14i %.12j %.10M %N'
