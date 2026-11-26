#!/bin/bash
# Live training dynamics of a con1 full-set job, from the login node: reads the node's /tmp through
# srun --overlap (the job's own allocation; nothing is started on another node).
#   bash /project/community/wxu2/con1/repo2m/scripts/con1/mon_full.sh <slurm job id> [run dir on shared, default full/run1]
#   tail of the trainer:  srun --jobid=<id> --overlap -N1 -n1 tail -5 /tmp/con1_full_<id>/logs/train.log
set -uo pipefail
JOB=${1:?slurm job id}
M=${2:-/project/community/wxu2/con1/full/run1}
REPO=${REPO:-/project/community/wxu2/con1/repo2m}
PY=/project/community/wxu2/rl/envs/train/bin/python
srun --jobid="$JOB" --overlap -N1 -n1 "$PY" "$REPO/scripts/con1/mon_full.py" "/tmp/con1_full_$JOB/run" --export "$M/export"
srun --jobid="$JOB" --overlap -N1 -n1 bash -c "grep -h '^step \|resumed\|rows check\|dev loss\|Error\|error' /tmp/con1_full_$JOB/logs/train.log | tail -3"
