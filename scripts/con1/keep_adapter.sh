#!/bin/bash
# con1: make sure a running main job's stage adapter lands on shared storage. Runs on the Orchard login node beside the
# job (reads the job's node /tmp through srun --overlap; never touches the job or its scripts). Once the adapter exists
# in /tmp it waits 10 min for the job's own copy, checks it byte for byte, and copies it itself (retrying while the
# shared volume is full) when the job's copy is missing or differs.
#   JOB=176033 STAGE=stage1 M=/project/community/wxu2/con1/main/run3 setsid nohup bash keep_adapter.sh > log 2>&1 &
set -uo pipefail
JOB="${JOB:?job id}"; STAGE="${STAGE:-stage1}"; M="${M:?run dir}"
T=/tmp/con1_main_$JOB/$STAGE/final
D=$M/$STAGE/final
on_node() { timeout 900 srun --jobid "$JOB" --overlap -n1 bash -c "$1" 2>/dev/null; }
echo "=== $(date '+%F %T') watching $T -> $D"
while squeue -h -j "$JOB" 2>/dev/null | grep -q .; do
  if on_node "test -e $T/adapter_config.json"; then
    echo "$(date '+%F %T') adapter in /tmp; waiting 10 min for the job's own copy"
    sleep 600
    for i in $(seq 1 60); do
      if on_node "cd $T && for f in *; do cmp -s \$f $D/\$f || exit 1; done"; then
        echo "$(date '+%F %T') KEPT: $D matches the job's adapter"; ls -la "$D"; exit 0
      fi
      echo "$(date '+%F %T') copy $i"
      on_node "rm -rf $D.keep && mkdir -p $D.keep && cp -r $T/. $D.keep/ && rm -rf $D && mv $D.keep $D" || sleep 60
    done
    echo "!!! could not keep the adapter in an hour"; exit 1
  fi
  sleep 600
done
echo "!!! job $JOB ended before its $STAGE adapter appeared in /tmp"
