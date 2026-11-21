#!/bin/bash
# stop_tuned.sh   (login node): cancel the sampler step(s) of job 173954 (steps named fcsampler, or bash from the first
# launches); never the job itself, its batch step or the scoring step (fcscore: stop that one with logs/swap/score.stop)
set -u
J=173954
squeue -h -s -j $J -o '%i %j' | while read -r S NAME; do
  case "$S" in $J.[0-9]|$J.[0-9][0-9]|$J.[0-9][0-9][0-9]) ;; *) echo "left alone: $S $NAME"; continue ;; esac
  case "$NAME" in fcsampler|bash) echo "scancel $S ($NAME)"; scancel "$S" ;; *) echo "left alone: $S $NAME" ;; esac
done
sleep 8
squeue -s -j $J -o '%.14i %.12j %.10M %N'
