#!/bin/bash
# kill_servers.sh   inside job 173954's allocation: end the launcher's vLLM servers (each one its own process group),
# so the GPUs are free for the training. The launcher itself stays (it waits on its stopped sampler).
set -u
PG=$(ps -u "$USER" -o pgid=,args= | grep "[v]llm serve" | awk '{print $1}' | sort -u | tr '\n' ' ')
echo "server process groups: ${PG:-none}"
for g in $PG; do kill -TERM -- "-$g" 2>/dev/null; done
for i in $(seq 1 12); do
  [ -z "$(ps -u "$USER" -o pgid=,args= | grep "[v]llm serve")" ] && break
  sleep 5
done
for g in $PG; do kill -KILL -- "-$g" 2>/dev/null; done
sleep 8
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
