#!/bin/bash
# after_g2r.sh (Orchard login node, nohup): when the grid2op share job (174748, advanced, 4 GPUs) ends, the g2e share 2
# (OUT g2e_share2) goes to the advanced GPUs: the preempt job 174746 is cancelled whatever its state (pending, or
# running on 2 GPUs; every stage resumes from OUT) and the same OUT is submitted on advanced with 4 servers.
set -u
A=174748
B=174746
V=$HOME/rl/logs/v3r
L=$V/after_g2r.log
R=/project/community/wxu2/rl/screen/open_round
while [ -n "$(squeue -h -j $A -o %T)" ]; do sleep 120; done
echo "$(date '+%F %T') $A ended: $(sacct -n -X -j $A -o state,exitcode | head -1)" >> $L
if [ -e $R/g2e_share2/rows.done ]; then echo "$(date '+%F %T') share 2 already done" >> $L; exit 0; fi
if [ -n "$(squeue -h -j $B -o %T)" ]; then
  echo "$(date '+%F %T') $B was $(squeue -h -j $B -o %T); cancelling" >> $L
  scancel $B
  for i in $(seq 1 60); do [ -z "$(squeue -h -j $B -o %T)" ] && break; sleep 5; done
  sleep 10
fi
cd ~/rl/repo || exit 5
export OUT=$R/g2e_share2 STATES_FILE=$R/g2e_in/states_share2.jsonl LABEL=dx38mtR ADAPTER=/project/community/wxu2/rl/inputs/dx38mtR_final
export K=8 TEMPERATURE=0.8 FIRST="engibench grid2op" TIER=open1 MAX_WORDS=1000
export NSERVERS=4 SAMPLE_WORKERS=128 SAMPLE_KV_BUDGET=270000 SAMPLE_REPLY_TOKENS=1700 SAMPLE_PROMPT_WEIGHT=1.0
export SCORE_ALONGSIDE=900 SCORE_WORKERS=84
G=$(sbatch --parsable -t 1-00:00:00 -J dx_open_g2s --export=ALL slurm/orchard_screen_data.sbatch)
echo "$(date '+%F %T') share 2 on advanced: $G" >> $L
echo "AFTER_G2R_DONE $G" >> $L
