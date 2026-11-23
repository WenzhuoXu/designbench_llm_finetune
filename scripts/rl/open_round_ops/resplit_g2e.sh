#!/bin/bash
# resplit_g2e.sh (Orchard login node): the g2e sampling job (174715, 1 general GPU, ~490 states/h, ~7 h left) is
# restarted on the same OUT with SAMPLE_LIMIT=N1 (its next N1 states, then its own scoring and rows); the states after
# those go to a second job on 2 preempt GPUs (requeue; every stage resumes). Disjoint shares, nothing sampled twice.
set -u
N1=${N1:-1200}
J=174715
V=$HOME/rl/logs/v3r
R=/project/community/wxu2/rl/screen/open_round
P=/project/community/wxu2/rl/screen/open_collect_g2e_rev
OUT1=$R/g2e_orchard
OUT2=$R/g2e_share2
IN=$R/g2e_in
[ -e "$OUT2" ] && { echo "ABORT: $OUT2 exists"; exit 2; }
mkdir -p "$IN"
for n in orchard-community-0 orchard-community-4; do
  scontrol show node $n | grep -oE "CPUAlloc=\S+|CPUTot=\S+|RealMemory=\S+|AllocMem=\S+|AllocTRES=\S+" | tr '\n' ' '; echo
done
scancel $J
for i in $(seq 1 60); do [ -z "$(squeue -h -j $J -o %T)" ] && break; sleep 5; done
[ -z "$(squeue -h -j $J -o %T)" ] || { echo "ABORT: $J still there"; exit 3; }
sleep 5
python3 $V/split_g2e.py $OUT1/states.jsonl $OUT1/samples.jsonl 8 "$N1" $IN/states_share2.jsonl || { echo "ABORT: split failed"; exit 4; }
cd ~/rl/repo || exit 5
export LABEL=dx38mtR ADAPTER=/project/community/wxu2/rl/inputs/dx38mtR_final
export K=8 TEMPERATURE=0.8 FIRST="engibench grid2op" TIER=open1 MAX_WORDS=1000
export SAMPLE_KV_BUDGET=270000 SAMPLE_REPLY_TOKENS=1700 SAMPLE_PROMPT_WEIGHT=1.0 SCORE_ALONGSIDE=900
G=$(OUT=$OUT1 OPEN_EPISODES=$P/episodes_orchard_only.jsonl NSERVERS=1 SAMPLE_WORKERS=32 SCORE_WORKERS=36 SAMPLE_LIMIT=$N1 \
    sbatch --parsable -p general --qos=general_qos --gres=gpu:1 --cpus-per-task=48 --mem=300G -t 12:00:00 \
    -J dx_open_g2e --export=ALL slurm/orchard_screen_data.sbatch)
echo "share 1: $G"
Q=$(OUT=$OUT2 STATES_FILE=$IN/states_share2.jsonl NSERVERS=2 SAMPLE_WORKERS=64 SCORE_WORKERS=52 \
    sbatch --parsable -p preempt --qos=preempt_qos --requeue --gres=gpu:2 --cpus-per-task=60 --mem=400G -t 1-00:00:00 \
    -J dx_open_g2s --export=ALL slurm/orchard_screen_data.sbatch)
echo "share 2: $Q"
sleep 10
squeue -u "$USER" -o '%.9i %.14j %.10P %.9T %.10M %R'
