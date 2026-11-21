#!/bin/bash
# submit_g1.sh <count> [skip=0]   (login node): one more GPU through the general queue (1 GPU, 12 h): its own server
# samples a share of the engibench states (the first <count> that still miss replies, after <skip>), into its own
# folder; the replies are merged into the main samples file later (merge_append.py).
set -u
P=/project/community/wxu2/rl/screen
MAIN=$P/dx38mtR_scr2_fcq
IN=$P/dx38mtR_scr2_fcq_g1_in; mkdir -p "$IN"
PY=/project/community/wxu2/rl/envs/sim/bin/python
[ -e "$IN/states_g1.jsonl" ] && { echo "ABORT: $IN/states_g1.jsonl is already there"; exit 2; }
"$PY" ~/rl/logs/fcswap/subset_states.py "$MAIN/states.jsonl" "$MAIN/samples.jsonl" engibench "${1:?count}" "$IN/states_g1.jsonl" "${2:-0}" || exit 3
wc -l "$IN/states_g1.jsonl"
export OUT=$P/dx38mtR_scr2_fcq_g1 LABEL=dx38mtR ADAPTER=/project/community/wxu2/rl/inputs/dx38mtR_final
export STATES_FILE=$IN/states_g1.jsonl SAMPLE_INSTRUCTION=$HOME/rl/repo/configs/rl/forecast_instruction_v1.txt
export PROMPT_TAG=served+forecast_v1 SAMPLE_CUE=FORECAST: ROWS=forecast K=8 NSERVERS=1 SAMPLE_WORKERS=40
export SAMPLE_KV_BUDGET=270000 STOP_AFTER=samples
cd ~/rl/repo && sbatch -J dx_fc_g1 -p general --qos=general_qos --gres=gpu:1 --cpus-per-task=16 --mem=120G -t 12:00:00 \
    --export=ALL slurm/orchard_screen_data.sbatch
sleep 5
squeue -u "$USER" -o '%.9i %.12j %.10P %.9T %.20S %.10M %R'
