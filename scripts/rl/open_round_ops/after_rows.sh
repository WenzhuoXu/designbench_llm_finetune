#!/bin/bash
# after_rows.sh (Orchard login node): once every share has its gain rows, the open-rules round's training and
# benchmark: merge (merge_open_rows.py: one row per state, Orchard shares first, drc last; every row checked), mix
# (new rows + replay of the open-rules restated dx38 rows, 1 per 2 new rows; dev = 1.2% of the new rows' problems,
# at most 240 rows), training from dx38 (advanced, 4 GPUs, dataxp_sft_scr1.yaml + sft.temperature=1.0), and the
# benchmark of its final adapter under the open rules (afterok on the training): photonics grid2op engibench on the
# general GPU, the other eight families on advanced, as dx38's open-rules baseline ran.
#   DRY=1: merge and mix into a scratch directory, nothing submitted.
set -u
V=$HOME/rl/logs/v3r
R=/project/community/wxu2/rl/screen/open_round
NAME=${NAME:-open1_dx38mtR_20261123}
RUN_DIR=/project/community/wxu2/rl/runs/$NAME
SHARES="grid2op_drc_rev grid2op_drc_tail g2e_orchard g2e_share2 photonics"
ROWS=()
for s in $SHARES; do
  if [ "$s" = grid2op_drc_tail ] && [ ! -e "$R/$s" ]; then continue; fi
  [ -e "$R/$s/rows.done" ] && [ -s "$R/$s/rows_gain.jsonl" ] || { [ -n "${DRY:-}" ] && { echo "dry: $s not done, left out"; continue; }; echo "ABORT: $s has no rows yet"; exit 2; }
  ROWS+=("$R/$s/rows_gain.jsonl")
done
if [ -s "$R/drc_rows/rows_gain.jsonl" ] && [ -e "$R/drc_rows/UPLOADED" ]; then ROWS+=("$R/drc_rows/rows_gain.jsonl")
elif [ -z "${DRY:-}" ]; then echo "ABORT: drc rows not uploaded"; exit 2
else echo "dry: drc rows not there, left out"; fi
D=$R; [ -n "${DRY:-}" ] && { D=$R/dry_merge; mkdir -p "$D"; }
[ -z "${DRY:-}" ] && [ -e "$RUN_DIR" ] && { echo "ABORT: $RUN_DIR exists"; exit 2; }
echo "=== $(date '+%F %T') merge: ${ROWS[*]}"
python3 $V/merge_open_rows.py --out $D/rows_all.jsonl --replay $R/replay_open.jsonl --tier open1_gain "${ROWS[@]}" \
    | tee $D/rows_all.report.txt
grep -q MERGE_OPEN_ROWS_DONE $D/rows_all.report.txt || { echo "ABORT: merge failed"; exit 3; }
python3 ~/rl/repo/scripts/rl_screen/mix_rows.py --new $D/rows_all.jsonl --replay $R/replay_open.jsonl \
    --out-train $D/train_open1.jsonl --out-dev $D/dev_open1.jsonl --replay-ratio 0.5 --dev-frac 0.012 --dev-max 240 --seed 0 \
    | tee $D/mix.report.txt
grep -q MIX_ROWS_DONE $D/mix.report.txt || { echo "ABORT: mix failed"; exit 4; }
[ -n "${DRY:-}" ] && { echo "DRY_DONE"; exit 0; }
cd ~/rl/repo || exit 5
T=$(sbatch --parsable -J dx_sft_open1 \
    --export=ALL,RUN_DIR=$RUN_DIR,CONFIG=configs/rl/dataxp_sft_scr1.yaml,ROWS=$D/train_open1.jsonl,DEV=$D/dev_open1.jsonl,OVERRIDES=sft.temperature=1.0 \
    slurm/orchard_sft_rows.sbatch) || { echo "ABORT: training not submitted"; exit 6; }
echo "training: $T (run $RUN_DIR)"
export LABEL=open1 ADAPTER=$RUN_DIR/final THINK=1 HARNESS_SCRIPT=dataxp_policy_eval_v3r.py HARNESS_ARGS=--open-rules
export PROBLEMS="/home/wxu2/rl/harness/problems_v2.jsonl /home/wxu2/rl/eval/problems_v3_native.jsonl"
E=/home/wxu2/rl/eval/${NAME}_open
G=$(OUT=${E}_weak VARIANTS="photonics grid2op engibench" NSERVERS=1 WORKERS=48 \
    sbatch --parsable --dependency=afterok:$T -p general --qos=general_qos --gres=gpu:1 --cpus-per-task=48 --mem=300G \
    -t 12:00:00 -J dx_eval_o1W --export=ALL slurm/orchard_dataxp_eval.sbatch) || echo "!!! weak-family benchmark not submitted"
A=$(OUT=${E}_rest VARIANTS="circuit2 frame3d2 grid2 seir2 traffic2 inventory2 molopt thermal2" NSERVERS=4 WORKERS=128 \
    sbatch --parsable --dependency=afterok:$T -p advanced --qos=adv_4gpu_qos -t 24:00:00 -J dx_eval_o1R \
    --export=ALL slurm/orchard_dataxp_eval.sbatch) || echo "!!! benchmark of the other families not submitted"
echo "benchmark: general $G -> ${E}_weak; advanced $A -> ${E}_rest (both after $T)"
sleep 8
squeue -u "$USER" -o '%.9i %.14j %.10P %.9T %.20S %.10M %R'
echo "AFTER_ROWS_DONE $T $G $A"
