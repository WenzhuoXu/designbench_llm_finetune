#!/bin/bash
# mix.sh   (login node): the round's training and dev files (scripts/rl_screen/mix_rows.py, the screened round's
# settings: 1 replay row of dx38's own training rows per 2 new rows, dev 1.2% of the new rows' problems, at most 240).
set -u
J=173954
P=/project/community/wxu2/rl/screen; OUT=$P/dx38mtR_scr2_fcq; H=$HOME/rl/logs/fcswap; D=$OUT/logs/swap
PY=/project/community/wxu2/rl/envs/sim/bin/python; S=$HOME/rl/repo/scripts/rl_screen
REPLAY=/project/community/wxu2/wenwenw/dx38/eval/dx38/tiers_mt/train_dx38mtR.jsonl
[ -e "$OUT/rows.done" ] && [ -s "$OUT/rows_fc.jsonl" ] || { echo "ABORT: the rows are not there"; exit 2; }
[ -e "$OUT/mix_fc/train.jsonl" ] && { echo "ABORT: $OUT/mix_fc is already there"; exit 2; }
mkdir -p "$OUT/mix_fc"
srun --jobid=$J --overlap -J fcmix -N1 -n1 -c2 "$PY" "$H/run_env.py" "$D" -- "$PY" "$S/mix_rows.py" --new "$OUT/rows_fc.jsonl" --replay "$REPLAY" \
    --out-train "$OUT/mix_fc/train.jsonl" --out-dev "$OUT/mix_fc/dev.jsonl" --replay-ratio 0.5 --dev-frac 0.012 --dev-max 240 --seed 0 \
    2>&1 | grep -v cgroup | tail -n 60
wc -l "$OUT/mix_fc/train.jsonl" "$OUT/mix_fc/dev.jsonl"
