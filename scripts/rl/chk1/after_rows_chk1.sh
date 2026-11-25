#!/bin/bash
# after_rows_chk1.sh (Orchard login node): the chk1 round's rows, training and benchmark.
#  rows   check_rows.py on Orchard's collection (collect_rev, + collect_adv when the advanced-node collector ran) + drc's rows (uploaded to $C/drc_rows/rows_fwd.jsonl with an
#         UPLOADED marker); an instance's rows come from one machine only (Orchard first, then drc)
#  mix    new rows + replay of the open-rules restated dx38 rows (replay_open.jsonl), 1 per 2 new rows; dev = 1.2% of the
#         new rows' problems, at most 240 rows (open1's recipe)
#  train  from dx38mtR (or INIT=<adapter dir>), advanced 4 GPUs, dataxp_sft_scr1.yaml + sft.temperature=1.0
#  bench  afterok, two pairs (RUNS=2 each): photonics grid2op engibench with harness v3rc --open-rules --checks 4
#         (the model may check up to 4 moves per turn; every check counted as a simulator call), the other eight
#         families with v3r --open-rules, as dx38's baseline ran
#   DRY=1: rows and mix into $C/dry, nothing submitted.
set -u
C=/project/community/wxu2/rl/chk1
NAME=${NAME:-chk1_dx38mtR_20261124}
RUN_DIR=${RUN_DIR:-/project/community/wxu2/rl/runs/$NAME}   # e.g. /home/wxu2/rl/runs/$NAME when /project/community is full
REPLAY=/project/community/wxu2/rl/screen/open_round/replay_open.jsonl
D=$C; [ -n "${DRY:-}" ] && { D=$C/dry; mkdir -p "$D"; }
cd ~/rl/repo || exit 2
for f in scripts/rl_screen/check_rows.py scripts/rl_screen/mix_rows.py $REPLAY ~/rl/harness/scripts/dataxp_policy_eval_v3rc.py \
         $C/collect_rev/dx38mtR.episodes.jsonl; do
  [ -s "$f" ] || { echo "ABORT: missing $f"; exit 2; }
done
[ -s $C/drc_rows/rows_fwd.jsonl ] && [ -e $C/drc_rows/UPLOADED ] || { [ -n "${DRY:-}" ] && echo "dry: no drc rows" || { echo "ABORT: drc rows not uploaded"; exit 2; }; }
[ -z "${DRY:-}" ] && [ -e "$RUN_DIR" ] && { echo "ABORT: $RUN_DIR exists"; exit 2; }
echo "=== $(date '+%F %T') rows"
EPS=$C/collect_rev/dx38mtR.episodes.jsonl            # + the advanced-node collector's (collect_adv.sbatch), if it ran
for f in $C/collect_adv*/*.episodes.jsonl; do [ -s "$f" ] && EPS="$EPS $f"; done
python3 scripts/rl_screen/check_rows.py --episodes $EPS --out $D/rows_rev.jsonl \
    --report $D/rows_rev.report.txt || { echo "ABORT: rows failed"; exit 3; }
python3 - "$D/rows_rev.jsonl" "$C/drc_rows/rows_fwd.jsonl" "$D/rows_all.jsonl" <<'EOF' | tee $D/rows_all.report.txt
import collections, json, os, sys
rev, fwd, out = sys.argv[1:4]
rows = [json.loads(l) for l in open(rev, encoding="utf-8")]
have = {(r["domain"], r["instance"]) for r in rows}
n_fwd = n_dup = 0
if os.path.exists(fwd):
    for l in open(fwd, encoding="utf-8"):
        r = json.loads(l)
        if (r["domain"], r["instance"]) in have:
            n_dup += 1
            continue
        rows.append(r); n_fwd += 1
with open(out, "w", encoding="utf-8") as fo:
    for r in rows:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
c = collections.Counter((r["domain"], r["kind"]) for r in rows)
print("rows %d (Orchard %d, drc %d, drc rows of instances Orchard has: %d dropped): %s" % (
    len(rows), len(rows) - n_fwd, n_fwd, n_dup, dict(sorted(c.items()))))
print("MERGE_CHK_ROWS_DONE")
EOF
grep -q MERGE_CHK_ROWS_DONE $D/rows_all.report.txt || { echo "ABORT: merge failed"; exit 3; }
python3 scripts/rl_screen/mix_rows.py --new $D/rows_all.jsonl --replay $REPLAY \
    --out-train $D/train_chk1.jsonl --out-dev $D/dev_chk1.jsonl --replay-ratio 0.5 --dev-frac 0.012 --dev-max 240 --seed 0 \
    | tee $D/mix.report.txt
grep -q MIX_ROWS_DONE $D/mix.report.txt || { echo "ABORT: mix failed"; exit 4; }
[ -n "${DRY:-}" ] && { echo "DRY_DONE"; exit 0; }
# INIT: the adapter training starts from (default the config's, dx38mtR_final), e.g. pref1's final adapter
[ -n "${INIT:-}" ] && { [ -s "$INIT/adapter_model.safetensors" ] || { echo "ABORT: INIT $INIT has no adapter"; exit 2; }; }
export OVERRIDES="sft.temperature=1.0${INIT:+ policy.init_adapter=$INIT}"
T=$(sbatch --parsable -J dx_sft_chk1 \
    --export=ALL,RUN_DIR=$RUN_DIR,CONFIG=configs/rl/dataxp_sft_scr1.yaml,ROWS=$D/train_chk1.jsonl,DEV=$D/dev_chk1.jsonl \
    slurm/orchard_sft_rows.sbatch) || { echo "ABORT: training not submitted"; exit 6; }
echo "training: $T (run $RUN_DIR, overrides '$OVERRIDES')"
export LABEL=chk1 ADAPTER=$RUN_DIR/final THINK=1
export PROBLEMS="/home/wxu2/rl/harness/problems_v2.jsonl /home/wxu2/rl/eval/problems_v3_native.jsonl"
E=/home/wxu2/rl/eval/${NAME}_open
for sfx in "" _b; do
  W=$(OUT=${E}_weak_chk4$sfx VARIANTS="photonics grid2op engibench" NSERVERS=4 WORKERS=96 \
      HARNESS_SCRIPT=dataxp_policy_eval_v3rc.py HARNESS_ARGS="--open-rules --checks 4" \
      sbatch --parsable --dependency=afterok:$T -p advanced --qos=adv_4gpu_qos -t 24:00:00 -J dx_eval_c1W$sfx \
      --export=ALL slurm/orchard_dataxp_eval.sbatch) || echo "!!! weak$sfx not submitted"
  R=$(OUT=${E}_rest$sfx VARIANTS="circuit2 frame3d2 grid2 seir2 traffic2 inventory2 molopt thermal2" NSERVERS=4 WORKERS=128 \
      HARNESS_SCRIPT=dataxp_policy_eval_v3r.py HARNESS_ARGS=--open-rules \
      sbatch --parsable --dependency=afterok:$T -p advanced --qos=adv_4gpu_qos -t 24:00:00 -J dx_eval_c1R$sfx \
      --export=ALL slurm/orchard_dataxp_eval.sbatch) || echo "!!! rest$sfx not submitted"
  echo "benchmark$sfx: weak (checks 4) $W -> ${E}_weak_chk4$sfx; rest $R -> ${E}_rest$sfx (after $T)"
done
sleep 8
squeue -u "$USER" -o '%.9i %.14j %.10P %.9T %.20S %.10M %R'
echo "AFTER_ROWS_CHK1_DONE $T"
