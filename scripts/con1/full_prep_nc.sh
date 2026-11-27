#!/bin/bash
# con1 full row set without the move request (11-27 owner order; con1 eff023a conseq_lib.consequence_user), CPU on
# Orchard. The same settings as the running set (facetm, later questions short, think-closing weight 0, Q 36, cap
# 22,000, seed 0, the same states and exclusions), built with the new repo (REPO, default repo2m_v3) and checked:
#  1. the running set rebuilt with its own repo (repo2m, read only): its sha1 must be the one in its stats file;
#  2. the new set (tag ..._nc): build, full_check.py, a second build with another worker count must give the same bytes;
#  3. stage 1's dev rows (the continuation's dev set) with the move request cut -> $F/dev_stage1_facetm_nc.jsonl;
#  4. full_noconf_check.py: no move-request mark in any row (new train, new dev, the cut stage-1 dev), the OPERATIONS
#     heading once per prompt, counts per split / family against the running set;
#  5. full_pair_check.py: row by row against the running set, nothing but the prompt cut differs.
# Rows stay on node /tmp (the training job rebuilds them at its start and checks the sha1 in the new stats file).
#   sbatch -p preempt --qos=preempt_qos -A ccm -c 64 --mem=200G -t 2:00:00 --requeue \
#     -o /project/community/wxu2/con1/full/logs/con1_full_prep_nc_%j.out --wrap "bash .../repo2m_v3/scripts/con1/full_prep_nc.sh"
set -uo pipefail
B=/project/community/wxu2/con1
R=${REPO:-$B/repo2m_v3}
R_OLD=$B/repo2m
F=$B/full
G=$B/gen
D=$B/data
TRPY=/project/community/wxu2/rl/envs/train/bin/python
MODEL=/project/community/wxu2/rl/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
T=/tmp/con1_full_prep_nc_${SLURM_JOB_ID:-x}
WORKERS=${WORKERS:-30}
mkdir -p "$F/logs" "$T"
export PYTHONPATH=$B/pylib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
keep() { local i; for i in $(seq 1 120); do cp "$1" "$2.part" 2>/dev/null && mv "$2.part" "$2" && return 0; sleep 30; done
  echo "!!! could not copy $1 to $2"; return 1; }
echo "=== $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-none} repo=$R (running set's repo $R_OLD)"
ST="$B/main/states_native_offmenu.jsonl.gz $F/states_native_offmenu_rest.jsonl.gz $G/out/states_native_orch.jsonl.gz $G/out/states_scripted.jsonl.gz"
EX="$B/main/rows/train_facet.jsonl $B/main/rows/dev_facet.jsonl $B/main/rows/dev_facetm.jsonl"
FORBID="$D/queries_v2.jsonl $G/bench_problems.jsonl"
SET="--target facetm --follow short --prefix-weight 0 --q 36 --max-tokens 22000 --seed 0"
OLD=facetm_short_pw0; NEW=facetm_short_pw0_nc
build() {   # repo tag workers
  $TRPY "$1/scripts/con1/build_full_rows.py" --states $ST --exclude-rows $EX --forbid-problems $FORBID --tokenizer "$MODEL" \
    $SET --workers "$3" --out-train "$T/train_$2.jsonl" --out-dev "$T/dev_$2.jsonl" --stats "$T/stats_$2.json" \
    --lens "$T/lens_$2.txt" > "$T/build_$2.log" 2>&1
  tail -2 "$T/build_$2.log"
}
echo "=== $(date '+%F %T') 1. the running set, rebuilt with $R_OLD"
build "$R_OLD" ${OLD}_q36 "$WORKERS"
WANT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_sha1"])' "$F/stats_${OLD}_q36.json")
GOT=$(sha1sum "$T/train_${OLD}_q36.jsonl" | cut -d' ' -f1)
[ "$GOT" = "$WANT" ] && echo "running set rebuilt: sha1 $GOT (as in its stats file)" || { echo "ABORT: running set rebuilt $GOT, stats say $WANT"; exit 3; }
echo "=== $(date '+%F %T') 2. the new set ($NEW) with $R"
T0=$(date +%s)
build "$R" ${NEW}_q36 "$WORKERS"
echo "build wall $(( $(date +%s) - T0 )) s"
sha1sum "$T/train_${NEW}_q36.jsonl" "$T/dev_${NEW}_q36.jsonl"
for x in stats_${NEW}_q36.json lens_${NEW}_q36.txt build_${NEW}_q36.log; do keep "$T/$x" "$F/$x"; done
$TRPY "$R/scripts/con1/full_check.py" --rows "$T/train_${NEW}_q36.jsonl" "$T/dev_${NEW}_q36.jsonl" --stats "$T/stats_${NEW}_q36.json" \
  --stage1 $EX --q 36 --like "$B/main/rows/train_facet.jsonl" | tee "$T/check_${NEW}_q36.log" | tail -8
keep "$T/check_${NEW}_q36.log" "$F/logs/check_${NEW}_q36.log"
$TRPY "$R/scripts/con1/dump_weights.py" "$T/train_${NEW}_q36.jsonl" "$MODEL" --line 0 | head -30
build "$R" ${NEW}_q36_again 64 > /dev/null
if cmp -s "$T/train_${NEW}_q36.jsonl" "$T/train_${NEW}_q36_again.jsonl" && cmp -s "$T/dev_${NEW}_q36.jsonl" "$T/dev_${NEW}_q36_again.jsonl"; then
  echo "REBUILD IDENTICAL (64 workers vs $WORKERS)"; else echo "!!! REBUILD DIFFERS"; fi
rm -f "$T"/*_again.jsonl
echo "=== $(date '+%F %T') 3. stage 1's dev rows with the move request cut"
DEVNC=$F/dev_stage1_facetm_nc.jsonl
$TRPY "$R/scripts/con1/noconf_rows.py" "$B/main/rows/dev_facetm.jsonl" "$T/dev_stage1_facetm_nc.jsonl" && keep "$T/dev_stage1_facetm_nc.jsonl" "$DEVNC"
sha1sum "$DEVNC"
echo "=== $(date '+%F %T') 4. move-request marks and counts against the running set"
$TRPY "$R/scripts/con1/full_noconf_check.py" --rows "$T/train_${NEW}_q36.jsonl" "$T/dev_${NEW}_q36.jsonl" "$DEVNC" \
  --stats "$T/stats_${NEW}_q36.json" --ref-stats "$F/stats_${OLD}_q36.json" | tee "$T/noconf_${NEW}_q36.log"
keep "$T/noconf_${NEW}_q36.log" "$F/logs/noconf_${NEW}_q36.log"
echo "=== $(date '+%F %T') 5. row by row against the running set"
for s in train dev; do
  echo "-- $s"; $TRPY "$R/scripts/con1/full_pair_check.py" --old "$T/${s}_${OLD}_q36.jsonl" --new "$T/${s}_${NEW}_q36.jsonl"
done | tee "$T/pair_${NEW}_q36.log"
echo "-- stage-1 dev"; $TRPY "$R/scripts/con1/full_pair_check.py" --old "$B/main/rows/dev_facetm.jsonl" --new "$DEVNC" | tee -a "$T/pair_${NEW}_q36.log"
keep "$T/pair_${NEW}_q36.log" "$F/logs/pair_${NEW}_q36.log"
ls -la "$T"; df -BG --output=avail /tmp | tail -1
echo "FULL_PREP_NC_DONE $(date '+%F %T')"
