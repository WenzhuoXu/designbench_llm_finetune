#!/bin/bash
# con1 full row set, CPU prep on Orchard (docs/con1_full_rows.md):
#  1. off-menu moves (add_offmenu.py, k = 2) for every native state stage 1's selection did not cover (train and dev);
#  2. the full builder at Q = 12 and Q = 36 into node /tmp (timed, exact token counts), then full_check.py on each;
#  3. stage 1's exact row lengths (the time estimate's calibration, full_estimate.py runs separately).
# Shared storage gets only the off-menu states (~70 MB), stats, lens files and logs; the row sets stay on node /tmp.
#   sbatch slurm/con1_full_prep.sbatch
set -uo pipefail
B=/project/community/wxu2/con1
R=${REPO:-$B/repo2m}
F=$B/full
G=$B/gen
D=$B/data
SIMPY=/project/community/wxu2/rl/envs/sim/bin/python
TRPY=/project/community/wxu2/rl/envs/train/bin/python
CODE=$HOME/rl/harness/dataxp_code_v3
HV=$G/h/dataxp_policy_eval_v3rc.py
MODEL=/project/community/wxu2/rl/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
T=/tmp/con1_full_prep_${SLURM_JOB_ID:-x}
WORKERS=${WORKERS:-30}                     # the builder's workers: the training job has 32 CPUs
mkdir -p "$F/logs" "$T"
export PYTHONPATH=$B/pylib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATAXP_SPELLING=multitool \
  PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
keep() {   # src dst: /project/community runs full for minutes at a time
  local i; for i in $(seq 1 120); do cp "$1" "$2.part" 2>/dev/null && mv "$2.part" "$2" && return 0; sleep 30; done
  echo "!!! could not copy $1 to $2"; return 1
}
echo "=== $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-none} repo=$R"
NATIVE="$G/out/states_native_orch.jsonl.gz $D/states_native.jsonl.gz $D/states_native_drc_snap.jsonl.gz"

echo "=== $(date '+%F %T') 1. off-menu moves for the native states outside stage 1's selection"
if [ ! -e "$F/offmenu_rest.done" ]; then
  $SIMPY - "$B/main/sel_native_sids.txt" "$T/rest_native_sids.txt" $NATIVE <<'EOF'
import collections, gzip, json, re, sys
sel = {l.strip() for l in open(sys.argv[1]) if l.strip()}
pat = re.compile(r'"sid":\s*"((?:[^"\\]|\\.)*)"')
var = re.compile(r'"variant":\s*"(\w+)"')
spl = re.compile(r'"split":\s*"(\w+)"')
seen, rest, c = set(), [], collections.Counter()
for p in sys.argv[3:]:
    for line in gzip.open(p, "rt", encoding="utf-8"):
        h = line[:600]
        sid = pat.search(h).group(1)
        if sid in seen:
            continue
        seen.add(sid)
        key = (var.search(h).group(1), spl.search(h).group(1), "selected" if sid in sel else "rest", p.split("/")[-1])
        c[key] += 1
        if sid not in sel:
            rest.append(sid)
open(sys.argv[2], "w").write("\n".join(rest) + "\n")
for k in sorted(c):
    print("  %-10s %-5s %-8s %-36s %d" % (k + (c[k],)))
print("native states %d: selected (off-menu since 11-25) %d, rest %d" % (len(seen), len(seen) - len(rest), len(rest)))
EOF
  T0=$(date +%s)
  $SIMPY "$R/scripts/con1/add_offmenu.py" --code "$CODE" --harness "$HV" --problems "$G/train_p2_datav4.jsonl" \
    --states $NATIVE --sids "$T/rest_native_sids.txt" --out "$T/states_native_offmenu_rest.jsonl.gz" \
    --workers "${OFFMENU_WORKERS:-90}" > "$T/offmenu.log" 2>&1
  grep "states to augment\|ADD_OFFMENU_DONE\|!!!" "$T/offmenu.log" | head -20
  echo "off-menu wall $(( $(date +%s) - T0 )) s"
  keep "$T/offmenu.log" "$F/logs/offmenu_rest.log"
  if grep -q "ADD_OFFMENU_DONE" "$T/offmenu.log"; then
    keep "$T/states_native_offmenu_rest.jsonl.gz" "$F/states_native_offmenu_rest.jsonl.gz" && touch "$F/offmenu_rest.done"
  else
    echo "ABORT: add_offmenu did not finish"; exit 3
  fi
fi

echo "off-menu moves added (regime as the facet answer states it):"
"$SIMPY" - "$F/states_native_offmenu_rest.jsonl.gz" <<'EOF'
import collections, gzip, json, sys
c = collections.Counter()
for line in gzip.open(sys.argv[1], "rt", encoding="utf-8"):
  st = json.loads(line)
  om = [m for m in st["moves"] if "offmenu" in m["src"]]
  c[(st["variant"], st["split"], "states")] += 1
  c[(st["variant"], st["split"], "offmenu moves")] += len(om)
  for m in om:
      reg = "refused" if m["verdict"] == "refused" else ("applied-feasible" if m.get("feasible") else "applied-infeasible")
      c[(st["variant"], st["split"], "offmenu " + reg)] += 1
for k in sorted(c):
  print("  %-10s %-5s %-22s %d" % (k + (c[k],)))
EOF

ST="$B/main/states_native_offmenu.jsonl.gz $F/states_native_offmenu_rest.jsonl.gz $G/out/states_native_orch.jsonl.gz $G/out/states_scripted.jsonl.gz"
EX="$B/main/rows/train_facet.jsonl $B/main/rows/dev_facet.jsonl $B/main/rows/dev_facetm.jsonl"
FOLLOW=${FOLLOW:-short}; CAP=${MAX_TOKENS:-22000}
TGT=${TARGET:-facetm}; TAG=$TGT; [ "$FOLLOW" = short ] && TAG=${TGT}_short
PW=${PREFIX_WEIGHT-0}; PWARG=; [ -n "$PW" ] && PWARG="--prefix-weight $PW" && TAG=${TAG}_pw$PW   # PREFIX_WEIGHT= : unset
FORBID="$D/queries_v2.jsonl $G/bench_problems.jsonl"
for Q in ${QS:-12 36}; do
  echo "=== $(date '+%F %T') 2. full build $TGT (later questions: $FOLLOW, prefix weight ${PW:-default}) Q=$Q cap $CAP (workers $WORKERS)"
  T0=$(date +%s)
  $TRPY "$R/scripts/con1/build_full_rows.py" --states $ST --exclude-rows $EX --forbid-problems $FORBID \
    --tokenizer "$MODEL" --target "$TGT" --follow "$FOLLOW" $PWARG --q "$Q" --max-tokens "$CAP" --seed 0 --workers "$WORKERS" \
    --out-train "$T/train_${TAG}_q$Q.jsonl" --out-dev "$T/dev_${TAG}_q$Q.jsonl" --stats "$T/stats_${TAG}_q$Q.json" --lens "$T/lens_${TAG}_q$Q.txt" \
    > "$T/build_${TAG}_q$Q.log" 2>&1
  echo "build wall $(( $(date +%s) - T0 )) s"; tail -3 "$T/build_${TAG}_q$Q.log"
  ls -la "$T/train_${TAG}_q$Q.jsonl" "$T/dev_${TAG}_q$Q.jsonl"
  $TRPY "$R/scripts/con1/dump_weights.py" "$T/train_${TAG}_q$Q.jsonl" "$MODEL" --line 0
  for f in stats_${TAG}_q$Q.json lens_${TAG}_q$Q.txt build_${TAG}_q$Q.log; do keep "$T/$f" "$F/$f"; done
  T0=$(date +%s)
  sha1sum "$T/train_${TAG}_q$Q.jsonl" | tee "$T/sha1_${TAG}_q$Q.txt"
  echo "sha1 of the rows file $(( $(date +%s) - T0 )) s"
  $TRPY "$R/scripts/con1/full_check.py" --rows "$T/train_${TAG}_q$Q.jsonl" "$T/dev_${TAG}_q$Q.jsonl" --stats "$T/stats_${TAG}_q$Q.json" \
    --stage1 $EX --q "$Q" --like "$B/main/rows/train_facet.jsonl" | tee "$T/check_${TAG}_q$Q.log"
  keep "$T/check_${TAG}_q$Q.log" "$F/logs/check_${TAG}_q$Q.log"
  # the same build twice must give the same bytes (the trainer's fingerprint holds the file's sha1): rebuild Q=$REBUILD_Q
  # with another worker count and compare
  if [ "$Q" = "${REBUILD_Q:-36}" ] && [ -z "${NO_REBUILD:-}" ]; then
    $TRPY "$R/scripts/con1/build_full_rows.py" --states $ST --exclude-rows $EX --forbid-problems $FORBID \
      --tokenizer "$MODEL" --target "$TGT" --follow "$FOLLOW" $PWARG --q "$Q" --max-tokens "$CAP" --seed 0 --workers 64 \
      --out-train "$T/train_${TAG}_q${Q}_again.jsonl" --out-dev "$T/dev_${TAG}_q${Q}_again.jsonl" --stats "$T/stats_${TAG}_q${Q}_again.json" \
      > "$T/build_${TAG}_q${Q}_again.log" 2>&1
    tail -2 "$T/build_${TAG}_q${Q}_again.log"
    if cmp -s "$T/train_${TAG}_q$Q.jsonl" "$T/train_${TAG}_q${Q}_again.jsonl" && cmp -s "$T/dev_${TAG}_q$Q.jsonl" "$T/dev_${TAG}_q${Q}_again.jsonl"; then
      echo "REBUILD IDENTICAL (64 workers vs $WORKERS)"
    else
      echo "!!! REBUILD DIFFERS"
    fi
    rm -f "$T/train_${TAG}_q${Q}_again.jsonl" "$T/dev_${TAG}_q${Q}_again.jsonl"
  fi
done

echo "=== $(date '+%F %T') 3. stage 1's exact row lengths"
[ -s "$F/lens_stage1.txt" ] || $TRPY - "$B/main/rows/train_facet.jsonl" "$T/lens_stage1.txt" "$MODEL" "$R" <<'EOF'
import sys
sys.path.insert(0, sys.argv[4] + "/scripts/con1")
sys.path.insert(0, sys.argv[4])
from full_estimate import stage1_lens
rows = stage1_lens(sys.argv[1], sys.argv[2], sys.argv[3], 48)
print("stage-1 rows %d, longest %d, tokens %.1fM" % (len(rows), max(r[0] for r in rows), sum(r[0] for r in rows) / 1e6))
EOF
[ -s "$F/lens_stage1.txt" ] || keep "$T/lens_stage1.txt" "$F/lens_stage1.txt"
df -BG --output=avail /tmp | tail -1
echo "FULL_PREP_DONE $(date '+%F %T')"
