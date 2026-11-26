#!/bin/bash
# eval_link.sh LABEL : on drc, the facets of one continuation checkpoint on round 1's held-out-problem questions, next to
# round 1's F, M, trees and con1's facetm arm Fm, with con1's own scripts (docs/con1_full_rows.md, per-link accuracy):
#   0. the checkpoint came off Orchard by scripts/con1/pull_slot.sh + the RL session's join_ckpt.sh (sha1-checked):
#      /storage/wxu/rl_open/orchard_runs/con1_full/ckpt/LABEL/slot.tar
#   1. its adapter -> con1/arm/run_full_LABEL/final (a rolling slot holds adapter/ + state.json, the F slots a final dir)
#   2. con1's drc_arm.sh with LAB=full_LABEL FMT=facetm: final exists, so no training; vLLM on GPU 4 (WAIT=1: waits until
#      GPU 4 holds < 5 GB), skill_client zero pass, 5 families (~45 min of GPU 4: ~14 min server start, ~27 min asking)
#   3. facets as con1's score_Fm.sh (drc has no simulator env: V_bal_phys n/a; regime_phys, cpl_dir_BA, cpl_size are not
#      affected): con1/full_eval/facets_LABEL.{txt,json}
#   4. con1/full_eval/curve.txt rewritten from every facets_*.txt: one line per checkpoint
#   bash /storage/wxu/con1/full_eval/eval_link.sh stage1_00248   (LABEL = <stage>_<step> as pull_slot.sh names it)
set -uo pipefail
L=${1:?label}
C=/storage/wxu/con1; A=$C/arm; E=$C/full_eval
TAR=/storage/wxu/rl_open/orchard_runs/con1_full/ckpt/$L/slot.tar
LAB=full_$L
RUN=$A/run_$LAB
FAM="engibench grid2op photonics circuit2 frame3d2"
mkdir -p "$E"
echo "=== $(date '+%F %T') $LAB from $TAR"
if [ ! -e "$RUN/final/adapter_config.json" ]; then
  [ -s "$TAR" ] || { echo "ABORT: no $TAR (pull_slot.sh + join_ckpt.sh first)"; exit 2; }
  rm -rf "$RUN/x" && mkdir -p "$RUN/x" && tar -x -C "$RUN/x" -f "$TAR" || { echo "ABORT: tar"; exit 2; }
  if [ -d "$RUN/x/adapter" ]; then mv "$RUN/x/adapter" "$RUN/final" && mv "$RUN/x/state.json" "$RUN/"; else mv "$RUN/x" "$RUN/final"; fi
  rm -rf "$RUN/x"
  [ -e "$RUN/final/adapter_config.json" ] || { echo "ABORT: no adapter_config.json in $TAR"; exit 2; }
  [ -e "$RUN/state.json" ] && echo "state: $(cat "$RUN/state.json")"
fi
# 2. serve + ask (con1's drc_arm.sh; its own facets step needs the simulator env, which drc lacks: step 3 replaces it)
[ -s "$A/preds_$LAB.jsonl" ] || LAB=$LAB FMT=facetm ROWS=none WAIT=1 bash "$C/repo/scripts/con1/drc_arm.sh" > "$E/arm_$L.out" 2>&1
grep "server ready\|SKILL_CLIENT_DONE\|ABORT" "$E/arm_$L.out" | tail -3
[ -s "$A/preds_$LAB.jsonl" ] || { echo "ABORT: no answers ($E/arm_$L.out)"; exit 3; }
# 3. facets (con1's score_Fm.sh, for this label)
cd "$C/repo"
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=
PY=${PY:-/opt/anaconda3/bin/python}
P="$A/preds_$LAB.jsonl"; for x in $A/preds_Fm.jsonl $A/r1/preds_F.jsonl $A/r1/preds_M.jsonl $A/r1/preds_trees.jsonl; do [ -s "$x" ] && P="$P $x"; done
$PY scripts/con1/facets.py --queries $A/queries.jsonl --preds $P --groups heldout_problem --variants $FAM --boot 1000 \
  --out "$E/facets_$L.txt" --json "$E/facets_$L.json" > "$E/score_$L.log" 2>&1
echo "facets rc=$?"
grep -A8 "group means over all" "$E/facets_$L.txt"
# 4. the curve over every scored checkpoint (labels sort by stage then step)
{ echo "# $(date '+%F %T') group means over all (held-out problems, 5 families, k = 0): one line per checkpoint"
  for f in $(ls "$E"/facets_*.txt | sort); do
    l=$(basename "$f" .txt); l=${l#facets_}
    awk -v lab="full_$l" '/group means over all/{on=1; next} on && $1==lab {print; exit}' "$f"
  done; } > "$E/curve.txt.tmp" && mv "$E/curve.txt.tmp" "$E/curve.txt"
cat "$E/curve.txt"
echo "EVAL_LINK_DONE $L $(date '+%F %T')"
