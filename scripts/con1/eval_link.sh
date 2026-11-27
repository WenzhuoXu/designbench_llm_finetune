#!/bin/bash
# eval_link.sh LABEL : on drc, the facets of one checkpoint of the full-set continuation on the main-run test set (held-out
# problems, all 8 training families, queries_v2: 3,687 questions with native off-menu moves), next to stage 1's answers
# (job 176033) and the main-run trees, with con1's generalised drc_arm.sh (docs/con1_full_rows.md §5):
#   0. the checkpoint came off Orchard by scripts/con1/pull_slot.sh + the RL session's join_ckpt.sh (sha1-checked):
#      /storage/wxu/rl_open/orchard_runs/con1_full/ckpt/LABEL/slot.tar
#   1. its adapter -> con1/arm/run_c2m_LABEL/final (a rolling slot holds adapter/ + state.json, the F slots a final dir);
#      slot.tar is deleted after a good extraction (drc /storage is ~98% full)
#   2. con1's drc_arm.sh: final exists, so no training; GPU 4 under arm/gpu4.lock (WAIT=1: until GPU 4 holds < 5 GB);
#      vLLM, skill_client zero pass (FMT=facetm, the continuation's format; MAXTOK 4096 as stage 1's questions), facets
#      without the harness (V_bal_phys n/a; regime_phys, cpl_dir_BA, cpl_size unaffected) next to CMP; ~60-75 min of
#      GPU 4 plus the server start
#   3. full_eval/curve.txt rewritten: one line per checkpoint (c2m_*), then the comparison lines of the newest file
#   bash /storage/wxu/con1/full_eval/eval_link.sh full_00700        (LABEL as pull_slot.sh names it: <stage>_<step>)
#   bash /storage/wxu/con1/full_eval/eval_link.sh stage1_final      (the curve's origin: the stage-1 final from F1-F3)
set -uo pipefail
L=${1:?label}
C=/storage/wxu/con1; A=$C/arm; E=${OUTDIR:-$C/full_eval}
TAR=/storage/wxu/rl_open/orchard_runs/con1_full/ckpt/$L/slot.tar
LAB=c2m_$L
RUN=$A/run_$LAB
# con1's script: drc_arm.sh once it carries the generalised options (after con1's k-shot run, ~11-26 10:00), else
# drc_arm_next.sh (the same script waiting to replace it; drc_arm.sh is never edited while a run executes it)
ARM=$C/repo/scripts/con1/drc_arm.sh
grep -q "OUTDIR" "$ARM" || ARM=$C/repo/scripts/con1/drc_arm_next.sh
STAGE1_PREDS=${STAGE1_PREDS:-$E/preds_stage1.jsonl}      # 176033's stage-1 answers (from slot P once written)
CMP=${CMP:-"$STAGE1_PREDS $A/main/preds_trees.jsonl"}
mkdir -p "$E"
echo "=== $(date '+%F %T') $LAB from $TAR with $ARM"
if [ ! -e "$RUN/final/adapter_config.json" ]; then
  [ -s "$TAR" ] || { echo "ABORT: no $TAR (pull_slot.sh + join_ckpt.sh first)"; exit 2; }
  rm -rf "$RUN/x" && mkdir -p "$RUN/x" && tar -x -C "$RUN/x" -f "$TAR" || { echo "ABORT: tar"; exit 2; }
  if [ -d "$RUN/x/adapter" ]; then mv "$RUN/x/adapter" "$RUN/final" && mv "$RUN/x/state.json" "$RUN/"; else mv "$RUN/x" "$RUN/final"; fi
  rm -rf "$RUN/x"
  [ -e "$RUN/final/adapter_config.json" ] || { echo "ABORT: no adapter_config.json in $TAR"; exit 2; }
  [ -e "$RUN/state.json" ] && echo "state: $(cat "$RUN/state.json")"
  rm -f "$TAR" && rmdir "$(dirname "$TAR")" 2>/dev/null; echo "slot.tar removed after extraction"
fi
for x in $CMP; do [ -s "$x" ] || echo "note: comparison $x not there yet (scored without it)"; done
[ -s "$E/facets_${LAB}_zero.txt" ] || LAB=$LAB FMT=facetm ROWS=none WAIT=1 QUERIES=$A/main/queries_v2.jsonl FAM=all \
  CMP="$CMP" MAXTOK=4096 OUTDIR="$E" PASSES=zero bash "$ARM" > "$E/arm_$L.out" 2>&1
grep "holding\|server ready\|SKILL_CLIENT_DONE\|ABORT\|DRC_ARM_DONE" "$E/arm_$L.out" | tail -5
[ -s "$E/facets_${LAB}_zero.txt" ] || { echo "ABORT: no facets ($E/arm_$L.out)"; exit 3; }
grep -A10 "group means over all" "$E/facets_${LAB}_zero.txt"
# held-out families (inventory2, molopt, thermal2; queries_v2 group heldout_family, 175 states, 1,400 questions): zero-shot
# and sequential passes (con1's script with a GROUPS hook: full_eval/drc_arm_groups.sh), then the sequential pass
# against its own zero-shot answers and the copiers (hf_score.sh); no trees exist for these families
H=$E/hf; mkdir -p "$H"
[ -s "$H/facets_${LAB}_seq.txt" ] || LAB=$LAB FMT=facetm ROWS=none WAIT=1 QUERIES=$A/main/queries_v2.jsonl FAM=all CMP=" " \
  MAXTOK=4096 OUTDIR="$H" PASSES="zero seq" QGROUPS=heldout_family bash "$E/drc_arm_groups.sh" > "$H/arm_$L.out" 2>&1
grep "server ready\|SKILL_CLIENT_DONE\|ABORT\|DRC_ARM_DONE" "$H/arm_$L.out" | tail -3
[ -s "$H/facets_${LAB}_seq_vs_zero.txt" ] || OUTDIR="$H" bash "$E/hf_score.sh" "$LAB" > "$H/hf_score_$L.out" 2>&1
[ -s "$H/report_$LAB.txt" ] || bash "$E/hf_report.sh" "$LAB" > /dev/null 2>&1   # per family, each with its own se
cat "$H/report_$LAB.txt" 2>/dev/null
# the parent session's error-anatomy scripts (review_tmp/ea, read only; ea4.py through a copy with its preds path
# replaced) on the unseen-family zero-shot answers; the two numbers asked for: PLACE_BLOCK agreement with the printed
# rule (ea2) and thermal2 answers missing FEASIBLE AFTER (ea1)
EA=$C/review_tmp/ea
if [ -s "$H/preds_$LAB.jsonl" ] && [ ! -s "$H/ea_$LAB.txt" ]; then
  { echo "## ea1"; CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /opt/anaconda3/bin/python $EA/ea1.py "$H/preds_$LAB.jsonl" zero inventory2,molopt,thermal2
    echo "## ea2"; CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /opt/anaconda3/bin/python $EA/ea2.py "$H/preds_$LAB.jsonl" zero
    echo "## ea4"; sed "s#/storage/wxu/con1/full_eval/hf/preds_c2m_stage1_final.jsonl#$H/preds_$LAB.jsonl#" $EA/ea4.py > "$H/ea4_$LAB.py" \
      && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /opt/anaconda3/bin/python "$H/ea4_$LAB.py"
  } > "$H/ea_$LAB.txt" 2>&1
fi
grep "^PLACE_BLOCK  .*_temp_rise" "$H/ea_$LAB.txt" 2>/dev/null
awk '/=+ thermal2 zero/{on=1} on && /missing FEASIBLE AFTER/{print "thermal2 " $0; exit}' "$H/ea_$LAB.txt" 2>/dev/null
line() {   # file label: the label's line of the primary group means
  awk -v lab="$2" '/group means over all \(primary\)/{on=1; next} on && NF > 2 && $2 != "select" {exit} on && $1==lab {print; exit}' "$1"
}
{ echo "# $(date '+%F %T') group means over all (primary), queries_v2, one line per checkpoint"
  echo "# trained families (8; held-out problems), zero-shot:"
  newest=
  for f in $(ls "$E"/facets_c2m_*_zero.txt | sort); do
    l=$(basename "$f" _zero.txt); l=${l#facets_}; newest=$f; line "$f" "$l"
  done
  echo "#   comparisons (from $(basename "$newest")):"
  awk '/group means over all \(primary\)/{on=1; next} on && NF > 2 && $2 != "select" {exit} on && $1 !~ /^c2m_/ && $2 == "select" {print}' "$newest"
  echo "# unseen families (inventory2, molopt, thermal2), zero-shot:"
  for f in $(ls "$H"/facets_c2m_*_zero.txt 2>/dev/null | sort); do
    l=$(basename "$f" _zero.txt); l=${l#facets_}; line "$f" "$l"
  done
  echo "# unseen families, sequential (positions >= 1) next to the same questions zero-shot and the copiers:"
  for f in $(ls "$H"/facets_c2m_*_seq_vs_zero.txt 2>/dev/null | sort); do
    l=$(basename "$f" _seq_vs_zero.txt); l=${l#facets_}
    for x in "$l" "${l}_zero" copy_last copy_mode; do line "$f" "$x"; done
  done
} > "$E/curve.txt.tmp" && mv "$E/curve.txt.tmp" "$E/curve.txt"
cat "$E/curve.txt"
echo "EVAL_LINK_DONE $L $(date '+%F %T')"
