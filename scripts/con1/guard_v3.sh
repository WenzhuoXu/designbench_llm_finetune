#!/bin/bash
# (con1 full-set copy of parent_check/guard_v3.sh. Changes: env RES, STAGES; FORCE=1 writes the newest checkpoint
#  whatever the age of the last slot copy; NO_FINAL=1 never writes a final; REUSE_SLOT=1 rewrites the other A/B slot
#  when the alternate target holds a final (one rolling slot left). Unset env = the original.)
# guard_v3.sh JOB : one cycle (run every ~10 min from the login node).
# Writes checkpoints INTO files we already hold on /project/community ("slots"), in place (dd conv=notrunc,
# oflag=dsync), so no space is ever released to the shared pool (v1/v2 lost the race: freed space was taken by
# other users within seconds, 2026-11-26 03:47 and 05:41).
#   slots/A.bin, slots/B.bin : rolling, alternate; each holds a tar of <ckpt>/adapter + state.json
#   slots/F1.bin F2.bin F3.bin : the final adapter of a stage (tar spread over the three files in order)
#   <slot>.meta : "stage step length sha1 time" of the tar it holds (written only after a verified write)
# Restore: cat <slot files in order> | head -c <length> | tar -x -C <dir>   (sha1 of the head -c stream = sha1)
# Prints SAVED / FAILED / JOB lines only.
JOB=$1; RES=${RES:-/project/community/wxu2/con1_reserve}; SL=$RES/slots; T=/tmp/con1_main_$JOB
onnode() { timeout 3000 srun --jobid=$JOB --overlap -N1 -n1 --gres=gpu:0 --cpus-per-task=2 --mem=4G bash -c "$1" 2>/dev/null; }
st=$(squeue -h -j $JOB -o %T 2>/dev/null)
[ "$st" = "RUNNING" ] || { echo "JOB $JOB ${st:-ended} $(date +%H:%M)"; exit 0; }
# write_slots LABEL SRC_TAR_ON_NODE FILE... : spread the node-side tar over the files in place; verify; write meta
write_slots() {
  local label=$1 tarf=$2; shift 2
  local files="$*"
  onnode "set -e; L=\$(stat -c %s $tarf); S=\$(sha1sum $tarf | cut -d' ' -f1); set -- $files
    if [ \$# -eq 1 ]; then dd if=$tarf of=\$1 bs=4M conv=notrunc oflag=dsync status=none
    else off=0; for f in \"\$@\"; do [ \$off -ge \$L ] && break; cap=\$(stat -c %s \$f)
      dd if=$tarf of=\$f bs=1M skip=\$((off/1048576)) count=\$((cap/1048576)) conv=notrunc oflag=dsync iflag=fullblock status=none
      off=\$((off+cap)); done; [ \$off -ge \$L ] || { echo short; exit 3; }; fi
    V=\$(cat $files | head -c \$L | sha1sum | cut -d' ' -f1); [ \"\$V\" = \"\$S\" ] || { echo badsha; exit 4; }
    echo \"$label \$L \$S \$(date +%F_%T)\""
}
for stage in ${STAGES:-stage1 stage2}; do
  # final adapter: stage1 -> F1-F3 (kept for good); stage2 -> the older of A/B (its rolling copies are then moot)
  if [ -z "${NO_FINAL:-}" ] && ! grep -qs "^$stage final" $SL/F.meta $SL/A.meta $SL/B.meta && onnode "test -e $T/$stage/final/adapter_config.json"; then
    if [ "$stage" = stage1 ]; then files="$SL/F1.bin $SL/F2.bin $SL/F3.bin"; mf=$SL/F.meta
    else newer=$(ls -t $SL/A.meta $SL/B.meta 2>/dev/null | head -1); if [ "$newer" = "$SL/A.meta" ]; then tg=B; else tg=A; fi
         files="$SL/$tg.bin"; mf=$SL/$tg.meta; rm -f $mf; fi
    out=$(onnode "rm -f /tmp/g_final.tar; tar -C $T/$stage/final -cf /tmp/g_final.tar . && echo ok")
    if [ "$out" = ok ] && m=$(write_slots "$stage final" /tmp/g_final.tar $files) && [ -n "$m" ] && [ "${m#*final }" != "$m" ]; then
      echo "$m" > $mf.tmp && mv $mf.tmp $mf; echo "SAVED $stage final into $(basename $mf .meta): $m $(date +%H:%M)"
    else echo "FAILED $stage final ($m) $(date +%H:%M)"; fi
  fi
  grep -qs "^$stage final" $SL/F.meta $SL/A.meta $SL/B.meta && [ "$stage" = stage2 ] && continue
  # held-out answers of the stage (small): tar into P.bin once the job log says the stage answered
  if [ -e $SL/P.bin ] && ! grep -qs "^$stage preds" $SL/P.meta && grep -qs "$stage answered" /project/community/wxu2/con1/logs/con1_main_$JOB.out \
     && onnode "test -s $T/preds_$stage.jsonl"; then
    out=$(onnode "rm -f /tmp/g_preds.tar; tar -C $T -cf /tmp/g_preds.tar preds_$stage.jsonl && echo ok")
    if [ "$out" = ok ] && m=$(write_slots "$stage preds" /tmp/g_preds.tar $SL/P.bin) && [ -n "$m" ]; then
      echo "$m" > $SL/P.meta.tmp && mv $SL/P.meta.tmp $SL/P.meta; echo "SAVED $stage answers into P: $m $(date +%H:%M)"
    else echo "FAILED $stage answers ($m) $(date +%H:%M)"; fi
  fi
  newest=$(onnode "for d in \$(ls -d $T/$stage/ckpt/step_* 2>/dev/null | sort -r); do [ -e \$d/COMPLETE ] && { echo \$d; break; }; done")
  [ -n "$newest" ] || continue
  step=$(basename "$newest")
  grep -q "^$stage $step " $SL/A.meta $SL/B.meta 2>/dev/null && continue
  # newest good slot must be older than ~110 min before we write again
  latest=$(ls -t $SL/A.meta $SL/B.meta 2>/dev/null | head -1)
  if [ -z "${FORCE:-}" ] && [ -n "$latest" ] && [ -z "$(find "$latest" -mmin +110 2>/dev/null)" ]; then continue; fi
  # target = the slot that is NOT the newest good one
  if [ "$latest" = "$SL/A.meta" ]; then tgt=B; else tgt=A; fi
  if grep -qs " final " $SL/$tgt.meta; then   # REUSE_SLOT=1: rewrite the other slot when this one holds a final
    [ -n "${REUSE_SLOT:-}" ] || continue; if [ "$tgt" = A ]; then tgt=B; else tgt=A; fi; grep -qs " final " $SL/$tgt.meta && continue; fi
  out=$(onnode "rm -f /tmp/g_roll.tar; tar -C $newest -cf /tmp/g_roll.tar adapter state.json && echo ok")
  rm -f $SL/$tgt.meta   # the slot is about to be overwritten: it no longer holds a valid copy
  if [ "$out" = ok ] && m=$(write_slots "$stage $step" /tmp/g_roll.tar $SL/$tgt.bin) && [ -n "$m" ] && [ "${m#*$step }" != "$m" ]; then
    echo "$m" > $SL/$tgt.meta.tmp && mv $SL/$tgt.meta.tmp $SL/$tgt.meta; echo "SAVED $stage $step adapter into slot $tgt: $m $(date +%H:%M)"
  else echo "FAILED $stage $step into slot $tgt ($m) $(date +%H:%M)"; fi
done
exit 0
