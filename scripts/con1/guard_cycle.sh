#!/bin/bash
# (con1 full-set copy of parent_check/guard_cycle.sh; only change: env RES, STAGES, FORCE=1 copies the newest checkpoint
#  whatever the age of the last copy; with none set it behaves exactly as the original)
# One cycle of the checkpoint guard for a con1 main job (run from the login node every ~10 min).
#  - keeps TARGET x 1 GB random-data placeholder chunks in $RES/chunks (zeros take no space on this NFS)
#  - copies each stage's final adapter from the node's /tmp to $RES/<stage>_final as soon as it exists
#  - copies the newest COMPLETE checkpoint of the running stage to $RES/<stage>_ckpt_<step> every ~2 h (keeps 1)
# Space for a copy comes from releasing placeholder chunks first. Prints SAVED / FAILED / JOB lines only.
JOB=$1; TARGET=${TARGET:-16}
RES=${RES:-/project/community/wxu2/con1_reserve}; CH=$RES/chunks; T=/tmp/con1_main_$JOB
mkdir -p $CH
free_mb() { df -B1M --output=avail /project/community | tail -1 | tr -dc 0-9; }
nch() { ls $CH | grep -c '^c[0-9]*$'; }
release1() { local f; f=$(ls $CH | grep '^c[0-9]*$' | sort | tail -1); [ -n "$f" ] && rm -f "$CH/$f"; sync; }
make_room() { local need=$1; while [ "$(free_mb)" -lt "$need" ] && [ "$(nch)" -gt 0 ]; do release1; done; sleep 2; }
grab() { local i; while [ "$(nch)" -lt "$TARGET" ] && [ "$(free_mb)" -gt 2500 ]; do i=$(date +%s%N)
  head -c 1073741824 /dev/urandom > "$CH/c$i.tmp" && mv "$CH/c$i.tmp" "$CH/c$i" || { rm -f "$CH/c$i.tmp"; break; }; done; }
onnode() { timeout 2400 srun --jobid=$JOB --overlap -N1 -n1 --gres=gpu:0 --cpus-per-task=2 --mem=2G bash -c "$1" 2>/dev/null; }
st=$(squeue -h -j $JOB -o %T 2>/dev/null)
if [ "$st" != "RUNNING" ]; then echo "JOB $JOB ${st:-ended} $(date +%H:%M)"; grab; exit 0; fi
for stage in ${STAGES:-stage1 stage2}; do
  if [ ! -e $RES/${stage}_final/adapter_config.json ] && onnode "test -e $T/$stage/final/adapter_config.json"; then
    sz=$(onnode "du -sm $T/$stage/final | cut -f1"); make_room $(( ${sz:-2000} + 400 ))
    if onnode "rm -rf $RES/${stage}_final.tmp && cp -r $T/$stage/final $RES/${stage}_final.tmp && mv $RES/${stage}_final.tmp $RES/${stage}_final"; then
      n1=$(onnode "du -sb $T/$stage/final | cut -f1"); n2=$(du -sb $RES/${stage}_final | cut -f1)
      echo "SAVED $stage final ${sz} MB (bytes node $n1 shared $n2) $(date +%H:%M)"
    else echo "FAILED $stage final copy $(date +%H:%M)"; fi
  fi
  newest=$(onnode "for d in \$(ls -d $T/$stage/ckpt/step_* 2>/dev/null | sort -r); do [ -e \$d/COMPLETE ] && { echo \$d; break; }; done")
  [ -n "$newest" ] || continue
  step=$(basename "$newest"); last=$(ls -d $RES/${stage}_ckpt_step_* 2>/dev/null | sort | tail -1)
  [ "$last" = "$RES/${stage}_ckpt_$step" ] && continue
  if [ -z "${FORCE:-}" ] && [ -n "$last" ] && [ -z "$(find "$last" -maxdepth 0 -mmin +110 2>/dev/null)" ]; then continue; fi
  sz=$(onnode "du -sm $newest | cut -f1"); make_room $(( ${sz:-6000} + 400 ))
  if onnode "rm -rf $RES/${stage}_ckpt.tmp && cp -r $newest $RES/${stage}_ckpt.tmp && mv $RES/${stage}_ckpt.tmp $RES/${stage}_ckpt_$step"; then
    [ -n "$last" ] && rm -rf "$last"
    echo "SAVED $stage checkpoint $step ${sz} MB $(date +%H:%M)"
  else echo "FAILED $stage checkpoint $step copy $(date +%H:%M)"; rm -rf $RES/${stage}_ckpt.tmp; fi
done
grab
exit 0
