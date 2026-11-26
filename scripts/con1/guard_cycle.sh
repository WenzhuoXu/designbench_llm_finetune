#!/bin/bash
# (con1 full-set copy of parent_check/guard_cycle.sh, 11-26 03:50 version. Changes: env RES, STAGES; FORCE=1 copies the
#  newest checkpoint whatever the age of the last copy; latest is empty when a stage has no copy yet (ls -td with no
#  operand prints ".", which made the age test look at the working directory). Unset env = the original.)
# One cycle of the checkpoint guard for a con1 main job (run from the login node every ~10 min).
# Space: 1 GB random-data placeholder chunks in $RES/chunks (zeros take no space on this NFS):
#   f* = reserve for FINAL adapters only (keep FRES of them), c* = for rolling checkpoint copies (up to TARGET).
# Final adapter of each stage -> $RES/<stage>_final as soon as it exists (released f* chunks pay for it).
# Rolling, every ~2 h: newest COMPLETE checkpoint -> $RES/<stage>_ckpt_<step> (full, with optimizer) when there is room
#   for it while the previous full copy is kept; otherwise its adapter only -> $RES/<stage>_adapter_<step>.
#   One full copy and one adapter copy are kept per stage (the newest of each).
# Prints SAVED / FAILED / JOB lines only. 2026-11-26: first version failed at step 80 (no room for a 2nd full copy).
JOB=$1; TARGET=${TARGET:-8}; FRES=${FRES:-3}
RES=${RES:-/project/community/wxu2/con1_reserve}; CH=$RES/chunks; T=/tmp/con1_main_$JOB
mkdir -p $CH
free_mb() { df -B1M --output=avail /project/community | tail -1 | tr -dc 0-9; }
n_of() { ls $CH | grep -c "^$1[0-9]*$"; }
rel() { local f; f=$(ls $CH | grep "^$1[0-9]*$" | sort | tail -1); [ -n "$f" ] && rm -f "$CH/$f"; sync; }
room_c() { local need=$1; while [ "$(free_mb)" -lt "$need" ] && [ "$(n_of c)" -gt 0 ]; do rel c; done; sleep 2; [ "$(free_mb)" -ge "$need" ]; }
room_f() { local need=$1; while [ "$(free_mb)" -lt "$need" ] && [ "$(n_of c)" -gt 0 ]; do rel c; done
           while [ "$(free_mb)" -lt "$need" ] && [ "$(n_of f)" -gt 0 ]; do rel f; done; sleep 2; [ "$(free_mb)" -ge "$need" ]; }
mk() { local n=$1$(date +%s%N); head -c 1073741824 /dev/urandom > "$CH/$n.tmp" && mv "$CH/$n.tmp" "$CH/$n" || { rm -f "$CH/$n.tmp"; return 1; }; }
grab() { while [ "$(n_of f)" -lt "$FRES" ] && [ "$(free_mb)" -gt 1300 ]; do mk f || break; done
         while [ "$(n_of c)" -lt "$TARGET" ] && [ "$(free_mb)" -gt 2500 ]; do mk c || break; done; }
room_a() { local need=$1; while [ "$(free_mb)" -lt "$need" ] && [ "$(n_of c)" -gt 0 ]; do rel c; done
           while [ "$(free_mb)" -lt "$need" ] && [ "$(n_of f)" -gt 1 ]; do rel f; done; sleep 2; [ "$(free_mb)" -ge "$need" ]; }
onnode() { timeout 2400 srun --jobid=$JOB --overlap -N1 -n1 --gres=gpu:0 --cpus-per-task=2 --mem=2G bash -c "$1" 2>/dev/null; }
st=$(squeue -h -j $JOB -o %T 2>/dev/null)
if [ "$st" != "RUNNING" ]; then echo "JOB $JOB ${st:-ended} $(date +%H:%M)"; exit 0; fi
for stage in ${STAGES:-stage1 stage2}; do
  if [ ! -e $RES/${stage}_final/adapter_config.json ] && onnode "test -e $T/$stage/final/adapter_config.json"; then
    sz=$(onnode "du -sm $T/$stage/final | cut -f1")
    if room_f $(( ${sz:-2000} + 300 )) && onnode "rm -rf $RES/${stage}_final.tmp && cp -r $T/$stage/final $RES/${stage}_final.tmp && mv $RES/${stage}_final.tmp $RES/${stage}_final"; then
      n1=$(onnode "du -sb $T/$stage/final | cut -f1"); n2=$(du -sb $RES/${stage}_final | cut -f1)
      echo "SAVED $stage final ${sz} MB (bytes node $n1 shared $n2) $(date +%H:%M)"
    else echo "FAILED $stage final copy (free $(free_mb) MB) $(date +%H:%M)"; rm -rf $RES/${stage}_final.tmp; fi
  fi
  newest=$(onnode "for d in \$(ls -d $T/$stage/ckpt/step_* 2>/dev/null | sort -r); do [ -e \$d/COMPLETE ] && { echo \$d; break; }; done")
  [ -n "$newest" ] || continue
  step=$(basename "$newest")
  lastf=$(ls -d $RES/${stage}_ckpt_step_* 2>/dev/null | sort | tail -1)
  lasta=$(ls -d $RES/${stage}_adapter_step_* 2>/dev/null | sort | tail -1)
  latest=; [ -n "$lastf$lasta" ] && latest=$(ls -td $lastf $lasta 2>/dev/null | head -1)
  case "$latest" in *"_$step") continue ;; esac
  if [ -z "${FORCE:-}" ] && [ -n "$latest" ] && [ -z "$(find "$latest" -maxdepth 0 -mmin +110 2>/dev/null)" ]; then continue; fi
  szf=$(onnode "du -sm $newest | cut -f1"); sza=$(onnode "du -sm $newest/adapter | cut -f1")
  if room_c $(( ${szf:-6000} + 400 )) && onnode "rm -rf $RES/${stage}_ckpt.tmp && cp -r $newest $RES/${stage}_ckpt.tmp && mv $RES/${stage}_ckpt.tmp $RES/${stage}_ckpt_$step"; then
    [ -n "$lastf" ] && rm -rf "$lastf"; [ -n "$lasta" ] && rm -rf "$lasta"
    echo "SAVED $stage full checkpoint $step ${szf} MB $(date +%H:%M)"
  else
    rm -rf $RES/${stage}_ckpt.tmp
    if room_a $(( ${sza:-2000} + 300 )) && onnode "rm -rf $RES/${stage}_ad.tmp && mkdir -p $RES/${stage}_ad.tmp && cp -r $newest/adapter $newest/state.json $RES/${stage}_ad.tmp/ && mv $RES/${stage}_ad.tmp $RES/${stage}_adapter_$step"; then
      [ -n "$lasta" ] && rm -rf "$lasta"
      echo "SAVED $stage adapter-only $step ${sza} MB (full copy kept: $(basename ${lastf:-none})) $(date +%H:%M)"
    else echo "FAILED $stage checkpoint $step (free $(free_mb) MB, c-chunks $(n_of c)) $(date +%H:%M)"; rm -rf $RES/${stage}_ad.tmp; fi
  fi
done
grab
exit 0
