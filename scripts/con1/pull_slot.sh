#!/bin/bash
# pull_slot.sh WHICH [LABEL] : copy the valid bytes of an Orchard checkpoint slot (con1_reserve/slots, guard_v3 layout)
# to the local disk for drc, by the RL session's route (pull_ckpt.sh: its orr.sh / orch.sh over the IAP forward,
# 1800 MiB byte-range chunks, each chunk's sha1 checked against Orchard's and retried up to 3 times). Run from the
# owner's PC (Git Bash); it only moves bytes.
#   WHICH = A | B (one file) or F (F1-F3 in order); the slot's meta gives "stage step length sha1 time"; only the first
#   <length> bytes are the tar (the rest of a slot file is stale), so the chunks cover [0, length).
#   out: $D/con1_full__LABEL/LABEL/slot.tar.partNN and $D/con1_full__LABEL.remote.sha1 ("<meta sha1>  LABEL/slot.tar"),
#   the layout of the RL session's join_ckpt.sh on drc (R=con1_full, T=LABEL). Prints PULL_OK or PULL_FAIL.
# Then: upload $D/con1_full__LABEL to drc:/storage/wxu/rl_open/orchard_xfer/ and the .remote.sha1 beside it, and on drc
#   bash /storage/wxu/rl_open/orchard_xfer/join_ckpt.sh con1_full LABEL      (-> "1 files DRC_SHA_MATCH")
#   bash /storage/wxu/con1/full_eval/eval_link.sh LABEL                     (serve, ask, score: scripts/con1/eval_link.sh)
set -u
S=${S:-/c/Users/xuwen/AppData/Local/Temp/claude/6677f76fa28a0973/9311749f-16a4-40d4-a1a2-48798138b96f/scratchpad}
D=${D:-/d/designbench_con1/full_xfer}
SL=${SL:-/project/community/wxu2/con1_reserve/slots}
W=${1:?A, B or F}
case "$W" in A|B) FILES="$SL/$W.bin";; F) FILES="$SL/F1.bin $SL/F2.bin $SL/F3.bin";; *) echo "WHICH: A, B or F"; exit 2;; esac
meta=$(timeout 100 bash "$S/orr.sh" ssh "cat $SL/$W.meta") || { echo "PULL_FAIL meta"; exit 1; }
set -- $meta
STAGE=$1 STEP=$2 L=$3 SHA=$4
[ -n "${SHA:-}" ] || { echo "PULL_FAIL meta '$meta'"; exit 1; }
LABEL=${LABEL:-${STAGE}_${STEP#step_}}
B="$D/con1_full__$LABEL"
CH=1800
MB=1048576
mkdir -p "$B/$LABEL"
echo "$(date '+%F %T') slot $W: $meta -> $B"
echo "$SHA  $LABEL/slot.tar" > "$B.remote.sha1"
n=$(( (L + CH*MB - 1) / (CH*MB) ))
fail=0
for ((k=0; k<n; k++)); do
  off=$((k*CH*MB)); len=$(( L - off < CH*MB ? L - off : CH*MB ))
  # the chunk: bytes [off, off+len) of the slot files read in order (tail -c +N counts from 1)
  stream="cat $FILES 2>/dev/null | tail -c +$((off+1)) 2>/dev/null | head -c $len"
  out=$(printf "%s/%s/slot.tar.part%02d" "$B" "$LABEL" "$k")
  want=$(timeout 900 bash "$S/orr.sh" ssh "$stream | sha1sum" | awk '{print $1}')
  ok=0
  for try in 1 2 3; do
    if [ -s "$out" ] && [ "$(stat -c %s "$out")" = "$len" ] && [ "$(sha1sum "$out" | awk '{print $1}')" = "$want" ]; then ok=1; break; fi
    timeout 1800 bash "$S/orch.sh" ssh "$stream" > "$out" < /dev/null
  done
  [ -s "$out" ] && [ "$(stat -c %s "$out")" = "$len" ] && [ "$(sha1sum "$out" | awk '{print $1}')" = "$want" ] && ok=1
  echo "part $k/$n $(stat -c %s "$out") bytes ok=$ok"
  [ "$ok" = 1 ] || fail=1
done
# the slot must not have been rewritten meanwhile, and the joined parts must give the meta's sha1
again=$(timeout 100 bash "$S/orr.sh" ssh "cat $SL/$W.meta")
[ "$again" = "$meta" ] || { echo "slot $W changed during the pull: '$again'"; fail=1; }
got=$(cat "$B/$LABEL"/slot.tar.part?? | sha1sum | awk '{print $1}')
[ "$got" = "$SHA" ] || { echo "joined sha1 $got, meta $SHA"; fail=1; }
[ "$fail" = 0 ] && echo "PULL_OK $LABEL $L bytes sha1 $SHA" || echo PULL_FAIL
date '+%F %T'
