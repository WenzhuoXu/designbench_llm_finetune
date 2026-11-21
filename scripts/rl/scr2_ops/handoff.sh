#!/bin/bash
# handoff.sh <local samples.jsonl.gz> <orchard job id>: put drc's replies into the Orchard data job's OUT as samples.jsonl,
# only while that job is still PENDING (never under a running sampler) and only if it holds more replies than what is there.
R="C:/Users/xuwen/AppData/Local/Temp/claude/6677f76fa28a0973/9311749f-16a4-40d4-a1a2-48798138b96f/scratchpad/orr.sh"
G="$1"; J="$2"
OUT=/project/community/wxu2/rl/screen/dx38mtR_scr2_fcq
[ -s "$G" ] || { echo "no file $G"; exit 2; }
SHA=$(sha1sum "$G" | cut -c1-40)
echo "local $G sha1 $SHA, $(gzip -dc "$G" | wc -l) lines"
bash "$R" scp "$G" "orchard:rl/logs/_handoff.jsonl.gz" || { echo "scp failed"; exit 3; }
bash "$R" ssh "set -u
S=\$(squeue -h -j $J -o %T 2>/dev/null)
echo \"job $J state: \${S:-gone}\"
[ \"\$S\" = PENDING ] || { echo 'NOT PENDING: nothing written'; rm -f ~/rl/logs/_handoff.jsonl.gz; exit 4; }
[ \"\$(sha1sum ~/rl/logs/_handoff.jsonl.gz | cut -c1-40)\" = $SHA ] || { echo 'checksum mismatch after the copy'; exit 5; }
mkdir -p $OUT
NEW=\$(gzip -dc ~/rl/logs/_handoff.jsonl.gz | wc -l)
OLD=\$( [ -e $OUT/samples.jsonl ] && wc -l < $OUT/samples.jsonl || echo 0 )
echo \"replies there now \$OLD, in the hand-off \$NEW\"
[ \"\$NEW\" -gt \"\$OLD\" ] || { echo 'nothing new'; rm -f ~/rl/logs/_handoff.jsonl.gz; exit 0; }
gzip -dc ~/rl/logs/_handoff.jsonl.gz > $OUT/samples.jsonl.tmp && [ \"\$(squeue -h -j $J -o %T)\" = PENDING ] && mv $OUT/samples.jsonl.tmp $OUT/samples.jsonl && echo HANDOFF_DONE
rm -f ~/rl/logs/_handoff.jsonl.gz $OUT/samples.jsonl.tmp
ls -la $OUT; wc -l < $OUT/samples.jsonl; squeue -u wxu2 -o '%.9i %.12j %.9T %.20S %R'"
