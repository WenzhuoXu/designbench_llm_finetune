#!/bin/bash
# pack.sh   inside job 173954's allocation: two archives of the round's data for the copy off the pool, with checksums
set -u
O=/project/community/wxu2/rl/screen/dx38mtR_scr2_fcq
B=$O/logs/backup; mkdir -p "$B"
cd "$O" || exit 2
sha1sum rows_fc.jsonl rows_fc.report.txt mix_fc/train.jsonl mix_fc/dev.jsonl mix_fc/train.jsonl.manifest.json samples.jsonl scored.jsonl \
    gate_rows.jsonl gate_rows.report.txt gate_scored.jsonl > "$B/sha1.txt"
wc -l rows_fc.jsonl mix_fc/train.jsonl mix_fc/dev.jsonl samples.jsonl scored.jsonl > "$B/lines.txt"
tar -czf "$B/scr2_fcq_rows.tgz.tmp" rows_fc.jsonl rows_fc.report.txt mix_fc gate_rows.jsonl gate_rows.report.txt gate_scored.jsonl \
    logs/samples.log logs/samples_tuned.log logs/gate.log logs/scored.log logs/scored_early.log \
    logs/swap/sids_drc_m1.txt logs/swap/early_rows.report.txt logs/backup/sha1.txt logs/backup/lines.txt *.done \
    && mv "$B/scr2_fcq_rows.tgz.tmp" "$B/scr2_fcq_rows.tgz"
tar -czf "$B/scr2_fcq_samples_scored.tgz.tmp" samples.jsonl scored.jsonl && mv "$B/scr2_fcq_samples_scored.tgz.tmp" "$B/scr2_fcq_samples_scored.tgz"
ls -la "$B"
sha1sum "$B"/*.tgz
echo PACK_DONE
