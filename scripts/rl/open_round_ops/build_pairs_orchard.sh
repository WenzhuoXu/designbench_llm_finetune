#!/bin/bash
# build_pairs_orchard.sh (Orchard login node, 2026-11-23): preference pairs of the five Orchard shares of the
# open-rules collection (scripts/rl_screen/build_pairs.py, copied to open_round/pairs/code), one file per share in
# open_round/pairs/<share>.jsonl with its report. drc's share is built on drc (same code) and uploaded next to them.
set -u
R=/project/community/wxu2/rl/screen/open_round
P=$R/pairs
cd $P/code || exit 2
for s in grid2op_drc_rev grid2op_drc_tail g2e_orchard g2e_share2 photonics; do
  python3 -I build_pairs.py --states $R/$s/states.jsonl --samples $R/$s/samples.jsonl --scored $R/$s/scored.jsonl \
    --out $P/$s.jsonl --min-pct 0 --max-words 1000 --tier pref1 > $P/$s.report.txt 2>&1 || { echo "FAILED $s"; tail -5 $P/$s.report.txt; continue; }
  tail -1 $P/$s.report.txt
done
