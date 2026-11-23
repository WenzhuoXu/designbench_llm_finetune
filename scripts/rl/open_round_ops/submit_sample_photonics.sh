#!/bin/bash
# submit_sample_photonics.sh: the open-rules round's photonics data on the advanced queue, right after its collection
# (174586) ends OK: states of the collected episodes (open_states.py, in the job), 8 replies per state at T=0.8, scoring
# alongside, gain rows (--min-pct 0 --max-words 1000). KV budget as measured best for photonics in the forecast round.
set -u
P=/project/community/wxu2/rl/screen
export OUT=$P/open_round/photonics LABEL=dx38mtR ADAPTER=/project/community/wxu2/rl/inputs/dx38mtR_final
export OPEN_EPISODES=$P/open_collect_photonics/dx38mtR.episodes.jsonl
export K=8 TEMPERATURE=0.8 FIRST=photonics TIER=open1 MAX_WORDS=1000
export SAMPLE_WORKERS=96 SAMPLE_KV_BUDGET=270000 SAMPLE_REPLY_TOKENS=1700 SAMPLE_PROMPT_WEIGHT=1.0
export SCORE_ALONGSIDE=900 SCORE_WORKERS=84
[ -e "$OUT" ] && { echo "ABORT: $OUT exists"; exit 2; }
for f in open_states.py score_loop.sh complete_states.py; do [ -e ~/rl/repo/scripts/rl_screen/$f ] || { echo "ABORT: $f not deployed"; exit 2; }; done
grep -q OPEN_EPISODES ~/rl/repo/scripts/rl/orchard_screen_data.sh || { echo "ABORT: launcher not deployed"; exit 2; }
cd ~/rl/repo && sbatch --parsable -t 1-00:00:00 -J dx_open_ph --dependency=afterok:174586 --export=ALL slurm/orchard_screen_data.sbatch
sleep 5
squeue -u "$USER" -o '%.9i %.14j %.10P %.9T %.20S %.10M %R'
