#!/bin/bash
# drc_tail.sh D2 (Orchard login node): drc's grid2op share [0, 2220) is sampled front to back on one GPU (~290 states/h);
# once the general GPU is free, the states [D2, 2220) go to a job on it (back to front) and drc's stop watcher is
# restarted with D2 (done from the PC: drc and Orchard cannot reach each other). Disjoint shares; states drc starts
# past D2 before it stops are sampled twice at most and the merge keeps one row per state.
set -u
D2=$1
END=${END:-2220}
R=/project/community/wxu2/rl/screen/open_round
IN=$R/drc_in
OUT=$R/grid2op_drc_tail
FIRST_SID='grid2op|grid2op_0018|0-97d170e1550e~ececbba8'
[ -e "$OUT" ] && { echo "ABORT: $OUT exists"; exit 2; }
python3 - $IN/states.jsonl $IN/states_tail.jsonl "$FIRST_SID" "$D2" "$END" <<'EOF' || { echo "ABORT: tail states failed"; exit 3; }
import json, os, sys
src, out, first, d2, end = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
lines = [l for l in open(src) if l.strip()]
if json.loads(lines[0])["sid"] != first:
    sys.exit("drc_in/states.jsonl does not start with drc's first state")
if not 0 < d2 < end <= len(lines):
    sys.exit("bad D2 %d (END %d, %d states)" % (d2, end, len(lines)))
tail = lines[d2:end][::-1]
open(out + ".tmp", "w").writelines(l if l.endswith("\n") else l + "\n" for l in tail)
os.replace(out + ".tmp", out)
print("tail states [%d, %d) = %d (back to front) -> %s" % (d2, end, len(tail), out))
EOF
echo "D2=$D2 END=$END $(date '+%F %T')" > $IN/split_D2.txt
cd ~/rl/repo || exit 5
export OUT LABEL=dx38mtR ADAPTER=/project/community/wxu2/rl/inputs/dx38mtR_final
export STATES_FILE=$IN/states_tail.jsonl
export K=8 TEMPERATURE=0.8 FIRST=grid2op TIER=open1 MAX_WORDS=1000
export NSERVERS=1 SAMPLE_WORKERS=32 SAMPLE_KV_BUDGET=330000 SAMPLE_REPLY_TOKENS=700 SAMPLE_PROMPT_WEIGHT=1.8
export SCORE_ALONGSIDE=900 SCORE_WORKERS=36
G=$(sbatch --parsable -p general --qos=general_qos --gres=gpu:1 --cpus-per-task=48 --mem=300G -t 12:00:00 \
    -J dx_open_g2t --export=ALL slurm/orchard_screen_data.sbatch)
echo "grid2op tail: $G"
sleep 10
squeue -u "$USER" -o '%.9i %.14j %.10P %.9T %.10M %R'
