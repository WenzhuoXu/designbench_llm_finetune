#!/bin/bash
# regrid.sh (Orchard login node): drc samples drc's 4,440 grid2op states front to back (index 1,141 at 03:13, ~310
# states/h); 174714 sampled them back to front with a KV budget that left 60% of the cache empty. Fixed shares instead
# of a meeting: drc keeps states [0, D), the Orchard job states [D, 4440) (back to front), restarted on the same OUT with
# its states file cut to that share and a KV budget of 330,000 per server.
set -u
D=${D:-2220}
J=174714
R=/project/community/wxu2/rl/screen/open_round
IN=$R/drc_in
OUT=$R/grid2op_drc_rev
FIRST_SID='grid2op|grid2op_0018|0-97d170e1550e~ececbba8'
python3 -c "import json,sys; s=json.loads(open(sys.argv[1]).readline())['sid']; sys.exit(0 if s==sys.argv[2] else 1)" \
    $IN/states.jsonl "$FIRST_SID" || { echo "ABORT: drc_in/states.jsonl does not start with drc's first state"; exit 2; }
scancel $J
for i in $(seq 1 60); do [ -z "$(squeue -h -j $J -o %T)" ] && break; sleep 5; done
[ -z "$(squeue -h -j $J -o %T)" ] || { echo "ABORT: $J still there"; exit 3; }
sleep 5
python3 - $IN/states.jsonl $OUT/states.jsonl $OUT/samples.jsonl $D <<'EOF' || { echo "ABORT: cut failed"; exit 4; }
import json, os, sys
src, out, samples, d = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
lines = [l for l in open(src) if l.strip()]
share = lines[d:][::-1]
have = set()
for raw in open(samples, "rb"):
    try:
        have.add(json.loads(raw)["sid"])
    except ValueError:
        pass
sids = [json.loads(l)["sid"] for l in share]
outside = have - set(sids)
open(out + ".tmp", "w").writelines(share)
os.replace(out + ".tmp", out)
print("states %d; Orchard share [%d, %d) = %d states, %d of them with replies already; states with replies outside the share %d"
      % (len(lines), d, len(lines), len(share), len(have & set(sids)), len(outside)))
EOF
echo "D=$D $(date '+%F %T')" > $IN/split_D.txt
cd ~/rl/repo || exit 5
export OUT LABEL=dx38mtR ADAPTER=/project/community/wxu2/rl/inputs/dx38mtR_final
export STATES_FILE=$IN/states_rev.jsonl
export K=8 TEMPERATURE=0.8 FIRST=grid2op TIER=open1 MAX_WORDS=1000
export SAMPLE_WORKERS=128 SAMPLE_KV_BUDGET=330000 SAMPLE_REPLY_TOKENS=700 SAMPLE_PROMPT_WEIGHT=1.8
export SCORE_ALONGSIDE=900 SCORE_WORKERS=84
G=$(sbatch --parsable -t 1-00:00:00 -J dx_open_g2r --export=ALL slurm/orchard_screen_data.sbatch)
echo "grid2op share: $G"
sleep 10
squeue -u "$USER" -o '%.9i %.14j %.10P %.9T %.10M %R'
