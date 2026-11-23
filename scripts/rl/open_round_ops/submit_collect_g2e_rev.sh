#!/bin/bash
# submit_collect_g2e_rev.sh: the grid2op + engibench collection under the open rules on Orchard's general GPU,
# working from the far end of drc's list (drc j2520 runs the same 865 problems front to back), so the two meet
# in the middle. Problems: the training pool's grid2op then engibench lines (drc's order), reversed, without the
# first SKIP of them (done or in flight on drc). Same keys as drc (dx38mtR/r1/<problem_id>).
set -u
SKIP=${SKIP:-215}
POOL=/home/wxu2/rl/problems/train_p2_datav4.jsonl
D=/project/community/wxu2/rl/screen/open_collect_g2e_rev
export OUT=$D
[ -e "$D" ] && { echo "ABORT: $D exists"; exit 2; }
mkdir -p "$D"
sha1sum "$POOL"
python3 - "$POOL" "$D/problems_rev.jsonl" "$SKIP" <<'EOF'
import json, sys
pool, out, skip = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [l for l in open(pool) if l.strip() and json.loads(l)["variant"] in ("grid2op", "engibench")]
order = [l for l in rows if json.loads(l)["variant"] == "grid2op"] + [l for l in rows if json.loads(l)["variant"] == "engibench"]
keep = order[skip:][::-1]
with open(out, "w") as f:
    f.writelines(keep)
print("pool grid2op+engibench", len(order), "| written (reversed, after the first %d)" % skip, len(keep),
      "| first", json.loads(keep[0])["spec"]["problem_id"], "| last", json.loads(keep[-1])["spec"]["problem_id"])
EOF
export LABEL=dx38mtR ADAPTER=/project/community/wxu2/rl/inputs/dx38mtR_final THINK=1 NSERVERS=1 WORKERS=48 RUNS=1
export VARIANTS="engibench grid2op" HARNESS_SCRIPT=dataxp_policy_eval_v3r.py HARNESS_ARGS="--open-rules --record-prompts"
export PROBLEMS=$D/problems_rev.jsonl
cd ~/rl/repo && sbatch --parsable -p general --qos=general_qos --gres=gpu:1 --cpus-per-task=48 --mem=300G -t 12:00:00 \
    -J dx_collect_g2e --export=ALL slurm/orchard_dataxp_eval.sbatch
sleep 5
squeue -u "$USER" -o '%.9i %.14j %.10P %.9T %.20S %.10M %R'
