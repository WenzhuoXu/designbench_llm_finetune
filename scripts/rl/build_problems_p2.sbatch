#!/bin/bash
# Training problems for dataxp RL, datav4 protocol (P2): the 8 training families' corpus problems
# (copied from bridges2 datav4_corpus to $HOME/rl/corpus_problems), minus every evaluation problem
# (harness problems_v2.jsonl + problems_v3_native.jsonl, by id and canonical spec), duplicates and
# feasible starts. CPU only; idempotent (rewrites its output). Run 2026-10-25 as Orchard job 161611.
#SBATCH -J rl_problems
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -A ccm
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /project/community/wxu2/rl/logs/rl_problems_%j.out
set -euo pipefail
B=/project/community/wxu2; H=$B/corpus/harness; R=$HOME/rl     # persistent files in $HOME
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
export DX_ORIG_SCRIPTS=$H/scripts DATAXP_MOLOPT_ORACLE_DIR=$H/molopt_oracle MPLBACKEND=Agg
export TMPDIR=$B/rl/cache/tmp XDG_CACHE_HOME=$B/rl/cache/xdg
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$R/problems"
CORPUS=""
for v in circuit2 frame3d2 grid2 seir2 traffic2 grid2op engibench photonics; do
  for f in "$R"/corpus_problems/$v/accepted*/problems.jsonl; do CORPUS="$CORPUS $v=$f"; done
done
echo "=== $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-none}"
cd "$R/repo"
time "$B/envs/corpus_o/bin/python" -m llm_finetune.rl_dataxp.problems \
  --script "$H/scripts/dataxp_policy_eval_v3.py" --code "$H/dataxp_code_v3" --corpus $CORPUS \
  --eval-problems "$H/problems_v2.jsonl" "$H/problems_v3_native.jsonl" \
  --heldout thermal2 inventory2 molopt --out "$R/problems/train_p2_datav4.jsonl"
sha1sum "$R/problems/train_p2_datav4.jsonl"
echo "RL_PROBLEMS_DONE $(date '+%F %T')"
