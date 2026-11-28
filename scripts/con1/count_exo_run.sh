#!/bin/bash
# count_exo_run.sh : Orchard CPU (preempt), count only: grid2op rows of the stage-1 set and of the 2M set (facetm_short_pw0_nc_q36)
# whose targets carry move-independent metric lines. Reads the states, exclusion rows and lens file; writes one report.
set -uo pipefail
B=/project/community/wxu2/con1; F=$B/full; G=$B/gen; O=$HOME/con1_exo
TRPY=/project/community/wxu2/rl/envs/train/bin/python
export PYTHONPATH=$B/pylib PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "=== $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-none}"
$TRPY $O/count_exo_rows.py --builder-dir $B/repo2m_v3/scripts/con1 \
  --states $B/main/states_native_offmenu.jsonl.gz $F/states_native_offmenu_rest.jsonl.gz $G/out/states_native_orch.jsonl.gz $G/out/states_scripted.jsonl.gz \
  --exclude-rows $B/main/rows/train_facet.jsonl $B/main/rows/dev_facet.jsonl $B/main/rows/dev_facetm.jsonl \
  --lens $F/lens_facetm_short_pw0_nc_q36.txt --stage1-rows $B/main/rows/train_facet.jsonl --q 36 --seed 0 --out $O/count_exo_rows.txt
echo "=== $(date '+%F %T') rc=$?"
