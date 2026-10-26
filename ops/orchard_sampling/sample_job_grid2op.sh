#!/bin/bash
# grid2op only, for a 1-GPU job on the general partition (general_qos: 1 GPU, 1 job per user).
# Writes rows/grid2op/general/samples.shard0.jsonl; the merge and reshard of any other job
# pick it up.
export OWN_DOMAINS=grid2op OWN_OUTSUB=general
exec bash /project/community/wxu2/corpus/scripts/sample_job.sh
