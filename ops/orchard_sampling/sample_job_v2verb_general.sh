#!/bin/bash
# v2verb only, for a 1-GPU job on the general partition (general_qos: 1 GPU, 1 job per user).
# Writes rows/v2verb/general/samples.shard0.jsonl; the merge and reshard of any other job
# (e.g. the 4-GPU advanced job writing rows/v2verb/samples.shard<i>.jsonl) pick it up.
export OWN_DOMAINS=v2verb OWN_OUTSUB=general
exec bash /project/community/wxu2/corpus/scripts/sample_job.sh
