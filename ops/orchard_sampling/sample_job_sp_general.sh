#!/bin/bash
# The set_parameter run on one GPU (general partition), as a hedge while the 4-GPU advanced job
# waits: writes rows_sp/<group>/general/samples.shard0.jsonl; the 4-GPU job seeds itself from it.
export OWN_OUTSUB=general
exec bash /project/community/wxu2/corpus/scripts/sample_job_sp.sh
