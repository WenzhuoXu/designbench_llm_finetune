#!/bin/bash
# Second sampling run: the same states re-selected in the set_parameter spelling
# (docs/dataxp_spellings.md), rows under corpus/rows_sp/<group>/ for the four groups.
# DATAXP_SPELLING is read once per dataxp process; every sampler / check / merge below inherits it.
export DATAXP_SPELLING=set_parameter
export OWN_ROWS=/project/community/wxu2/corpus/rows_sp
export OWN_DOMAINS=${OWN_DOMAINS:-"grid2op v2 engibench photonics"}
exec bash /project/community/wxu2/corpus/scripts/sample_job.sh
