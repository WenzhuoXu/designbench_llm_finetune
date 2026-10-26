#!/bin/bash
# python -m dataxp.smoke <variant> for the four native-verb variants, run concurrently
# (one core each, BLAS pinned) from /storage/wxu/corpus/code with the corpus venv.
# Per-variant logs: /storage/wxu/corpus/smoke/<variant>.log, last line "SMOKE_RC <v> rc=<rc> wall=<s>s".
# The molopt smoke also performs PyTDC's one-time oracle download into
# DATAXP_MOLOPT_ORACLE_DIR (shared by every worker of the later 16-worker job).
set -u
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DATAXP_MOLOPT_ORACLE_DIR=/home/wxu/.dataxp/molopt_oracle
P=/storage/wxu/corpus/env/bin/python
cd /storage/wxu/corpus/code || exit 2
mkdir -p /storage/wxu/corpus/smoke "$DATAXP_MOLOPT_ORACLE_DIR"
"$P" --version
for v in grid2op photonics molopt engibench; do
  ( t0=$SECONDS; "$P" -m dataxp.smoke "$v" > /storage/wxu/corpus/smoke/$v.log 2>&1; rc=$?;
    echo "SMOKE_RC $v rc=$rc wall=$((SECONDS-t0))s" >> /storage/wxu/corpus/smoke/$v.log ) &
done
wait
for v in grid2op photonics molopt engibench; do
  echo; echo "===== $v"; tail -n 30 /storage/wxu/corpus/smoke/$v.log
done
echo; echo "===== summary"; grep -h SMOKE_RC /storage/wxu/corpus/smoke/*.log
ls -la "$DATAXP_MOLOPT_ORACLE_DIR"
