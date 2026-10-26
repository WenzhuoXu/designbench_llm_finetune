#!/bin/bash
# dx_validate_orchard.sh (2026-10-23): the harness loop on Orchard without a model, CPU only.
# Non-LLM policies on the 3 lowest evaluation problems of each native-verb variant
# (test/problems_native3.jsonl: grid2op 69-71, photonics/molopt/engibench 70-72) and on the 3
# old-variant problems of drc's acceptance (test/problems_old3.jsonl: circuit2_e10000, thermal2_e10001,
# frame3d2_e10000), PYTHONHASHSEED=0, one thread per simulator, ONE worker per run (episodes are written
# as they finish, so a multi-worker file order would depend on timing). Every per-turn numeric must
# equal drc's: test/verbs_<pol> for the old variants (drc, d27_drc python; verbs_nat_random under the
# corpus env), test/refs_native3/<variant>_<pol> for the native variants (drc, corpus env,
# make_refs_native3_drc.sh). Outputs go to test/orchard/; compare_validation.py does the comparison.
#   [CLIENT_PY_OVERRIDE=<python>] [TAG=orchard] bash dx_validate_orchard.sh
set -uo pipefail
B=/project/community/wxu2
H="${H:-$B/corpus/harness}"
T=$H/test
O=$T/${TAG:-orchard}; mkdir -p "$O"
PY="${CLIENT_PY_OVERRIDE:-$B/envs/corpus_o/bin/python}"
V3S=$H/scripts/dataxp_policy_eval_v3.py
V3C=$H/dataxp_code_v3
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export DX_ORIG_SCRIPTS=$H/scripts DATAXP_MOLOPT_ORACLE_DIR="${DATAXP_MOLOPT_ORACLE_DIR:-$H/molopt_oracle}"
export TMPDIR=$B/corpus/cache/tmp; mkdir -p "$TMPDIR"
cd "$H"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} python=$PY ($("$PY" -c 'import sys, numpy, scipy; print(sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__)'))"
echo "=== script sha1 $(sha1sum < $V3S | cut -c1-12); code: $(cd $V3C && sha1sum -c SHA1SUMS 2>/dev/null | grep -c ': OK$') of $(wc -l < $V3C/SHA1SUMS) files match SHA1SUMS"
OLD=(--variants circuit2 thermal2 frame3d2 --n 0 --turns 20 --obs v2 --param-line --instruction-file $H/configs/task_instruction_V2.txt
     --problems-file $T/problems_old3.jsonl --record-prompts --workers 1)
echo "=== old variants (SET_PARAM data, verb surface): random greedy repeat_last phi_select, in parallel"
for pol in random greedy repeat_last phi_select; do
  "$PY" "$V3S" episodes --code "$V3C" "${OLD[@]}" --policy $pol --out "$O/old_$pol" > "$O/old_$pol.log" 2>&1 &
done
echo "=== native variants: random greedy phi_select x grid2op photonics molopt engibench, in parallel"
for v in grid2op photonics molopt engibench; do
  for pol in random greedy phi_select; do
    "$PY" "$V3S" episodes --code "$V3C" --variants $v --n 0 --turns 20 --policy $pol \
       --problems-file "$T/problems_native3.jsonl" --record-prompts --workers 1 --out "$O/${v}_$pol" > "$O/${v}_$pol.log" 2>&1 &
  done
done
wait
echo "=== $(date) runs finished"
for f in "$O"/*.log; do
  n=$(basename "$f" .log)
  if grep -q '^elapsed ' "$f"; then echo "--- $n: $(grep -A1 '^variant' "$f" | tail -1 | tr -s ' ')"; else echo "--- $n: NO SUMMARY (rc?)"; tail -3 "$f"; fi
done
for f in "$O"/*.episodes.jsonl; do echo "$(sha1sum < "$f" | cut -c1-12) $(wc -c < "$f") $(basename "$f")"; done
echo "VALIDATE_DONE $(date)"
