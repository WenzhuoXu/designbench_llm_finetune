#!/bin/bash
# Normalise the spelling label of the multitool rows (states selected before the spelling switch
# carry "verb_surface" or nothing), then build datav4_multitool, datav4_set_parameter and the shared
# datav4_corpus under /storage/wxu/datav4, and run --check on both.
set -uo pipefail
P=/storage/wxu/corpus
export PYTHONPATH=$P/code_v3
PY=$P/env/bin/python
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export DATAXP_MOLOPT_ORACLE_DIR=/home/wxu/.dataxp/molopt_oracle/oracle
$PY - <<'EOF'
import json, os, glob
P = "/storage/wxu/corpus/rows"
for f in glob.glob(P + "/*/tiers/*.jsonl"):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    changed = 0
    for r in rows:
        if r.get("spelling") in (None, "verb_surface", "native"):
            r["spelling"] = "multitool"; changed += 1
    if changed:
        with open(f, "w") as fh:
            for r in rows: fh.write(json.dumps(r) + "\n")
    print("%s: %d rows, %d relabelled multitool" % (os.path.relpath(f, P), len(rows), changed))
EOF
OUT=/storage/wxu/datav4; mkdir -p $OUT
COMMON="--native-corpus $P/out --v2-corpus run2_supp=/storage/wxu/sftprobe/src/dataxp/run2_supp/batch1_idx0130-0259 --v2-corpus-ref run2=/ocean/projects/mch250030p/shared/dataxp/run2 --eval-problems-native $P/harness/problems_v3_native.jsonl --eval-problems-v2 /storage/wxu/sftprobe/problems_v2.jsonl --eval-episodes $P/harness/eval/r1 --corpus-mode shared --corpus-dir $OUT/datav4_corpus --force"
echo "=== $(date '+%F %T') package multitool"
$PY -m dataxp.package_datav4 --spelling multitool --rows-root $P/rows --out $OUT/datav4_multitool $COMMON --stamp-spelling 2>&1 | tail -25
echo "=== $(date '+%F %T') package set_parameter"
DATAXP_SPELLING=set_parameter $PY -m dataxp.package_datav4 --spelling set_parameter --rows-root $P/rows_sp --out $OUT/datav4_set_parameter $COMMON 2>&1 | tail -25
echo "=== $(date '+%F %T') check"
$PY -m dataxp.package_datav4 --check --out $OUT/datav4_multitool --corpus-dir $OUT/datav4_corpus --parse-check 2>&1 | tail -6
DATAXP_SPELLING=set_parameter $PY -m dataxp.package_datav4 --check --out $OUT/datav4_set_parameter --corpus-dir $OUT/datav4_corpus --parse-check 2>&1 | tail -6
du -sh $OUT/*; find $OUT -type f | wc -l
echo "PACKAGE_DONE $(date '+%F %T')"
