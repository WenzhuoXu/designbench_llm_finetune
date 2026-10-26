#!/bin/bash
# States for the set_parameter spelling (datav4_set_parameter): same states as the multitool rows
# (same problems, same draw, same seed), re-selected with DATAXP_SPELLING=set_parameter so the
# stored prompt parts carry that spelling; Φ caches are reused by state id. Output rows_sp/<group>/.
set -uo pipefail
P=/storage/wxu/corpus
export PYTHONPATH=$P/code_v3 DATAXP_SPELLING=set_parameter
PY=$P/env/bin/python
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export DATAXP_MOLOPT_ORACLE_DIR=/home/wxu/.dataxp/molopt_oracle/oracle
SUPP=/storage/wxu/sftprobe/src/dataxp/run2_supp/batch1_idx0130-0259
R=$P/rows_sp; mkdir -p $R/logs
sel() {  # group data variants per_domain maxidx
  g=$1; data=$2; vs=$3; pd=$4; mx=$5
  mkdir -p $R/$g
  $PY -m dataxp.ownrows.select_states --data $data --variants $vs --out $R/$g/all --per-domain $pd --dev-per-domain 12 --seed 0 > $R/logs/select_$g.log 2>&1 || { echo "select $g failed"; tail -3 $R/logs/select_$g.log; return; }
  $PY - "$g" "$mx" <<'EOF'
import json, os, sys
g, mx = sys.argv[1], int(sys.argv[2]); R = "/storage/wxu/corpus/rows_sp/" + g
n = k = 0
with open(os.path.join(R, "states.jsonl"), "w") as out:
    for l in open(os.path.join(R, "all", "states.jsonl")):
        o = json.loads(l); n += 1
        pid = o.get("instance") or o.get("problem_id") or ""
        try: idx = int(str(pid).rsplit("_", 1)[-1])
        except ValueError: idx = -1
        if mx < 0 or 0 <= idx <= mx:
            out.write(l); k += 1
sp = {json.loads(l).get("spelling") for l in open(os.path.join(R, "states.jsonl"))}
print("%s: %d of %d states kept; spelling %s" % (g, k, n, sorted(x for x in sp if x)))
EOF
}
sel v2 $SUPP "circuit2 frame3d2 grid2 seir2 traffic2" 600 -1
sel grid2op $P/out/grid2op grid2op 0 68
sel engibench $P/out/engibench engibench 0 69
sel photonics $P/out/photonics photonics 0 69
# reuse the Φ caches (keyed by sid) from the multitool run
$PY - <<'EOF'
import json, os
P = "/storage/wxu/corpus"
for g, src in (("v2", "v2verb"), ("grid2op", "grid2op"), ("engibench", "engibench"), ("photonics", "photonics")):
    cache = os.path.join(P, "rows", src, "states_phi.jsonl")
    if not os.path.exists(cache): print(g, "no cache"); continue
    old = {json.loads(l)["sid"]: l for l in open(cache)}
    sids = [json.loads(l)["sid"] for l in open(os.path.join(P, "rows_sp", g, "states.jsonl"))]
    hit = sum(s in old for s in sids)
    print("%s: phi cache %d of %d" % (g, hit, len(sids)))
    if hit == len(sids):
        with open(os.path.join(P, "rows_sp", g, "states_phi.jsonl"), "w") as f:
            for s in sids: f.write(old[s])
EOF
wc -l $R/*/states.jsonl $R/*/states_phi.jsonl 2>/dev/null
echo "SELECT_SP_DONE $(date '+%F %T')"
