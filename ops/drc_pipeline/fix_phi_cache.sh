#!/bin/bash
# The Φ caches copied from the first run carried the OLD state records (pre-rewrite prompt text and
# metadata). Rebuild each states_phi.jsonl as the FRESH state record (states.jsonl) + the Φ-only
# fields of the cache, keyed by sid; then re-score and rebuild tiers for both spellings.
set -uo pipefail
P=/storage/wxu/corpus
$P/env/bin/python - <<'EOF'
import json, os
P = "/storage/wxu/corpus"
PHI_KEYS = ("moves", "gold_phi", "gold_pct", "gold_in_legal", "n_candidates", "n_illegal_enumerated", "n_nan_phi", "offset_rule")
for root, groups in (("rows", ("v2verb", "grid2op", "engibench", "photonics")), ("rows_sp", ("v2", "grid2op", "engibench", "photonics"))):
    for g in groups:
        d = os.path.join(P, root, g)
        sp, st = os.path.join(d, "states_phi.jsonl"), os.path.join(d, "states.jsonl")
        if not (os.path.exists(sp) and os.path.exists(st)):
            print(root, g, "missing files"); continue
        cache = {}
        for l in open(sp):
            o = json.loads(l); cache[o["sid"]] = {k: o[k] for k in PHI_KEYS if k in o}
        out, n, miss = [], 0, 0
        for l in open(st):
            o = json.loads(l); n += 1
            c = cache.get(o["sid"])
            if c is None: miss += 1; continue
            o.update(c); out.append(json.dumps(o))
        os.replace(sp, sp + ".stale")
        with open(sp, "w") as f: f.write("\n".join(out) + "\n")
        spelling = sorted({json.loads(l).get("spelling") for l in open(sp)} - {None})
        print("%s/%s: %d states, %d without cache, spelling %s" % (root, g, n, miss, spelling))
EOF
ROOT=$P/rows    SPELLING=multitool     GRPS="v2verb grid2op engibench photonics" bash $P/bw_score.sh
ROOT=$P/rows_sp SPELLING=set_parameter GRPS="v2 grid2op engibench photonics"     bash $P/bw_score.sh
$P/env/bin/python - <<'EOF'
import json
P = "/storage/wxu/corpus"
for f in ("rows/v2verb/tiers/train_own_act.jsonl", "rows/grid2op/tiers/train_own_act.jsonl", "rows_sp/v2/tiers/train_own_act.jsonl", "rows_sp/photonics/tiers/train_own_act.jsonl"):
    o = json.loads(open(P + "/" + f).readline()); u = o["messages"][1]["content"]
    print(f, "| spelling", o.get("spelling"), "sha", o.get("instruction_sha1"), "| action", o["messages"][-1]["content"][:60])
    print("   user tail:", u[-200:].replace("\n", " | "))
EOF
echo "FIX_PHI_DONE $(date '+%F %T')"
