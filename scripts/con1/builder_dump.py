"""Render rows and eval prompts of a few states with one library copy, for builder_compare.py (CPU, seconds; 11-28).

    python builder_dump.py LIBDIR MODE STATES.jsonl.gz QUERIES.jsonl OUT.jsonl [N_PER_FAMILY]

MODE: 'old' (the copy's own defaults; lib_v6 = every set before 11-28), 'optout' (configure(no_defined_terms=True,
'keep')), 'default' (lib_v8 defaults), 'zero', 'drop' (configure(False, mode)). Per state (the first N of each family):
build_rows.conseq_row(state, its first 6 moves, fmt) for fmt in facet, facetm, dir, delta, absolute; per query (the
first N of each family): skill_client.messages_for(q, 0, False, 'facetm'). One JSON line per rendered object.
"""
import gzip
import json
import sys

lib, mode, sp, qp, out = sys.argv[1:6]
n = int(sys.argv[6]) if len(sys.argv) > 6 else 20
sys.path.insert(0, lib)
sys.dont_write_bytecode = True
import build_rows as BR  # noqa: E402
import skill_client as SC  # noqa: E402

if mode != "old":
    BR.configure(mode == "optout", mode if mode in ("zero", "drop") else "keep")
cnt = {}
with open(out, "w") as fo:
    for line in gzip.open(sp, "rt", encoding="utf-8"):
        st = json.loads(line)
        v = st["variant"]
        if cnt.get(v, 0) >= n or not st.get("moves"):
            continue
        cnt[v] = cnt.get(v, 0) + 1
        for fmt in ("facet", "facetm", "dir", "delta", "absolute"):
            fo.write(json.dumps({"kind": "row", "sid": st["sid"], "fmt": fmt, "row": BR.conseq_row(st, st["moves"][:6], fmt)},
                                sort_keys=True) + "\n")
    qc = {}
    for line in open(qp, encoding="utf-8"):
        q = json.loads(line)
        if qc.get(q["variant"], 0) >= n:
            continue
        qc[q["variant"]] = qc.get(q["variant"], 0) + 1
        fo.write(json.dumps({"kind": "eval", "sid": q["sid"], "msgs": SC.messages_for(q, 0, False, "facetm")},
                            sort_keys=True) + "\n")
print("BUILDER_DUMP_DONE", mode, cnt, sum(qc.values()))
