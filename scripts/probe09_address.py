"""
EXPERIMENT 9 -- address corruption vs value corruption.

The derivation claims the placebo's -35.6pp is caused by ADDRESS corruption: a permutation
preserves the extrema the policy's decision rule keys on (the smallest FOS in the table is
still the smallest number) while destroying WHERE they point. Prediction:

    P(emitted element = the TRUE argmin | shuffled table)  ~  chance (about 1/m)

against roughly 0.927 with a truthful table. If it holds, the policy is a transcriber of the
observation's addresses and carries no independent prior that could override a false one --
which is why a confidently wrong table is worse than no table.

Needs the placebo run's per-turn records. No simulator.
"""
import json, re, sys, glob, collections, statistics
from pathlib import Path
P = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")

f = P / "results/api_guidance/grammar_placebo.jsonl"
rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
print("rows:", len(rows))
print("arms:", collections.Counter((r.get("obs"), r.get("act")) for r in rows))
print("keys on row 0:", sorted(rows[0].keys()))
for k in sorted(rows[0].keys()):
    v = rows[0][k]
    s = json.dumps(v)[:170] if not isinstance(v, str) else v[:170]
    print("   %-26s %s" % (k, s.replace("\n", " | ")))

# If per-turn detail exists, test the address hypothesis.
turnkey = None
for cand in ("turns", "trace", "steps", "history"):
    if isinstance(rows[0].get(cand), list) and rows[0][cand]:
        turnkey = cand
        break
print("\nper-turn key:", turnkey)
if turnkey:
    t0 = rows[0][turnkey][0]
    print("turn keys:", sorted(t0.keys()) if isinstance(t0, dict) else type(t0))
    for k in sorted(t0.keys()) if isinstance(t0, dict) else []:
        v = t0[k]
        s = json.dumps(v)[:200] if not isinstance(v, str) else v[:200]
        print("   %-26s %s" % (k, s.replace("\n", " | ")))
