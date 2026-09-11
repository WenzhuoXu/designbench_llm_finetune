#!/usr/bin/env python
"""Shape/size profile of a gen_expert JSONL, and what the existing SFT pipeline makes of it."""
import json, sys, statistics
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
sys.path.insert(0, str(PROJECT))
from llm_finetune.data.processors.warmstart_transform import WarmstartTransform

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
turns = [r["n_action_turns"] for r in recs]
chars = [sum(len(m["content"]) for m in r["messages"]) for r in recs]
acts = [a for r in recs for a in r["actions"]]
comp = [a for a in acts if " ; " in a]
parts = [len(a.split(" ; ")) for a in acts]

print("records                %d" % len(recs))
print("action turns           total %d | mean %.2f | median %d | max %d"
      % (sum(turns), statistics.mean(turns), statistics.median(turns), max(turns)))
print("assistant turns        %d" % sum(r["n_assistant_turns"] for r in recs))
print("conversation chars     mean %.0f | median %.0f | max %d"
      % (statistics.mean(chars), statistics.median(chars), max(chars)))
print("actions                %d total | %d compound (%.1f%%) | parts mean %.2f max %d"
      % (len(acts), len(comp), 100.0 * len(comp) / max(len(acts), 1),
         statistics.mean(parts), max(parts)))
print("distinct action heads  %s"
      % sorted({a.split("(")[0] for p in acts for a in p.split(" ; ")}))

tm = WarmstartTransform()
kept, lost = 0, 0
for r in recs:
    out = tm.transform_messages(r["messages"])
    for m, a in zip([x for x in out if x["role"] == "assistant"], r["actions"] + [None]):
        if a is None:
            continue
        if ("<action>%s</action>" % a) in m["content"]:
            kept += 1
        else:
            lost += 1
print("WarmstartTransform     %d/%d action turns survive whole; %d truncated to their first part"
      % (kept, kept + lost, lost))
