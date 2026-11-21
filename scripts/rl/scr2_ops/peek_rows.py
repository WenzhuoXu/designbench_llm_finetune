"""peek_rows.py <train.jsonl>: structure checks of the training rows of a forecast round."""
import collections
import json
import sys

n = 0
c = collections.Counter()
shown = 0
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    n += 1
    m = r["messages"]
    c["roles " + ",".join(x["role"] for x in m)] += 1
    c["tier " + str(r.get("tier"))] += 1
    if not str(r.get("tier", "")).startswith("scr2"):
        continue
    a = m[-1]["content"]
    t = m[-1].get("reasoning_content") or ""
    u = " ".join(x["content"] for x in m[:-1])
    c["new: instruction text in the prompt"] += "BEFORE YOU DECIDE" in u or "FORECAST" in u
    c["new: rationale present (reasoning_content)"] += bool(t.strip())
    c["new: reply is one <action> line"] += a.startswith("<action>") and a.endswith("</action>") and a.count("<action>") == 1
    c["new: 2-3 FORECAST lines in the rationale"] += 2 <= t.count("FORECAST:") <= 3
    c["new: stored forecasts all appear in the rationale"] += all(f[0] in t for f in r.get("forecasts", []))
    c["new: the action is one of the forecast moves, said better"] += any(f[0] == a[8:-9] and f[1] == "better" for f in r.get("forecasts", []))
    c["new: one <action> after </think>"] += a.split("</think>")[-1].count("<action>") == 1
    c["new: sample_prompt " + str(r.get("sample_prompt"))] += 1
    if shown < 1:
        shown += 1
        print("--- one new row:", r["sid"], "| words", r.get("words"), "| forecasts", r.get("forecasts"))
        print("rationale end:", repr(t[-700:]))
        print("reply:", repr(a))
print("rows", n)
for k, v in sorted(c.items()):
    print("  %-60s %d" % (k, v))
