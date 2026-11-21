"""reread.py <samples.jsonl> <scored.jsonl> <dir of forecasts.py>: the scored replies whose forecast lines, read again from
the reply's rationale, are not the (move, verdict) pairs stored when it was scored. Shows the first few."""
import json
import sys

sys.path.insert(0, sys.argv[3])
import forecasts  # noqa: E402

think = {}
for raw in open(sys.argv[1], "rb"):
    try:
        o = json.loads(raw)
    except ValueError:
        continue
    if not o.get("error"):
        think[(o["sid"], int(o["sample"]))] = o.get("thinking") or ""
n = bad = 0
shown = 0
twice = 0
seen = set()
for raw in open(sys.argv[2], "rb"):
    try:
        x = json.loads(raw)
    except ValueError:
        continue
    if x.get("state") or "sample" not in x:
        continue
    key = (x["sid"], int(x["sample"]))
    if key in seen:
        twice += 1
    seen.add(key)
    n += 1
    lines, unread = forecasts.read(think.get(key, ""))
    stored = [(c.get("move"), c.get("said")) for c in (x.get("forecasts") or [])]
    if len(lines) <= forecasts.MAX_SCORED and stored != lines:
        bad += 1
        if shown < 6:
            shown += 1
            print("---", key, "stored lines", x.get("n_forecast_lines"), "unread", x.get("forecast_unread"), "error", x.get("forecast_error"))
            print("   stored:", stored)
            print("   now   :", lines, "unread", unread)
print("scored replies %d (recorded twice %d), reading differently %d" % (n, twice, bad))
