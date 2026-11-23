"""row_shape.py <rows.jsonl> [n=1]: the message structure of the first n rows (roles, lengths, head/tail of the last
message) and the non-message fields, to compare a new rows file with the replay file."""
import json
import sys

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= n:
            break
        r = json.loads(line)
        msgs = r.get("messages") or []
        print("--- row %d: %s" % (i, {k: v for k, v in r.items() if k != "messages" and len(str(v)) < 100}))
        for m in msgs:
            c = m.get("content")
            c = c if isinstance(c, str) else json.dumps(c)
            extra = [k for k in m if k not in ("role", "content")]
            print("  %-9s %6d chars %s" % (m.get("role"), len(c), extra))
        last = msgs[-1]
        c = last.get("content") if isinstance(last.get("content"), str) else json.dumps(last.get("content"))
        print("  last head:", repr(c[:300]))
        print("  last tail:", repr(c[-400:]))
        for k in last:
            if k not in ("role", "content"):
                print("  last.%s:" % k, repr(json.dumps(last[k])[:400]))
