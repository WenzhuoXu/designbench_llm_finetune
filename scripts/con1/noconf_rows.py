"""An existing consequence row file with the move request cut from each row's prompt (CPU; con1-2m-rows, 11-27; owner
order, conseq_lib.consequence_user, con1 eff023a). For stage 1's dev rows, which the full-set continuation evaluates on.

    python noconf_rows.py IN.jsonl OUT.jsonl

A row's first user message is the state's served user text + the first QUESTION (build_rows.conseq_row); it is split
where that question starts (the last '\\n\\nQUESTION (no move is made now)'), consequence_user is applied to the state
text, the question is appended unchanged. Every other field and message is copied as is. Stops on a row whose prompt
is not in the harness's wording (consequence_user raises).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conseq_lib import consequence_user  # noqa: E402

QMARK = "\n\nQUESTION (no move is made now)"
src, dst = sys.argv[1:3]
n = 0
with open(dst + ".part", "w", encoding="utf-8") as fo:
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        k = next(i for i, m in enumerate(r["messages"]) if m["role"] == "user")
        u = r["messages"][k]["content"]
        cut = u.rfind(QMARK)
        if cut < 0:
            raise SystemExit("row %d (%s): no first QUESTION in the prompt" % (n, r.get("sid")))
        r["messages"][k]["content"] = consequence_user(u[:cut]) + u[cut:]
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1
os.replace(dst + ".part", dst)
print("NOCONF_ROWS_DONE %d rows -> %s" % (n, dst))
