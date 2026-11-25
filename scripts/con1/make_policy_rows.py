"""con1 stage 2: the policy rows (tier ownrows_own: the open-rules restated dx38 replay rows, rationale then action,
8 training families) from the chk1 training file; 60 rows held out as dev (seeded).
    python make_policy_rows.py IN.jsonl[.gz] OUT_DIR"""
import collections
import gzip
import json
import random
import sys

src, out = sys.argv[1], sys.argv[2]
f = gzip.open(src, "rt", encoding="utf-8") if src.endswith(".gz") else open(src, encoding="utf-8")
rows = [json.loads(l) for l in f]
rows = [r for r in rows if r.get("tier") == "ownrows_own"]
assert not any(r.get("domain") in {"inventory2", "molopt", "thermal2"} for r in rows)
rng = random.Random(0)
rng.shuffle(rows)
dev, train = rows[:60], rows[60:]
for name, rs in (("policy_train.jsonl", train), ("policy_dev.jsonl", dev)):
    with open(out + "/" + name, "w", encoding="utf-8") as fo:
        for r in rs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print("train", len(train), dict(collections.Counter(r["domain"] for r in train)), "dev", len(dev))
