"""The replay rows of a round under the open rules: dx38's own training rows with the prompt's rule statements
changed to the open rules (open_rules.open_prompt: photonics' fabrication sentence replaced, grid2op's rule line
added), so that no training row states a rule the round does not use. Every other byte of every row is kept.

    python open_replay.py --in train_dx38mtR.jsonl --out replay_open.jsonl
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from open_rules import open_prompt  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cnt = collections.Counter()
    tmp = a.out + ".tmp"
    with open(a.src, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as fo:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            m = r["messages"]
            if [x["role"] for x in m[:2]] != ["system", "user"]:
                raise SystemExit("row %s: messages start %s" % (r.get("sid"), [x["role"] for x in m[:2]]))
            s, u = open_prompt(r["domain"], m[0]["content"], m[1]["content"])
            changed = (s, u) != (m[0]["content"], m[1]["content"])
            m[0]["content"], m[1]["content"] = s, u
            if changed:
                r["prompt_rules"] = "open"
            cnt["rows, %s" % r["domain"]] += 1
            cnt["rows changed, %s" % r["domain"]] += changed
            fo.write(json.dumps(r) + "\n")
    os.replace(tmp, a.out)
    for k in sorted(cnt):
        print("%-40s %d" % (k, cnt[k]))
    print("OPEN_REPLAY_DONE %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
