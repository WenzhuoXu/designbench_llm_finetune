#!/usr/bin/env python
"""Bring the built corpus's system prompt in line with the evaluator's.

The corpus stores each conversation's system message as text, generated at build
time. The prompt has since gained a worked example of the wire format, because
an un-finetuned model emitted "<SCALE ids=[E4,E6] factor=1.3></SCALE>" on 12 of
12 calls -- wrong wrapper, and element labels where the tool wants integers.

If training kept the old prompt and evaluation used the new one, every reported
number would be confounded by that difference. Rewriting the stored text is far
cheaper than regenerating 2.5 GB of trajectories, and it is exact: the old
closing line is replaced by the new block, and any row that does not carry the
old line verbatim is left alone and counted.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(PROJECT / "design_agent"), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

OLD_TAIL = "Reply with a one-line reason, then exactly one tool call in <tool></tool> tags."

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(PROJECT / "data/corpus_v2"))
    a = ap.parse_args()

    from gen_corpus import FORMAT_HELP

    if not FORMAT_HELP.startswith(OLD_TAIL):
        sys.exit("FORMAT_HELP no longer starts with the old line; refusing to guess")

    for f in sorted(Path(a.corpus).rglob("sft.jsonl")):
        tmp = f.with_suffix(".jsonl.tmp")
        n = fixed = already = missing = 0
        with f.open() as fh, tmp.open("w") as out:
            for line in fh:
                r = json.loads(line)
                n += 1
                m = r["messages"][0]
                c = m["content"]
                if c.endswith(FORMAT_HELP):
                    already += 1
                elif c.endswith(OLD_TAIL):
                    m["content"] = c[: -len(OLD_TAIL)] + FORMAT_HELP
                    fixed += 1
                else:
                    missing += 1
                out.write(json.dumps(r) + "\n")
        tmp.replace(f)
        print("  %-12s %7d rows | %7d rewritten, %d already current, %d unmatched"
              % (f.parent.name, n, fixed, already, missing))
