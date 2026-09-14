#!/usr/bin/env python
"""Bring stored conversations' system prompt in line with gen_corpus.FORMAT_HELP.

The corpus and the tier files store each conversation's system message as text,
generated at build time, and the prompt's closing block has changed twice:
  1. it gained a worked example of the wire format, because an un-finetuned model
     emitted "<SCALE ids=[E4,E6] factor=1.3></SCALE>" on 12 of 12 calls;
  2. that example was replaced, because it had been lifted from a model's reply on an
     evaluation problem and was close to that problem's answer.
If training kept an old prompt and evaluation used the new one, every reported number
would be confounded by the difference. Rewriting the stored text is far cheaper than
regenerating the trajectories, and it is exact: a known old closing block is replaced by
the current one, every rewrite is checked to reverse back to the original row, and a
row that carries none of the known blocks is left alone and counted.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT), str(PROJECT / "design_agent"), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

OLD_TAILS = (
    # version 0: no worked example
    "Reply with a one-line reason, then exactly one tool call in <tool></tool> tags.",
    # version 1: example lifted from hard_problem_0000
    "Reply with a one-line reason, then exactly one tool call in <tool></tool> tags.\n"
    "Write the call as NAME(arg=value, ...) inside the tags. Element ids are bare\n"
    "integers: the member shown as E4 in the table is id 4.\n\n"
    "Example reply:\n"
    "E6 is worst at margin 0.62; enlarging it and E4 lifts both above requirement.\n"
    "<tool>SCALE(ids=[4, 6], factor=1.30)</tool>",
)


def rewrite(f, help_text):
    tmp = f.with_suffix(".jsonl.tmp")
    n = fixed = already = missing = 0
    with f.open() as fh, tmp.open("w") as out:
        for line in fh:
            r = json.loads(line)
            n += 1
            m = r["messages"][0]
            c = m["content"]
            if c.endswith(help_text):
                already += 1
            else:
                old = next((t for t in sorted(OLD_TAILS, key=len, reverse=True) if c.endswith(t)), None)
                if old is None:
                    missing += 1
                else:
                    m["content"] = c[: -len(old)] + help_text
                    back = json.loads(json.dumps(r))
                    back["messages"][0]["content"] = back["messages"][0]["content"][: -len(help_text)] + old
                    assert back == json.loads(line), "rewrite does not reverse in %s row %d" % (f, n)
                    fixed += 1
            out.write(json.dumps(r) + "\n")
    tmp.replace(f)
    print("  %-40s %7d rows | %7d rewritten, %d already current, %d unmatched"
          % (str(f.relative_to(f.parents[1])), n, fixed, already, missing), flush=True)
    return missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(PROJECT / "data/corpus_v2"))
    ap.add_argument("--tiers", default=str(PROJECT / "data/tiers"))
    a = ap.parse_args()

    from gen_corpus import FORMAT_HELP

    files = sorted(Path(a.corpus).rglob("sft.jsonl")) + sorted(Path(a.tiers).glob("*.jsonl"))
    unmatched = sum(rewrite(f, FORMAT_HELP) for f in files)
    sys.exit(1 if unmatched else 0)
