#!/usr/bin/env python
"""Rewrite stored conversations' system prompt to the one gen_corpus builds today.

The corpus and the tier files store each conversation's system message as text,
generated at build time, and that text has changed three times:
  1. it gained a worked example of the wire format, because an un-finetuned model
     emitted "<SCALE ids=[E4,E6] factor=1.3></SCALE>" on 12 of 12 calls;
  2. the example was replaced, having been lifted from a model's reply on an
     evaluation problem, close to that problem's answer;
  3. the tool list gained the real signatures of each domain's own tools, which had
     been advertised as NAME(...) with no argument names at all.
If training keeps an old prompt and evaluation uses the new one, every reported number
is confounded by the difference. Rewriting the stored text is far cheaper than
regenerating the trajectories, and it is exact: `system_prompt` is a pure function of
the domain, so the correct text is recomputed per domain and written in whole, and
every row is checked to differ from its original in nothing but that one message.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT), str(PROJECT / "design_agent"), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

INTRO = "You are sizing a design to meet every requirement without exceeding its budget."


def rewrite(f, prompt_for):
    tmp = f.with_suffix(".jsonl.tmp")
    n = fixed = already = unexpected = 0
    with f.open() as fh, tmp.open("w") as out:
        for line in fh:
            r = json.loads(line)
            n += 1
            m = r["messages"][0]
            want = prompt_for(r["domain"], r["instance"])
            if m["content"] == want:
                already += 1
            elif m["content"].startswith(INTRO):
                original = json.loads(line)
                m["content"] = want
                check = json.loads(json.dumps(r))
                check["messages"][0]["content"] = original["messages"][0]["content"]
                assert check == original, "row %d of %s changed outside the system prompt" % (n, f)
                fixed += 1
            else:
                unexpected += 1
            out.write(json.dumps(r) + "\n")
    tmp.replace(f)
    print("  %-40s %7d rows | %7d rewritten, %d already current, %d unrecognised"
          % (str(f.relative_to(f.parents[1])), n, fixed, already, unexpected), flush=True)
    return unexpected


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(PROJECT / "data/corpus_v2"))
    ap.add_argument("--tiers", default=str(PROJECT / "data/tiers"))
    a = ap.parse_args()

    from gen_corpus import make, system_prompt

    cache = {}

    def prompt_for(domain, instance):
        if domain not in cache:
            # The procedural domains are keyed by an integer seed, stored as text.
            # Any instance of the domain gives the same prompt; the tools are the domain's.
            if domain != "truss" and str(instance).lstrip("-").isdigit():
                instance = int(instance)
            dom, st, _ = make(domain, instance)
            cache[domain] = system_prompt(dom, st)
        return cache[domain]

    files = sorted(Path(a.corpus).rglob("sft.jsonl")) + sorted(Path(a.tiers).glob("*.jsonl"))
    bad = sum(rewrite(f, prompt_for) for f in files)
    sys.exit(1 if bad else 0)
