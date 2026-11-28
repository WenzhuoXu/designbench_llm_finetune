"""Before the effort arm (11-28, parent's order C): render arm D's prompts with the chat template's reasoning_effort
unset, 'xhigh', 'medium' and 'low', and check what changes (CPU, tokenizer only).

    python effort_render_check.py --queries queries_v3.jsonl --tokenizer MODEL --subset subset_300.tsv

Checks, per prompt (arm D's message layout: system, user + the dir question with the metrics named):
  unset == xhigh            token ids identical (the template's default is xhigh)
  medium / low vs unset     the decoded text differs only in the reasoning-instruction paragraph at the top of the
                            system turn: unset carries the xhigh paragraph, medium none, low the low paragraph; the
                            rest of the text is byte-identical
Prints the paragraphs found and the token-count change. Exit 1 on any failure.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("CON1_DIR") or HERE)
import skill_client as SC  # noqa: E402
from targets import question  # noqa: E402

XHIGH = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
         "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\n")
LOW = ("Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
       "unnecessary elaboration.\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--subset", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    sub = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open(a.subset) if l.strip()}
    c = collections.Counter()
    dtok = collections.defaultdict(list)
    first_fail = None
    for l in open(a.queries, encoding="utf-8"):
        q = json.loads(l)
        for j, m in enumerate(q["moves"]):
            if (q["sid"], j) not in sub:
                continue
            msgs = [{"role": "system", "content": q["system"]},
                    {"role": "user", "content": q["user"] + question("dir", m["move"], True, sorted(q["metrics0"]))}]
            base = SC.prompt_ids(tok, msgs)
            r = {}
            for e in ("xhigh", "medium", "low"):
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, reasoning_effort=e)
                r[e] = list(ids if isinstance(ids, list) else ids["input_ids"])
            c["prompts"] += 1
            ok = r["xhigh"] == base
            c["unset == xhigh"] += ok
            t0 = tok.decode(base)
            for e, para in (("medium", ""), ("low", LOW)):
                te = tok.decode(r[e])
                good = XHIGH in t0 and t0.count(XHIGH) == 1 and te == t0.replace(XHIGH, para, 1)
                c["%s differs only in the paragraph" % e] += good
                dtok[e].append(len(r[e]) - len(base))
                ok = ok and good
            if not ok and first_fail is None:
                first_fail = (q["sid"], j)
            c["first 200 chars identical after the system tag (xhigh)"] += t0.startswith("<|im_start|>system\n" + XHIGH)
    print(dict(c))
    for e in dtok:
        print("token change %s vs unset: %s" % (e, sorted(set(dtok[e]))))
    print("first failure:", first_fail)
    bad = c["prompts"] - min(c["unset == xhigh"], c["medium differs only in the paragraph"],
                             c["low differs only in the paragraph"])
    print("EFFORT_RENDER_CHECK %s (%d of %d prompts failed)" % ("OK" if not bad else "FAILED", bad, c["prompts"]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
