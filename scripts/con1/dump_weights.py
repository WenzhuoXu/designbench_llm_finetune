"""Per-token loss weights of one consequence row, as the trainer computes them (conseq_format.encode, the model's
tokenizer): the first answer token by token, then a summary over the whole row.

    python dump_weights.py ROWS.jsonl TOKENIZER_DIR [--line 0]
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("tokenizer")
    ap.add_argument("--line", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from llm_finetune.rl_dataxp.conseq_format import encode
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    with open(a.rows, encoding="utf-8") as f:
        for k, line in enumerate(f):
            if k == a.line:
                row = json.loads(line)
                break
    msgs = row["messages"]
    e = encode(tok, msgs)
    ids, mask = e["completion_ids"], e["mask"]
    first = msgs[2]
    print("row %d: %s %s, %d questions; answer keys %s; prefix_weight %s, fixed_weight %s" % (
        a.line, row["domain"], row["sid"], len(row["con1"]["moves"]), sorted(first), first.get("prefix_weight"),
        first.get("fixed_weight")))
    if len(msgs) > 3:
        print("second question: %r" % msgs[3]["content"])
    # the first answer = the completion of the row cut after it
    n_first = len(encode(tok, msgs[:3])["completion_ids"])
    print("first answer, token by token (weight  token):")
    for t, w in zip(ids[:n_first], mask[:n_first]):
        print("  %-5s %r" % (("%g" % w), tok.decode([t])))
    c = collections.Counter("%g" % w for w in mask)
    think = [w for t, w in zip(ids, mask) if tok.decode([t]) == "</think>"]
    print("whole row: %d completion tokens; weights %s; '</think>' tokens %d, their weights %s" % (
        len(mask), dict(sorted(c.items())), len(think), sorted(set(think))))


if __name__ == "__main__":
    main()
