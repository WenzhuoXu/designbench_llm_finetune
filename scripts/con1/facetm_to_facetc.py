"""con1: facetc rows from the facetm rows (round 1's Fm arm) and the chain truth (chain_truth.py).

    python facetm_to_facetc.py rows_facetm.jsonl chain_r1.jsonl rows_facetc.jsonl

In CHAIN_FAMILIES each applied answer gets the LOCAL and REDISTRIBUTION blocks (targets.chain_block / chain_text from
the per-element state before and after the move) right after 'VERDICT: applied', before MARGINS AFTER; the questions
get facetc's wording. Everything else of every answer is byte-identical to the facetm row, and the other families'
rows are copied unchanged (facetm), so the arm differs from Fm only by the chain blocks of the chain families.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from targets import CHAIN_FAMILIES, NL, chain_block, chain_text, question  # noqa: E402

HEAD = "VERDICT: applied" + NL


def convert(r: dict, ch: dict, c: collections.Counter) -> dict:
    moves = r["con1"]["moves"]
    out, qi, ai = [], 0, 0
    for m in r["messages"]:
        m = dict(m)
        if m["role"] == "user":
            old, new = question("facetm", moves[qi], qi == 0), question("facetc", moves[qi], qi == 0)
            assert m["content"].endswith(old), (r["sid"], qi)
            m["content"] = m["content"][:len(m["content"]) - len(old)] + new
            qi += 1
        elif m["role"] == "assistant":
            body = m["content"]
            if body.startswith(HEAD):
                after = ch["after"].get(moves[ai])
                if after is None:
                    c["applied answer, refused or missing in the re-simulation"] += 1
                else:
                    cb = chain_block(r["domain"], moves[ai], ch["before"], after)
                    lines, spans = chain_text(cb)
                    block = NL.join(lines) + NL
                    n, h = len(block), len(HEAD)
                    m["content"] = HEAD + block + body[h:]
                    m["supervise_spans"] = ([list(s) for s in m["supervise_spans"] if s[0] < h] +
                                            [[h + a, h + b] for a, b in spans] +
                                            [[s[0] + n, s[1] + n] for s in m["supervise_spans"] if s[0] >= h])
                    c["answers with the chain"] += 1
                    c["chain chars"] += n
                    c["LOCAL none"] += not cb["local"]
                    c["no redistribution"] += not cb["redis"]
            ai += 1
        out.append(m)
    r = dict(r)
    r["messages"], r["format"] = out, "facetc"
    return r


def main():
    src, chain_path, dst = sys.argv[1:4]
    chains = {}
    for line in open(chain_path, encoding="utf-8"):
        x = json.loads(line)
        if x.get("split") == "train":
            chains[x["sid"]] = x
    c = collections.Counter()
    with open(dst + ".tmp", "w", encoding="utf-8") as fo:
        for line in open(src, encoding="utf-8"):
            r = json.loads(line)
            if r["domain"] in CHAIN_FAMILIES:
                if r["sid"] not in chains:
                    raise SystemExit("no chain truth for %s" % r["sid"])
                r = convert(r, chains[r["sid"]], c)
                c["rows converted %s" % r["domain"]] += 1
            else:
                c["rows unchanged %s" % r["domain"]] += 1
            c["answer chars"] += sum(len(m["content"]) for m in r["messages"] if m["role"] == "assistant")
            c["answers"] += sum(1 for m in r["messages"] if m["role"] == "assistant")
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(dst + ".tmp", dst)
    for k in sorted(c):
        print("%-60s %d" % (k, c[k]))
    print("chars per answer %.0f; chain chars per chain answer %.0f" % (
        c["answer chars"] / max(1, c["answers"]), c["chain chars"] / max(1, c["answers with the chain"])))


if __name__ == "__main__":
    main()
