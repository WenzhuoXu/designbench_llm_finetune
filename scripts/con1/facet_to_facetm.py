"""con1: facetm rows from a facet row set and the masked (verbatim) row set built on the same states and moves.

    python facet_to_facetm.py rows_facet.jsonl rows_masked.jsonl rows_facetm.jsonl

Each facet answer of an applied move gets the MARGINS AFTER block (targets.margins_after of the verbatim result in the
masked row's answer at the same position) right after 'VERDICT: applied'; the questions get facetm's wording. The rest
of every answer (FEASIBLE, VIOLATED, CHANGES) is byte-identical to the facet row, so a facetm arm differs from the
facet arm only by the margins. Rows must align one to one (same sid and moves, in the same order).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from targets import NL, margins_after, question  # noqa: E402

HEAD = "VERDICT: applied" + NL


def convert(rf: dict, rm: dict) -> dict:
    assert rf["sid"] == rm["sid"] and rf["con1"]["moves"] == rm["con1"]["moves"], rf["sid"]
    moves = rf["con1"]["moves"]
    out, qi, ai = [], 0, 0
    am = [m for m in rm["messages"] if m["role"] == "assistant"]
    for m in rf["messages"]:
        m = dict(m)
        if m["role"] == "user":
            old, new = question("facet", moves[qi], qi == 0), question("facetm", moves[qi], qi == 0)
            assert m["content"].endswith(old), (rf["sid"], qi)
            m["content"] = m["content"][:len(m["content"]) - len(old)] + new
            qi += 1
        elif m["role"] == "assistant":
            c = m["content"]
            if c.startswith(HEAD):
                block, spans = "MARGINS AFTER:" + NL, []
                for k, side, size in margins_after(am[ai]["content"]):
                    fixed, info = "  %-22s " % k, ("%s %s" % (side, size)).strip()
                    s0 = len(HEAD) + len(block) + len(fixed)
                    spans.append([s0, s0 + len(info)])
                    block += fixed + info + NL
                n = len(block)
                m["content"] = HEAD + block + c[len(HEAD):]
                m["supervise_spans"] = ([list(s) for s in m["supervise_spans"] if s[0] < len(HEAD)] + spans +
                                        [[s[0] + n, s[1] + n] for s in m["supervise_spans"] if s[0] >= len(HEAD)])
            ai += 1
        out.append(m)
    r = dict(rf)
    r["messages"], r["format"] = out, "facetm"
    return r


def main():
    fa, ma, dst = sys.argv[1:4]
    n = na = 0
    with open(dst + ".tmp", "w", encoding="utf-8") as fo:
        for lf, lm in zip(open(fa, encoding="utf-8"), open(ma, encoding="utf-8")):
            r = convert(json.loads(lf), json.loads(lm))
            n += 1
            na += sum(1 for m in r["messages"] if m["role"] == "assistant" and "MARGINS AFTER:" in m["content"])
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(dst + ".tmp", dst)
    print("rows %d, answers with margins %d -> %s" % (n, na, dst))


if __name__ == "__main__":
    main()
