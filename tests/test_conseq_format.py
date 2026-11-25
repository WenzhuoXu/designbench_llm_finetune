"""con1: the consequence-row token layout (llm_finetune/rl_dataxp/conseq_format.py) and sft_rows' use of it.

    CON1_TOKENIZER=<Qwen3.8-27B snapshot dir> python -m pytest tests/test_conseq_format.py -q
(skipped without a tokenizer). Checks: the supervised tokens decode to exactly the answers; the prompt for question
j is a prefix of the full row's tokens (so a model asked at test time sees what training showed); sft_rows.row_sample
routes a consequence row through the layout and leaves a policy row unchanged.
"""
import os

import pytest

TOK = os.environ.get("CON1_TOKENIZER")
pytestmark = pytest.mark.skipif(not TOK, reason="set CON1_TOKENIZER to a Qwen tokenizer directory")


def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK)


def _row(n=3):
    msgs = [{"role": "system", "content": "DOMAIN: test\nGOAL: test"},
            {"role": "user", "content": "PROBLEM\nstate text\n\nQUESTION (no move is made now): if the move A(1) ..."}]
    for k in range(n):
        if k:
            msgs.append({"role": "user", "content": "QUESTION (no move is made now): if the move A(%d) ..." % (k + 1)})
        msgs.append({"role": "assistant", "content": "CHECK RESULT for A(%d)\n\n  x   1.5 -> %d.25\n" % (k + 1, k),
                     "reasoning_content": ""})
    return {"messages": msgs, "supervise": "every_assistant", "domain": "t", "tier": "con1_conseq"}


def test_mask_decodes_to_answers():
    from llm_finetune.rl_dataxp.conseq_format import answer_text, encode
    tok = _tok()
    row = _row()
    e = encode(tok, row["messages"])
    assert len(e["mask"]) == len(e["completion_ids"])
    sup = [t for t, m in zip(e["completion_ids"], e["mask"]) if m]
    want = "".join(answer_text(m) for m in row["messages"] if m["role"] == "assistant")
    assert tok.decode(sup) == want
    full = tok.decode(e["prompt_ids"] + e["completion_ids"])
    assert full.count("<think>\n\n</think>\n\n") == 3
    assert full.endswith("<|im_end|>")


def test_prompt_is_prefix():
    from llm_finetune.rl_dataxp.conseq_format import encode, prompt_ids
    tok = _tok()
    row = _row()
    e = encode(tok, row["messages"])
    full = e["prompt_ids"] + e["completion_ids"]
    for j in (1, 2):
        # messages up to the j-th question (index 2*j+1 is the j-th later user turn)
        msgs = row["messages"][: 2 + 2 * j]
        p = prompt_ids(tok, msgs)
        assert full[: len(p)] == p
        assert tok.decode(p).endswith("<|im_start|>assistant\n<think>\n")
    p0 = prompt_ids(tok, row["messages"][:2])
    assert p0 == e["prompt_ids"]


def test_sft_rows_routing():
    from llm_finetune.rl_dataxp.sft_rows import completion_text, row_sample
    tok = _tok()
    s = row_sample(_row(), tok)
    assert "mask" in s and sum(s["mask"]) > 0
    pol = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "<action>A(1)</action>", "reasoning_content": "because"}]}
    s2 = row_sample(pol, tok)
    assert "mask" not in s2
    assert tok.decode(s2["completion_ids"]) == completion_text(pol["messages"][-1])
