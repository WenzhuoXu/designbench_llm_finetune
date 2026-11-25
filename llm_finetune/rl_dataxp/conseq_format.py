"""con1 consequence rows: one token layout for training (sft_rows.py) and for asking a served model (skill test).

A consequence row is a conversation [system, user, assistant, user, assistant, ...]: the state as the harness serves
it plus a question about one move, the simulator's result message as the answer, then further questions about other
moves of the same state, each answered. Every assistant turn is supervised.

Layout (Qwen chat template for the first two messages, then the trainer's own single-turn convention for each turn):
    apply_chat_template([system, user], add_generation_prompt=True)        ... ends '<|im_start|>assistant\\n<think>\\n'
    answer      reasoning.strip() + '\\n</think>\\n\\n' + content.strip() + '<|im_end|>'        (supervised)
    next user   '\\n<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n<think>\\n'   (not supervised)
    answer      ...
So every answer is generated after '<think>\\n' exactly as a single-turn reply of the policy rows is (sft_rows.
completion_text), and a model asked the j-th question gets prompt = the layout up to and including the j-th 'next
user' piece. The pieces are tokenised separately, as sft_rows tokenises prompt and completion separately.
"""
from __future__ import annotations

END = "<|im_end|>"
GLUE = "\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n"


def answer_text(message: dict) -> str:
    return (message.get("reasoning_content") or "").strip() + "\n</think>\n\n" + (message.get("content") or "").strip() + END


def _head_ids(tok, messages):
    ids = tok.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True)
    ids = list(ids if isinstance(ids, list) else ids["input_ids"])
    if not tok.decode(ids[-4:]).endswith("<think>\n"):
        raise ValueError("the served prompt does not end with '<think>\\n': %r" % tok.decode(ids[-8:]))
    return ids


def check_roles(messages) -> None:
    roles = [m.get("role") for m in messages]
    if len(roles) < 2 or roles[0] != "system" or roles[1] != "user":
        raise ValueError("a consequence row starts with system, user: %s" % roles[:3])
    for k, r in enumerate(roles[2:]):
        if r != ("assistant" if k % 2 == 0 else "user"):
            raise ValueError("a consequence row alternates assistant / user after the first user turn: %s" % roles)


def encode(tok, messages: list) -> dict:
    """prompt_ids (system + first user + generation prompt), completion_ids (every later piece) and mask (1 on the
    tokens of assistant answers). A conversation that ends with a user turn ends with its '<think>\\n' piece (a
    prompt for the next answer)."""
    check_roles(messages)
    ids = _head_ids(tok, messages)
    comp, mask = [], []
    for m in messages[2:]:
        if m["role"] == "assistant":
            t = tok.encode(answer_text(m), add_special_tokens=False)
            mask += [1] * len(t)
        else:
            t = tok.encode(GLUE % m["content"], add_special_tokens=False)
            mask += [0] * len(t)
        comp += list(t)
    return {"prompt_ids": ids, "completion_ids": comp, "mask": mask}


def prompt_ids(tok, messages: list) -> list:
    """Token ids that ask for the next answer: messages must end with a user turn."""
    if messages[-1]["role"] != "user":
        raise ValueError("a prompt ends with a user turn")
    e = encode(tok, messages)
    return e["prompt_ids"] + e["completion_ids"]
