"""Stream instruction, side question and the writer loop (docs/cot_stream_validation.md §2-§3).

`run_stream(ss, chat, cap)` drives one stream: the writer's messages, each probe result injected as
a user turn, and the held-out side question asked after the first message ("pre") and after the
stream ends ("post"). `chat(messages) -> dict(text, finish, usage, items)` is the model adapter;
`items` (OpenAI reasoning items) is carried on the assistant message under "_items".
"""
from __future__ import annotations

import json
import re
from typing import Callable, List, Optional

CAP = 6
TAG_RE = re.compile(r"<(check|action)>(.*?)</\1>", re.S)
PRED_RE = re.compile(r"<predictions>(.*?)</predictions>", re.S)
REVIEW_RE = re.compile(r"<review>(.*?)</review>", re.S)

STREAM_INSTRUCTION = """Reason in writing before you act, in this order.
1. Status: which constraints are violated and by how much, which are within 2% of their limit, and what the last change did (from the observed effect and history) compared with what it was meant to do.
2. Mechanism: the relation that governs each violated or near-limit quantity, taken from the problem's own rules, and which elements are coupled to it (shared load path, panel, node, line, stage, signal path). Use the problem's numbers.
3. Candidates: the changes worth comparing, including at least the direct fix on the element with the worst value and one change on another element that acts through the coupling. For each candidate predict the new value of every violated or near-limit metric, of the objective, and of the element-level values that drive them, each as an 80% interval [low, expected, high]: you expect the true value inside it four times out of five. Derive the numbers from the problem's rules and the observation and show the arithmetic. Make each interval as wide as your real uncertainty, for example wide where load or flow redistributes in ways the observation does not give.
4. Checks: you may check a candidate with the simulator by ending your message with <check>action</check>. The reply shows what the design would look like after that change alone; nothing is applied. Each check is one simulator call and you have at most {cap} for this step, so check where your prediction is least certain or where the decision depends on it, not every candidate. After a check result, start your next message with a <review> block, explain in physical terms what you predicted wrong and why (which load path, coupling or effect you misjudged), and update the predictions of the remaining candidates (add new candidates if the result suggests them).
5. Decision: choose the change with the best predicted or checked outcome: every constraint satisfied first, then the lowest objective; on a near-limit constraint a small margin is worth a small objective cost. Size the change by calculation. A checked change is not applied until you choose it in <action>.
Use only the information in the observation and in the check results.
Blocks:
<predictions>{{"candidates": [{{"action": "<action>", "metrics": {{"<metric name exactly as in CONSTRAINT STATUS>": [low, expected, high], ...}}, "elements": {{"<element id, e.g. m15>": {{"<quantity, e.g. N_kN, fos_b, fos_y>": [low, expected, high]}}}}}}]}}</predictions>
<review>{{"action": "<checked action>", "observed": {{"<metric, or element.quantity such as m15.N_kN>": <value from the check result>, ...}}, "predicted": {{"<same keys>": <your expected value>}}}}</review>
Every message that makes or updates predictions includes one <predictions> block. End every message with exactly one of <check>action</check> or <action>action</action>, written in the problem's syntax."""

USER_TAIL = "\n\nChoose exactly ONE change for THIS step. You may check up to {cap} candidate changes with the simulator first."

# Version 2 (2026-10-30, docs/cot_stream_validation.md §6): version 1 did not say that the chosen change is not
# applied, and asked after the final <action> the GPT writers answered for the design after that action.
SIDE_VERSION = 2
SIDE_HEAD = ("SIDE QUESTION (separate from your design work: it uses no check and changes nothing). Nothing has "
             "been applied to the design: not the changes you checked and not any change you chose. Predict the "
             "outcome of each of these changes, each applied alone to the current design (the design in the "
             "observation at the start of this step), as an 80% interval [low, expected, high] for every quantity "
             "listed. Use everything you know so far. Reply with only a <predictions> block in the same JSON form, "
             "one candidate per change, in this order:")

NUDGE_FORMAT = "End your message with exactly one of <check>action</check> or <action>action</action>."
NUDGE_CAP = "No checks are left for this step ({cap} used). Choose your change now and end with <action>action</action>."


NO_CHECKS = ("4. Checks: none are available for this step; decide from your predictions. (When checks are "
             "available you may end a message with <check>action</check> to see a change's result before "
             "choosing.)")
USER_TAIL_NO_CHECKS = "\n\nChoose exactly ONE change for THIS step. No simulator checks are available for this step."


def system_text(rec: dict, cap: int = CAP) -> str:
    text = STREAM_INSTRUCTION.format(cap=cap)
    if cap == 0:
        lines = text.split("\n")
        lines = [NO_CHECKS if l.startswith("4. Checks:") else l for l in lines]
        text = "\n".join(lines)
    return rec["system"] + "\n\n" + text


def user_text(rec: dict, cap: int = CAP) -> str:
    return rec["observation"] + (USER_TAIL_NO_CHECKS if cap == 0 else USER_TAIL.format(cap=cap))


def side_question(ss, H: List[str]) -> str:
    lines = [SIDE_HEAD]
    for n, h in enumerate(H, 1):
        q = ss.queries(h)
        el = "; ".join("%s: %s" % (e, ", ".join(qs)) for e, qs in (q.get("elements") or {}).items())
        lines.append("%d. %s | metrics: %s%s" % (n, h, ", ".join(q["metrics"]), (" | elements: " + el) if el else ""))
    return "\n".join(lines)


def last_tag(text: str):
    """('check' | 'action', inner) of the LAST check/action tag, or (None, None)."""
    m = TAG_RE.findall(text or "")
    if not m:
        return None, None
    tag, inner = m[-1]
    return tag, inner.strip()


def _strip(msgs: List[dict]) -> List[dict]:
    return [dict(m) for m in msgs]


def run_stream(ss, chat: Callable[[List[dict]], dict], cap: int = CAP, side: bool = True,
               max_nudges: int = 2) -> dict:
    rec = ss.r
    msgs = [{"role": "system", "content": system_text(rec, cap)},
            {"role": "user", "content": user_text(rec, cap)}]
    H = ss.hset() if side else []
    sq = side_question(ss, H) if side else None
    probes, calls, side_ans = [], [], {}
    status, nudges = "running", 0
    while True:
        rep = chat(msgs)
        calls.append({k: rep.get(k) for k in ("finish", "usage", "latency_s", "error")})
        if rep.get("error"):
            # a stream that outgrew the context is a final outcome, not a retryable error
            status = "context_length" if rep["error"] == "context_length" else "error"
            break
        msgs.append({"role": "assistant", "content": rep["text"], "_items": rep.get("items")})
        if rep.get("finish") == "length":
            status = "truncated"
            break
        n_asst = sum(1 for m in msgs if m["role"] == "assistant")
        if side and n_asst == 1:
            side_ans["pre"] = _side(chat, msgs, sq)
        tag, inner = last_tag(rep["text"])
        if tag == "action":
            status = "done"
            break
        if tag == "check" and len(probes) < cap:
            res = ss.check(inner)
            probes.append({"text": inner, "ok": res["ok"], "err": res.get("err"),
                           "applied": res.get("applied"), "metrics": res.get("metrics"),
                           "nom": res.get("nom"), "after_msg": n_asst})
            msgs.append({"role": "user", "content": ss.result_message(res, len(probes), cap), "_probe": len(probes) - 1})
            continue
        if nudges >= max_nudges:
            status = "no_action"
            break
        nudges += 1
        msgs.append({"role": "user", "content": NUDGE_CAP.format(cap=cap) if tag == "check" else NUDGE_FORMAT,
                     "_nudge": True})
    if side and status in ("done", "no_action") and "pre" in side_ans:
        side_ans["post"] = _side(chat, msgs, sq)
    final = last_tag(msgs[-1]["content"])[1] if msgs[-1]["role"] == "assistant" and status == "done" else None
    return {"sid": rec["sid"], "variant": rec["variant"], "group": (rec.get("pilot") or {}).get("group"),
            "status": status, "cap": cap, "n_probes": len(probes), "probes": probes, "final_action": final,
            "H": H, "side_question": sq, "side_version": SIDE_VERSION, "side": side_ans, "calls": calls,
            "messages": [{k: v for k, v in m.items() if k != "_items"} for m in msgs]}


def _side(chat, msgs, sq) -> dict:
    rep = chat(msgs + [{"role": "user", "content": sq}])
    return {"text": rep.get("text"), "finish": rep.get("finish"), "usage": rep.get("usage"),
            "error": rep.get("error")}


def parse_json_block(text: str, which: str = "predictions"):
    rx = PRED_RE if which == "predictions" else REVIEW_RE
    m = rx.findall(text or "")
    if not m:
        return None, None
    return loads_lenient(m[-1])


CAND_START = re.compile(r'\{\s*"action"\s*:')
UNTAGGED = re.compile(r'\{\s*"candidates"')


def _close_open(s: str) -> str:
    """s with the brackets it leaves open closed (brackets inside strings ignored)."""
    stack, ins, esc = [], False, False
    for ch in s:
        if ins:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                ins = False
        elif ch == '"':
            ins = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    return s + "".join(reversed(stack))


def parse_side(text: str):
    """(predictions dict, how) for a side answer. how: strict | lenient (the tagged block), bare (no tags,
    JSON in the reply), recovered (one candidate at a time, after a nesting error; a candidate that still
    does not parse is None), or None."""
    blocks = PRED_RE.findall(text or "")
    if blocks:
        body = blocks[-1]
    else:
        m = UNTAGGED.search(text or "")
        body = text[m.start():] if m else ""
    d, how = loads_lenient(body) if body.strip() else (None, None)
    if isinstance(d, dict):
        return d, (how if blocks else "bare")
    starts = [m.start() for m in CAND_START.finditer(body)]
    if not starts:
        return None, None
    cands = []
    for i, s in enumerate(starts):
        seg = body[s:starts[i + 1] if i + 1 < len(starts) else len(body)]
        seg = re.sub(r"[\s,\]\}]*$", "", seg)
        c, _ = loads_lenient(_close_open(seg))
        cands.append(c if isinstance(c, dict) else None)
    return {"candidates": cands}, "recovered"


def loads_lenient(s: str):
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        return json.loads(s), "strict"
    except ValueError:
        pass
    t = re.sub(r",\s*([}\]])", r"\1", s)
    t = re.sub(r"(?<![\w.])\+(\d)", r"\1", t)
    t = re.sub(r"\bInfinity\b|\binf\b", "1e309", t)
    t = re.sub(r"//[^\n]*", "", t)
    # a bare word for a value (N/A, None, limit) is that value missing, not the whole candidate
    t = re.sub(r"(?<=[\[,\s])(N/A|None|limit)(?=\s*[,\]])", "null", t)
    try:
        return json.loads(t), "lenient"
    except ValueError:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1]), "lenient"
        except ValueError:
            pass
    return None, None
