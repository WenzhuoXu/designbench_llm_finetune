"""Build dataxp_policy_eval_v3rc.py from dataxp_policy_eval_v3r.py (sha1 4978bab2...): one opt-in flag, --checks C.
Without the flag the file behaves as v3r. Every replacement must match exactly once.

--checks C (2026-11-24; native variants only: photonics, grid2op, engibench). The model may query the simulator
before it commits a move (the owner's rule: the model hypothesises and may ask the simulator to check, each check
counted as a simulator call). Each turn's prompt gets a CHECKS paragraph after the user text. A reply whose last
tag is <check>MOVE</check> (instead of <action>MOVE</action>) is a check while fewer than C checks have been made in
this turn: the move is applied to a copy of the current design under the same rules (--open-rules as given) and
scored with the seed the harness scores the next design with (so a check of a move predicts its commit exactly);
the model then gets, as the next user message of the same conversation, the copy's CONSTRAINT STATUS and the move's
EFFECT (before -> after), in the observation's own format, or the simulator's refusal; and replies again. A reply
whose last tag is <action> commits as in a plain turn. After C checks the model is told to commit; a reply that
still ends with <check> then commits that move (the turn record gets check_as_action = true; the answer recorded
carries the <action> line appended to it). The turn record gets checks_max = C and checks = [{i, text, parse |
refused | move/feasible/phi/viol, result (the text shown), finish, completion_tokens, reasoning, answer}]; the
turn's simulator calls are 1 + the checks whose move was read (an unreadable check calls no simulator). The
conversation of a turn rebuilds from the record: user + check_rule(C), then per check its answer and its result
followed by the budget line. Earlier assistant messages are sent as their answer text (the chat template drops
their thinking). Review 11-24 (one agent): probe prompts have no YOUR MOVE section (handled); the native-only check
runs per episode (a non-native variant crashes its episodes: give native variants only).

--check-actions (with --checks C; data collection for a check-trained adapter, 2026-11-24). dx38 never writes a
<check> tag on its own (0 checks in 563 turns of the zero-shot pilot), so while fewer than C checks have been made,
a reply ending with an <action> line whose move has not been checked in this turn (or cannot be read) is checked
instead: its last <action>...</action> is renamed <check>...</check> in the conversation and in the record
(check entry converted = true), so the model's later replies see the history a check-trained model would write.
A reply naming a move already checked in this turn commits it; after C checks the next reply commits.

usage: python make_v3rc.py <v3r> <out>
"""
import hashlib
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
assert hashlib.sha1(open(src, "rb").read()).hexdigest() == "4978bab2dd8c68a8bbfedcc6cdac176df37cfab8", "not the expected v3r"


def rep(old, new):
    global s
    assert s.count(old) == 1, "%d matches for %r" % (s.count(old), old[:70])
    s = s.replace(old, new)


# 1. helpers, before the observation section
rep('''# ------------------------------------------------------------------ observation
''', '''# ------------------------------------------------------------------ checks (--checks)
# v3rc (2026-11-24). Without the flag this file behaves as dataxp_policy_eval_v3r.py (sha1 4978bab2).

TAG_LAST = re.compile(r"<(action|check)>\\s*(.*?)\\s*</\\1>", re.S)


def check_rule(c):
    """The CHECKS paragraph appended to the user text of a turn when --checks C > 0."""
    return ("\\n\\nCHECKS: before you commit a move you may check up to %d candidate moves in the simulator. To check "
            "a move, end your reply with <check>VERB(arg, ...)</check> instead of the <action> line: the simulator "
            "applies it to a copy of the current design and shows you the result; the design itself stays as it is, "
            "and you then reply again. Each check costs one simulator call. Commit a move with the <action> line as "
            "usual whenever you are ready; after %d checks you must commit.\\n" % (c, c))


def last_tag(answer):
    """(kind, text) of the last <action>/<check> tag of the answer, or (None, None)."""
    m = TAG_LAST.findall(answer or "")
    return (m[-1][0], " ".join(m[-1][1].split())) if m else (None, None)


def as_check(answer):
    """The answer with its last <action>...</action> tag renamed to <check>...</check> (--check-actions)."""
    i = answer.rfind("<action>")
    j = answer.find("</action>", i)
    if not 0 <= i < j:
        return answer
    return answer[:i] + "<check>" + answer[i + len("<action>"):j] + "</check>" + answer[j + len("</action>"):]


def llm_chat(llm, messages):
    """LLM.__call__ on a whole conversation (same request settings, same reply format)."""
    a = llm.a
    body = {"model": a.model, "messages": messages, "temperature": a.temperature, "max_tokens": a.max_tokens,
            "chat_template_kwargs": {"enable_thinking": not a.no_think}}
    if a.temperature > 0:
        body.update(top_p=0.95, top_k=20)
    req = urllib.request.Request(a.base_url.rstrip("/") + "/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                j = json.load(r)
            break
        except Exception as e:                       # noqa: BLE001
            if attempt == 3:
                return {"err": str(e)[:300]}
            time.sleep(5 * (attempt + 1))
    ch = j["choices"][0]
    msg = ch["message"]
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    closed = True
    if "</think>" in text:
        reasoning, text = text.rsplit("</think>", 1)
    elif not a.no_think and not reasoning:
        closed = False
    u = j.get("usage") or {}
    with LOCK:
        llm.tokens["prompt"] += u.get("prompt_tokens", 0) or 0
        llm.tokens["completion"] += u.get("completion_tokens", 0) or 0
    return {"answer": text, "reasoning": reasoning.replace("<think>", "").strip(),
            "think_closed": closed, "finish": ch.get("finish_reason"),
            "completion_tokens": u.get("completion_tokens")}


def check_text(C, R, L, domain, spec, child, sc, mv, history, metrics, variant, native_prompt, ev):
    """The result of a checked move as the observation would show it after that move: CONSTRAINT STATUS of the copy
    and the move's EFFECT (before -> after), cut from observation_native of the copy."""
    cm = {k: float(v) for k, v in sc.agg_metrics.items()}
    ad = mv.as_dict()
    h2 = list(history) + [(R._label_for(domain, child, ad), ad)]
    _, u2 = observation_native(C, domain, spec, child, cm, L._binding_constraint(ev.program, cm), h2, metrics,
                               variant, native_prompt)
    i, j = u2.find("CONSTRAINT STATUS"), u2.find("CHANGES ALREADY MADE")
    k, e = u2.find("OBSERVED EFFECT OF THE LAST CHANGE"), u2.find("YOUR MOVE")
    if e < 0:                                  # --native-prompt probe: no YOUR MOVE section, the effect runs to the end
        e = len(u2)
    if not (0 <= i < j <= k < e):
        raise RuntimeError("check_text: observation sections not found in order (%d %d %d %d)" % (i, j, k, e))
    status = u2[i:j].strip().replace("CONSTRAINT STATUS (current design)", "CONSTRAINT STATUS (after this move)")
    effect = u2[k:e].strip().replace("OBSERVED EFFECT OF THE LAST CHANGE", "EFFECT OF THIS MOVE")
    lines = ["CHECK RESULT for %s (simulated on a copy; the current design is unchanged)" % domain.format_move(mv),
             status, effect]
    bad = float(cm.get("min_feature_violations") or 0.0) if variant == "photonics" else 0.0
    if bad > 0:
        lines.append("FABRICATION: after this move the design breaks the minimum feature rule at %d pixels; it does "
                     "not count as feasible until that is 0." % int(round(bad)))
    lines.append("Feasible after this move: %s." % ("yes" if sc.feasible else "no"))
    return "\\n\\n".join(lines)


def check_turn(a, llm, C, D, L, R, domain, spec, st, ev, pid, t, history, metrics, variant, opening, native_prompt,
               system, user, nchk):
    """(the committing reply, the turn's check records): see --checks. Called with SIM held."""
    keep = getattr(a, "record_prompts", False)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user + check_rule(nchk)}]
    log, seen = [], set()
    while True:
        SIM.release()
        try:
            reply = llm_chat(llm, msgs)
        finally:
            SIM.acquire()
        if "err" in reply or not reply.get("think_closed", True):
            return reply, log
        kind, text = last_tag(reply.get("answer"))
        conv = False
        if kind == "action" and getattr(a, "check_actions", False) and len(log) < nchk:
            mv0, _ = C.parse_reply(domain, "<action>%s</action>" % text)
            if mv0 is None or domain.format_move(mv0) not in seen:      # a move not checked yet: check it
                kind, conv = "check", True
                reply = dict(reply, answer=as_check(reply.get("answer", "")))
        if kind != "check":
            return reply, log
        if len(log) >= nchk:                       # told to commit: the checked move is the commitment
            return dict(reply, answer=reply["answer"] + "\\n<action>%s</action>" % text, check_as_action=True), log
        ent = {"i": len(log), "text": text, "converted": conv, "finish": reply.get("finish"),
               "completion_tokens": reply.get("completion_tokens"),
               "reasoning": reply.get("reasoning", "") if keep else reply.get("reasoning", "")[:4000],
               "answer": reply.get("answer", "") if keep else reply.get("answer", "")[:600]}
        mv, why = C.parse_reply(domain, "<action>%s</action>" % text)
        if mv is None:
            ent["parse"] = why
            result = ("CHECK RESULT: the move %s could not be read (%s); write it exactly as the operations list "
                      "gives it." % (text, why))
        else:
            child = domain.clone(st)
            try:
                open_apply(domain, variant, child, mv, opening)
            except Exception as e:                   # noqa: BLE001
                ent.update(move=domain.format_move(mv), refused=str(e)[:300])
                result = ("CHECK RESULT for %s: refused by the simulator: %s. The current design is unchanged."
                          % (domain.format_move(mv), str(e)[:300]))
            else:
                sc = ev.score(child, seed0=D.stable_seed(pid, "eval", "s", t + 1))
                viol = float(sc.agg_metrics.get("min_feature_violations") or 0.0) if variant == "photonics" else 0.0
                ent.update(move=domain.format_move(mv), feasible=bool(sc.feasible), phi=float(sc.mean), viol=viol)
                result = check_text(C, R, L, domain, spec, child, sc, mv, history, metrics, variant, native_prompt, ev)
        ent["result"] = result
        if ent.get("move"):
            seen.add(ent["move"])
        log.append(ent)
        left = nchk - len(log)
        tail = ("You may check %d more move%s, or commit one with the <action> line." % (left, "" if left == 1 else "s")
                if left > 0 else "No checks left: commit a move now with the <action> line.")
        msgs += [{"role": "assistant", "content": reply.get("answer", "")},
                 {"role": "user", "content": result + "\\n\\n" + tail}]


# ------------------------------------------------------------------ observation
''')

# 2. the model call of a turn
rep('''                SIM.release()
                try:
                    reply = llm(system, user + note)
                finally:
                    SIM.acquire()
''', '''                nchk = int(getattr(a, "checks", 0) or 0)
                if nchk > 0 and not native:
                    raise SystemExit("--checks takes native variants only (got %s)" % variant)
                if getattr(a, "check_actions", False) and nchk <= 0:
                    raise SystemExit("--check-actions needs --checks C > 0")
                if nchk > 0:
                    reply, rec["checks"] = check_turn(a, llm, C, D, L, R, domain, spec, st, ev, pid, t, history,
                                                      metrics, variant, opening, native_prompt, system, user + note,
                                                      nchk)
                    rec["checks_max"] = nchk
                    if reply.get("check_as_action"):
                        rec["check_as_action"] = True
                else:
                    SIM.release()
                    try:
                        reply = llm(system, user + note)
                    finally:
                        SIM.acquire()
''')

# 3. the flag
rep('''    ap.add_argument("--native-prompt", choices=["served", "probe"], default="served",''',
    '''    ap.add_argument("--checks", type=int, default=0,
                    help="episodes, llm, native variants (v3rc): up to C checks per turn -- a reply ending with "
                         "<check>MOVE</check> is simulated on a copy and its result shown before the model commits")
    ap.add_argument("--check-actions", action="store_true",
                    help="with --checks (data collection, v3rc): a reply ending with an <action> line whose move "
                         "was not checked yet in this turn is checked instead (its tag renamed to <check> in the "
                         "conversation and the record); naming a checked move again commits it")
    ap.add_argument("--native-prompt", choices=["served", "probe"], default="served",''')

open(dst, "w", encoding="utf-8", newline="").write(s)
print("wrote %s (sha1 %s)" % (dst, hashlib.sha1(open(dst, "rb").read()).hexdigest()))
