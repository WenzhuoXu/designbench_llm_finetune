"""Build dataxp_policy_eval_v3n.py from dataxp_policy_eval_v3.py (sha1 540512fd...): one opt-in flag,
--reject-note. Every replacement must match exactly once.

usage: python make_v3n.py <orig> <out>
"""
import hashlib
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
assert hashlib.sha1(open(src, "rb").read()).hexdigest() == "540512fdd0601b7df6773e86b34d0d117ca333fe", "not the expected original"


def rep(old, new):
    global s
    assert s.count(old) == 1, "%d matches for %r" % (s.count(old), old[:70])
    s = s.replace(old, new)


# 1. helpers
rep('''# ------------------------------------------------------------------ observation
''', '''# ------------------------------------------------------------------ refused moves (--reject-note)
# v3n (2026-11-12). Without the flag this file behaves as dataxp_policy_eval_v3.py (sha1 540512fd).
# A move the simulator refuses spends the turn and leaves the design unchanged, so the next prompt
# is the same prompt: the model is not told. At T=0 it then proposes the same move for every
# remaining turn (dx38mtR, evaluation problems: 32 of 60 grid2op episodes end that way), the case
# the loop-guard comment below describes for a dropped move. With --reject-note the next prompts
# carry, after the instruction, one line per move refused at the CURRENT design with the
# simulator's own reason, until a move is applied. The turn is still spent: one model call per
# turn, the turn budget and every other rule unchanged.

ACTION_TEXT = re.compile(r"<action>\\s*(.*?)\\s*</action>", re.S)


def action_text(reply, mv):
    """The refused move as the model wrote it (the last <action>...</action> of its answer); a
    move with no such text is shown as its serialised form."""
    m = ACTION_TEXT.findall((reply or {}).get("answer") or "")
    if m:
        return " ".join(m[-1].split())
    return json.dumps(mv.as_dict() if hasattr(mv, "as_dict") else mv, sort_keys=True)


def reject_note(rejected):
    """The text appended to the user turn: '' when no move has been refused at this design."""
    if not rejected:
        return ""
    lines = ["", "", "REFUSED MOVES (the simulator did not apply them; the design is unchanged since):"]
    for text, why in rejected:
        lines.append("- %s" % (why if text and text in why else "%s: %s" % (text, why)))
    lines.append("Propose a different move.")
    return "\\n".join(lines) + "\\n"


def note_refusal(rejected, reply, mv, why):
    item = (action_text(reply, mv), str(why))
    if item not in rejected:
        rejected.append(item)


# ------------------------------------------------------------------ observation
''')

# 2. episode state
rep('''    history, turns = [], []
    v2 = getattr(a, "obs", "v1") == "v2"
''', '''    history, turns = [], []
    noting = bool(getattr(a, "reject_note", False)) and a.policy == "llm"
    rejected = []                        # --reject-note: (move as written, reason) refused at the current design
    v2 = getattr(a, "obs", "v1") == "v2"
''')
rep('''    native_prompt = getattr(a, "native_prompt", "served")
    if native:
''', '''    native_prompt = getattr(a, "native_prompt", "served")
    if noting:
        out["reject_note"] = True
    if native:
''')

# 3. the prompt
rep('''            if getattr(a, "record_prompts", False):
                rec.update(system=system, user=user)       # the exact prompt, for SFT data (DX-21)
''', '''            if noting and rejected:
                user = user + reject_note(rejected)
                rec["reject_note"] = len(rejected)
            if getattr(a, "record_prompts", False):
                rec.update(system=system, user=user)       # the exact prompt, for SFT data (DX-21)
''')

# 4. native path: refused / applied
rep('''            except Exception as e:                     # noqa: BLE001
                why = "illegal:%s" % str(e)[:80]
                rec.update(move=ad, action=ad)
            else:
''', '''            except Exception as e:                     # noqa: BLE001
                why = "illegal:%s" % str(e)[:80]
                rec.update(move=ad, action=ad)
                if noting:
                    note_refusal(rejected, reply, mv, str(e)[:300])
            else:
                del rejected[:]                        # the design changes: earlier refusals no longer describe it
''')

# 5. SET_PARAM path: refused / applied
rep('''            ok, why, v_old = check_move(domain, st, mv)
            if ok:
                child = domain.clone(st)
''', '''            ok, why, v_old = check_move(domain, st, mv)
            if not ok and noting:
                note_refusal(rejected, reply, mv, why)
            if ok:
                del rejected[:]
                child = domain.clone(st)
''')

# 6. the flag
rep('''    ap.add_argument("--guard-retries", type=int, default=2)
''', '''    ap.add_argument("--guard-retries", type=int, default=2)
    ap.add_argument("--reject-note", action="store_true",
                    help="episodes, llm (v3n): after a move the simulator refused, the next prompts list the moves "
                         "refused at the current design with the simulator's reason (the turn is still spent)")
''')
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst, hashlib.sha1(s.encode("utf-8")).hexdigest())
