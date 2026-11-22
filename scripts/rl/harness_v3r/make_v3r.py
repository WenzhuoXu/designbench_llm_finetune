"""Build dataxp_policy_eval_v3r.py from dataxp_policy_eval_v3n.py (sha1 2b34b25e...): one opt-in flag, --open-rules.
Without the flag the file behaves as v3n (and so, without --reject-note, as the original v3). Every replacement
must match exactly once.

--open-rules (2026-11-22) changes the simulator's refusal rules, in every family:
  1. every refused move is reported back: the next prompt lists the moves refused at the current design with the
     simulator's reason (v3n's --reject-note, switched on), until a move is applied;
  2. grid2op: a move that would end the episode (a diverged power flow, an infeasible redispatch) is undone: the
     grid stays as it was, the turn is spent, and the move is reported as refused with the cause. Without the flag
     such a move is applied, the episode is over and every later move is refused ("the episode has ended");
  3. photonics: a move that breaks the minimum-feature (fabrication) rule is applied instead of refused. The
     design is outside the validity envelope until min_feature_violations is 0 (potential.DesignProgram.is_valid),
     so it never counts as feasible before it is fixed; the prompt says how many pixels break the rule.
  The prompt's own statements of rules 2 and 3 are changed to match (photonics: the sentence is replaced; grid2op:
  a rule line is added after the grammar, as its rule text appears in several places). Moves that are malformed,
  change nothing, or name what does not exist are refused as before (and now reported). The turn budget is the same.

usage: python make_v3r.py <v3n> <out>
"""
import hashlib
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
assert hashlib.sha1(open(src, "rb").read()).hexdigest() == "2b34b25e9c3288d8b3e010849dbd99c2e2b760da", "not the expected v3n"


def rep(old, new):
    global s
    assert s.count(old) == 1, "%d matches for %r" % (s.count(old), old[:70])
    s = s.replace(old, new)


# 1. helpers
rep('''# ------------------------------------------------------------------ observation
''', '''# ------------------------------------------------------------------ open rules (--open-rules)
# v3r (2026-11-22). Without the flag this file behaves as dataxp_policy_eval_v3n.py (sha1 2b34b25e).

PHOTONICS_RULE_OLD = "A change that breaks this is rejected and the design stays as it was."
PHOTONICS_RULE_NEW = ("A change that breaks this is still applied; min_feature_violations counts the pixels that "
                      "break it, and the design does not count as feasible until that count is 0.")
GRID2OP_RULE = ("\\n\\nRULE IN THIS RUN: a move that would end the episode (a diverged power flow, or a CURTAIL / "
                "SET_STORAGE order the dispatchable units cannot balance) is not applied: the grid stays as it was, "
                "the turn is spent, and the next prompt says why.\\n")


def episode_end(child):
    """The cause, when the state after a move is an ended episode (grid2op's blackout), else None."""
    cell = child.get("cell") if isinstance(child, dict) else None
    if isinstance(cell, dict) and cell.get("done"):
        return str(cell.get("reason") or "blackout")
    return None


def open_apply(domain, variant, child, mv, opening):
    """domain.apply, under the open rules when `opening`: photonics without the fabrication refusal, and a move
    that ends the episode refused (ValueError) instead of applied."""
    if opening and variant == "photonics":
        domain._apply_unchecked(child, mv)       # judged by min_feature_violations and the validity envelope
    else:
        domain.apply(child, mv)
    if opening:
        ended = episode_end(child)
        if ended:
            raise ValueError("%s would end the episode (%s); it was undone and the grid is as it was"
                             % (domain.format_move(mv), ended))


def open_rules_prompt(variant, system, user, metrics):
    """(system, user) with the prompt's statements of the changed rules changed to match."""
    if variant == "photonics":
        n = system.count(PHOTONICS_RULE_OLD) + user.count(PHOTONICS_RULE_OLD)
        if n != 1:
            raise RuntimeError("photonics prompt: %d copies of the fabrication refusal sentence" % n)
        system = system.replace(PHOTONICS_RULE_OLD, PHOTONICS_RULE_NEW)
        user = user.replace(PHOTONICS_RULE_OLD, PHOTONICS_RULE_NEW)
        bad = float((metrics or {}).get("min_feature_violations") or 0.0)
        if bad > 0:
            user = user + ("\\n\\nFABRICATION: the current design breaks the minimum feature rule at %d pixels; it does "
                           "not count as feasible until that is 0.\\n" % int(round(bad)))
    elif variant == "grid2op":
        user = user + GRID2OP_RULE
    return system, user


# ------------------------------------------------------------------ observation
''')

# 2. episode state
rep('''    noting = bool(getattr(a, "reject_note", False)) and a.policy == "llm"
''', '''    opening = bool(getattr(a, "open_rules", False))
    noting = (bool(getattr(a, "reject_note", False)) or opening) and a.policy == "llm"
''')
rep('''    if noting:
        out["reject_note"] = True
''', '''    if noting:
        out["reject_note"] = True
    if opening:
        out["open_rules"] = True
''')

# 3. the prompt
rep('''            if noting and rejected:
                user = user + reject_note(rejected)
''', '''            if opening:
                system, user = open_rules_prompt(variant, system, user, metrics)
            if noting and rejected:
                user = user + reject_note(rejected)
''')

# 4. native path: apply under the open rules
rep('''            ad = mv.as_dict()
            child = domain.clone(st)
            try:
                domain.apply(child, mv)
            except Exception as e:                     # noqa: BLE001
''', '''            ad = mv.as_dict()
            child = domain.clone(st)
            try:
                open_apply(domain, variant, child, mv, opening)
            except Exception as e:                     # noqa: BLE001
''')

# 5. the flag
rep('''    ap.add_argument("--native-prompt",''', '''    ap.add_argument("--open-rules", action="store_true",
                    help="episodes (v3r): every refused move is reported back (implies --reject-note for llm); a grid2op "
                         "move that would end the episode is undone and refused; a photonics move that breaks the "
                         "fabrication rule is applied (the design is invalid until min_feature_violations is 0)")
    ap.add_argument("--native-prompt",''')
with open(dst, "w", encoding="utf-8", newline="") as f:
    f.write(s)
print("written", dst, hashlib.sha1(s.encode("utf-8")).hexdigest())
