"""The open rules of dataxp_policy_eval_v3r.py --open-rules, for the data tools (states, scoring).

  photonics  a move that breaks the minimum-feature rule is applied (domain._apply_unchecked); the design is outside
             the validity envelope until min_feature_violations is 0, so it is never feasible before it is fixed
  grid2op    a move that would end the episode (blackout) is refused: the state stays as it was
  others     unchanged (domain.apply, through dataxp.ownrows.common's own helpers)

The harness's open_apply and these functions must agree; tests/test_rl_screen.py checks the rules on stand-in
domains and scripts/rl/harness_v3r/test_v3r.py checks the harness against stored episodes.
"""


# the prompt's statements of the changed rules, as dataxp_policy_eval_v3r.py changes them (test_rl_screen.py checks
# these three strings against the harness file)
PHOTONICS_RULE_OLD = "A change that breaks this is rejected and the design stays as it was."
PHOTONICS_RULE_NEW = ("A change that breaks this is still applied; min_feature_violations counts the pixels that "
                      "break it, and the design does not count as feasible until that count is 0.")
GRID2OP_RULE = ("\n\nRULE IN THIS RUN: a move that would end the episode (a diverged power flow, or a CURTAIL / "
                "SET_STORAGE order the dispatchable units cannot balance) is not applied: the grid stays as it was, "
                "the turn is spent, and the next prompt says why.\n")


def open_prompt(variant: str, system: str, user: str):
    """(system, user) of a prompt served under the strict rules, with the rule statements of the open rules (the
    harness's open_rules_prompt, without the per-design fabrication line: a design reached under the strict rules
    never breaks the fabrication rule)."""
    if variant == "photonics":
        n = system.count(PHOTONICS_RULE_OLD) + user.count(PHOTONICS_RULE_OLD)
        if n != 1:
            raise ValueError("photonics prompt: %d copies of the fabrication refusal sentence" % n)
        return system.replace(PHOTONICS_RULE_OLD, PHOTONICS_RULE_NEW), user.replace(PHOTONICS_RULE_OLD, PHOTONICS_RULE_NEW)
    if variant == "grid2op":
        if GRID2OP_RULE in user:
            raise ValueError("grid2op prompt already carries the open-rules line")
        return system, user + GRID2OP_RULE
    return system, user


def episode_end(st):
    """The cause, when `st` is an ended episode (grid2op's blackout), else None."""
    cell = st.get("cell") if isinstance(st, dict) else None
    if isinstance(cell, dict) and cell.get("done"):
        return str(cell.get("reason") or "blackout")
    return None


def apply_open(domain, variant: str, st, move) -> None:
    """Apply a Move on `st` in place under the open rules (ValueError when refused)."""
    if variant == "photonics":
        domain._apply_unchecked(st, move)
    else:
        domain.apply(st, move)
    ended = episode_end(st)
    if ended:
        raise ValueError("%s would end the episode (%s); it was undone and the grid is as it was"
                         % (domain.format_move(move), ended))


def try_apply_open(domain, variant: str, st, move, try_apply):
    """dataxp.ownrows.common.try_apply under the open rules: (child, move applied, info) or (None, None, info).
    SET_PARAM moves go to try_apply unchanged (the open rules change native moves only)."""
    if getattr(move, "is_set_param", False):
        return try_apply(domain, st, move)
    info = {"clamped": False, "noop": False, "err": None}
    child = domain.clone(st)
    try:
        apply_open(domain, variant, child, move)
    except Exception as e:                                   # noqa: BLE001
        info["err"] = "illegal:%s" % str(e)[:80]
        return None, None, info
    return child, move, info


def bad_pixels(variant: str, st) -> int:
    """Pixels of `st` that break photonics' minimum-feature rule (min_feature_violations); 0 in other families."""
    if variant != "photonics":
        return 0
    from dataxp.domains.photonics import _drc_count
    return int(_drc_count(st["fx"], st["pat"]))


def open_rank(bad: int, phi: float) -> tuple:
    """How a design ranks under the open rules in photonics: fewer rule-breaking pixels first, then Phi. Phi alone
    cannot rank it: the potential gives every design outside the validity envelope a flat objective of 10, so on a
    problem whose valid objective is above 10 breaking the rule would raise Phi and a repair would lower it."""
    return (-int(bad), float(phi))


def open_gain(bad: int, phi: float, bad0: int, phi0: float, tol: float = 1e-9) -> bool:
    """A move improves the design under the open rules (photonics): fewer rule-breaking pixels, or as many and a
    higher Phi."""
    return bad < bad0 or (bad == bad0 and phi > phi0 + tol)


def rebuild_open(domain, variant: str, spec: dict, history: list, apply_action, move_from_dict):
    """The state after the applied moves `history` (serialised) under the open rules. Only photonics differs from
    dataxp.ownrows.common.rebuild_state: a history may hold moves that break the fabrication rule."""
    st = domain.load(spec)
    for a in history:
        if variant == "photonics":
            domain._apply_unchecked(st, move_from_dict(a))
        else:
            apply_action(domain, st, a)
    return st
