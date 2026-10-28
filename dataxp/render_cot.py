"""Render one trajectory turn as: observation -> reasoning -> <action>.

The observation is SELF-CONTAINED: the trajectory is replayed from its problem
spec to the turn being rendered, and the domain's describe() prints the real
design at that step - problem setup, every element with its role and current
values, and the simulator's per-element results - followed by the constraint
status, the changes already made this episode, and the (not yet evaluated)
change to justify. Replay is exact: for every deterministic variant the
replayed state reproduces the recorded metrics to <1e-12.

The reasoning may use ONLY what the policy will have at inference, and it reads
the current state only: what is violated, where the per-element results place
the problem, and why this element. It makes no prediction of what the change
(or any alternative) does - not numerically and not directionally. The policy
gets one simulator call per turn and never has those outcomes; a version that
allowed directional claims contradicted the measured outcome in 27/56 traces.
This is also why preference pairs are not a CoT substrate - a pair's content is
two simulated outcomes at one state, only one of which the policy can ever see.

Every number in the output must appear in the observation or be a simple
derivation (difference, percentage) of the key quantities; ungrounded
generations are rejected rather than repaired.

    # against a running vLLM server (drc: code/serve_qwen.sh -> 127.0.0.1:8766)
    python -m dataxp.render_cot --data <run1> --out <dir> --variants grid circuit \
        --endpoint http://127.0.0.1:8766/v1 --concurrency 16 --limit 8 --thinking
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .domain import (Categorical, Continuous, Move, action_form, apply_action, grammar_lead,
                     speaks_verbs)
from .domains import get_domain

MODEL = "Qwen/Qwen3.8-27B"

SYSTEM = """You write the internal reasoning of an engineer solving a constrained design problem, at one step of an iterative process. You are given a full observation of the current design (the problem setup, every element with its current settings, and the per-element results of the last evaluation), the status of each constraint, the changes already made, and the change about to be made. Write the reasoning that leads to that change.

The reasoning is a reading of the CURRENT state and the choice it leads to. It contains no predictions.

Cover, in your own order:
- which constraints are violated and by how much;
- where the per-element results place the problem (which elements carry it, which do not);
- the chosen element: its role, its current setting and remaining range, and what the current results show about it compared with the other elements (it carries the most load, has the least margin, dominates the cost or area; others are at a bound or were already changed);
- a closing sentence naming the change and the violation or element it is aimed at.

Rules:
- Use ONLY information in the observation, and read it exactly: values, comparisons and their size, element identities, and topology (which element feeds which, upstream/downstream).
- Do NOT predict. Do not say what the change does, will do or is for in terms of its effect on anything: the changed element, a constraint, the objective, another element, a flow or a gradient. This applies in every tense and form. Not allowed: "increasing the radius raises its buckling capacity", "I cut G3 to reduce the flow on L5", "this relieves the queue", "adding a safety buffer", "which brings G3 to 69 MW", "without affecting the other members", "this should fix...". Allowed: "m8 carries the governing deficit and has room below its upper bound", "G5's listed sensitivity on L23 is -1 MW/MW", "I raise m8's radius, aimed at the buckling deficit it carries". Stating what a current result depends on is fine ("GBW is set by gm1 and Cc"); stating how this change moves it is not, even when the observation states the rule.
- Do not use the words potential, score, ranking, candidate, option, or say that a search was run.
- First person, present tense. Follow the STYLE line; do not reuse a stock opening.
- End with exactly one action line, on its own line, in this form:
  <action>SET_PARAM(<element>, <parameter>, <value>)</action>"""

# The action-format section above is replaced by the domain's own grammar()
# for native-verb domains (docs/dataxp_move_contract.md) and for v2 domains
# with a verb surface (docs/dataxp_verb_surface.md); old variants keep SYSTEM
# exactly. The placeholder line is action_form(domain): VERB(arg, ...) under
# the multitool spelling, SET_PARAM(element, parameter, value) under
# set_parameter (docs/dataxp_spellings.md); the rest of the text is the same.
_ACTION_FORMAT = "  <action>SET_PARAM(<element>, <parameter>, <value>)</action>"
assert _ACTION_FORMAT in SYSTEM


def _surface(domain) -> bool:
    """A v2 domain whose SET_PARAM moves are spelled as verbs: the stored
    action is still {i, p, v, v_old}, the printed text is format_move."""
    return bool(getattr(domain, "verb_surface", False))


_DOMAINS = {}


def _domain_for(variant):
    d = _DOMAINS.get(variant)
    if d is None:
        d = _DOMAINS[variant] = get_domain(variant)
    return d


def system_prompt(domain=None) -> str:
    if domain is None or not speaks_verbs(domain):
        return SYSTEM
    g = "\n".join("    " + l for l in domain.grammar().rstrip().splitlines())
    return SYSTEM.replace(_ACTION_FORMAT, (
        "  <action>%s</action>\n  %s\n" % (action_form(domain), grammar_lead()) + g))


def _is_native(action: dict) -> bool:
    """A serialised move of a native-verb domain ({verb, args}); the legacy
    SET_PARAM form is {i, p, v, v_old}."""
    return "verb" in action


def _num_args(action: dict):
    return [float(v) for v in (action.get("args") or {}).values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]


def _label_for(domain, st, action: dict) -> str:
    """Role label for a history / change line: element_label for SET_PARAM,
    the domain's move_label (optional) for a native verb."""
    if not _is_native(action):
        return domain.element_label(st, int(action["i"]))
    ml = getattr(domain, "move_label", None)
    return str(ml(st, Move.from_dict(action))) if callable(ml) else ""


DOMAIN_BLURB = {
    "grid": ("power grid economic dispatch", "minimise total generation cost"),
    "circuit": ("two-stage Miller op-amp sizing", "minimise area"),
    "traffic": ("traffic signal timing on a road network", "minimise mean delay"),
    "thermal": ("heat-sink allocation on a thermal network", "minimise cooling cost"),
    "seir": ("epidemic control across population groups", "minimise intervention cost"),
    "inventory": ("multi-echelon inventory control", "minimise total cost"),
    "frame3d": ("3D truss member sizing", "minimise mass"),
    # v2: full action sets
    "frame3d2": ("3D truss design: member sizing, materials, bracing topology and joint positions",
                 "minimise mass"),
    "circuit2": ("two-stage op-amp design: device sizing, channel lengths, compensation and cascode",
                 "minimise area"),
    "grid2": ("power grid operation: dispatch, unit commitment, line switching and phase shifters",
              "minimise total operating cost"),
    "traffic2": ("traffic signal design: cycle, green split, offset, phase order and lanes",
                 "minimise mean delay"),
    "thermal2": ("thermal design: heat-sink placement and sizing, spreader links and floorplan",
                 "minimise cooling cost"),
    "seir2": ("epidemic response: intervention strength and timing, testing and vaccine allocation",
              "minimise intervention cost"),
    "inventory2": ("inventory network design: stock levels, review periods, stocking points and expediting",
                   "minimise total cost"),
    # native-verb domains (docs/corpus_native_domains.md)
    "grid2op": ("power grid operation under grid2op: busbar topology, line status, redispatch, "
                "curtailment and storage, one 5-minute step per move",
                "minimise operating cost"),
    "photonics": ("photonic inverse design on a 2-D silicon design region: fill, clear, shift, "
                  "dilate and erode features (ceviche FDFD)",
                  "minimise the optical loss to the intended output port"),
    "molopt": ("molecular design by graph edits: attach, remove or swap fragments, substitute "
               "atoms, close or break ring bonds",
               "minimise the heavy-atom count"),
    "engibench": ("structural / thermal / photonic topology design on a solid-void field: fill, "
                  "clear, shift, dilate and erode regions (EngiBench simulators)",
                  "minimise the problem's objective (compliance, thermal dissipation or optical loss)"),
}

BANNED = re.compile(
    r"\b(phi|score[sd]?|ranking|ranked|search|candidates?|option [AB]\b|"
    r"simulat\w+ (?:show|return|gave))\b", re.I)
# "potential" is banned everywhere except thermal, where temperature IS a
# thermal potential and the word is ordinary physics.
POTENTIAL = re.compile(r"\bpotential\b", re.I)
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")

# Rotated per sample: without it 43% of a 56-sample run opened with the same
# four words, the low-entropy signature that made the previous distillation
# derail at temperature 0.8.
STYLES = (
    "Open by naming the quantity that is out of bounds and by how much.",
    "Open from where the objective currently stands, then bring in the violation.",
    "Open with the physical mechanism that is currently failing, then the measurement.",
    "Open mid-deliberation, as if continuing a train of thought.",
    "Open by stating which element you are about to change and why that one.",
    # was "the tension between the violations and the objective": that asks
    # for a trade-off, i.e. an effect claim (1/56 in v4)
    "Open by comparing how far each violated constraint sits from its limit.",
    "Open by setting aside another element, citing what the current results show about it.",
    "Open with what the per-element results say about where the problem sits.",
)

# Forward-looking language. The trace reads the current state and nothing
# else: an earlier prompt that allowed effect claims produced contradicted
# causal claims in 27/56 traces (second-order and network effects the model
# cannot predict). This only catches surface forms; generic present-tense
# predictions ("raising B raises I2") are rare with --thinking (0 in 56 by an
# independent audit) but are not machine-checked.
PREDICTIVE = re.compile(
    r"\b(will|would|could|should|might|expect\w*|likely|going to|so that|"
    r"in order to|thereby|"
    # "room to increase" states remaining range, not a purpose
    r"(?<!room )(?<!range )(?<!headroom )(?<!margin )"
    r"to (?:reduce|raise|lower|increase|decrease|relieve|restore|improve|fix|"
    r"resolve|clear|boost|lift|cut|recover|bring|shift|push|pull|drain|close))\b"
    r"|'ll\b", re.I)


def _fmt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f or f in (float("inf"), float("-inf")):
        return str(f)
    if f == 0.0:
        return "0"
    # significant figures for small magnitudes: 0.000132 must not print as "0"
    if abs(f) < 1e-2:
        return "%.3g" % f
    if abs(f) < 1e4:
        return ("%.3f" % f).rstrip("0").rstrip(".")
    return "%.2f" % f


def _fmtv(v) -> str:
    """Action values and observed effects: enough digits to reproduce the
    applied value and to show small changes (3 decimals printed a radius
    change 0.0094 -> 0.011 as 0.009 -> 0.011 and deflection changes as 0)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f or f in (float("inf"), float("-inf")):
        return str(f)
    if f == int(f) and abs(f) < 1e9:
        return str(int(f))
    return "%.6g" % f


def _limits(goals: dict, state: dict) -> dict:
    out = {}
    for gk, gv in (goals or {}).items():
        try:
            lim = float(gv)
        except (TypeError, ValueError):
            continue
        for pref in ("minimum_", "maximum_", "min_", "max_"):
            if gk.startswith(pref):
                metric = gk if gk in state else gk[len(pref):]
                out[metric] = (">=" if pref.startswith("min") else "<=", lim)
                break
    return out


def policy_observation(domain, spec, st, turn, history, variant):
    """(lines, key_numbers): what the policy sees at this turn - domain/goal,
    describe(), constraint status, the changes already made, and the observed
    effect of the last change. The renderer and the information probe both use
    this, so training prompts and probed observations are identical."""
    base = variant.replace("_stoch", "")
    dom, goal = DOMAIN_BLURB.get(base, (base, "optimise"))
    state = turn.get("state_metrics") or {}
    binding = turn.get("binding_constraint")
    limits = _limits(spec.get("goals") or {}, state)

    lines = ["DOMAIN: %s" % dom, "GOAL: %s subject to the constraints below" % goal,
             "", domain.describe(st, spec).rstrip(),
             "", "CONSTRAINT STATUS (current design)"]
    # Mark EVERY violated constraint, not just the most severe one: marking only
    # the binding constraint hid that a circuit state also failed GBW and slew,
    # which is exactly what the gold move (lower Cc: PM falls, GBW rises, area
    # drops sharply) was trading for.
    key = []
    for k, v in sorted(state.items()):
        lim = limits.get(k)
        if not lim:
            continue
        key += [float(v), lim[1]]
        viol = float(v) < lim[1] if lim[0] == ">=" else float(v) > lim[1]
        tag = ""
        if viol:
            tag = "   VIOLATED" + (" (most severe)" if k == binding else "")
        lines.append("  %-22s %-10s limit %s %s%s" % (k, _fmt(v), lim[0], _fmt(lim[1]), tag))
    for k, v in sorted(state.items()):
        if k not in limits:
            key.append(float(v))
            lines.append("  %-22s %s" % (k, _fmt(v)))

    if history:
        lines += ["", "CHANGES ALREADY MADE THIS EPISODE (oldest first)"]
        for n, (lab, a) in enumerate(history, 1):
            if _is_native(a):
                lines.append("  %d. %s%s" % (n, domain.format_move(Move.from_dict(a)),
                                             (" (%s)" % lab) if lab else ""))
                continue
            if _surface(domain):
                # verb surface: the stored action is {i, p, v, v_old}; the
                # line spells it as its verb and names the value it replaced
                # (a toggle's verb pair already says what it was)
                was = domain.format_value(a["p"], a.get("v_old"))
                note = "; ".join(x for x in (lab, ("was " + was) if was else "") if x)
                lines.append("  %d. %s%s" % (n, domain.format_move(Move.from_dict(a)),
                                             (" (%s)" % note) if note else ""))
                continue
            lines.append("  %d. element %d, %s: %s %s -> %s" % (
                n, int(a["i"]), lab, a["p"], _fmtv(a["v_old"]), _fmtv(a["v"])))
    # The observed effect of the previous change (history, not a forecast).
    # Without it state-only reasoning oscillated (collaborator DX-20: grid
    # moves undoing the previous move 0.41 vs 0.06 without reasoning).
    prev = turn.get("prev_state_metrics")
    if history and prev:
        lines += ["", "OBSERVED EFFECT OF THE LAST CHANGE (before -> after)"]
        for k, v in sorted(state.items()):
            if k in prev:
                lines.append("  %-22s %s -> %s" % (k, _fmtv(prev[k]), _fmtv(v)))
    return lines, key


def build_prompt(domain, spec, st, turn, history, variant, hash_idx=0):
    """Return (prompt, exact_numbers, derived_base, label)."""
    act = turn["action"]
    if _is_native(act):
        return _build_prompt_native(domain, spec, st, turn, history, variant, hash_idx)
    if _surface(domain):
        return _build_prompt_surface(domain, spec, st, turn, history, variant, hash_idx)
    label = domain.element_label(st, int(act["i"]))
    lines, key = policy_observation(domain, spec, st, turn, history, variant)

    up = float(act["v"]) > float(act["v_old"])
    direction = "increase" if up else "decrease"
    # The range that matters is the one in the move's direction. Traces quoted
    # upward headroom for a decrease; the element's own limit is read back by
    # clamping a copy (per-element limits, e.g. a generator's p_max, differ
    # from the parameter's global bounds).
    probe = domain.clone(st)
    lo, hi = domain.bounds(st, act["p"])
    domain.set(probe, int(act["i"]), act["p"], hi if up else lo)
    limit = float(domain.get(probe, int(act["i"]), act["p"]))
    room = abs(limit - float(act["v_old"]))
    key += [float(act["v"]), float(act["v_old"]), float(act["i"]), limit, room]
    lines += ["", "CHANGE YOU ARE ABOUT TO MAKE (not yet evaluated)",
              "  element %d, %s: %s %s -> %s  (%s; room in this direction %s, to the %s bound %s)"
              % (int(act["i"]), label, act["p"], _fmtv(act["v_old"]), _fmtv(act["v"]), direction,
                 _fmt(room), "upper" if up else "lower", _fmtv(limit)),
              "", "STYLE FOR THIS ONE: %s" % STYLES[hash_idx % len(STYLES)],
              "", "End with: <action>SET_PARAM(%d, %s, %s)</action>"
              % (int(act["i"]), act["p"], _fmtv(act["v"]))]
    prompt = "\n".join(lines)
    exact = [float(t.replace(",", "")) for t in NUMBER.findall(prompt)]
    return prompt, exact, key, label


def _build_prompt_native(domain, spec, st, turn, history, variant, hash_idx=0):
    """build_prompt for a native-verb domain: the change is printed with the
    domain's format_move (plus its move_label when the domain has one) and
    the required action line is that same text inside <action>...</action>.
    The move's numeric arguments join the key quantities."""
    act = turn["action"]
    move = Move.from_dict(act)
    text = domain.format_move(move)
    label = _label_for(domain, st, act)
    lines, key = policy_observation(domain, spec, st, turn, history, variant)
    key += _num_args(act)
    lines += ["", "CHANGE YOU ARE ABOUT TO MAKE (not yet evaluated)",
              "  %s%s" % (text, ("  (%s)" % label) if label else ""),
              "", "STYLE FOR THIS ONE: %s" % STYLES[hash_idx % len(STYLES)],
              "", "End with: <action>%s</action>" % text]
    prompt = "\n".join(lines)
    exact = [float(t.replace(",", "")) for t in NUMBER.findall(prompt)]
    return prompt, exact, key, label


def _build_prompt_surface(domain, spec, st, turn, history, variant, hash_idx=0):
    """build_prompt for a verb-surface domain: the stored action is a SET_PARAM
    dict, the printed change and the required action line are its verb
    spelling (format_move), the label is the element's, and for a continuous
    parameter the old value, direction and room to the bound are stated as
    the SET_PARAM prompt states them (a categorical change names both codes;
    a toggle's verb says it all). The key quantities are the SET_PARAM ones."""
    act = turn["action"]
    move = Move.from_dict(act)
    text = domain.format_move(move)
    label = domain.element_label(st, int(act["i"]))
    lines, key = policy_observation(domain, spec, st, turn, history, variant)
    key += [float(act["v"]), float(act["v_old"]), float(act["i"])]
    pv = domain.verb_param(act["p"])
    detail = ""
    if pv is not None and isinstance(pv.codec, Continuous):
        up = float(act["v"]) > float(act["v_old"])
        probe = domain.clone(st)
        lo, hi = domain.bounds(st, act["p"])
        domain.set(probe, int(act["i"]), act["p"], hi if up else lo)
        limit = float(domain.get(probe, int(act["i"]), act["p"]))
        room = abs(limit - float(act["v_old"]))
        key += [limit, room]
        detail = "%s %s -> %s (%s; room in this direction %s, to the %s bound %s)" % (
            pv.codec.arg, _fmtv(act["v_old"]), _fmtv(act["v"]),
            "increase" if up else "decrease", _fmt(room), "upper" if up else "lower",
            _fmtv(limit))
    elif pv is not None and isinstance(pv.codec, Categorical):
        detail = "%s %s -> %s" % (pv.codec.arg, domain.format_value(act["p"], act["v_old"]),
                                  domain.format_value(act["p"], act["v"]))
    note = "; ".join(x for x in (label, detail) if x)
    lines += ["", "CHANGE YOU ARE ABOUT TO MAKE (not yet evaluated)",
              "  %s%s" % (text, ("  (%s)" % note) if note else ""),
              "", "STYLE FOR THIS ONE: %s" % STYLES[hash_idx % len(STYLES)],
              "", "End with: <action>%s</action>" % text]
    prompt = "\n".join(lines)
    exact = [float(t.replace(",", "")) for t in NUMBER.findall(prompt)]
    return prompt, exact, key, label


def _derived(numbers):
    """Legitimate computations on the KEY quantities: margins ("exceeds the 1.0
    limit by 0.042") and percentage changes ("a 59% overload"). Restricted to
    the key quantities - not every number in the observation - so the allowed
    set stays sparse enough for the check to mean something."""
    out = set()
    uniq = list({round(float(n), 6) for n in numbers if n == n})
    for i, a in enumerate(uniq):
        for b in uniq[i + 1:]:
            out.add(round(abs(a - b), 6))
            if abs(b) > 1e-9:
                out.add(round(abs(100.0 * (a - b) / b), 4))
            if abs(a) > 1e-9:
                out.add(round(abs(100.0 * (b - a) / a), 4))
    for a in uniq:                      # fraction <-> percent ("10.5%" for 0.105)
        out.add(round(100.0 * abs(a), 4))
    return out


def grounded(text, exact, key, tol=0.01, derived_tol=0.01):
    """(ok, offending). Small integers are exempt (counts and ordinals).

    Compared by MAGNITUDE: the observation prints signed quantities (a grid
    sensitivity of -0.825) that prose correctly states unsigned ("reduces the
    flow by 0.825"). Observation fractions may also be quoted as percentages.
    Quantities the model computes for the post-change design (a new capacity, a
    new safety stock) are deliberately NOT admitted: they cannot be checked
    against the observation, and a sample run got one wrong (capacity 0.475 at
    g=0.5 where the stated lost time gives 0.45)."""
    bad, derived = [], None
    mags = [abs(r) for r in exact]
    pct = [100.0 * m for m in mags if m <= 1.0]
    for tok in NUMBER.findall(text):
        try:
            v = abs(float(tok.replace(",", "")))
        except ValueError:
            continue
        if v == int(v) and v <= 12:
            continue
        if any(abs(v - r) <= tol * max(1.0, r) for r in mags):
            continue
        if any(abs(v - p) <= tol * max(1.0, p) for p in pct):
            continue
        if derived is None:
            derived = _derived(key)
        if any(abs(v - d) <= derived_tol * max(1.0, d) for d in derived):
            continue
        bad.append(tok)
    return (not bad), bad


def check_row(r, text=None, domain=None):
    """check() for a rendered row, rebuilding the grounding sets from its prompt."""
    exact = [float(t.replace(",", "")) for t in NUMBER.findall(r["prompt"])]
    key = [float(v) for v in (r.get("state_metrics") or {}).values()]
    if _is_native(r["action"]):
        key += _num_args(r["action"])
    else:
        key += [float(r["action"][k]) for k in ("v", "v_old", "i")]
    if domain is None:
        # native rows and verb-surface rows are parsed by the domain; the
        # SET_PARAM regex path ignores it
        domain = _domain_for(r["variant"])
    blk = r["prompt"].split("CONSTRAINT STATUS")[1].split("\n\n")[0]
    key += [float(x) for x in re.findall(r"limit [<>]= (-?[\d.eE+-]+)", blk)]
    return check(r["cot"] if text is None else text, r, r["variant"], exact, key,
                 domain=domain)


def rescore(path):
    """Re-apply check() to a rendered file (CPU only); rewrites it in place."""
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        reasons, bad = check_row(r)
        r["accepted"], r["reject_reasons"], r["ungrounded_numbers"] = not reasons, reasons, bad
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return sum(r["accepted"] for r in rows), len(rows)


def strip_think(text):
    """Drop a thinking block if the server returns it inline."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def call(endpoint, user, temperature, max_tokens, model=MODEL, thinking=False,
         system=SYSTEM):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature, "top_p": 0.95, "top_k": 20,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        text = json.load(r)["choices"][0]["message"].get("content") or ""
    # with thinking on, an answer cut off inside the think block has no body
    if thinking and "<think>" in text and "</think>" not in text:
        return ""
    return strip_think(text)


def iter_states(data_dir, variant, pre_feasible_only=True):
    """Replay each accepted trajectory and yield (domain, spec, st, turn, history)
    at every eligible turn. Stratified to the repair phase by default: 59% of
    turns are post-feasibility trimming, and a set that was 87% trimming is what
    taught a previous model to shrink unconditionally (ledger Turn 8b). The
    yielded st is live - consume it before advancing the generator."""
    domain = get_domain(variant)
    acc = os.path.join(data_dir, variant, "accepted")
    specs = {}
    for line in open(os.path.join(acc, "problems.jsonl")):
        s = json.loads(line)
        specs[s["problem_id"]] = s
    for line in open(os.path.join(acc, "trajectories.jsonl")):
        t = json.loads(line)
        spec = specs[t["problem_id"]]
        st = domain.load(spec)
        history = []
        for k, turn in enumerate(t["turns"]):
            if k > 0:
                a = t["actions"][k - 1]
                assert a == t["turns"][k - 1]["action"], "actions/turns misaligned"
                apply_action(domain, st, a)
                # label the element as it is AFTER the change: labelled before,
                # a history line read "(..., Al): mat 0 -> 2"
                history.append((_label_for(domain, st, a), a))
            if pre_feasible_only and turn.get("binding_constraint") is None:
                continue
            turn = dict(turn)
            turn["problem_id"] = t["problem_id"]
            if k > 0 and t["turns"][k - 1]["turn"] == turn["turn"] - 1:
                turn["prev_state_metrics"] = t["turns"][k - 1].get("state_metrics")
            yield domain, spec, st, turn, list(history)


ACTION_RE = re.compile(r"<action>\s*SET_PARAM\(\s*(\d+)\s*,\s*([A-Za-z_]\w*)\s*,"
                       r"\s*(-?[\d.eE+-]+)\s*\)\s*</action>")
ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.S)


def check(text, turn, variant, exact, key, domain=None):
    """Reject reasons for one rendered trace. `domain` is needed only for a
    native-verb or verb-surface row (its parse_move reads the action line)."""
    reasons, bad_nums = [], []
    if not text:
        return ["empty"], []
    ok_g, bad_nums = grounded(text, exact, key)
    if not ok_g:
        reasons.append("ungrounded")
    if BANNED.search(text) or (variant.replace("_stoch", "") != "thermal"
                               and POTENTIAL.search(text)):
        reasons.append("banned")
    if PREDICTIVE.search(text):
        reasons.append("predictive")
    a = turn["action"]
    if _is_native(a) or _surface(domain):
        m = ACTION_TAG_RE.search(text)
        if not m:
            reasons.append("no_action")
        else:
            parsed = domain.parse_move(m.group(1)) if domain is not None else None
            if parsed is None or not parsed.matches(Move.from_dict(a), rel=1e-2):
                reasons.append("wrong_action")
        return reasons, bad_nums
    m = ACTION_RE.search(text)
    if not m:
        reasons.append("no_action")
    elif (int(m.group(1)) != int(a["i"]) or m.group(2) != a["p"]
          or abs(float(m.group(3)) - float(a["v"])) > 1e-2 * max(1.0, abs(float(a["v"])))):
        reasons.append("wrong_action")
    return reasons, bad_nums


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8765/v1")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="parallel requests; the server is shared, keep this modest")
    ap.add_argument("--limit", type=int, default=0, help="turns per variant, 0 = all")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="default 450, or 8000 with --thinking")
    ap.add_argument("--thinking", action="store_true",
                    help="let the renderer think first; only the answer is kept")
    ap.add_argument("--all-phases", action="store_true",
                    help="do NOT stratify to the repair phase (not recommended)")
    ap.add_argument("--select", default=None,
                    help="JSON file: list of [problem_id, turn] to render (any phase)")
    ap.add_argument("--prompts-only", action="store_true",
                    help="write prompts without calling the model")
    args = ap.parse_args(argv)
    if not args.max_tokens:
        args.max_tokens = 8000 if args.thinking else 450

    os.makedirs(args.out, exist_ok=True)
    keys = ("n", "ok", "ungrounded", "banned", "predictive", "no_action",
            "wrong_action", "empty", "err")
    grand = dict.fromkeys(keys, 0)

    select = None
    if args.select:
        select = {(str(p), int(t)) for p, t in json.load(open(args.select))}
    batches = {}
    domains = {}            # variant -> domain (system prompt, native checks)
    for variant in args.variants:
        batch = []
        for domain, spec, st, turn, hist in iter_states(
                args.data, variant, not (args.all_phases or select)):
            domains[variant] = domain
            if select is not None and (turn["problem_id"], turn["turn"]) not in select:
                continue
            if args.limit and len(batch) >= args.limit:
                break
            prompt, exact, key, label = build_prompt(domain, spec, st, turn, hist,
                                                     variant, hash_idx=len(batch))
            batch.append((turn, prompt, exact, key, label, len(hist)))
        batches[variant] = batch

    # one pool across all variants: per-variant batches of 8 left the server
    # half idle, which costs most with --thinking
    flat = [(v, b) for v in args.variants for b in batches[v]]
    if args.prompts_only:
        flat_texts = [""] * len(flat)
    else:
        systems = {v: system_prompt(domains.get(v)) for v in args.variants}

        def _one(vb):
            try:
                return call(args.endpoint, vb[1][1], args.temperature,
                            args.max_tokens, args.model, thinking=args.thinking,
                            system=systems[vb[0]])
            except Exception as e:           # recorded, not fatal
                return "__ERR__ %s" % e
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            flat_texts = list(ex.map(_one, flat))
    texts_by = {v: [t for (fv, _), t in zip(flat, flat_texts) if fv == v]
                for v in args.variants}

    for variant in args.variants:
        batch, texts = batches[variant], texts_by[variant]
        stats = dict.fromkeys(keys, 0)
        stats["n"] = len(batch)
        if not batch:
            print("%-15s no eligible turns" % variant, flush=True)
            continue

        rows = []
        for (turn, prompt, exact, key, label, n_hist), text in zip(batch, texts):
            if text.startswith("__ERR__"):
                stats["err"] += 1
                reasons, bad = ["err"], []
                text = text[8:]
            elif args.prompts_only:
                reasons, bad = [], []
            else:
                reasons, bad = check(text, turn, variant, exact, key,
                                     domain=domains.get(variant))
                for r in reasons:
                    stats[r] = stats.get(r, 0) + 1
            if not reasons:
                stats["ok"] += 1
            row = {
                "problem_id": turn["problem_id"], "turn": turn["turn"],
                "variant": variant, "cot": text, "action": turn["action"],
                "element_label": label, "binding_constraint": turn["binding_constraint"],
                "state_metrics": turn["state_metrics"], "n_prior_changes": n_hist,
                "accepted": not reasons, "reject_reasons": reasons,
                "ungrounded_numbers": bad, "prompt": prompt,
                "prompt_chars": len(prompt),
                "render_model": args.model,
                "render_params": {"temperature": args.temperature, "top_p": 0.95,
                                  "top_k": 20, "enable_thinking": bool(args.thinking)},
            }
            if _is_native(turn["action"]) or _surface(domains.get(variant)):
                # the system prompt carries the domain's grammar; keep it with
                # the row so the SFT builder does not have to recompute it
                row["system_prompt"] = system_prompt(domains.get(variant))
            rows.append(row)
        path = os.path.join(args.out, "%s.jsonl" % variant)
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        for k in keys:
            grand[k] += stats.get(k, 0)
        print("%-12s n=%-4d ok=%-4d ungrounded=%-3d banned=%-3d predictive=%-3d "
              "no_action=%-3d wrong_action=%-3d err=%-3d max_prompt=%d chars -> %s"
              % (variant, stats["n"], stats["ok"], stats["ungrounded"], stats["banned"],
                 stats["predictive"], stats["no_action"],
                 stats["wrong_action"], stats["err"],
                 max(len(b[1]) for b in batch), path), flush=True)

    n = max(grand["n"], 1)
    print("\nTOTAL n=%d accepted=%d (%.0f%%)  ungrounded=%.0f%%  banned=%.0f%%  "
          "predictive=%.0f%%  no_action=%.0f%%  wrong_action=%.0f%%  err=%d"
          % (grand["n"], grand["ok"], 100 * grand["ok"] / n,
             100 * grand["ungrounded"] / n, 100 * grand["banned"] / n,
             100 * grand["predictive"] / n,
             100 * grand["no_action"] / n, 100 * grand["wrong_action"] / n, grand["err"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
