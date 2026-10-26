"""Shared pieces of the own-action row pipeline.

Everything that two steps must agree on lives here: how a corpus is found, how
a state is rebuilt from (spec, action history), how the served prompt is split
into system / user / instruction, which seeds score a move, and how a reply's
<action> line is read. Ported from the sftprobe scratchpad (sftprobe_states,
up_tools/cand_phi, up_tools/own_phi, up_tools/build_own) and
dataxp.gold_probe (own_nothink mode), generalised from SET_PARAM to Move.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import List, Optional, Tuple

import numpy as np

from ..domain import (SET_PARAM, Move, PhiEvaluator, action_form, apply_action, current_spelling,
                      grammar_lead, speaks_verbs, stable_seed)
from ..domains import get_domain
from ..render_cot import ACTION_TAG_RE, _fmtv, _label_for, policy_observation

# ---------------------------------------------------------------------------
# The served task instruction (collaborator's instruction V2, DX-24 onward).
# This is the byte content of their task_instruction_V2.txt: it starts with two
# newlines (it is appended to the observation) and ends with one. Its sha1[:12]
# is the `instruction_sha1` every DX-26 / sftprobe tier row carries.
# ---------------------------------------------------------------------------
INSTRUCTION_V2 = (
    "\n\nYOUR MOVE\n"
    "Make exactly one change: set one parameter of one element to a new value. First reason, "
    "in 60-150 words and in any order: which constraints are violated and by how much, what the "
    "results above say about where the problem is, and why this change; close by naming the "
    "change and what it is aimed at. Use only what is shown above.\n"
    "Then end your reply with exactly this one line:\n"
    "<action>SET_PARAM(<element>, <parameter>, <value>)</action>\n"
    "<element> is the element's number as the problem defines it, written as a bare integer (the "
    "number in brackets or after \"element\", or as the problem says to write it, e.g. 5 for m5), "
    "never an id with letters. <parameter> is the parameter's name exactly as the problem writes "
    "it. <value> is the new absolute value, not a change, as a plain number with no units: 1 or 0 "
    "for an on/off switch, the bare code number for a categorical choice.\n"
)
INSTRUCTION_V2_SHA1 = "3d4867180337"
assert hashlib.sha1(INSTRUCTION_V2.encode()).hexdigest()[:12] == INSTRUCTION_V2_SHA1

# Native-verb domains had no served instruction before this package. This one
# is V2 with the SET_PARAM sentences replaced by the verb-call form; the
# domain's grammar() follows it the way the parameter line follows V2. A v2
# domain with a verb surface (docs/dataxp_verb_surface.md) is served this
# instruction too: its grammar() lists the verbs its SET_PARAM moves are
# spelled in.
INSTRUCTION_NATIVE = (
    "\n\nYOUR MOVE\n"
    "Make exactly one change: apply one of this problem's operations, with its arguments, to the "
    "current design. First reason, in 60-150 words and in any order: which constraints are "
    "violated and by how much, what the results above say about where the problem is, and why "
    "this change; close by naming the change and what it is aimed at. Use only what is shown "
    "above.\n"
    "Then end your reply with exactly this one line:\n"
    "<action>VERB(arg, ...)</action>\n"
    "VERB is one of the operations listed below, written exactly as listed, with every argument "
    "in the listed order and of the listed type: integers as bare integers, numbers as plain "
    "numbers with no units, names exactly as the problem writes them.\n"
)
INSTRUCTION_NATIVE_SHA1 = hashlib.sha1(INSTRUCTION_NATIVE.encode()).hexdigest()[:12]

# The set_parameter spelling (docs/dataxp_spellings.md): INSTRUCTION_NATIVE
# with its action line replaced by the wrapper-tool form and the sentence
# that explained VERB now explaining SET_PARAM (no served text says VERB
# under set_parameter, where no verb exists); nothing else changes, so the
# two served instructions differ in how the action is written only; the
# grammar that follows it is the domain's SET_PARAM operation list.
_NATIVE_ACTION_LINE = "<action>VERB(arg, ...)</action>"
_NATIVE_VERB_SENTENCE = "VERB is one of the operations listed below, written exactly as listed, "
assert _NATIVE_ACTION_LINE in INSTRUCTION_NATIVE and _NATIVE_VERB_SENTENCE in INSTRUCTION_NATIVE
INSTRUCTION_SET_PARAM = INSTRUCTION_NATIVE.replace(
    _NATIVE_ACTION_LINE, "<action>SET_PARAM(element, parameter, value)</action>").replace(
    _NATIVE_VERB_SENTENCE,
    "SET_PARAM performs one of the operations listed below, written exactly as listed, ")
assert "VERB" not in INSTRUCTION_SET_PARAM
INSTRUCTION_SET_PARAM_SHA1 = hashlib.sha1(INSTRUCTION_SET_PARAM.encode()).hexdigest()[:12]
assert INSTRUCTION_SET_PARAM_SHA1 != INSTRUCTION_NATIVE_SHA1

# gold_probe.SYSTEM_OWN in its own_nothink form (written reasoning, thinking off)
SYSTEM_OWN_NOTHINK = (
    "You are a design engineer working on a constrained design problem, one change per step. You "
    "see the full current observation. Write your reasoning first (under 250 words: what is "
    "violated, where the results place the problem, which change and why), then choose exactly "
    "ONE change that best moves the design toward meeting every constraint at low objective cost "
    "(feasibility first, then the objective). End your answer with exactly one line:\n"
    "<action>SET_PARAM(<element index>, <parameter>, <value>)</action>"
)
_OWN_ACTION_LINE = "<action>SET_PARAM(<element index>, <parameter>, <value>)</action>"
assert _OWN_ACTION_LINE in SYSTEM_OWN_NOTHINK
USER_TAIL_OWN = "\n\nChoose exactly ONE change for THIS step."


def is_native(domain) -> bool:
    """A native-verb domain (its moves are {verb, args})."""
    return bool(getattr(domain, "native_moves", False))


def verb_spelled(domain) -> bool:
    """Prompts, action lines and parsing go through format_move / parse_move /
    grammar: a native-verb domain or a v2 domain with a verb surface (whose
    stored actions stay {i, p, v, v_old})."""
    return speaks_verbs(domain)


def spelling_of(domain) -> str:
    """What every states.jsonl line and served row records: the process's
    DATAXP_SPELLING (multitool | set_parameter) for a domain that speaks
    verbs; 'SET_PARAM' for a plain v1/v2 domain, which has only the classic
    spelling and the V2 instruction."""
    return current_spelling() if speaks_verbs(domain) else "SET_PARAM"


def instruction_for(domain) -> Tuple[str, str]:
    """(instruction text appended to the observation, instruction_sha1).

    SET_PARAM domains: V2 + the parameter line (collaborator's --param-line).
    Native and verb-surface domains: INSTRUCTION_NATIVE (multitool) or
    INSTRUCTION_SET_PARAM (set_parameter) + the domain's grammar, as
    grammar() prints it under the process's spelling. The sha1 is that of
    the base text, as the collaborator hashed the instruction file and not
    the per-domain line."""
    if verb_spelled(domain):
        base, sha = ((INSTRUCTION_NATIVE, INSTRUCTION_NATIVE_SHA1)
                     if current_spelling() == "multitool"
                     else (INSTRUCTION_SET_PARAM, INSTRUCTION_SET_PARAM_SHA1))
        g = domain.grammar().rstrip()
        return base + "The operations in this problem are exactly:\n" + g + "\n", sha
    return (INSTRUCTION_V2 + "The parameter names in this problem are exactly: %s.\n"
            % ", ".join(domain.params), INSTRUCTION_V2_SHA1)


def own_system_prompt(domain) -> str:
    """gold_probe's own_nothink system prompt; on a native or verb-surface
    domain its action line is replaced by the spelling's action form
    (VERB(arg, ...) or SET_PARAM(element, parameter, value)) plus grammar(),
    the substitution render_cot.system_prompt makes on the renderer's SYSTEM."""
    if not verb_spelled(domain):
        return SYSTEM_OWN_NOTHINK
    g = "\n".join("    " + l for l in domain.grammar().rstrip().splitlines())
    return SYSTEM_OWN_NOTHINK.replace(_OWN_ACTION_LINE, (
        "<action>%s</action>\n%s\n" % (action_form(domain), grammar_lead()) + g))


# ---------------------------------------------------------------------------
# corpus and states
# ---------------------------------------------------------------------------
def corpus_dir(data: str, variant: str, tmp: Optional[str] = None) -> str:
    """The directory holding problems.jsonl + trajectories.jsonl for a variant:
    <data>/<variant>/accepted (a gated run_generation output) or, when the
    gates did not pass, the scratch run under --tmp (<tmp>/<variant>)."""
    acc = os.path.join(data, variant, "accepted")
    if os.path.isfile(os.path.join(acc, "trajectories.jsonl")):
        return acc
    if tmp:
        scratch = os.path.join(tmp, variant)
        if os.path.isfile(os.path.join(scratch, "trajectories.jsonl")):
            return scratch
    raise FileNotFoundError("no trajectories.jsonl under %s (accepted) %s"
                            % (acc, ("or " + os.path.join(tmp, variant)) if tmp else ""))


def read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows, mode: str = "w") -> None:
    with open(path, mode, encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def load_corpus(data: str, variant: str, tmp: Optional[str] = None):
    d = corpus_dir(data, variant, tmp)
    specs = {s["problem_id"]: s for s in read_jsonl(os.path.join(d, "problems.jsonl"))}
    trajs = read_jsonl(os.path.join(d, "trajectories.jsonl"))
    return d, specs, trajs


def sid_of(variant: str, pid: str, turn: int) -> str:
    return "%s|%s|%d" % (variant, pid, int(turn))


def rebuild_state(domain, spec: dict, history: List[dict]):
    """The state after replaying `history` (serialised moves) on load(spec)."""
    st = domain.load(spec)
    for a in history:
        apply_action(domain, st, a)
    return st


def labelled_history(domain, spec: dict, history: List[dict]):
    """(state, [(label, action), ...]) with each action labelled AFTER it is
    applied, as render_cot.iter_states does for the observation's history."""
    st = domain.load(spec)
    out = []
    for a in history:
        apply_action(domain, st, a)
        out.append((_label_for(domain, st, a), a))
    return st, out


def observation_parts(domain, spec, st, turn: dict, hist_labelled, variant) -> Tuple[str, str]:
    """(system, observation): render_cot.policy_observation split the way the
    collaborator's observation_v2 splits it - system = the DOMAIN and GOAL
    head, observation = the rest (the served user text is observation +
    instruction)."""
    lines, _ = policy_observation(domain, spec, st, turn, hist_labelled, variant)
    head, sep, rest = "\n".join(lines).partition("\n\n")
    if not (head.startswith("DOMAIN:") and sep and rest.strip()):
        raise RuntimeError("observation layout not recognised for %s" % spec.get("problem_id"))
    return head, rest.strip()


def served_messages(state: dict) -> List[dict]:
    """The system and user messages of a training row for this state."""
    return [{"role": "system", "content": state["system"]},
            {"role": "user", "content": state["observation"] + state["instruction"]}]


def probe_prompt(domain, state: dict) -> Tuple[str, str]:
    """gold_probe own_nothink: (system, user) with user = the full observation
    (system head + body, as the sftprobe states stored it) + the closing line."""
    obs = state["system"] + "\n\n" + state["observation"]
    return own_system_prompt(domain), obs + USER_TAIL_OWN


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def legal_rng(pid: str, k: int) -> np.random.Generator:
    """The rng gold_probe.prepare / cand_phi draw the legal set with."""
    return np.random.default_rng(stable_seed(pid, "gp", int(k)))


def move_seed(pid: str, domain, move: Move) -> int:
    """Per-move Phi seed: gold_probe's (pid, 'gp', i, p) for SET_PARAM; the
    formatted move for a native verb (probe_info does the same with its own
    tag). Deterministic domains ignore it."""
    if move.is_set_param:
        d = move.as_dict()
        return stable_seed(pid, "gp", d["i"], d["p"])
    return stable_seed(pid, "gp", domain.format_move(move))


def evaluator(domain, spec: dict, offset_rule: str = "infeasible") -> PhiEvaluator:
    return PhiEvaluator(domain, spec, offset_rule=offset_rule)


def percentile(cand: np.ndarray, phi: float) -> Optional[float]:
    """Share of enumerated legal moves whose Phi is <= phi, in percent (the
    rule of up_tools/own_phi.py, whose output build_own consumed)."""
    if cand.size == 0:
        return None
    return float((cand <= phi).mean() * 100.0)


# ---------------------------------------------------------------------------
# reading a reply
# ---------------------------------------------------------------------------
def action_text(answer: str) -> Optional[str]:
    """Inside of the LAST <action>...</action> in the reply, or None."""
    m = ACTION_TAG_RE.findall(answer or "")
    return m[-1].strip() if m else None


def reasoning_of(answer: str) -> str:
    """The reply before its last <action> (build_own: answer.split('<action>')[0])."""
    a = answer or ""
    i = a.rfind("<action>")
    return (a if i < 0 else a[:i]).strip()


def parse_reply(domain, answer: str) -> Tuple[Optional[Move], Optional[str]]:
    """(move, reason): the move named in the reply's last <action> line, or
    (None, 'no_action' | 'unparsed')."""
    txt = action_text(answer)
    if txt is None:
        return None, "no_action"
    mv = domain.parse_move(txt)
    if mv is None:
        return None, "unparsed"
    return mv, None


def try_apply(domain, st, move: Move):
    """Apply `move` on a clone of st. Returns (child, move_applied, info) or
    (None, None, info) when the state does not admit it. For SET_PARAM the
    applied value is read back (the domain may clamp it): move_applied carries
    that value and v_old, info['clamped'] / info['noop'] say what happened."""
    info = {"clamped": False, "noop": False, "err": None}
    if move.is_set_param:
        i, p = int(move.args["i"]), move.args["p"]
        if p not in domain.params:
            info["err"] = "unknown_parameter"
            return None, None, info
        try:
            cur = domain.get(st, i, p)
        except Exception as e:  # out-of-range element in some domains raises
            info["err"] = "element_out_of_range:%s" % str(e)[:60]
            return None, None, info
        if cur is None or not np.isfinite(float(cur)):
            info["err"] = "element_out_of_range"
            return None, None, info
        v = float(move.args["v"])
        if not np.isfinite(v):
            info["err"] = "bad_number"
            return None, None, info
        child = domain.clone(st)
        try:
            domain.apply(child, Move(SET_PARAM, {"i": i, "p": p, "v": v, "v_old": float(cur)}))
        except Exception as e:
            info["err"] = "illegal:%s" % str(e)[:80]
            return None, None, info
        v_new = float(domain.get(child, i, p))
        info["clamped"] = abs(v_new - v) > 1e-9 * max(1.0, abs(v))
        info["noop"] = abs(v_new - float(cur)) <= 1e-12 * max(1.0, abs(float(cur)))
        return child, Move(SET_PARAM, {"i": i, "p": p, "v": v_new, "v_old": float(cur)}), info
    child = domain.clone(st)
    try:
        domain.apply(child, move)
    except Exception as e:
        info["err"] = "illegal:%s" % str(e)[:80]
        return None, None, info
    return child, move, info


def fmt_move(domain, move: Move) -> str:
    return domain.format_move(move)


def domain_of(variant: str):
    return get_domain(variant)


__all__ = [
    "INSTRUCTION_V2", "INSTRUCTION_V2_SHA1", "INSTRUCTION_NATIVE", "INSTRUCTION_NATIVE_SHA1",
    "INSTRUCTION_SET_PARAM", "INSTRUCTION_SET_PARAM_SHA1",
    "SYSTEM_OWN_NOTHINK", "USER_TAIL_OWN", "is_native", "verb_spelled", "spelling_of",
    "instruction_for", "own_system_prompt",
    "corpus_dir", "read_jsonl", "write_jsonl", "load_corpus", "sid_of", "rebuild_state",
    "labelled_history", "observation_parts", "served_messages", "probe_prompt", "legal_rng",
    "move_seed", "evaluator", "percentile", "action_text", "reasoning_of", "parse_reply",
    "try_apply", "fmt_move", "domain_of", "Move", "_fmtv",
]
