"""Domain contract for multi-domain data generation, plus the Phi evaluator.

The contract extends analysis/remote/code_new/llm_finetune/design_agent/
da_domain.py with what generation needs: a problem GENERATOR, a DesignProgram
factory, a reference optimum, and seeded evaluation for stochastic variants.

Everything above this file (candidate enumeration, ladder policies, gates,
harvest) touches only this interface. A new domain is one subclass.

Moves (docs/dataxp_move_contract.md): every action is a ``Move(verb, args)``
in the domain's own vocabulary. The v1/v2 ``Candidate(i, p, v, v_old)`` is
the move ``SET_PARAM`` and keeps its serialised form, so run1/run2 replay bit
for bit; native-verb domains set ``native_moves = True`` and override
``legal_moves`` / ``apply`` / ``format_move`` / ``parse_move`` / ``grammar``.

Verb surfaces (docs/dataxp_verb_surface.md): a v2 domain keeps SET_PARAM as
its move, its enumeration, ``set`` and the stored ``{i, p, v, v_old}`` dict,
and sets ``verb_surface = True`` with a ``verb_table`` of ``ParamVerb`` rows;
``format_move`` / ``parse_move`` / ``grammar`` then spell every SET_PARAM move
in the field's own verbs (``INSTALL_MEMBER(3)``, ``SET_MATERIAL(3, Ti6Al4V)``,
``MOVE_JOINT(12, x, 0.25)``) and read them back to the same SET_PARAM move.
Only the prompt-facing text changes.

Set-parameter spelling (docs/dataxp_spellings.md; owner's decision of
2026-10-23, "two spellings"): the SAME legal moves can be written in a second
way, through one wrapper tool ``SET_PARAM(element, parameter, value)`` whose
grammar documents every operation of the verb table operation by operation
("Install a member: SET_PARAM(member, active, 1)", "Set a member's material:
SET_PARAM(member, mat, code)  code: 0 = Al6061T6, 1 = S355, ..."). The
spelling is chosen per process by the environment variable DATAXP_SPELLING
(``multitool`` = the verbs as built, the default; ``set_parameter``), read
once by ``current_spelling()``. Only ``format_move`` / ``parse_move`` /
``grammar`` (and ``format_value``) look at it, through the ``set_param_*``
hooks; ``legal_moves``, ``apply``, Phi and the stored ``{i, p, v, v_old}`` /
``{verb, args}`` dicts never do, so one corpus serves both spellings.

Judgment calls of the set_parameter spelling (recorded here because the
spelling is generated, not hand-written per domain):

* the action text of a verb-surface domain is the classic v2 line
  ``SET_PARAM(i, p, v)`` byte for byte (element index as SET_PARAM used it,
  the parameter name, the value as stored: 6 significant digits, a
  categorical choice as its stored code, a toggle as 1 / 0), and
  ``parse_move`` reads back exactly that line, as the v2 harness did: a
  category NAME in the value slot is not accepted (the grammar names the
  codes), nor is any verb;
* a parameter that shares a verb through ``select`` (dx / dy / dz) is its own
  operation block, in the verb table's order, since SET_PARAM names it
  directly; a toggle is two blocks (value 1, value 0);
* ``format_value`` (the "was ..." of history lines) prints the stored code of
  a categorical parameter under this spelling, '' for a toggle as before;
* the operation phrase ("Install a member") is ``ParamVerb.phrase`` /
  ``Toggle.off_phrase``, defaulting to the verb name in words;
* the observation's "how to write a change" paragraph of a verb-surface
  domain follows ``spelled_in_verbs``: the verb paragraph under multitool,
  the historical SET_PARAM paragraph (the one the run2 data were generated
  with) under set_parameter, so the two prompts differ in action syntax only;
* a native-verb domain gets the generic encoding
  ``SET_PARAM(element, operation, value)`` from its ``verbs`` table (element =
  the verb's first int argument, 0 when it has none; operation = the verb in
  lower case; value = the one remaining argument, a JSON list of them in
  order when there are several, 0 when there are none); a native domain that
  overrides ``format_move`` / ``parse_move`` without the matching
  ``set_param_*`` hook fails at construction under this spelling instead of
  silently serving verbs (``check_native_set_param``);
* the observation of a native domain names its operations through
  ``Domain.op_name`` (``CURTAIL`` under multitool, ``curtail`` under
  set_parameter) and shows how one is written through ``Domain.call_form``
  (``CURTAIL(g, frac)`` / ``SET_PARAM(g, curtail, frac)``, the head of that
  operation's grammar block), so the describe() paragraph that teaches the
  call syntax agrees with the grammar in either spelling and the multitool
  text stays byte for byte what it was; a grammar() wrapper (molopt,
  engibench) re-reads its own prose the same way, and a verb named inside a
  table doc ("comes back only through RECONNECT_LINE") prints as its
  operation name in the generated grammar (``_name_ops``);
* a value of several arguments (grid2op set_bus / redispatch, the rectangle
  and disk operations of photonics and engibench) is written as a JSON list,
  so a name inside it is double-quoted (``SET_PARAM(4, set_bus, ["line3_ex",
  2])``); the native preamble says so (``SET_PARAM_NATIVE_PREAMBLE``) and
  each such operation's block head ends with the list it expects and which
  members are names ("value: JSON list [element, bus] in this order, element
  in double quotes", ``_native_list_note``). The parser also accepts the list
  as the block head literally shows it, with bare identifiers
  (``[line3_ex, 2]``; single quotes pass too), splitting it on its top-level
  commas and coercing each item by its ArgSpec.type exactly as a multitool
  call's arguments are (``_parse_list_value``), so a reply that copies the
  block head is not lost in one condition and kept in the other; the
  multitool parse is untouched. format_move still writes JSON only, so the
  bijection on the legal moves stands (one canonical text per move; the
  bare form is a parse-side alias, as a Categorical alias is under multitool);
* under set_parameter no served text says VERB: the system prompts' line
  between the action placeholder and the grammar block is ``grammar_lead()``
  ("where the operations, with their argument types and ranges, are listed
  here:" instead of "where VERB is one of this domain's verbs, ..."), the
  ownrows instruction says "SET_PARAM performs one of the operations listed
  below, written exactly as listed, ..." and probe_info's tail says
  "performing one of these operations:"; the multitool texts are byte for
  byte what they were (the lead is the same string).

Compatibility: Python >= 3.8, numpy/scipy only.
"""

from __future__ import annotations

import json
import math
import os
import re
import zlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .potential import (
    DesignProgram,
    aggregate_metrics,
    auto_feasibility_offset,
    compute_potential_v2,
)


def stable_seed(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary string/int parts (crc32, not
    Python hash(), which is salted per process)."""
    text = "|".join(str(p) for p in parts)
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


# ── spelling: how moves are written in prompts and action lines ──────────────
#
# Two spellings of the SAME legal moves (docs/dataxp_spellings.md). The
# spelling is read once per process from DATAXP_SPELLING; every prompt-facing
# path (render_cot, ownrows, probe_info, gold_probe) goes through format_move /
# parse_move / grammar, which honour it, and records which one it served.

SPELLING_ENV = "DATAXP_SPELLING"
SPELLINGS = ("multitool", "set_parameter")
VERB_FORM = "VERB(arg, ...)"                               # the multitool action placeholder
SET_PARAM_FORM = "SET_PARAM(element, parameter, value)"   # the set_parameter one
SET_PARAM_PREAMBLE = ("The only tool is SET_PARAM(element, parameter, value). The operations "
                      "it performs in this problem:")
# the native four: `parameter` names the operation, and a value of several
# arguments is a JSON list (so a name inside it is double-quoted)
SET_PARAM_NATIVE_PREAMBLE = (
    "The only tool is SET_PARAM(element, parameter, value); here `parameter` names the "
    "operation, and when an operation takes several arguments after the element, `value` is a "
    "JSON list of them in the listed order, with names in double quotes. The operations it "
    "performs in this problem:")
# the line between the action placeholder and the indented grammar block in
# every served system prompt (render_cot, ownrows, gold_probe)
VERB_LEAD = ("where VERB is one of this domain's verbs, with the argument types and ranges "
             "listed here:")
SET_PARAM_LEAD = "where the operations, with their argument types and ranges, are listed here:"
_SPELLING: List[str] = []


def current_spelling() -> str:
    """The spelling this process writes and reads moves in: DATAXP_SPELLING
    (``multitool`` = default, ``set_parameter``), read once on first use."""
    if not _SPELLING:
        raw = os.environ.get(SPELLING_ENV, "").strip()
        s = raw.lower() or "multitool"
        if s not in SPELLINGS:
            raise ValueError("%s=%r: expected one of %s" % (SPELLING_ENV, raw, SPELLINGS))
        _SPELLING.append(s)
    return _SPELLING[0]


def speaks_verbs(domain) -> bool:
    """True when prompts, action lines and parsing go through format_move /
    parse_move / grammar: a native-verb domain or a verb-surface one. True
    under BOTH spellings (the methods honour the spelling themselves)."""
    return bool(getattr(domain, "native_moves", False)
                or getattr(domain, "verb_surface", False))


def spelled_in_verbs(domain) -> bool:
    """True when the domain's moves are written as its field verbs: a
    speaks_verbs domain under the multitool spelling. False under
    set_parameter (its observation then describes the SET_PARAM form) and for
    plain v1/v2 domains."""
    return speaks_verbs(domain) and current_spelling() == "multitool"


def action_form(domain) -> str:
    """The action-line placeholder a speaks_verbs domain is served:
    ``VERB(arg, ...)`` under multitool, ``SET_PARAM(element, parameter,
    value)`` under set_parameter."""
    return VERB_FORM if current_spelling() == "multitool" else SET_PARAM_FORM


def grammar_lead() -> str:
    """The line that introduces the grammar block after the action placeholder
    in a served system prompt: ``VERB_LEAD`` under multitool, ``SET_PARAM_LEAD``
    under set_parameter (where no verb is served, so no text may say VERB)."""
    return VERB_LEAD if current_spelling() == "multitool" else SET_PARAM_LEAD


class Domain:
    """Minimal contract a generation domain must satisfy.

    States are opaque dicts owned by the domain; metrics dicts are flat
    {str: float}. ``evaluate(st, seed)`` MUST be deterministic in (st, seed);
    a deterministic domain ignores ``seed`` entirely.
    """

    name = "abstract"
    params: tuple = ()            # tunable per-element parameter names
    factors: tuple = (0.8, 0.9, 1.1, 1.25)   # default scaling moves
    stochastic = False            # True => evaluate() varies with seed
    reference_class = "heuristic"  # "exact" | "heuristic" | "bound"

    def __init__(self, stochastic: bool = False):
        self.stochastic = bool(stochastic)
        self.calls = 0
        if self.verb_surface:
            check_verb_table(self)
        if self.native_moves and current_spelling() == "set_parameter":
            check_native_set_param(self)

    # -- problems ------------------------------------------------------------
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        """Return a JSON-serialisable problem spec. Must contain at least
        problem_id, family, goals (minimum_/maximum_ convention). The initial
        design should usually be INFEASIBLE (that is where the data is)."""
        raise NotImplementedError

    def load(self, spec: dict):
        raise NotImplementedError

    def clone(self, st):
        raise NotImplementedError

    # -- simulation ----------------------------------------------------------
    def evaluate(self, st, seed: int = 0) -> dict:
        raise NotImplementedError

    # -- design variables ----------------------------------------------------
    def n_elements(self, st) -> int:
        raise NotImplementedError

    def get(self, st, i: int, p: str) -> Optional[float]:
        raise NotImplementedError

    def set(self, st, i: int, p: str, v: float) -> None:
        raise NotImplementedError

    def bounds(self, st, p: str) -> Tuple[float, float]:
        raise NotImplementedError

    # -- program / reference ---------------------------------------------------
    def program(self, spec: dict) -> DesignProgram:
        """DesignProgram incl. objective and VALIDITY BOUNDS for this problem.
        Do not call with_violation_scale here; the evaluator does that."""
        raise NotImplementedError

    def reference(self, spec: dict) -> Optional[float]:
        """Reference optimum for obj/ref reporting (class in reference_class)."""
        return None

    # -- optional depth-0 feedback hint ---------------------------------------
    def hint(self, st) -> Optional[Tuple[int, str, str]]:
        """(element, param, 'up'|'down') the domain's cheap diagnostic blames,
        or None. This is the greedy policy's move and the anchor of the
        coupling statistic (feedback-sufficiency)."""
        return None

    # -- moves: the domain's own verbs -----------------------------------------
    # A native-verb domain sets native_moves = True and overrides legal_moves /
    # apply / format_move / parse_move / grammar (the last three come for free
    # when it fills the `verbs` table). v1/v2 domains keep these defaults,
    # which wrap the Candidate path without changing it.
    native_moves = False
    verbs: tuple = ()             # Tuple[VerbSpec, ...]; optional
    # A v2 domain that spells its SET_PARAM moves in the field's own verbs
    # (docs/dataxp_verb_surface.md). The move set, apply, Phi and the stored
    # {i, p, v, v_old} dict do not change; format_move / parse_move / grammar
    # come from verb_table (one ParamVerb per parameter), and every
    # prompt-facing path treats (native_moves or verb_surface) as "speak in
    # verbs". A domain overrides the three methods only for what the table
    # cannot express.
    verb_surface = False
    verb_table: tuple = ()        # Tuple[ParamVerb, ...]

    def legal_moves(self, st, rng: np.random.Generator,
                    max_candidates: Optional[int] = None) -> "List[Move]":
        """Legal move set at st, deterministic in (st, rng). Default: the
        Candidate enumeration as SET_PARAM moves (same rng consumption, same
        order). Overrides must honour max_candidates (phi_d2 asks for 24;
        subsample_moves does the stable subsample)."""
        return [Move.from_candidate(c) for c in
                enumerate_candidates(self, st, rng, max_candidates=max_candidates)]

    @staticmethod
    def subsample_moves(moves: "List[Move]", rng: np.random.Generator,
                        max_candidates: Optional[int]) -> "List[Move]":
        """The subsample rule enumerate_candidates uses: a seeded choice of
        max_candidates indices, kept in their original order."""
        moves = list(moves)
        if max_candidates is not None and len(moves) > int(max_candidates):
            keep = rng.choice(len(moves), size=int(max_candidates), replace=False)
            moves = [moves[k] for k in sorted(keep.tolist())]
        return moves

    def apply(self, st, move: "Move") -> None:
        """Apply move to st IN PLACE. Default: SET_PARAM -> self.set with the
        values exactly as the Candidate held them. Any other verb is an error
        here; native domains override and raise ValueError for a move the
        state does not admit."""
        if move.verb == SET_PARAM:
            a = move.args
            self.set(st, a["i"], a["p"], a["v"])
            return
        raise ValueError("domain %s has no verb %r" % (self.name, move.verb))

    def verb_spec(self, verb: str) -> "Optional[VerbSpec]":
        for vs in self.verbs:
            if vs.name == verb:
                return vs
        return None

    def verb_param(self, p: str) -> "Optional[ParamVerb]":
        """The verb_table row spelling parameter p (verb-surface domains)."""
        for pv in self.verb_table:
            if pv.param == p:
                return pv
        return None

    def format_value(self, p: str, v) -> str:
        """The value of parameter p as its verb spells it: the name of a
        categorical code, the number of a continuous value, '' for a toggle
        (the verb pair carries it). Used for the 'was ...' part of history
        lines on verb-surface domains; plain fmt_arg elsewhere."""
        pv = self.verb_param(p) if self.verb_surface else None
        if pv is None or v is None:
            return fmt_arg(v) if v is not None else ""
        if current_spelling() == "set_parameter":
            # the stored code / number, as the SET_PARAM line writes it
            return "" if isinstance(pv.codec, Toggle) else fmt_arg(float(v))
        return surface_value(pv, v)

    def format_move(self, move: "Move") -> str:
        """'VERB(a, b, ...)': the text inside <action>...</action>. Under the
        set_parameter spelling a speaks_verbs domain writes
        SET_PARAM(element, parameter, value) through set_param_format."""
        if current_spelling() == "set_parameter" and speaks_verbs(self):
            return self.set_param_format(move)
        if move.verb == SET_PARAM:
            if self.verb_surface:
                return surface_format(self, move)
            a = move.args
            return "SET_PARAM(%d, %s, %s)" % (int(a["i"]), a["p"], fmt_arg(a["v"]))
        vs = self.verb_spec(move.verb)
        names = [x.name for x in vs.args] if vs is not None else list(move.args.keys())
        return "%s(%s)" % (move.verb, ", ".join(fmt_arg(move.args[n]) for n in names))

    def parse_move(self, text: str) -> "Optional[Move]":
        """First balanced VERB(...) call in text whose verb this domain knows
        (SET_PARAM for non-native domains, the `verbs` table otherwise, the
        verb_table spellings for a verb-surface domain, whose grammar no
        longer lists SET_PARAM), with arguments coerced by type; None when
        there is none. Under the set_parameter spelling a speaks_verbs domain
        accepts SET_PARAM(...) calls only, through set_param_parse."""
        if current_spelling() == "set_parameter" and speaks_verbs(self):
            for verb, toks in parse_verb_calls(text):
                mv = self.set_param_parse(verb, toks)
                if mv is not None:
                    return mv
            return None
        for verb, toks in parse_verb_calls(text):
            if self.verb_surface:
                mv = surface_parse(self, verb, toks)
                if mv is not None:
                    return mv
            if verb == SET_PARAM and not self.native_moves and not self.verb_surface:
                mv = parse_classic_set_param(toks)
                if mv is None:
                    continue
                return mv
            vs = self.verb_spec(verb)
            if vs is None or len(toks) != len(vs.args):
                continue
            try:
                args = {a.name: coerce_arg(t, a.type) for a, t in zip(vs.args, toks)}
            except ValueError:
                continue
            return Move(verb, args)
        return None

    # -- time-stepping domains ---------------------------------------------------
    # A domain whose moves advance simulated time (grid2op: one 5-minute step
    # per move) sets advances_time = True and implements null_move(st): the
    # move that lets the step pass with no operator action. When no candidate
    # improves on the incumbent, the ladder applies it so the world advances
    # instead of the same state being scored and recorded again; harvest then
    # drops exact duplicate pairs by (problem, time_key, chosen, rejected).
    # v1/v2 domains keep the defaults and their path does not change
    # (docs/corpus_native_domains.md, "Time-stepping domains").
    advances_time = False
    time_key = "step"             # the metric that carries simulated time

    def null_move(self, st) -> "Optional[Move]":
        """The move that advances time without acting (None: no such move)."""
        return None

    def grammar(self) -> str:
        """Verb list with argument types and ranges, for the system prompt.
        Under the set_parameter spelling a speaks_verbs domain returns the
        SET_PARAM operation list instead (set_param_grammar)."""
        if current_spelling() == "set_parameter" and speaks_verbs(self):
            return self.set_param_grammar()
        if self.verb_surface and self.verb_table:
            return surface_grammar(self)
        if self.verbs:
            lines = []
            for vs in self.verbs:
                sig = ", ".join("%s: %s%s" % (a.name, a.type, (" " + a.doc) if a.doc else "")
                                for a in vs.args)
                lines.append("%s(%s)" % (vs.name, sig))
                if vs.doc:
                    lines.append("    " + vs.doc)
            return "\n".join(lines)
        params = ", ".join(self.params) if self.params else "<parameter>"
        return ("SET_PARAM(element: int 0..n_elements-1, parameter: one of {%s}, "
                "value: float within bounds(parameter))\n"
                "    set one parameter of one element (the only verb of this domain)"
                % params)

    # -- the set_parameter spelling (docs/dataxp_spellings.md) --------------------
    # The hooks format_move / parse_move / grammar dispatch to under
    # DATAXP_SPELLING=set_parameter. Defaults: a verb-surface domain writes
    # the classic SET_PARAM(i, p, v) line and documents its verb_table
    # operation by operation; a native domain gets the generic encoding
    # SET_PARAM(element, operation, value) from its `verbs` table. Override
    # these three (never format_move / parse_move / grammar themselves) for
    # what the tables cannot express, keep the bijection, and record why in
    # the module docstring section "Set-parameter spelling".
    def set_param_format(self, move: "Move") -> str:
        """The SET_PARAM(...) text of a move under the set_parameter spelling."""
        if move.verb == SET_PARAM:
            a = move.args
            return "SET_PARAM(%d, %s, %s)" % (int(a["i"]), a["p"], fmt_arg(a["v"]))
        return native_set_param_format(self, move)

    def set_param_parse(self, verb: str, toks: "List[str]") -> "Optional[Move]":
        """One VERB(...) call (as parse_verb_calls splits it) -> the move it
        spells under the set_parameter spelling, or None. Only SET_PARAM
        calls are admitted."""
        if verb != SET_PARAM or len(toks) != 3:
            return None
        if self.native_moves:
            return native_set_param_parse(self, toks)
        return parse_classic_set_param(toks)

    def set_param_grammar(self) -> str:
        """The operation list of the set_parameter spelling: a preamble line,
        then one block per operation in the table's order."""
        if self.verb_surface and self.verb_table:
            return surface_set_param_grammar(self)
        if self.native_moves and self.verbs:
            return native_set_param_grammar(self)
        raise NotImplementedError("domain %s has no table to derive its set_parameter grammar "
                                  "from: override set_param_grammar (docs/dataxp_spellings.md)"
                                  % self.name)

    # -- spelling-aware operation names for describe() and grammar wrappers ------
    def op_name(self, verb: str) -> str:
        """A `verbs`-table operation named as the served texts spell it: the
        verb itself under multitool (and for a domain that does not speak
        verbs), its lower-case operation name under set_parameter
        (``CURTAIL`` -> ``curtail``), so a describe() sentence that names an
        operation agrees with the grammar in either spelling."""
        if current_spelling() == "set_parameter" and speaks_verbs(self):
            return verb.lower()
        return verb

    def call_form(self, verb: str) -> str:
        """How a `verbs`-table verb is written, as the served texts show it:
        ``VERB(a, b, ...)`` with the table's argument names under multitool,
        the head of its set_parameter grammar block (``SET_PARAM(g, curtail,
        frac)``) under that spelling. For the describe() paragraph that
        teaches the call syntax."""
        vs = self.verb_spec(verb)
        if vs is None:
            raise ValueError("domain %s: verb %r is not in its verbs table" % (self.name, verb))
        if current_spelling() == "set_parameter" and speaks_verbs(self):
            return native_set_param_call(vs)
        return "%s(%s)" % (vs.name, ", ".join(a.name for a in vs.args))


@dataclass
class Candidate:
    i: int
    p: str
    v: float
    v_old: float

    def as_dict(self) -> dict:
        return {"i": int(self.i), "p": self.p, "v": float(self.v),
                "v_old": float(self.v_old)}


def enumerate_candidates(domain: Domain, st, rng: np.random.Generator,
                         max_elements: Optional[int] = None,
                         max_candidates: Optional[int] = None) -> List[Candidate]:
    """Legal move set.

    v1 domains (single continuous knob per element): per (element, param),
    each scale factor within bounds; elements subsampled when too many. This
    path is unchanged so run1 replays bit for bit.

    v2 domains (typed moves) may override either hook:
      candidate_moves(st, rng) -> List[Candidate]   full control of the set
      candidate_values(st, i, p, cur) -> [values]   per-(element, param)
        values: scaled for continuous knobs, shifted for additive/circular
        ones, every other option for categorical ones, the flip for toggles.
    and may set class attributes max_elements / max_candidates."""
    if max_elements is None:
        max_elements = int(getattr(domain, "max_elements", 24))
    if max_candidates is None:
        max_candidates = int(getattr(domain, "max_candidates", 160))
    custom = getattr(domain, "candidate_moves", None)
    if callable(custom):
        cands = list(custom(st, rng))
        if len(cands) > max_candidates:
            keep = rng.choice(len(cands), size=max_candidates, replace=False)
            cands = [cands[k] for k in sorted(keep.tolist())]
        return cands
    values_fn = getattr(domain, "candidate_values", None)
    n = domain.n_elements(st)
    idx = list(range(n))
    if n > max_elements:
        idx = sorted(rng.choice(n, size=max_elements, replace=False).tolist())
    cands: List[Candidate] = []
    for i in idx:
        for p in domain.params:
            cur = domain.get(st, i, p)
            if cur is None or not math.isfinite(cur):
                continue
            if callable(values_fn):
                for v in values_fn(st, i, p, cur):
                    v = float(v)
                    if math.isfinite(v) and abs(v - cur) > 1e-12 * max(1.0, abs(cur)):
                        cands.append(Candidate(i=i, p=p, v=v, v_old=float(cur)))
                continue
            lo, hi = domain.bounds(st, p)
            for f in domain.factors:
                v = min(hi, max(lo, cur * f))
                if abs(v - cur) > 1e-12 * max(1.0, abs(cur)):
                    cands.append(Candidate(i=i, p=p, v=float(v), v_old=float(cur)))
    if len(cands) > max_candidates:
        keep = rng.choice(len(cands), size=max_candidates, replace=False)
        cands = [cands[k] for k in sorted(keep.tolist())]
    return cands


# ── moves: the domain's own verbs ────────────────────────────────────────────
#
# A Move is one action in the field's own language: a verb and typed
# arguments. The v1/v2 path is a special case - Candidate(i, p, v, v_old) is
# the Move SET_PARAM(i, p, v) - and it serialises to exactly the Candidate dict
# (keys i, p, v, v_old), so run1/run2 files are unchanged. Native-verb domains
# set native_moves = True and override legal_moves / apply / format_move /
# parse_move / grammar (or fill the `verbs` table and keep the defaults for
# the last three). See docs/dataxp_move_contract.md.

SET_PARAM = "SET_PARAM"


def _plain(v):
    """JSON-friendly copy of an argument value (numpy scalars -> python)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return v


def _is_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


@dataclass
class Move:
    verb: str
    args: dict = field(default_factory=dict)

    # -- serialisation ----------------------------------------------------------
    def as_dict(self) -> dict:
        """SET_PARAM -> {i, p, v, v_old} (byte-identical to Candidate.as_dict);
        any other verb -> {"verb": ..., "args": {...}}."""
        if self.verb == SET_PARAM:
            a = self.args
            out = {"i": int(a["i"]), "p": a["p"], "v": float(a["v"])}
            if a.get("v_old") is not None:
                out["v_old"] = float(a["v_old"])
            return out
        return {"verb": self.verb, "args": {str(k): _plain(v) for k, v in self.args.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Move":
        """Inverse of as_dict; also reads the legacy {i, p, v, v_old} form
        (run1/run2 actions, pairs and trajectories)."""
        if "verb" in d:
            return cls(verb=str(d["verb"]), args=dict(d.get("args") or {}))
        args = {"i": int(d["i"]), "p": d["p"], "v": float(d["v"]),
                "v_old": (float(d["v_old"]) if d.get("v_old") is not None else None)}
        return cls(verb=SET_PARAM, args=args)

    # -- Candidate adapter ----------------------------------------------------
    @classmethod
    def from_candidate(cls, c: Candidate) -> "Move":
        # values are kept as the Candidate holds them (no int()/float()
        # coercion) so Domain.apply hands set() exactly what the ladder did
        return cls(verb=SET_PARAM, args={"i": c.i, "p": c.p, "v": c.v, "v_old": c.v_old})

    def to_candidate(self) -> Candidate:
        if self.verb != SET_PARAM:
            raise ValueError("not a SET_PARAM move: %s" % self.verb)
        a = self.args
        v_old = a.get("v_old")
        return Candidate(i=a["i"], p=a["p"], v=a["v"],
                         v_old=(v_old if v_old is not None else float("nan")))

    @property
    def is_set_param(self) -> bool:
        return self.verb == SET_PARAM

    # -- comparison -------------------------------------------------------------
    def matches(self, other: "Move", rel: float = 1e-9) -> bool:
        """Same verb and same arguments: numbers within rel (relative to
        max(1, |other|)), everything else by equality. v_old is ignored (a
        parsed move does not know it)."""
        if other is None or self.verb != other.verb:
            return False
        keys = (set(self.args) | set(other.args)) - {"v_old"}
        for k in keys:
            if k not in self.args or k not in other.args:
                return False
            a, b = self.args[k], other.args[k]
            if _is_number(a) and _is_number(b):
                if abs(float(a) - float(b)) > rel * max(1.0, abs(float(b))):
                    return False
            elif _plain(a) != _plain(b):
                return False
        return True


def candidate_to_move(c: Candidate) -> Move:
    return Move.from_candidate(c)


def move_to_candidate(m: Move) -> Candidate:
    return m.to_candidate()


def as_move(x) -> Optional[Move]:
    """None / Candidate / Move -> Move (hint_move may return either)."""
    if x is None or isinstance(x, Move):
        return x
    if isinstance(x, Candidate):
        return Move.from_candidate(x)
    raise TypeError("expected Move or Candidate, got %r" % type(x))


def apply_action(domain: Domain, st, action: dict) -> None:
    """Replay one serialised action (legacy {i, p, v, v_old} or {verb, args})
    on st in place."""
    domain.apply(st, Move.from_dict(action))


# -- verb table (optional; drives the default grammar/format/parse) ----------
@dataclass(frozen=True)
class ArgSpec:
    name: str
    type: str = "float"      # "int" | "float" | "str" | "bool" | "json"
    doc: str = ""            # range / meaning, printed in grammar()


@dataclass(frozen=True)
class VerbSpec:
    name: str
    args: Tuple[ArgSpec, ...] = ()
    doc: str = ""            # what the verb does physically (one line)
    phrase: str = ""         # the operation in words ("Reconnect a line"), for the
                             # set_parameter grammar; default: the verb name in words


# -- verb surface over SET_PARAM (v2 domains; docs/dataxp_verb_surface.md) ---
#
# A v2 domain keeps its typed parameters, candidate enumeration, set() and the
# {i, p, v, v_old} serialisation, and adds a spelling layer: every legal
# SET_PARAM move has exactly one verb text and every verb text parses back to
# the same SET_PARAM move (a bijection on the existing legal moves). The domain
# sets verb_surface = True and fills verb_table with one ParamVerb per
# parameter; format_move / parse_move / grammar then come from the table.
# Value codecs: Toggle (0/1 as a verb pair), Categorical (integer code as a
# name), Continuous (a number, its unit named in grammar()).

@dataclass(frozen=True)
class Toggle:
    """0/1 parameter as a verb pair: value 1 <-> ParamVerb.verb (the ON verb),
    value 0 <-> `off`. The call carries the element argument only."""
    off: str
    off_doc: str = ""        # what the OFF verb does (grammar); ParamVerb.doc is the ON verb's
    off_phrase: str = ""     # the OFF operation in words ("Remove a member"), for the
                             # set_parameter grammar; default: the OFF verb name in words


@dataclass(frozen=True)
class Categorical:
    """Integer-coded parameter spelled by name: code k <-> names[k]. parse_move
    also accepts `aliases` (extra spelling -> code) and ignores case; the
    canonical spelling format_move writes is names[k]."""
    arg: str                                   # argument name, e.g. "material"
    names: Tuple[str, ...]                     # index = code
    aliases: Tuple[Tuple[str, int], ...] = ()
    doc: str = ""                              # what each name is (grammar)


@dataclass(frozen=True)
class Continuous:
    """Numeric parameter: the argument is the new absolute value, written as a
    plain number; the unit is named in grammar() only."""
    arg: str                                   # argument name, e.g. "radius"
    unit: str = ""
    doc: str = ""                              # range / meaning (grammar)


@dataclass(frozen=True)
class ParamVerb:
    """How one SET_PARAM parameter is spelled. `verb` is the verb name (for a
    Toggle codec, the ON verb); `elem` the element argument (type int: the
    element index SET_PARAM used); `codec` one of Toggle / Categorical /
    Continuous; `select` an optional (argument name, literal) written between
    the element and the value, which lets several parameters share one verb
    (dx / dy / dz -> MOVE_JOINT(joint, axis, offset) with axis = x / y / z)."""
    param: str
    verb: str
    elem: ArgSpec
    codec: object
    select: Optional[Tuple[str, str]] = None
    doc: str = ""
    phrase: str = ""         # the operation in words ("Install a member", "Move a joint
                             # along x"), for the set_parameter grammar; default from
                             # the verb name (and the select literal)


def default_phrase(verb: str, select: Optional[Tuple[str, str]] = None) -> str:
    """'INSTALL_MEMBER' -> 'Install member'; with select ('axis', 'x'):
    'Move joint (axis x)'. The fallback when a table row sets no phrase."""
    words = " ".join(w for w in verb.lower().split("_") if w)
    text = words[:1].upper() + words[1:]
    if select is not None:
        text += " (%s %s)" % (select[0], select[1])
    return text


def phrase_of(pv: ParamVerb) -> str:
    return pv.phrase or default_phrase(pv.verb, pv.select)


def off_phrase_of(pv: ParamVerb) -> str:
    c = pv.codec
    return (getattr(c, "off_phrase", "") or default_phrase(c.off, pv.select))


def check_verb_table(domain: Domain) -> None:
    """A verb-surface domain's table must spell every parameter in
    domain.params and nothing else, with distinct (verb, select) spellings;
    a domain without a table must override format_move / parse_move /
    grammar itself. Raised at construction so a mistake never reaches a run."""
    if not domain.verb_table:
        own = all(getattr(type(domain), m) is not getattr(Domain, m)
                  for m in ("format_move", "parse_move", "grammar"))
        if not own:
            raise ValueError("domain %s: verb_surface without a verb_table must override "
                             "format_move, parse_move and grammar" % domain.name)
        return
    params = tuple(domain.params)
    seen_p, seen_v = set(), set()
    for pv in domain.verb_table:
        if not isinstance(pv, ParamVerb):
            raise ValueError("domain %s: verb_table rows must be ParamVerb" % domain.name)
        if pv.param not in params or pv.param in seen_p:
            raise ValueError("domain %s: verb_table parameter %r unknown or repeated"
                             % (domain.name, pv.param))
        seen_p.add(pv.param)
        if not isinstance(pv.codec, (Toggle, Categorical, Continuous)):
            raise ValueError("domain %s: %r has no codec" % (domain.name, pv.param))
        verbs = [pv.verb] + ([pv.codec.off] if isinstance(pv.codec, Toggle) else [])
        for v in verbs:
            if not re.match(r"^[A-Z][A-Z0-9_]*$", v) or v == SET_PARAM:
                raise ValueError("domain %s: bad verb name %r" % (domain.name, v))
            key = (v, pv.select[1] if pv.select else None)
            if key in seen_v:
                raise ValueError("domain %s: spelling %r is used twice" % (domain.name, key))
            seen_v.add(key)
        if isinstance(pv.codec, Categorical) and not pv.codec.names:
            raise ValueError("domain %s: %r has no category names" % (domain.name, pv.param))
    missing = [p for p in params if p not in seen_p]
    if missing:
        raise ValueError("domain %s: verb_table does not spell %s" % (domain.name, missing))


def surface_value(pv: ParamVerb, v) -> str:
    """The value of pv.param as the verb writes it ('' for a toggle)."""
    c = pv.codec
    if isinstance(c, Toggle):
        return ""
    if isinstance(c, Categorical):
        k = int(round(float(v)))
        if not 0 <= k < len(c.names):
            raise ValueError("%s: code %r has no name" % (pv.verb, v))
        return c.names[k]
    return fmt_arg(float(v))


def surface_format(domain: Domain, move: "Move") -> str:
    """SET_PARAM(i, p, v) -> its one verb spelling from domain.verb_table."""
    a = move.args
    pv = domain.verb_param(a["p"])
    if pv is None:
        raise ValueError("domain %s: parameter %r has no verb in verb_table"
                         % (domain.name, a["p"]))
    args = [fmt_arg(int(a["i"]))]
    if pv.select is not None:
        args.append(pv.select[1])
    c = pv.codec
    v = float(a["v"])
    if isinstance(c, Toggle):
        verb = pv.verb if v >= 0.5 else c.off
    else:
        verb = pv.verb
        args.append(surface_value(pv, v))
    return "%s(%s)" % (verb, ", ".join(args))


def _category_code(c: Categorical, name: str) -> Optional[int]:
    if name in c.names:
        return c.names.index(name)
    low = name.lower()
    for k, n in enumerate(c.names):
        if n.lower() == low:
            return k
    for alias, k in c.aliases:
        if alias == name or alias.lower() == low:
            return int(k)
    return None


def surface_parse(domain: Domain, verb: str, toks: List[str]) -> "Optional[Move]":
    """One VERB(...) call (as parse_verb_calls splits it) -> the SET_PARAM
    move it spells, or None when no verb_table row admits it."""
    for pv in domain.verb_table:
        c = pv.codec
        on = pv.verb == verb
        off = isinstance(c, Toggle) and c.off == verb
        if not (on or off):
            continue
        n = 1 + (1 if pv.select is not None else 0) + (0 if isinstance(c, Toggle) else 1)
        if len(toks) != n:
            continue
        try:
            i = coerce_arg(toks[0], "int")
            pos = 1
            if pv.select is not None:
                if coerce_arg(toks[pos], "str").lower() != pv.select[1].lower():
                    continue
                pos += 1
            if isinstance(c, Toggle):
                v = 1.0 if on else 0.0
            elif isinstance(c, Categorical):
                k = _category_code(c, coerce_arg(toks[pos], "str"))
                if k is None:
                    continue
                v = float(k)
            else:
                v = coerce_arg(toks[pos], "float")
        except ValueError:
            continue
        return Move(SET_PARAM, {"i": i, "p": pv.param, "v": v, "v_old": None})
    return None


def surface_grammar(domain: Domain) -> str:
    """The verb list of a verb-surface domain, one verb per line in the style
    of the VerbSpec grammar: VERB(arg: type range, ...) then an indented
    line saying what it does. Parameters sharing a verb through `select`
    print once, with every literal listed."""
    lines = []
    done = set()
    for pv in domain.verb_table:
        if pv.verb in done:
            continue
        done.add(pv.verb)
        group = [q for q in domain.verb_table if q.verb == pv.verb]
        c = pv.codec
        elem = "%s: int%s" % (pv.elem.name, (" " + pv.elem.doc) if pv.elem.doc else "")
        sel = ""
        if pv.select is not None:
            sel = ", %s: one of %s" % (pv.select[0], " | ".join(q.select[1] for q in group))
        if isinstance(c, Toggle):
            lines.append("%s(%s%s)" % (pv.verb, elem, sel))
            if pv.doc:
                lines.append("    " + pv.doc)
            lines.append("%s(%s%s)" % (c.off, elem, sel))
            if c.off_doc:
                lines.append("    " + c.off_doc)
            continue
        if isinstance(c, Categorical):
            val = ", %s: one of %s" % (c.arg, " | ".join(c.names))
            doc = " ".join(x for x in (pv.doc, c.doc) if x)
        else:
            val = ", %s: float%s%s" % (c.arg, (" [%s]" % c.unit) if c.unit else "",
                                       (" " + c.doc) if c.doc else "")
            doc = pv.doc
        lines.append("%s(%s%s%s)" % (pv.verb, elem, sel, val))
        if doc:
            lines.append("    " + doc)
    return "\n".join(lines)


# ── the set_parameter spelling (docs/dataxp_spellings.md) ────────────────────
#
# One wrapper tool, SET_PARAM(element, parameter, value), documented operation
# by operation from the same tables the verbs come from. The functions below
# are the defaults behind Domain.set_param_format / set_param_parse /
# set_param_grammar; nothing here touches the move set.

def parse_classic_set_param(toks: List[str]) -> "Optional[Move]":
    """The v2 rule for a SET_PARAM(i, p, v) call: a bare integer element, an
    identifier parameter, a plain number value (a category name or a verb is
    not a value). None when the tokens do not fit."""
    if len(toks) != 3:
        return None
    try:
        i = coerce_arg(toks[0], "int")
        p = coerce_arg(toks[1], "str")
        v = coerce_arg(toks[2], "float")
    except ValueError:
        return None
    if not re.match(r"^[A-Za-z_]\w*$", p):
        return None
    return Move(SET_PARAM, {"i": i, "p": p, "v": v, "v_old": None})


def set_param_code_list(c: Categorical) -> str:
    """'code: 0 = Al6061T6, 1 = S355, 2 = Ti6Al4V' - how the set_parameter
    grammar names a categorical parameter's stored codes (a domain whose
    codes are per problem replaces this exact string by a rule, thermal2)."""
    return "code: " + ", ".join("%d = %s" % (k, n) for k, n in enumerate(c.names))


def _arg_desc(a: ArgSpec) -> str:
    return "%s: %s%s" % (a.name, a.type, (" " + a.doc) if a.doc else "")


def surface_set_param_grammar(domain: Domain) -> str:
    """The set_parameter grammar of a verb-surface domain: the preamble, then
    one block per operation in verb_table order - a toggle is two blocks
    (value 1, value 0), a parameter sharing a verb through `select` is its own
    block - each '<phrase>: SET_PARAM(<elem>, <param>, <value>)  <element
    description>; <value description>' followed by the indented verb doc.
    The descriptions are the ParamVerb / codec texts the multitool grammar
    prints inside the verb signature."""
    lines = [SET_PARAM_PREAMBLE]
    for pv in domain.verb_table:
        elem = _arg_desc(pv.elem)
        c = pv.codec
        if isinstance(c, Toggle):
            for phrase, val, doc in ((phrase_of(pv), "1", pv.doc),
                                     (off_phrase_of(pv), "0", c.off_doc)):
                lines.append("%s: SET_PARAM(%s, %s, %s)  %s"
                             % (phrase, pv.elem.name, pv.param, val, elem))
                if doc:
                    lines.append("    " + doc)
            continue
        if isinstance(c, Categorical):
            val = "code"
            vdesc = set_param_code_list(c) + (("; " + c.doc) if c.doc else "")
        else:
            val = c.arg
            vdesc = "%s: float%s%s" % (c.arg, (" [%s]" % c.unit) if c.unit else "",
                                       (" " + c.doc) if c.doc else "")
        lines.append("%s: SET_PARAM(%s, %s, %s)  %s; %s"
                     % (phrase_of(pv), pv.elem.name, pv.param, val, elem, vdesc))
        if pv.doc:
            lines.append("    " + pv.doc)
    return "\n".join(lines)


# -- native-verb domains: the generic encoding SET_PARAM(element, operation, value)
def _native_elem_index(vs: VerbSpec) -> Optional[int]:
    """Position of the verb's first int argument (the element), None when it
    has none (DO_NOTHING)."""
    for k, a in enumerate(vs.args):
        if a.type == "int":
            return k
    return None


def _json_item(v) -> str:
    """One member of the JSON-list value: numbers as fmt_arg writes them,
    strings quoted, anything else as JSON."""
    if _is_number(v):
        return fmt_arg(v)
    return json.dumps(_plain(v), separators=(",", ":"))


def _coerce_value(x, typ: str):
    """coerce_arg for a value that json.loads already produced."""
    if typ == "int":
        if isinstance(x, bool) or not _is_number(x) or float(x) != int(x):
            raise ValueError("not an integer: %r" % (x,))
        return int(x)
    if typ == "float":
        if isinstance(x, bool) or not _is_number(x):
            raise ValueError("not a number: %r" % (x,))
        return float(x)
    if typ == "str":
        if not isinstance(x, str) or not x:
            raise ValueError("not a name: %r" % (x,))
        return x
    if typ == "bool":
        return x if isinstance(x, bool) else coerce_arg(str(x), "bool")
    if typ == "json":
        return x
    raise ValueError("unknown argument type %r" % typ)


def native_set_param_format(domain: Domain, move: "Move") -> str:
    """VERB(a, b, ...) of a native domain -> SET_PARAM(element, verb_lower,
    value): element = the first int argument (0 when none), value = the one
    remaining argument as fmt_arg writes it, '[a, b]' (JSON list) when
    several remain, 0 when none."""
    vs = domain.verb_spec(move.verb)
    if vs is None:
        raise ValueError("domain %s: verb %r is not in its verbs table" % (domain.name, move.verb))
    ei = _native_elem_index(vs)
    elem = fmt_arg(move.args[vs.args[ei].name]) if ei is not None else "0"
    rest = [move.args[a.name] for k, a in enumerate(vs.args) if k != ei]
    if not rest:
        val = "0"
    elif len(rest) == 1:
        val = fmt_arg(rest[0])
    else:
        val = "[" + ", ".join(_json_item(x) for x in rest) + "]"
    return "SET_PARAM(%s, %s, %s)" % (elem, move.verb.lower(), val)


def _parse_list_value(tok: str, rest: "List[ArgSpec]") -> "Optional[list]":
    """The value slot of an operation with several remaining arguments: a JSON
    list (as native_set_param_format writes it, ``["line3_ex", 2]``) or, as the
    block head literally shows it, the same list with bare identifiers
    (``[line3_ex, 2]``; single quotes pass too): then the inside is split on
    its top-level commas (parse_verb_calls' tokenizer, so a nested list or a
    quoted string with a comma survives) and each item coerced by its
    ArgSpec.type, exactly as a multitool call's arguments are. None when the
    list does not have exactly the remaining arguments or an item does not
    fit its type."""
    try:
        lst = json.loads(tok)
    except ValueError:
        lst = None
    if isinstance(lst, list):
        if len(lst) != len(rest):
            return None
        return [_coerce_value(x, a.type) for x, a in zip(lst, rest)]
    if not (tok.startswith("[") and tok.endswith("]")):
        return None
    calls = parse_verb_calls("L(" + tok[1:-1] + ")")
    items = calls[0][1] if calls else []
    if len(items) != len(rest):
        return None
    return [coerce_arg(x, a.type) for x, a in zip(items, rest)]


def native_set_param_parse(domain: Domain, toks: List[str]) -> "Optional[Move]":
    """Inverse of native_set_param_format on the three tokens of a SET_PARAM
    call: the operation is matched to the verbs table ignoring case, the
    element and the value(s) are coerced by the table's types, and the
    arguments are rebuilt in table order. Strict: a verb without an element
    needs element 0, one without further arguments needs value 0, a list
    value must have exactly the remaining arguments (written as JSON or with
    bare identifiers, _parse_list_value)."""
    try:
        e = coerce_arg(toks[0], "int")
        op = coerce_arg(toks[1], "str").lower()
    except ValueError:
        return None
    vs = None
    for cand in domain.verbs:
        if cand.name.lower() == op:
            vs = cand
            break
    if vs is None:
        return None
    ei = _native_elem_index(vs)
    rest = [a for k, a in enumerate(vs.args) if k != ei]
    tok = toks[2].strip()
    try:
        if ei is None and e != 0:
            return None
        if not rest:
            if float(tok) != 0.0:
                return None
            vals = []
        elif len(rest) == 1:
            vals = [coerce_arg(tok, rest[0].type)]
        else:
            vals = _parse_list_value(tok, rest)
            if vals is None:
                return None
    except (ValueError, TypeError):
        return None
    args = {}
    it = iter(vals)
    for k, a in enumerate(vs.args):
        args[a.name] = e if k == ei else next(it)
    return Move(vs.name, args)


def native_set_param_call(vs: VerbSpec) -> str:
    """'SET_PARAM(<elem>, <verb_lower>, <value>)' of one VerbSpec, with the
    table's argument names: element = the first int argument (0 when none),
    value = the one remaining argument, '[a, b]' when several, 0 when none.
    The head of the verb's set_parameter grammar block, and what
    Domain.call_form prints for it in an observation."""
    ei = _native_elem_index(vs)
    rest = [a for k, a in enumerate(vs.args) if k != ei]
    elem = vs.args[ei].name if ei is not None else "0"
    if not rest:
        val = "0"
    elif len(rest) == 1:
        val = rest[0].name
    else:
        val = "[" + ", ".join(a.name for a in rest) + "]"
    return "SET_PARAM(%s, %s, %s)" % (elem, vs.name.lower(), val)


def _native_list_note(rest: "List[ArgSpec]") -> str:
    """The value description of an operation with several remaining
    arguments, appended to its grammar block head: the JSON list the value
    is, in table order, and which of its members are names (double-quoted)."""
    note = "value: JSON list [%s] in this order" % ", ".join(a.name for a in rest)
    quoted = [a.name for a in rest if a.type == "str"]
    if quoted:
        note += ", %s in double quotes" % " and ".join(quoted)
    return note


def _name_ops(text: str, verbs) -> str:
    """A verb name mentioned inside a table doc ('the line comes back only
    through RECONNECT_LINE') -> its lower-case operation name, the name the
    set_parameter grammar gives that operation."""
    for vs in sorted(verbs, key=lambda v: -len(v.name)):
        text = re.sub(r"\b%s\b" % re.escape(vs.name), vs.name.lower(), text)
    return text


def native_set_param_grammar(domain: Domain) -> str:
    """The set_parameter grammar of a native domain from its verbs table: the
    native preamble (`parameter` names the operation; a value of several
    arguments is a JSON list, names double-quoted), then one block per verb:
    '<phrase>: SET_PARAM(<elem>, <verb_lower>, <value>)  <argument
    descriptions>[; value: JSON list [...] ...]' and the indented verb doc.
    A verb named inside a doc is written as its operation name (_name_ops)."""
    lines = [SET_PARAM_NATIVE_PREAMBLE]
    for vs in domain.verbs:
        ei = _native_elem_index(vs)
        rest = [a for k, a in enumerate(vs.args) if k != ei]
        descs = ([_arg_desc(vs.args[ei])] if ei is not None else []) + [_arg_desc(a) for a in rest]
        if len(rest) > 1:
            descs.append(_native_list_note(rest))
        head = "%s: %s" % (vs.phrase or default_phrase(vs.name), native_set_param_call(vs))
        lines.append(_name_ops(head + (("  " + "; ".join(descs)) if descs else ""), domain.verbs))
        if vs.doc:
            lines.append("    " + _name_ops(vs.doc, domain.verbs))
    return "\n".join(lines)


def check_native_set_param(domain: Domain) -> None:
    """Under the set_parameter spelling a native domain must either leave
    format_move / parse_move to the table-driven defaults (then the generic
    encoding needs a non-empty `verbs` table) or override the matching
    set_param_* hook for each one it overrides. Raised at construction so a
    domain the encoding cannot spell fails loudly instead of serving verbs
    (a grammar() override is allowed when it wraps Domain.grammar(self), as
    molopt and engibench do)."""
    cls = type(domain)
    for base, hook in (("format_move", "set_param_format"), ("parse_move", "set_param_parse")):
        own_base = getattr(cls, base) is not getattr(Domain, base)
        own_hook = getattr(cls, hook) is not getattr(Domain, hook)
        if own_base and not own_hook:
            raise NotImplementedError(
                "domain %s overrides %s but not %s: it has no set_parameter spelling yet "
                "(docs/dataxp_spellings.md); unset %s or add the hook"
                % (domain.name, base, hook, SPELLING_ENV))
    hooks_own = all(getattr(cls, h) is not getattr(Domain, h)
                    for h in ("set_param_format", "set_param_parse", "set_param_grammar"))
    if not domain.verbs and not hooks_own:
        raise NotImplementedError(
            "domain %s has no `verbs` table for the generic set_parameter encoding: override "
            "set_param_format / set_param_parse / set_param_grammar (docs/dataxp_spellings.md)"
            % domain.name)


def fmt_arg(v) -> str:
    """Argument text: ints bare, floats to 6 significant digits (integral
    floats as ints), strings bare, lists/dicts as JSON."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return str(f)
        if f == int(f) and abs(f) < 1e9:
            return str(int(f))
        return "%.6g" % f
    if isinstance(v, str):
        return v
    return json.dumps(_plain(v), separators=(",", ":"))


def coerce_arg(tok: str, typ: str):
    """Parse one argument token by its declared type; ValueError when it
    does not fit."""
    tok = tok.strip()
    if typ == "int":
        f = float(tok)
        if f != int(f):
            raise ValueError("not an integer: %r" % tok)
        return int(f)
    if typ == "float":
        return float(tok)
    if typ == "bool":
        t = tok.lower()
        if t in ("1", "true", "yes", "on"):
            return True
        if t in ("0", "false", "no", "off"):
            return False
        raise ValueError("not a bool: %r" % tok)
    if typ == "str":
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
            tok = tok[1:-1]
        if not tok:
            raise ValueError("empty string argument")
        return tok
    if typ == "json":
        return json.loads(tok)
    raise ValueError("unknown argument type %r" % typ)


_CALL_HEAD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def parse_verb_calls(text: str) -> List[Tuple[str, List[str]]]:
    """Every balanced VERB(a, b, ...) call in text, in order, as (verb, raw
    argument tokens). Nested brackets and quoted strings are respected, so
    list arguments like [1, 2] survive. Callers pick the first call whose
    verb they know."""
    out: List[Tuple[str, List[str]]] = []
    if not text:
        return out
    for m in _CALL_HEAD.finditer(text):
        depth, quote, buf, toks = 1, None, [], []
        i = m.end()
        while i < len(text):
            ch = text[i]
            if quote:
                buf.append(ch)
                if ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
                buf.append(ch)
            elif ch in "([{":
                depth += 1
                buf.append(ch)
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    tok = "".join(buf).strip()
                    if tok or toks:
                        toks.append(tok)
                    out.append((m.group(1), toks))
                    break
                buf.append(ch)
            elif ch == "," and depth == 1:
                toks.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
            i += 1
    return out


@dataclass
class PhiScore:
    mean: float           # E[Phi] over replicates (each pays its own hinge)
    se: float             # standard error of the mean (0 for deterministic)
    n_rep: int
    feasible: bool        # chance-feasibility on the quantile-aggregated state
    valid_frac: float     # fraction of replicates inside the validity envelope
    agg_metrics: dict     # quantile-aggregated metrics (for reporting/harvest)


class PhiEvaluator:
    """Scores states under Phi_v2 with per-problem calibration and, for
    stochastic domains, replicate-based noise accounting.

    Scoring rule: Phi is computed PER REPLICATE and averaged (a violating draw
    pays, so E[Phi] prices violation probability); feasibility is judged on
    the adverse-quantile aggregate (chance constraint at level q).
    """

    def __init__(self, domain: Domain, spec: dict, *, alpha: float = 5.0,
                 tau: float = 0.05, q: float = 0.9,
                 n_rep_min: int = 4, n_rep_max: int = 24,
                 valid_frac_min: float = 0.8, offset_rule: str = "infeasible"):
        self.domain = domain
        self.alpha = alpha
        self.tau = tau
        self.q = q
        self.n_rep_min = 1 if not domain.stochastic else n_rep_min
        self.n_rep_max = 1 if not domain.stochastic else n_rep_max
        self.valid_frac_min = valid_frac_min
        self.sim_calls = 0

        base = domain.program(spec)
        st0 = domain.load(spec)
        m0 = self._sample(st0, self.n_rep_min, seed0=stable_seed(spec.get("problem_id"), "init"))
        agg0 = aggregate_metrics(_to_samples(m0), base, q=self.q)
        self.program = base.with_violation_scale(agg0, tau=tau)
        # run1 was generated with offset_rule="legacy_scaled" (offset never on)
        self.offset_rule = offset_rule
        self.feasibility_offset = auto_feasibility_offset(
            self.program, agg0, alpha=alpha, tau=tau, rule=offset_rule)
        self.initial_agg = agg0

    # -- internals -------------------------------------------------------------
    def _sample(self, st, n_rep: int, seed0: int) -> List[dict]:
        out = []
        for r in range(n_rep):
            out.append(self.domain.evaluate(self.domain.clone(st),
                                            seed=(seed0 + 1009 * r) & 0xFFFFFFFF))
            self.sim_calls += 1
        return out

    def _phi_one(self, metrics: dict) -> float:
        return compute_potential_v2(
            metrics, self.program, alpha=self.alpha, tau=self.tau,
            feasibility_offset=self.feasibility_offset,
            offset_on_hard=self.offset_rule != "legacy_scaled")

    def score(self, st, *, seed0: int, n_rep: Optional[int] = None) -> PhiScore:
        n = n_rep if n_rep is not None else self.n_rep_min
        reps = self._sample(st, n, seed0)
        phis = [self._phi_one(m) for m in reps]
        mean = sum(phis) / len(phis)
        if len(phis) > 1:
            var = sum((x - mean) ** 2 for x in phis) / (len(phis) - 1)
            se = math.sqrt(max(var, 0.0) / len(phis))
        else:
            se = 0.0
        agg = aggregate_metrics(_to_samples(reps), self.program, q=self.q)
        valid = sum(1 for m in reps if self.program.is_valid(m)) / len(reps)
        feasible = valid >= self.valid_frac_min and self.program.is_feasible(agg)
        return PhiScore(mean=mean, se=se, n_rep=len(phis), feasible=feasible,
                        valid_frac=valid, agg_metrics=agg)

    def refine(self, st, score: PhiScore, *, seed0: int) -> PhiScore:
        """Double the replicate count (up to n_rep_max) and re-score."""
        n = min(self.n_rep_max, max(score.n_rep * 2, score.n_rep + 1))
        if n <= score.n_rep:
            return score
        return self.score(st, seed0=seed0, n_rep=n)

    def decided(self, a: PhiScore, b: PhiScore, *, k_sigma: float = 2.0,
                eps: float = 1e-9) -> bool:
        """Reliability gate: is the (a - b) Phi margin outside the noise band?"""
        band = k_sigma * math.sqrt(a.se ** 2 + b.se ** 2)
        return abs(a.mean - b.mean) > max(band, eps)


def _to_samples(reps: List[dict]) -> dict:
    keys = set()
    for m in reps:
        keys.update(m.keys())
    return {k: [m.get(k, float("nan")) for m in reps] for k in keys}
