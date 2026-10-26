"""DX-20: evaluate a policy on dataxp -- one move at replayed teacher states, or whole episodes
in the real simulator loop. LLM policies are served by vLLM; scripted ones (gold, greedy,
random) run the same loop and are how the loop itself is checked.

  states    python scripts/dataxp_policy_eval.py states --tier <dev_in_cot.jsonl> \\
                --code <dir holding the dataxp package> --run1 <run1 dir> \\
                --policy llm --model <served name> --base-url http://127.0.0.1:8771/v1 --out <prefix>
            One move per row of the tier (the tier's own system/user text: training's exact
            prompt). Writes <prefix>.rows.jsonl and <prefix>.actions.jsonl, the latter in
            dataxp_regret.py's --actions format.

  episodes  python scripts/dataxp_policy_eval.py episodes --code ... --variants circuit grid \\
                --n 60 --turns 20 --policy llm --model ... --out <prefix>
            DX-14's fresh problems (index >= 10000, run1's seeding, the first N that start
            infeasible), up to --turns moves each. The observation is built by the renderer's
            own build_prompt and cut by the assembler's own split_prompt -- the code path the
            training rows went through -- so there is no second renderer to drift. The model
            is asked only while a constraint is violated, as in training (repair phase only);
            the episode ends at the first feasible design. Writes <prefix>.episodes.jsonl.

  v2        --obs v2 (DX-23, dataxp run2's typed-action domains *2): the observation is
            render_cot.policy_observation -- the text run2's training rows and the information probe use --
            with the history labelled after each change and the observed effect of the last change; system =
            the DOMAIN and GOAL lines, user = the rest + the instruction. --problems-file reads the problem
            specs from a file written by the `problems` mode, so every arm sees the same problems
            (make_problem is not reproducible across machines for grid2/seir2_stoch, DX-22b).

  v3        native-verb variants (grid2op, photonics, molopt, engibench: the dataxp Move contract,
            docs/dataxp_move_contract.md). A variant whose domain has native_moves = True takes this
            path automatically; every SET_PARAM variant runs the v1/v2 code below unchanged (checked
            byte-for-byte against the original script, see the handoff). On a native variant:
              observation  render_cot.policy_observation through dataxp.ownrows.common.observation_parts
                           (history lines = format_move + move_label, labelled after the change as
                           iter_states / select_states label them); served as the own-row training rows
                           are: system = the DOMAIN/GOAL head, user = the rest + common.instruction_for
                           (INSTRUCTION_NATIVE + the domain's grammar(), verbatim the text the own-row
                           sampler's rows carry; its sha1[:12] is recorded per episode). --native-prompt
                           probe serves the sampler's probe prompt instead (common.own_system_prompt, i.e.
                           render_cot.system_prompt's grammar substitution, and the full observation +
                           USER_TAIL_OWN).
              parsing      the LAST <action>...</action> of the reply after the last </think>
                           (common.parse_reply -> domain.parse_move); fail = no_action | unparsed.
              applying     domain.apply on a clone of the live state; a move the state refuses
                           (ValueError) spends the turn with the design unchanged, fail = illegal:<why>,
                           as an invalid SET_PARAM does; a move that leaves every metric unchanged is a
                           noop, as a SET_PARAM landing on its old value is.
              scoring      the same PhiEvaluator / program the ladder uses (unchanged).
              record       every existing per-turn key, plus "action": Move.as_dict() (the serialised
                           move, Move.from_dict re-applies it); v_old / v_new are None, clamped False.
              policies     random (a random legal move), greedy (hint_move, else a random legal move),
                           repeat_last (the previous move again when the state still admits it, else
                           greedy), replay, llm, and phi_select: the ladder's Phi-select teacher (argmax
                           E[Phi] over legal_moves, envelope-valid first, applied only when it improves
                           on the incumbent; the null move on a time-stepping domain otherwise). phi_select
                           also runs on SET_PARAM variants (a new policy, not part of the byte-identity
                           check). Native problems come from --problems-file only (the generated corpus;
                           make_problem is not used).
            --loop-guard is ignored on native variants (an exact undo is SET_PARAM-specific).

  v3, verb surfaces (docs/dataxp_verb_surface.md, 2026-10-23): a v2 domain with verb_surface = True
            keeps its SET_PARAM moves, data, legal moves and apply path and is only SPOKEN in the
            field's verbs. Every domain for which dataxp.domain.speaks_verbs (native_moves or
            verb_surface) is true takes the verb-spelled prompt and parse path above: observation
            through common.observation_parts + common.instruction_for (INSTRUCTION_NATIVE + the
            domain's grammar(): the own-row training rows' exact text), history lines through
            format_move, --native-prompt served|probe as for a native variant, the reply's last
            <action> through domain.parse_move (fail = no_action | unparsed, as on a native variant).
            On a verb-surface domain the parsed move is a SET_PARAM Move and is handed on as the
            {i, p, v} dict the v2 apply path takes, so check_move / domain.set / v_old / v_new /
            clamped / noop, the scripted policies (random, greedy, repeat_last, phi_select) and the
            loop guard run byte for byte as before; the per-turn record gains "action" = the applied
            {i, p, v, v_old} (what the native path records) and the episode gains verb_surface /
            instruction_sha1 / native_prompt. Problems come from --problems-file as for any v2 run
            (problems_v2.jsonl); only native variants are required to have one.

Move handling: the move is read from the reply after the LAST </think> (a thinking reply whose
think block never closes counts as unparsed), last <action> line wins. A move naming a
non-existent element or parameter, or a non-finite value, is invalid and the turn is spent with
the design unchanged. A valid value is applied through domain.set, which clamps to the element's
own limits; the clamp and a move that changes nothing (no-op) are counted.
"""
import argparse
import collections
import json
import math
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    from assemble_dataxp_sft import split_prompt  # noqa: E402  (a copy sits next to this file)
except ImportError:                                # ...else the collaborator's scripts dir
    sys.path.append(os.environ.get("DX_ORIG_SCRIPTS", "/storage/wxu/sftprobe/repo/scripts"))
    from assemble_dataxp_sft import split_prompt  # noqa: E402

VARIANTS = ["circuit", "frame3d", "grid", "inventory", "seir", "thermal", "traffic"]
ACTION = re.compile(r"<action>\s*SET_PARAM\(\s*(-?\d+)\s*,\s*([A-Za-z_]\w*)\s*,"
                    r"\s*(-?[\d.eE+-]+)\s*\)\s*</action>")
LOCK = threading.Lock()
# The simulators call native code (sparse solvers, LAPACK) that is not safe to run from many threads at
# once: j81's episodes (32 threads) died with a segmentation fault. Every simulator touch holds SIM;
# only the model call, which is where the time goes, runs unlocked.
SIM = threading.RLock()


def import_dataxp(code):
    sys.path.insert(0, code)
    import numpy as np  # noqa: F401
    from dataxp import domain as D, ladder as L, render_cot as R
    from dataxp.domains import get_domain
    return D, L, R, get_domain


def import_common():
    """dataxp.ownrows.common: the native task instruction (instruction_for) and the own-row
    pipeline's observation split / reply parsing. Imported only when a native variant runs (the
    v2 code copies carry no ownrows package)."""
    from dataxp.ownrows import common as C
    return C


def is_native(domain):
    return bool(getattr(domain, "native_moves", False))


def speaks_verbs(D, domain):
    """dataxp.domain.speaks_verbs: native_moves or verb_surface, the test every prompt-facing path
    of the package uses (render_cot, ownrows). The getattr form is the same test for a code copy
    that predates the helper."""
    f = getattr(D, "speaks_verbs", None)
    if callable(f):
        return bool(f(domain))
    return bool(getattr(domain, "native_moves", False) or getattr(domain, "verb_surface", False))


# ------------------------------------------------------------------ model client

class LLM:
    def __init__(self, a):
        self.a = a
        self.tokens = collections.Counter()

    def __call__(self, system, user):
        a = self.a
        body = {"model": a.model, "messages": [{"role": "system", "content": system},
                                               {"role": "user", "content": user}],
                "temperature": a.temperature, "max_tokens": a.max_tokens,
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
            self.tokens["prompt"] += u.get("prompt_tokens", 0) or 0
            self.tokens["completion"] += u.get("completion_tokens", 0) or 0
        return {"answer": text, "reasoning": reasoning.replace("<think>", "").strip(),
                "think_closed": closed, "finish": ch.get("finish_reason"),
                "completion_tokens": u.get("completion_tokens")}


def parse_move(reply):
    if "err" in reply:
        return None, "error"
    if not reply["think_closed"]:
        return None, "think_unclosed"
    m = ACTION.findall(reply["answer"])
    if not m:
        return None, "no_action"
    i, p, v = m[-1]
    try:
        return {"i": int(i), "p": p, "v": float(v)}, None
    except ValueError:
        return None, "bad_number"


def parse_move_native(C, domain, reply):
    """(move, None) or (None, reason) for a verb-spelled variant: the last <action>...</action>
    of the answer through domain.parse_move (common.parse_reply, the own-row scorer's reader).
    Native variant: the Move. Verb-surface variant: the parsed SET_PARAM move as the {i, p, v}
    dict the v2 apply path takes (check_move / domain.set), as a SET_PARAM text was read before."""
    if "err" in reply:
        return None, "error"
    if not reply["think_closed"]:
        return None, "think_unclosed"
    mv, why = C.parse_reply(domain, reply["answer"])
    if mv is not None and not is_native(domain):
        if not mv.is_set_param:
            return None, "unparsed"
        d = mv.as_dict()
        mv = {"i": d["i"], "p": d["p"], "v": d["v"]}
    return mv, why


def check_move(domain, st, mv):
    """(ok, reason, v_old): does the move name something this design can change?"""
    if not (0 <= mv["i"] < domain.n_elements(st)):
        return False, "element_out_of_range", None
    if mv["p"] not in domain.params:
        return False, "unknown_parameter", None
    v_old = domain.get(st, mv["i"], mv["p"])
    if v_old is None or not math.isfinite(float(v_old)):
        return False, "parameter_not_on_element", None
    if not math.isfinite(mv["v"]):
        return False, "non_finite_value", None
    return True, None, float(v_old)


# ------------------------------------------------------------------ observation

def observation(R, domain, spec, st, metrics, binding, history, variant, instruction):
    """(system, user) for this state, through the SAME build_prompt + split_prompt as training.
    build_prompt needs a move to render its CHANGE block; the block is cut, so any valid
    (element, parameter) serves -- the first one the design has."""
    dummy = None
    for i in range(domain.n_elements(st)):
        for p in domain.params:
            v = domain.get(st, i, p)
            if v is not None and math.isfinite(float(v)):
                dummy = {"i": i, "p": p, "v": float(v) * 1.1 + 1e-6, "v_old": float(v)}
                break
        if dummy:
            break
    turn = {"state_metrics": {k: float(v) for k, v in metrics.items()},
            "binding_constraint": binding, "action": dummy}
    prompt, _, _, _ = R.build_prompt(domain, spec, st, turn, history, variant, hash_idx=0)
    sp = split_prompt(prompt, False)
    if sp is None:
        raise RuntimeError("observation layout not recognised for %s" % spec.get("problem_id"))
    system, user = sp
    return system, user + instruction


def observation_v2(R, domain, spec, st, metrics, binding, history, prev_metrics, variant, instruction):
    """(system, user) for the v2 domains: render_cot.policy_observation, as run2's renderer builds it
    (DX-23). The effect block appears whenever a change has been applied before (prev_metrics = the
    metrics just before the last applied change)."""
    turn = {"state_metrics": {k: float(v) for k, v in metrics.items()}, "binding_constraint": binding}
    if history and prev_metrics is not None:
        turn["prev_state_metrics"] = {k: float(v) for k, v in prev_metrics.items()}
    lines, _ = R.policy_observation(domain, spec, st, turn, history, variant)
    head, sep, rest = "\n".join(lines).partition("\n\n")
    if not (head.startswith("DOMAIN:") and sep and rest.strip()):
        raise RuntimeError("v2 observation layout not recognised for %s" % spec.get("problem_id"))
    return head, rest.strip() + instruction


def observation_native(C, domain, spec, st, metrics, binding, history, prev_metrics, variant, mode):
    """(system, user) for a verb-spelled variant (v3: native, or a v2 domain with a verb surface). The
    observation is render_cot.policy_observation split by common.observation_parts (the own-row
    pipeline's split: head = DOMAIN/GOAL, rest = the body); the history lines print format_move +
    move_label (native) or format_move + element label + "was <old value>" (verb surface), labelled
    after each change.
    mode 'served': the training-row messages (common.served_messages) with the instruction
    common.instruction_for(domain) -- INSTRUCTION_NATIVE + the domain's grammar(), verbatim.
    mode 'probe': the own-row sampler's probe prompt (common.probe_prompt: own_system_prompt with the
    grammar substituted in, full observation + USER_TAIL_OWN)."""
    turn = {"state_metrics": {k: float(v) for k, v in metrics.items()}, "binding_constraint": binding}
    if history and prev_metrics is not None:
        turn["prev_state_metrics"] = {k: float(v) for k, v in prev_metrics.items()}
    head, rest = C.observation_parts(domain, spec, st, turn, history, variant)
    instruction, _ = C.instruction_for(domain)
    state = {"system": head, "observation": rest, "instruction": instruction}
    if mode == "probe":
        return C.probe_prompt(domain, state)
    m = C.served_messages(state)
    return m[0]["content"], m[1]["content"]


# ------------------------------------------------------------------ loop guard (DX-25b)

def guard_reason(domain, st, history, mv):
    """None, or (code, message) when the proposed move exactly undoes the previous applied move: same
    (element, parameter), and the value -- legalised through domain.set, so a clamped proposal that
    lands back on the old value counts -- equals the value before that move. Only valid moves are
    judged (invalid ones spend the turn unchanged, as always), and only when the previous move actually
    changed the value (a repeated no-op, e.g. pinned at a bound, is not an undo; review 10-05). This is
    the single pre-registered intervention of DX-25b -- no other rule, so the reading stays 'is the
    exact A-B-A oscillation causal'."""
    ok, _, _ = check_move(domain, st, mv)
    if not ok or not history:
        return None
    prev = history[-1][1]
    if int(prev["i"]) != mv["i"] or prev["p"] != mv["p"]:
        return None
    po = float(prev["v_old"])
    if abs(float(prev["v"]) - po) <= 1e-12 * max(1.0, abs(po)):   # previous move was a no-op
        return None
    child = domain.clone(st)
    domain.set(child, mv["i"], mv["p"], mv["v"])
    v_new = float(domain.get(child, mv["i"], mv["p"]))
    if abs(v_new - po) <= 1e-9 * max(1.0, abs(po)):
        return ("exact_undo",
                "Your proposed move would set parameter %s of element %d back to exactly its value "
                "before your previous move, undoing that move. An exact undo is not allowed. Choose a "
                "different move: a different value, a different parameter, or a different element."
                % (mv["p"], mv["i"]))
    return None


# ------------------------------------------------------------------ scripted policies

def scripted(name, D, L, domain, st, rng, history):
    if name == "random":
        cands = D.enumerate_candidates(domain, st, rng)
        c = cands[int(rng.integers(len(cands)))] if cands else None
        return {"i": c.i, "p": c.p, "v": c.v} if c else None
    if name in ("greedy", "repeat_last"):
        if name == "repeat_last" and history:
            a = history[-1][1]
            ratio = float(a["v"]) / float(a["v_old"]) if float(a["v_old"]) else 1.0
            cur = domain.get(st, int(a["i"]), a["p"])
            if cur is not None:
                return {"i": int(a["i"]), "p": a["p"], "v": float(cur) * ratio}
        h = domain.hint(st)
        if h is not None:
            i, p, direction = h
            cur = domain.get(st, i, p)
            if cur is not None and math.isfinite(cur):
                f = max(domain.factors) if direction == "up" else min(domain.factors)
                lo, hi = domain.bounds(st, p)
                v = min(hi, max(lo, cur * f))
                if abs(v - cur) > 1e-12 * max(1.0, abs(cur)):
                    return {"i": i, "p": p, "v": float(v)}
        cands = D.enumerate_candidates(domain, st, rng)
        c = cands[int(rng.integers(len(cands)))] if cands else None
        return {"i": c.i, "p": c.p, "v": c.v} if c else None
    raise ValueError(name)


def _admits(domain, st, move):
    """Does the state admit the move (domain.apply on a clone does not raise)?"""
    try:
        domain.apply(domain.clone(st), move)
    except Exception:            # noqa: BLE001
        return False
    return True


def scripted_native(name, D, domain, st, rng, history):
    """The scripted policies on a native-verb domain, as Move objects (ladder.run_episode's random /
    greedy rules: a random legal move; hint_move, else a random legal move)."""
    if name == "random":
        moves = domain.legal_moves(st, rng)
        return moves[int(rng.integers(len(moves)))] if moves else None
    if name in ("greedy", "repeat_last"):
        if name == "repeat_last" and history:
            last = D.Move.from_dict(history[-1][1])
            if _admits(domain, st, last):
                return last
        hm = getattr(domain, "hint_move", None)
        mv = D.as_move(hm(st)) if callable(hm) else None
        if mv is not None:
            return mv
        moves = domain.legal_moves(st, rng)
        return moves[int(rng.integers(len(moves)))] if moves else None
    raise ValueError(name)


def phi_select(D, domain, st, rng, ev, pid, t, cur):
    """The ladder's Phi-select teacher (dataxp.ladder.run_episode, policy 'phi_select', on a
    deterministic domain): every legal move applied on a clone and scored at seed
    stable_seed(pid, 'phi_select', t, ci); envelope-valid candidates first; the top one is applied
    only when it improves on the incumbent, else the domain's null move (time-stepping domains),
    else no move. With --policy phi_select the harness's rng is default_rng(stable_seed(pid,
    'phi_select', 0)), the ladder's, so the chosen moves reproduce the corpus trajectory of the same
    problem while the episode is in the repair phase. The ladder's stochastic refinement loops are
    not reproduced (no stochastic variant is evaluated here). Returns (Move, None) or (None, why)."""
    cands = list(domain.legal_moves(st, rng))
    if not cands:
        return None, "no_move"
    scored = []
    for ci, cand in enumerate(cands):
        child = domain.clone(st)
        domain.apply(child, cand)
        scored.append((cand, ev.score(child, seed0=D.stable_seed(pid, "phi_select", t, ci))))
    scored.sort(key=lambda s: s[1].mean, reverse=True)
    valid = [s for s in scored if s[1].valid_frac >= ev.valid_frac_min]
    pick = (valid or scored)[0]
    if pick[1].mean > cur.mean + 1e-9:
        return pick[0], None
    if getattr(domain, "advances_time", False):
        nm = D.as_move(domain.null_move(st))
        if nm is not None:
            return nm, None
    return None, "no_improvement"


# ------------------------------------------------------------------ episodes

EVAL_START = 10000     # DX-14's fresh evaluation problems; training data (DX-21) is drawn from 20000 on


def fresh_problems(D, get_domain, variant, n, run1, start=EVAL_START):
    """The first n initially-infeasible problems with index >= start. Never one of run1's, and a
    training draw (start != EVAL_START) never one of the 60 evaluation problems per domain."""
    import numpy as np
    d = get_domain(variant)
    # run1 specs carry an extra "domain_variant" key; the guard must compare the problem itself
    canon = lambda s: json.dumps({k: v for k, v in s.items() if k not in ("problem_id", "domain_variant")},
                                 sort_keys=True, default=str)
    seen = set()
    if run1:
        for line in open(os.path.join(run1, variant, "accepted", "problems.jsonl")):
            seen.add(canon(json.loads(line)))
    if start != EVAL_START:
        seen |= {canon(s) for s in fresh_problems(D, get_domain, variant, 60, run1)}
    keep, i = [], start
    while len(keep) < n and i < start + 20 * n:
        rng = np.random.default_rng(D.stable_seed(variant, 0, i))
        spec = d.make_problem(i, rng)
        spec["problem_id"] = "%s_e%05d" % (variant, i)
        assert canon(spec) not in seen, spec["problem_id"]
        ev = D.PhiEvaluator(d, spec)
        if not ev.program.is_feasible(ev.initial_agg):
            keep.append(spec)
        i += 1
    return keep


def run_episode(mods, variant, spec, a, llm, instruction):
    with SIM:
        return _run_episode(mods, variant, spec, a, llm, instruction)


def _run_episode(mods, variant, spec, a, llm, instruction):
    """Called with SIM held; releases it only around the model call."""
    import numpy as np
    D, L, R, get_domain = mods
    domain = get_domain(variant)
    native = is_native(domain)            # Move path: legal_moves / apply / Move.as_dict
    spoken = speaks_verbs(D, domain)      # verb-spelled prompt and parse path (native or verb surface)
    C = import_common() if spoken else None
    pid = spec["problem_id"]
    ev = D.PhiEvaluator(domain, spec)
    st = domain.load(spec)
    rng = np.random.default_rng(D.stable_seed(pid, a.policy, 0))
    cur = ev.score(st, seed0=D.stable_seed(pid, "eval", "s", 0))
    okey = ev.program.objective_key
    history, turns = [], []
    v2 = getattr(a, "obs", "v1") == "v2"
    prev_metrics = None                  # v2: metrics just before the last applied change
    out = {"variant": variant, "problem_id": pid, "policy": a.policy_label or a.policy,
           "feasible_start": bool(cur.feasible), "first_feasible_turn": None, "obj_at_first_feasible": None,
           "reference": L._ref_of(domain, spec),
           "phi_start": float(cur.mean), "valid_start": float(cur.valid_frac)}
    native_prompt = getattr(a, "native_prompt", "served")
    if native:
        out.update(native=True, instruction_sha1=C.instruction_for(domain)[1], native_prompt=native_prompt)
    elif spoken:                          # verb surface: SET_PARAM data, verb-spelled prompt
        out.update(verb_surface=True, instruction_sha1=C.instruction_for(domain)[1], native_prompt=native_prompt)

    def obs_native():
        return observation_native(C, domain, spec, st, metrics, binding, history, prev_metrics,
                                  variant, native_prompt)

    for t in range(a.turns):
        if cur.feasible:
            break
        metrics = dict(cur.agg_metrics)
        binding = L._binding_constraint(ev.program, metrics)
        rec = {"t": t, "binding": binding}
        if a.policy == "llm":
            if spoken:
                system, user = obs_native()
            elif v2:
                system, user = observation_v2(R, domain, spec, st, metrics, binding, history, prev_metrics,
                                              variant, instruction)
            else:
                system, user = observation(R, domain, spec, st, metrics, binding, history, variant, instruction)
            if getattr(a, "record_prompts", False):
                rec.update(system=system, user=user)       # the exact prompt, for SFT data (DX-21)
            # DX-25b loop guard: a degenerate proposal is refused and the model re-asked in the SAME
            # turn (the rejection note appended to the user text, so T=0 can answer differently), up to
            # --guard-retries times; a proposal still degenerate after that is applied anyway
            # (guard_forced), never silently dropped -- a dropped move would leave state and prompt
            # unchanged and, at T=0, burn every remaining turn on the same proposal.
            # (v3: no guard on a native variant -- an exact undo is SET_PARAM-specific.)
            retries = (int(getattr(a, "guard_retries", 2))
                       if getattr(a, "loop_guard", False) and not native else 0)
            note, guard_events, refused = "", [], None
            for g_try in range(retries + 1):
                SIM.release()
                try:
                    reply = llm(system, user + note)
                finally:
                    SIM.acquire()
                mv, why = parse_move_native(C, domain, reply) if spoken else parse_move(reply)
                if native or not getattr(a, "loop_guard", False) or mv is None:
                    break
                g = guard_reason(domain, st, history, mv)
                if g is None:
                    break
                refused = dict(mv)           # valid when refused; the state has not changed since
                guard_events.append({"try": g_try, "reason": g[0], "proposed": dict(mv)})
                if g_try == retries:
                    rec["guard_forced"] = True
                    break
                note = "\n[loop guard] " + g[1] + "\n"
            # A refused valid move is never silently dropped: if the retry produced nothing applicable
            # (unparsed, or naming a non-existent element/parameter), apply the refused move instead --
            # dropping it would leave state and prompt unchanged and, at T=0, burn every remaining turn
            # on the same exchange. A server error stays loud (fail="error" -> n_err -> exit 3).
            if refused is not None and not rec.get("guard_forced") and why != "error" and (
                    mv is None or not check_move(domain, st, mv)[0]):
                mv, why = refused, None
                rec["guard_forced"] = True
                rec["guard_fallback"] = "retry_unusable"
            if guard_events:
                rec["guard"] = guard_events
            keep = getattr(a, "record_prompts", False)       # SFT data keeps the whole thinking (A2 may use it)
            rec.update(reasoning=reply.get("reasoning", "") if keep else reply.get("reasoning", "")[:4000],
                       answer=reply.get("answer", "") if keep else reply.get("answer", "")[:600],
                       finish=reply.get("finish"), completion_tokens=reply.get("completion_tokens"))
        else:
            if getattr(a, "record_prompts", False):   # the same observation, for tests of the SFT tier builder
                rec["system"], rec["user"] = (
                    obs_native() if spoken else
                    observation_v2(R, domain, spec, st, metrics, binding, history, prev_metrics, variant, instruction)
                    if v2 else observation(R, domain, spec, st, metrics, binding, history, variant, instruction))
            if a.policy == "replay":                   # a recorded move sequence (DX-21's phi_select control)
                acts = getattr(a, "replay_actions", None) or []
                mv = dict(acts[len(history)]) if len(history) < len(acts) else None
                if native and mv is not None:
                    mv = D.Move.from_dict(mv)
                why = None if mv else "replay_exhausted"
            elif a.policy == "phi_select":             # v3: the ladder's Phi-select teacher
                mv, why = phi_select(D, domain, st, rng, ev, pid, t, cur)
                if mv is not None and not native:      # SET_PARAM: into the v1/v2 apply path below
                    d = mv.as_dict()
                    mv = {"i": d["i"], "p": d["p"], "v": d["v"]}
            elif native:
                mv = scripted_native(a.policy, D, domain, st, rng, history)
                why = None if mv is not None else "no_move"
            else:
                mv = scripted(a.policy, D, L, domain, st, rng, history)
                why = None if mv else "no_move"
        if native and mv is not None:
            # v3: apply the Move on a clone of the live state. A refused move (the state does not
            # admit it) spends the turn with the design unchanged and is a fail, as an invalid
            # SET_PARAM is; an applied move that changes no metric is a noop.
            ad = mv.as_dict()
            child = domain.clone(st)
            try:
                domain.apply(child, mv)
            except Exception as e:                     # noqa: BLE001
                why = "illegal:%s" % str(e)[:80]
                rec.update(move=ad, action=ad)
            else:
                # the history is labelled AFTER the change, as render_cot.iter_states /
                # ownrows.select_states label it (the v2 path labels after the change too)
                history.append((R._label_for(domain, child, ad), ad))
                prev_metrics = metrics
                st = child
                cur = ev.score(st, seed0=D.stable_seed(pid, "eval", "s", t + 1))
                after = {k: float(v) for k, v in cur.agg_metrics.items()}
                rec.update(move=ad, v_old=None, v_new=None, clamped=False,
                           noop=(after == {k: float(v) for k, v in metrics.items()}), action=ad)
                if cur.feasible and out["first_feasible_turn"] is None:
                    out["first_feasible_turn"] = t + 1
                    ob = cur.agg_metrics.get(okey) if okey else None
                    out["obj_at_first_feasible"] = float(ob) if ob is not None else None
        elif mv is not None:
            ok, why, v_old = check_move(domain, st, mv)
            if ok:
                child = domain.clone(st)
                domain.set(child, mv["i"], mv["p"], mv["v"])
                v_new = float(domain.get(child, mv["i"], mv["p"]))
                rec.update(move=mv, v_old=v_old, v_new=v_new,
                           clamped=abs(v_new - mv["v"]) > 1e-9 * max(1.0, abs(mv["v"])),
                           noop=abs(v_new - v_old) <= 1e-12 * max(1.0, abs(v_old)))
                # v1 labels the element as it was before the change; v2 after it, as run2's
                # render_cot.iter_states does (a verb-surface domain is labelled after it too, as
                # ownrows.labelled_history labels its training rows)
                applied = {"i": mv["i"], "p": mv["p"], "v": v_new, "v_old": v_old}
                history.append((domain.element_label(child if (v2 or spoken) else st, mv["i"]), applied))
                if spoken:                   # verb surface: the applied move, as the native path records it
                    rec["action"] = applied
                prev_metrics = metrics
                st = child
                cur = ev.score(st, seed0=D.stable_seed(pid, "eval", "s", t + 1))
                if cur.feasible and out["first_feasible_turn"] is None:
                    out["first_feasible_turn"] = t + 1
                    ob = cur.agg_metrics.get(okey) if okey else None
                    out["obj_at_first_feasible"] = float(ob) if ob is not None else None
        rec["fail"] = why
        rec["feasible_after"] = bool(cur.feasible)
        # DX-27: Phi / validity / metrics of the design AFTER this turn (unchanged design if the turn
        # failed), so closed-loop readouts need no replay; "before" is the previous turn's "after"
        # (turn 0: phi_start / valid_start on the episode).
        rec["phi_after"] = float(cur.mean)
        rec["valid_after"] = float(cur.valid_frac)
        rec["metrics_after"] = {k: float(v) for k, v in cur.agg_metrics.items()}
        turns.append(rec)
    out["turns"] = turns
    for k in (12, 20):
        out["feasible@%d" % k] = out["first_feasible_turn"] is not None and out["first_feasible_turn"] <= k
    ref = out["reference"]
    out["obj_over_ref"] = (out["obj_at_first_feasible"] / ref
                           if out["obj_at_first_feasible"] is not None and ref not in (None, 0) else None)
    return out


def _one_episode():
    """Hidden mode: one episode in this process. Payload (pickle) on stdin, result (pickle) on stdout."""
    import argparse
    import pickle
    p = pickle.load(sys.stdin.buffer)
    a = argparse.Namespace(**p["a"])
    a.replay_actions = p.get("replay")
    if os.environ.get("DX_CRASH_TEST") == p["spec"]["problem_id"]:      # exercises the crash path only
        import signal
        os.kill(os.getpid(), signal.SIGSEGV)
    llm = LLM(a) if a.policy == "llm" else None
    out = run_episode(import_dataxp(a.code), p["variant"], p["spec"], a, llm, p["instruction"])
    out["_tokens"] = dict(llm.tokens) if llm else None
    sys.stdout.buffer.write(pickle.dumps(out))


def cmd_episodes(a):
    """Every episode runs in its OWN child process (the threads here only wait on children), and is
    written as it finishes. j81 and j100 (episodes in 32 threads, with or without a lock around the
    simulators) died of a segmentation fault a few minutes in, which greedy through the same threads
    on the same problems never does; the cause is not pinned yet. Now a native crash costs one
    episode: it is retried once, and if it crashes again it is recorded as {"crashed": true} (not
    feasible), counted, and its faulthandler stack is printed."""
    import pickle
    import subprocess
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(k, "1")
    D, L, R, get_domain = import_dataxp(a.code)
    instruction = open(a.instruction_file).read() if a.instruction_file else ""
    domains = {v: get_domain(v) for v in a.variants}
    # verb-spelled variants (native, or a v2 domain with a verb surface) build their instruction
    # in-process (common.instruction_for); the file is theirs to ignore, as is --param-line
    instr_of = {v: ("" if speaks_verbs(D, domains[v]) else instruction_for(instruction, domains[v], a.param_line))
                for v in a.variants}
    natives = [v for v in a.variants if is_native(domains[v])]
    surfaces = [v for v in a.variants if speaks_verbs(D, domains[v]) and not is_native(domains[v])]
    if natives and not a.problems_file:
        raise SystemExit("native variants %s take their problems from --problems-file (the generated "
                         "corpus; make_problem is not used for them)" % natives)
    jobs = []
    if a.problems_file:                           # the same specs for every arm (DX-23)
        have = collections.defaultdict(list)
        for line in open(a.problems_file):
            r = json.loads(line)
            have[r["variant"]].append(r["spec"])
        for v in a.variants:
            if len(have[v]) < a.n:
                raise SystemExit("--problems-file has %d problems for %s, need %d" % (len(have[v]), v, a.n))
            jobs += [(v, spec) for spec in (have[v][: a.n] if a.n > 0 else have[v])]   # --n 0: every problem in the file
    else:
        for v in a.variants:
            for spec in fresh_problems(D, get_domain, v, a.n, a.run1, a.start_index):
                jobs.append((v, spec))
    if a.shard:                                   # "k/m" or "k1,k2/m": every m-th problem from k (each k)
        ks, m = a.shard.split("/")
        m = int(m)
        jobs = [j for k in (int(x) for x in ks.split(",")) for j in jobs[k::m]]
    print("%d episodes over %s, policy %s, up to %d turns, %d at a time (one process each)%s%s"
          % (len(jobs), a.variants, a.policy_label or a.policy, a.turns, a.workers,
             (" | native: %s (prompt %s)" % (natives, a.native_prompt)) if natives else "",
             (" | verb surface: %s (prompt %s)" % (surfaces, a.native_prompt)) if surfaces else ""), flush=True)
    t0 = time.time()
    res, tokens, crashed = {}, collections.Counter(), []
    fo = open(a.out + ".episodes.jsonl", "w")
    payload_a = {k: v for k, v in vars(a).items()}
    replay = {}
    if a.policy == "replay":
        for line in open(a.replay):
            r = json.loads(line)
            replay[r["problem_id"]] = r["actions"]
        missing = [s["problem_id"] for _, s in jobs if s["problem_id"] not in replay]
        if missing:
            raise SystemExit("--replay has no actions for %d problems, e.g. %s" % (len(missing), missing[:3]))

    def one(k):
        v, spec = jobs[k]
        data = pickle.dumps({"a": payload_a, "variant": v, "spec": spec, "instruction": instr_of[v],
                             "replay": replay.get(spec["problem_id"])})
        for attempt in (1, 2):
            pr = subprocess.run([sys.executable, "-X", "faulthandler", os.path.abspath(__file__), "_one_episode"],
                                input=data, capture_output=True, env=env)
            if pr.returncode == 0:
                return k, pickle.loads(pr.stdout), None
            err = pr.stderr.decode(errors="replace")
            print("!!! %s: child exited rc=%d (attempt %d)\n%s" % (spec["problem_id"], pr.returncode, attempt,
                  "\n".join(l for l in err.splitlines() if "arning" not in l)[-3000:]), flush=True)
        return k, {"variant": v, "problem_id": spec["problem_id"], "policy": a.policy_label or a.policy,
                   "crashed": True, "turns": [], "first_feasible_turn": None, "feasible@12": False,
                   "feasible@20": False, "obj_at_first_feasible": None, "obj_over_ref": None}, pr.returncode

    from concurrent.futures import as_completed
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, k) for k in range(len(jobs))]
        for fu in as_completed(futs):                    # written as each finishes, not in submission order
            k, r, rc = fu.result()
            tokens.update(r.pop("_tokens", None) or {})
            if rc is not None:
                crashed.append((r["problem_id"], rc))
            res[k] = r
            fo.write(json.dumps(r) + "\n")
            fo.flush()
    fo.close()
    out = [res[k] for k in sorted(res)]
    summarize_episodes(out)
    # a turn whose model call failed after all retries ("error") is the server's failure, not the
    # model's: it must not pass silently as an unsolved episode
    n_err = sum(1 for r in out for t in r["turns"] if t.get("fail") == "error")
    err_eps = sorted({r["problem_id"] for r in out for t in r["turns"] if t.get("fail") == "error"})
    print("elapsed %.0f s | crashed %d %s | server-error turns %d in %d episodes %s%s" % (
        time.time() - t0, len(crashed), crashed[:10], n_err, len(err_eps), err_eps[:10],
        (" | tokens %s" % dict(tokens)) if tokens else ""))
    if n_err:
        sys.exit(3)


def instruction_for(instruction, domain, param_line):
    """The task instruction served for one domain. With --param-line (DX-24) one line naming the domain's
    settable parameters follows it -- the grid2 control of DX-23 (task_instruction_V2_grid2params.txt) is
    exactly V2 + this line for grid2."""
    if not param_line:
        return instruction
    return instruction + "The parameter names in this problem are exactly: %s.\n" % ", ".join(domain.params)


def cmd_problems(a):
    """Write the evaluation problems once (DX-23): {variant, spec} per line, the first --n initially
    infeasible problems from --start-index, none equal to a problem in --run1 (pass run2 for v2)."""
    D, L, R, get_domain = import_dataxp(a.code)
    n = 0
    with open(a.out, "w") as fo:
        for v in a.variants:
            for spec in fresh_problems(D, get_domain, v, a.n, a.run1, a.start_index):
                fo.write(json.dumps({"variant": v, "spec": spec}, default=_json_default) + "\n")
                n += 1
            print(v, "done", flush=True)
    print("wrote %d problems to %s" % (n, a.out))


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:            # noqa: BLE001
        pass
    return str(o)


def summarize_episodes(res):
    by = collections.defaultdict(list)
    for r in res:
        by[r["variant"]].append(r)
    print("%-10s %4s %9s %9s %7s %7s %7s %7s %8s" % ("variant", "n", "feas@12", "feas@20", "moves", "fail", "noop", "clamp", "obj/ref"))
    for v in sorted(by):
        rs = by[v]
        moves = [t for r in rs for t in r["turns"]]
        orr = sorted(r["obj_over_ref"] for r in rs if r["obj_over_ref"] is not None)
        print("%-10s %4d %9.3f %9.3f %7d %7d %7d %7d %8s" % (
            v, len(rs), sum(r["feasible@12"] for r in rs) / len(rs), sum(r["feasible@20"] for r in rs) / len(rs),
            len(moves), sum(1 for t in moves if t.get("fail")), sum(1 for t in moves if t.get("noop")),
            sum(1 for t in moves if t.get("clamped")), ("%.3f" % orr[len(orr) // 2]) if orr else "-"))


# ------------------------------------------------------------------ teacher states

def cmd_states(a):
    D, L, R, get_domain = import_dataxp(a.code)
    rows = [json.loads(l) for l in open(a.tier)]
    if a.limit:
        rows = rows[: a.limit]
    # replay each needed trajectory to the rows' turns (render_cot.iter_states, repair phase)
    need = collections.defaultdict(set)
    for r in rows:
        need[r["domain"]].add((r["instance"], r["turn"]))
    states = {}
    for v in need:
        for domain, spec, st, turn, hist in R.iter_states(a.run1, v, True):
            key = (turn["problem_id"], turn["turn"])
            if key in need[v]:
                states[(v,) + key] = (domain, domain.clone(st))
    llm = LLM(a) if a.policy == "llm" else None
    # a verb-surface domain's tier rows are verb-spelled: read the reply with domain.parse_move
    # (SET_PARAM domains keep the ACTION regex; native domains have no states tiers)
    doms = {v: get_domain(v) for v in need}
    spoken = {v: speaks_verbs(D, d) and not is_native(d) for v, d in doms.items()}
    C = import_common() if any(spoken.values()) else None

    def one(r):
        rec = {"domain": r["domain"], "problem_id": r["instance"], "turn": r["turn"], "gold": r.get("action")}
        if a.policy == "llm":
            reply = llm(r["messages"][0]["content"], r["messages"][1]["content"])
            mv, why = (parse_move_native(C, doms[r["domain"]], reply) if spoken.get(r["domain"])
                       else parse_move(reply))
            full = getattr(a, "full_text", False)     # DX-29c: own rationales become training text
            rec.update(reasoning=reply.get("reasoning", "") if full else reply.get("reasoning", "")[:4000],
                       answer=reply.get("answer", "") if full else reply.get("answer", "")[:600],
                       finish=reply.get("finish"), completion_tokens=reply.get("completion_tokens"))
        elif a.policy == "gold":
            g = r["action"]
            mv, why = {"i": int(g["i"]), "p": g["p"], "v": float(g["v"])}, None
        else:
            raise SystemExit("states mode takes --policy llm or gold")
        rec["move"], rec["fail"] = mv, why
        with SIM:                       # simulator calls; see SIM
            ds = states.get((r["domain"], r["instance"], r["turn"]))
            if mv is not None and ds is not None:
                domain, st = ds
                ok, why2, v_old = check_move(domain, st, mv)
                rec["valid"] = ok
                if not ok:
                    rec["fail"] = why2
                else:
                    child = domain.clone(st)
                    domain.set(child, mv["i"], mv["p"], mv["v"])
                    v_new = float(domain.get(child, mv["i"], mv["p"]))
                    g = r.get("action") or {}
                    rec.update(v_old=v_old, v_new=v_new,
                               clamped=abs(v_new - mv["v"]) > 1e-9 * max(1.0, abs(mv["v"])),
                               noop=abs(v_new - v_old) <= 1e-12 * max(1.0, abs(v_old)),
                               same_ip=(mv["i"] == int(g.get("i", -1)) and mv["p"] == g.get("p")),
                               same_dir=(mv["i"] == int(g.get("i", -1)) and mv["p"] == g.get("p")
                                         and (v_new > v_old) == (float(g["v"]) > float(g["v_old"]))),
                               exact=(mv["i"] == int(g.get("i", -1)) and mv["p"] == g.get("p")
                                      and abs(v_new - float(g["v"])) <= 1e-2 * max(1.0, abs(float(g["v"])))),
                               step_ratio=(abs(v_new - v_old) / abs(float(g["v"]) - float(g["v_old"]))
                                           if g and abs(float(g["v"]) - float(g["v_old"])) > 0 else None))
            elif mv is not None:
                rec["fail"] = "state_not_replayed"
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, rows))
    with open(a.out + ".rows.jsonl", "w") as fr, open(a.out + ".actions.jsonl", "w") as fa:
        for rec in res:
            fr.write(json.dumps(rec) + "\n")
            if rec.get("valid"):
                fa.write(json.dumps({"problem_id": rec["problem_id"], "turn": rec["turn"],
                                     "action": {"i": rec["move"]["i"], "p": rec["move"]["p"], "v": rec["v_new"]}}) + "\n")
    by = collections.defaultdict(list)
    for rec in res:
        by[rec["domain"]].append(rec)
    print("%-10s %4s %7s %7s %7s %7s %7s %7s %7s" % ("domain", "n", "parsed", "valid", "noop", "same_ip", "samedir", "exact", "step~"))
    for d in sorted(by):
        rs = by[d]
        n = len(rs)
        sr = sorted(x["step_ratio"] for x in rs if x.get("step_ratio") is not None)
        print("%-10s %4d %7.3f %7.3f %7d %7.3f %7.3f %7.3f %7s" % (
            d, n, sum(x["move"] is not None for x in rs) / n, sum(bool(x.get("valid")) for x in rs) / n,
            sum(bool(x.get("noop")) for x in rs), sum(bool(x.get("same_ip")) for x in rs) / n,
            sum(bool(x.get("same_dir")) for x in rs) / n, sum(bool(x.get("exact")) for x in rs) / n,
            ("%.2f" % sr[len(sr) // 2]) if sr else "-"))
    fails = collections.Counter(x["fail"] for x in res if x.get("fail"))
    print("fail reasons:", dict(fails), "| elapsed %.0f s%s" % (time.time() - t0, (" | tokens %s" % dict(llm.tokens)) if llm else ""))


def main():
    import faulthandler
    faulthandler.enable()               # a native crash prints every thread's Python stack
    ap = argparse.ArgumentParser()
    if len(sys.argv) > 1 and sys.argv[1] == "_one_episode":
        return _one_episode()
    ap.add_argument("mode", choices=["states", "episodes", "problems"])
    ap.add_argument("--code", required=True)
    ap.add_argument("--run1", default=None)
    ap.add_argument("--tier")
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--policy", default="llm",
                    help="llm | gold (states) | greedy | random | repeat_last | replay | phi_select (episodes)")
    ap.add_argument("--replay", default=None,
                    help="episodes, --policy replay: jsonl of {problem_id, actions:[{i,p,v} | {verb,args}]}")
    ap.add_argument("--policy-label", default=None, help="name recorded in the output, e.g. the adapter")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--base-url", default="http://127.0.0.1:8771/v1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--instruction-file", default=None, help="appended to the user turn in episodes (F3)")
    ap.add_argument("--start-index", type=int, default=10000,
                    help="episodes: first problem index (10000 = DX-14's evaluation problems; 20000+ = DX-21 training draws)")
    ap.add_argument("--shard", default=None, help="episodes: k/m (or k1,k2/m), every m-th problem starting at k")
    ap.add_argument("--record-prompts", action="store_true", help="episodes: keep each turn's system/user prompt")
    ap.add_argument("--full-text", action="store_true", help="states: keep the whole reasoning and answer (default truncates)")
    ap.add_argument("--obs", choices=["v1", "v2"], default="v1",
                    help="episodes: v1 = build_prompt + split_prompt (run1 domains); v2 = render_cot.policy_observation (run2's *2 domains)")
    ap.add_argument("--problems-file", default=None, help="episodes: read {variant, spec} lines written by the problems mode")
    ap.add_argument("--param-line", action="store_true",
                    help="episodes: append one line naming the domain's parameters to the instruction (DX-24)")
    ap.add_argument("--loop-guard", action="store_true",
                    help="episodes, llm: refuse an exact undo of the previous applied move; re-ask up to "
                         "--guard-retries times in the same turn, then apply the proposal anyway (DX-25b)")
    ap.add_argument("--guard-retries", type=int, default=2)
    ap.add_argument("--native-prompt", choices=["served", "probe"], default="served",
                    help="episodes, verb-spelled variants (v3: native, or a v2 domain with a verb surface): "
                         "served = the own-row training rows' messages (DOMAIN/GOAL head; observation + "
                         "INSTRUCTION_NATIVE + grammar); probe = the own-row sampler's probe prompt "
                         "(own_system_prompt + observation + tail)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"states": cmd_states, "episodes": cmd_episodes, "problems": cmd_problems}[a.mode](a)


if __name__ == "__main__":
    main()
