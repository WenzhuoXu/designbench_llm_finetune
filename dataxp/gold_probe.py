"""Does the model land on the gold move? And can it rationalize gold truthfully?

Two questions about synthetic CoT for SFT on dataxp v2 (run2) gold trajectories:

  Option 1 - own choice: the model sees the policy observation (optionally with
  the full previous trace, and/or the LATER gold steps as privileged context)
  and picks a move with thinking on. How often does its own choice match the
  gold move? If often and stably, its own reasoning traces (filtered to the
  matching ones) are causal CoT for the gold action.

  Option 2 - rationalization: the model is GIVEN the gold action (and the
  previous and post traces, i.e. it knows what happened) and writes the
  reasoning. Is that reasoning true (state facts and effect claims check out
  against the observation and the actual replayed outcome), stable across
  samples, and sensible?

    python -m dataxp.gold_probe prepare --data <run2> --out <dir> --n 20
    python -m dataxp.gold_probe run --states <dir>/states.jsonl --mode own --k 4 \
        --endpoint http://127.0.0.1:8766/v1 --out <dir>/own.jsonl
    python -m dataxp.gold_probe score --states <dir>/states.jsonl --runs <dir>/own.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .domain import (Move, PhiEvaluator, action_form, apply_action, current_spelling,
                     grammar_lead, parse_verb_calls, speaks_verbs, stable_seed)
from .domains import get_domain
from .render_cot import (MODEL, NUMBER, _fmtv, grounded, policy_observation,
                         strip_think)

VARIANTS = ("frame3d2", "circuit2", "grid2", "traffic2", "thermal2", "seir2", "inventory2")

KIND = {  # how a move on each param is compared with gold
    "r": "cont", "active": "tog", "mat": "cat", "dx": "add", "dy": "add", "dz": "add",
    "W": "cont", "L": "cont", "B": "cont", "Cc": "cont", "comp": "cat", "Ib": "cont",
    "Rz": "cont", "rz_on": "tog", "cas_on": "tog",
    "p_set": "cont", "on": "tog", "closed": "tog", "shift": "add",
    "cycle": "cat", "green_frac": "cont", "offset": "add", "phase_order": "tog",
    "lanes_main": "cat",
    "site": "cat", "g": "cont", "slot": "cat",
    "contact_reduction": "cont", "start_day": "add", "duration": "add", "test_rate": "cont",
    "vax_share": "cont", "school_policy": "cat", "mask_mandate": "tog",
    "base_stock": "cont", "review": "cat", "expedite": "tog",
}

ACTION_RE = re.compile(r"SET_PARAM\(\s*(\d+)\s*,\s*([A-Za-z_]\w*)\s*,\s*(-?[\d.eE+-]+)\s*\)")

# Verb-surface domains (docs/dataxp_verb_surface.md): the stored actions are
# still {i, p, v, v_old}, but every printed action and every parsed reply goes
# through the domain's format_move / parse_move. On a plain v2 domain these
# helpers reproduce the historical SET_PARAM text and the ACTION_RE rule
# byte for byte.
_DOMAINS = {}


def _domain(variant):
    d = _DOMAINS.get(variant)
    if d is None:
        d = _DOMAINS[variant] = get_domain(variant)
    return d


def _verbs(domain) -> bool:
    return speaks_verbs(domain)


def _fmt_action(domain, a: dict) -> str:
    """An action dict as the prompts write it."""
    if _verbs(domain) or "verb" in a:
        return domain.format_move(Move.from_dict(a))
    return "SET_PARAM(%d, %s, %s)" % (int(a["i"]), a["p"], _fmtv(a["v"]))


def _was(domain, a: dict, template: str) -> str:
    """template % <the value the action replaced>, or '' when the verb pair
    already says it (a toggle on a verb-surface domain)."""
    if _verbs(domain):
        w = domain.format_value(a["p"], a.get("v_old")) if "p" in a else ""
    else:
        w = _fmtv(a["v_old"])
    return (template % w) if w else ""


def _verb_of(domain, p: str) -> str:
    """How a parameter is named in a hint: the verb (pair) that spells it
    under the multitool spelling; 'parameter p' under set_parameter (the
    SET_PARAM line names it) and on a plain domain."""
    pv = (domain.verb_param(p) if getattr(domain, "verb_surface", False)
          and current_spelling() == "multitool" else None)
    if pv is None:
        return "parameter %s" % p
    name = pv.verb
    c = pv.codec
    if getattr(c, "off", None):
        name += " / " + c.off
    if pv.select is not None:
        name += " with %s %s" % pv.select
    return name


def _last_action(domain, text: str):
    """(i, p, v) of the action named LAST in text: the last SET_PARAM(...)
    anywhere (ACTION_RE, the historical rule) on a plain domain; the last
    verb call parse_move accepts on a verb-surface domain."""
    if not _verbs(domain):
        m = ACTION_RE.findall(text or "")
        return (int(m[-1][0]), m[-1][1], float(m[-1][2])) if m else None
    found = None
    for verb, toks in parse_verb_calls(text or ""):
        mv = domain.parse_move("%s(%s)" % (verb, ", ".join(toks)))
        if mv is not None and mv.is_set_param:
            found = (int(mv.args["i"]), mv.args["p"], float(mv.args["v"]))
    return found


def _last_action_text(domain, text: str):
    """The last action in text as a string: the matched SET_PARAM text on a
    plain domain (as prompt_compress printed it), the canonical format_move
    spelling on a verb-surface domain; None when there is none."""
    if not _verbs(domain):
        m = ACTION_RE.findall(text or "")
        return ("SET_PARAM(%s, %s, %s)" % m[-1]) if m else None
    a = _last_action(domain, text)
    if a is None:
        return None
    return domain.format_move(Move("SET_PARAM", {"i": a[0], "p": a[1], "v": a[2], "v_old": None}))


_SYSTEM_ACTION_LINE = "<action>SET_PARAM(<element index>, <parameter>, <value>)</action>"


def _system_for(domain, system: str) -> str:
    """A probe system prompt on a verb-surface domain: its SET_PARAM action
    line becomes the spelling's action form (VERB(arg, ...), or
    SET_PARAM(element, parameter, value) under DATAXP_SPELLING=set_parameter)
    plus grammar() (the substitution render_cot.system_prompt makes) and,
    under multitool, the candidates template names a verb call. Plain
    domains get the text unchanged."""
    if not _verbs(domain):
        return system
    g = "\n".join("    " + l for l in domain.grammar().rstrip().splitlines())
    out = system.replace(_SYSTEM_ACTION_LINE, (
        "<action>%s</action>\n%s\n" % (action_form(domain), grammar_lead()) + g))
    if current_spelling() == "multitool":
        out = out.replace('"SET_PARAM(<i>, <p>, <v>)"', '"VERB(arg, ...)"')
    return out


# ---------------------------------------------------------------- prepare ---
def _metrics_line(before, after):
    keys = [k for k in after if k in before]
    return "; ".join("%s %s -> %s" % (k, _fmtv(before[k]), _fmtv(after[k])) for k in sorted(keys))


def prepare(args):
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for variant in args.variants:
        domain = get_domain(variant)
        acc = os.path.join(args.data, variant, "accepted")
        specs = {json.loads(l)["problem_id"]: json.loads(l)
                 for l in open(os.path.join(acc, "problems.jsonl"))}
        rng = np.random.default_rng(stable_seed("gold_probe", variant))
        n = 0
        for line in open(os.path.join(acc, "trajectories.jsonl")):
            if n >= args.n:
                break
            t = json.loads(line)
            turns = t["turns"]
            rep = [k for k, tt in enumerate(turns) if tt.get("binding_constraint") is not None]
            if not rep:
                continue
            k = int(rep[int(rng.integers(len(rep)))])
            spec = specs[t["problem_id"]]
            st = domain.load(spec)
            history = []
            for j in range(k):
                a = turns[j]["action"]
                apply_action(domain, st, a)
                history.append((domain.element_label(st, int(a["i"])), a))
            turn = dict(turns[k])
            turn["problem_id"] = t["problem_id"]
            if k > 0 and turns[k - 1]["turn"] == turn["turn"] - 1:
                turn["prev_state_metrics"] = turns[k - 1]["state_metrics"]
            lines, _ = policy_observation(domain, spec, st, turn, history, variant)
            obs = "\n".join(lines)

            prev = []
            for j in range(k):
                a = turns[j]["action"]
                prev.append("step %d: %s%s -> %s" % (
                    j + 1, _fmt_action(domain, a), _was(domain, a, " [was %s]"),
                    _metrics_line(turns[j]["state_metrics"], turns[j + 1]["state_metrics"])))
            # post trace: what happened AFTER this step's (hidden) change
            post = []
            for j in range(k + 1, len(turns)):
                a = turns[j]["action"]
                after = turns[j + 1]["state_metrics"] if j + 1 < len(turns) else None
                post.append("step %d: %s%s%s" % (
                    j + 1, _fmt_action(domain, a), _was(domain, a, " [was %s]"),
                    (" -> " + _metrics_line(turns[j]["state_metrics"], after)) if after else ""))
            outcome = turns[k + 1]["state_metrics"] if k + 1 < len(turns) else None

            # score the legal moves at this state (for Phi percentiles)
            ev = PhiEvaluator(domain, spec, offset_rule="infeasible")
            crng = np.random.default_rng(stable_seed(t["problem_id"], "gp", k))
            moves = []
            for c in domain.legal_moves(st, crng):
                ch = domain.clone(st)
                domain.apply(ch, c)
                d = c.as_dict()      # {i, p, v, v_old} for the v2 variants this probe targets
                moves.append({"i": d["i"], "p": d["p"], "v": d["v"],
                              "phi": ev.score(ch, seed0=stable_seed(t["problem_id"], "gp", d["i"], d["p"])).mean})
            g = turns[k]["action"]
            ch = domain.clone(st)
            apply_action(domain, ch, g)
            gold_phi = ev.score(ch, seed0=stable_seed(t["problem_id"], "gp", int(g["i"]), g["p"])).mean
            rows.append({
                "sid": "%s|%s|%d" % (variant, t["problem_id"], k), "variant": variant,
                "spelling": current_spelling() if _verbs(domain) else "SET_PARAM",
                "problem_id": t["problem_id"], "k": k, "spec": spec,
                "history": [a for _, a in history], "gold": g, "gold_phi": gold_phi,
                "moves": moves, "obs": obs, "prev_trace": prev, "post_trace": post,
                "gold_outcome": outcome, "state_metrics": turns[k]["state_metrics"],
            })
            n += 1
        print("%-11s %d states" % (variant, n), flush=True)
    with open(os.path.join(args.out, "states.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# ----------------------------------------------------------------- prompts ---
SYSTEM_OWN = """You are a design engineer working on a constrained design problem, one change per step. You see the full current observation. Think it through carefully, then choose exactly ONE change that best moves the design toward meeting every constraint at low objective cost (feasibility first, then the objective). End your answer with exactly one line:
<action>SET_PARAM(<element index>, <parameter>, <value>)</action>"""


def prompt_own(row, with_prev=False, with_post=False):
    parts = [row["obs"]]
    if with_prev and row["prev_trace"]:
        parts += ["", "FULL RECORD OF THE PREVIOUS STEPS (change -> effect on every metric)"]
        parts += ["  " + s for s in row["prev_trace"]]
    if with_post and row["post_trace"]:
        parts += ["", "FOR REFERENCE: THE LATER STEPS OF A SUCCESSFUL EPISODE FROM THIS STATE "
                      "(the change for THIS step is not shown; the first line starts from the state "
                      "after it)"]
        parts += ["  " + s for s in row["post_trace"]]
    parts += ["", "Choose exactly ONE change for THIS step."]
    return "\n".join(parts)


SYSTEM_OWN_WRITTEN = SYSTEM_OWN.replace(
    "Think it through carefully, then choose",
    "Write your reasoning first (what is violated, what each change you consider "
    "would do according to the observation's own numbers, and why the one you choose wins), then choose")


def prompt_hint(row, sample, mode):
    """Own choice with a hint that does not state the move:
    own_elem : the element and parameter of the gold move (value free);
    own_cands: a shuffled list of 4 candidate moves (gold + one of the five best other moves by Phi
               + two random legal moves), the model evaluates and picks."""
    g = row["gold"]
    domain = _domain(row["variant"])
    parts = [row["obs"], ""]
    if mode == "own_elem":
        parts.append("HINT: the change for this step is to element %d, %s; the value is "
                     "for you to decide." % (int(g["i"]), _verb_of(domain, g["p"])))
    else:
        rng = np.random.default_rng(stable_seed(row["sid"], "cands", int(sample)))
        moves = [m for m in row.get("moves") or []
                 if not _match(g, int(m["i"]), m["p"], float(m["v"]))[1]]
        top = sorted(moves, key=lambda m: -m["phi"])[:5]
        alts = []
        if top:
            alts.append(top[int(rng.integers(len(top)))])
        rest = [m for m in moves if m not in alts]
        for _ in range(2):
            if rest:
                alts.append(rest.pop(int(rng.integers(len(rest)))))
        cands = alts + [{"i": int(g["i"]), "p": g["p"], "v": g["v"]}]
        order = rng.permutation(len(cands))
        parts.append("CANDIDATE CHANGES (at least one of these is a good move; evaluate each from the "
                     "observation's own numbers and pick the best; pick a change outside this list only "
                     "if the observation clearly supports it):")
        parts += ["  " + _fmt_action(domain, cands[j]) for j in order]
    parts += ["", "Choose exactly ONE change for THIS step."]
    return "\n".join(parts)


SYSTEM_RAT = """You write the internal reasoning of a design engineer at one step of an iterative design process. You are given the observation the engineer had, the change the engineer made at this step, and - for your information only - what actually happened afterwards. Write the reasoning that leads to that change.

Rules:
- The reasoning may use ONLY information in the observation: never quote a number that appears only in the outcome or later steps.
- Every statement about the current design must be correct for the observation (values, comparisons, rankings, which element is which).
- You may say what the change is expected to do, but only qualitatively (direction), and only if the actual outcome confirms it. Do not claim effects the outcome contradicts.
- Explain why THIS element and THIS kind of change (and roughly this size), from what the observation shows.
- First person, present tense.
- After the reasoning, put the action line, then a claims block:
<action>SET_PARAM(<element index>, <parameter>, <value>)</action>
<claims>{"effects": [{"metric": "<exact metric name from CONSTRAINT STATUS>", "direction": "up|down|unchanged"}]}</claims>
List in "effects" every effect on an aggregate metric that your reasoning states or implies (empty list if none)."""


SYSTEM_COMPRESS = """You condense an engineer's own working notes into the short reasoning they would write down before making a change. You get the observation, the engineer's private working (long, exploratory), and the change they chose. Write the reasoning that actually led to that change, keeping the decisive comparisons and calculations from the working notes.

Rules:
- Keep only what the working notes actually establish; do not add new arguments.
- Every number must come from the observation (or be simple arithmetic on observation numbers that the notes performed). Drop numbers you cannot trace to the observation.
- Do not mention the notes, a summary, or alternatives you did not consider in the notes.
- First person, present tense.
- End with exactly the chosen action line, unchanged:
<action>SET_PARAM(<element index>, <parameter>, <value>)</action>"""


def prompt_compress(row, own):
    act = _last_action_text(_domain(row["variant"]), own.get("answer") or "") or "(none)"
    return "\n".join(["=== OBSERVATION ===", row["obs"], "",
                      "=== ENGINEER'S PRIVATE WORKING NOTES ===", own.get("thinking") or "", "",
                      "=== CHANGE CHOSEN ===", act])


# v2: targets the four failure patterns found by the independent audit of v1
# (wrong mechanisms, unsupported comparisons with alternatives, "stays within the
# limit" claims the outcome breaks, adjectives that misstate size)
SYSTEM_RAT2 = SYSTEM_RAT.replace(
    "- Explain why THIS element and THIS kind of change (and roughly this size), from what the observation shows.",
    """- Explain why THIS element and THIS kind of change (and roughly this size), from what the observation shows.
- Mechanisms: only use relations the observation itself states (its rules, formulas, sensitivity or what-if tables). If the observation does not state how the change acts, say what it is aimed at instead of inventing a mechanism.
- Alternatives: mention another option only with the observation's own numbers for it; never claim other options are worse without those numbers.
- Never claim a constraint is met, stays within its limit, or that a violation closes, unless the actual outcome shows exactly that.
- Comparisons: give the numbers (e.g. "3.41 vs a 1.55 limit") rather than adjectives like "slightly", "modestly", "far".""")


SYSTEM_RAT3 = """You write the internal reasoning of a design engineer at one step of an iterative design process. You are given the observation the engineer had, the change the engineer made at this step, and - for your information only - what actually happened afterwards. Write the reasoning as a short, explicit comparison of options that leads to that change.

Structure (first person, present tense, plain prose, no headings):
1. One or two sentences: which limits are violated and by how much, from CONSTRAINT STATUS.
2. Options, 2 to 4 in total, one short sentence each; the made change must be one of them. Each option is a concrete change (element, parameter, new value) with its predicted effect on the violated metrics and on whatever it harms, each prediction derived from a number the observation gives for it (a sensitivity row, a what-if or preview line, a per-step row, a stated formula or scaling rule) and its cost in the objective. Include at least one alternative that attacks the same violation. If a previous step made some metric worse, include the option of reverting that step. Say in a few words why each rejected option loses.
3. The decision: why the made change wins the comparison, in the terms the observation uses to compare designs (shortfalls against the objective).

Rules:
- Use ONLY information in the observation: never quote a number that appears only in the outcome or later steps. Arithmetic on observation numbers is allowed; say what it comes from.
- Every statement about the current design must be correct for the observation (values, comparisons, rankings, which element is which).
- Predicted effects are directions, with a rough size only when the observation supports it. Never claim a limit becomes satisfied unless the actual outcome shows exactly that. Do not claim effects of the made change that the outcome contradicts.
- Comparisons: give the numbers (e.g. "3.41 vs a 1.55 limit") rather than adjectives like "slightly", "modestly", "far".
- After the reasoning put the action line, then two blocks:
<action>SET_PARAM(<element index>, <parameter>, <value>)</action>
<claims>{"effects": [{"metric": "<exact metric name from CONSTRAINT STATUS>", "direction": "up|down|unchanged"}]}</claims>
<candidates>[{"move": "SET_PARAM(<i>, <p>, <v>)", "effects": [{"metric": "<exact metric name>", "direction": "up|down|unchanged"}]}]</candidates>
"claims" lists every effect on an aggregate metric that your reasoning states or implies for the MADE change. "candidates" lists every OTHER option you wrote, as a concrete SET_PARAM with the effects you predicted for it (an empty effects list if you predicted none)."""


def prompt_rat(row, with_post=True):
    g = row["gold"]
    domain = _domain(row["variant"])
    parts = ["=== OBSERVATION (what the engineer saw) ===", row["obs"], "",
             "=== THE CHANGE MADE AT THIS STEP ===",
             _fmt_action(domain, g) + _was(domain, g, "   (was %s)")]
    if row["prev_trace"]:
        parts += ["", "=== EARLIER STEPS (change -> effect) ==="] + ["  " + s for s in row["prev_trace"]]
    if with_post and row.get("gold_outcome"):
        parts += ["", "=== FOR YOUR INFORMATION ONLY: WHAT THIS CHANGE ACTUALLY DID ===",
                  "  " + _metrics_line(row["state_metrics"], row["gold_outcome"])]
        if row["post_trace"]:
            parts += ["", "=== FOR YOUR INFORMATION ONLY: LATER STEPS ==="] + ["  " + s for s in row["post_trace"]]
    return "\n".join(parts)


# -------------------------------------------------------------------- run ---
def call(endpoint, system, user, temperature, max_tokens, thinking=True, model=MODEL):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature, "top_p": 0.95, "top_k": 20, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    if "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "")
    elif thinking and not reasoning:
        # cut off inside the think block: there is no answer
        reasoning, text = text.replace("<think>", ""), ""
    return {"answer": text.strip(), "thinking": reasoning.strip(),
            "finish": out["choices"][0].get("finish_reason"),
            "tokens": out.get("usage", {}).get("completion_tokens")}


def run(args):
    rows = [json.loads(l) for l in open(args.states, encoding="utf-8")]
    if args.variants:
        rows = [r for r in rows if r["variant"] in args.variants]
    if args.per_variant:
        kept, cnt = [], {}
        for r in rows:
            if cnt.get(r["variant"], 0) < args.per_variant:
                kept.append(r)
                cnt[r["variant"]] = cnt.get(r["variant"], 0) + 1
        rows = kept
    # skip (sid, sample) pairs already in the output file (resume)
    done_keys = set()
    if os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8"):
            try:
                o = json.loads(l)
                # truncated generations are re-run (e.g. with a larger budget)
                if (o.get("answer") or o.get("thinking")) and o.get("finish") != "length":
                    done_keys.add((o["sid"], o["sample"]))
            except ValueError:
                pass
    jobs = []
    for r in rows:
        # verb-surface domains get the verb-call action line + grammar in every
        # system prompt; plain domains get the texts unchanged
        dom = _domain(r["variant"])
        for s in range(args.k):
            if args.mode == "own":
                jobs.append((r, s, _system_for(dom, SYSTEM_OWN), prompt_own(r)))
            elif args.mode == "own_prev":
                jobs.append((r, s, _system_for(dom, SYSTEM_OWN), prompt_own(r, with_prev=True)))
            elif args.mode == "own_post":
                jobs.append((r, s, _system_for(dom, SYSTEM_OWN),
                             prompt_own(r, with_prev=True, with_post=True)))
            elif args.mode in ("own_nothink", "own_prev_nothink", "own_post_nothink"):
                jobs.append((r, s, _system_for(dom, SYSTEM_OWN.replace(
                    "Think it through carefully, then choose",
                    "Write your reasoning first (what is violated, where the "
                    "results place the problem, which change and why), then choose")),
                    prompt_own(r, with_prev=args.mode != "own_nothink",
                               with_post=args.mode == "own_post_nothink")))
            elif args.mode == "rat":
                jobs.append((r, s, _system_for(dom, SYSTEM_RAT), prompt_rat(r)))
            elif args.mode == "rat2":
                jobs.append((r, s, _system_for(dom, SYSTEM_RAT2), prompt_rat(r)))
            elif args.mode == "rat3":
                jobs.append((r, s, _system_for(dom, SYSTEM_RAT3), prompt_rat(r)))
            elif args.mode in ("own_cands", "own_elem"):
                jobs.append((r, s, _system_for(dom, SYSTEM_OWN_WRITTEN),
                             prompt_hint(r, s, args.mode)))
            elif args.mode == "rat_nopost":
                jobs.append((r, s, _system_for(dom, SYSTEM_RAT), prompt_rat(r, with_post=False)))
            else:
                raise SystemExit("unknown mode")
    jobs = [j for j in jobs if (j[0]["sid"], j[1]) not in done_keys]

    def one(j):
        r, s, system, user = j
        try:
            out = call(args.endpoint, system, user, args.temperature, args.max_tokens,
                       thinking=not args.no_think)
        except Exception as e:  # recorded, not fatal
            out = {"answer": "", "thinking": "", "error": str(e)}
        out.update({"sid": r["sid"], "sample": s, "mode": args.mode})
        return out

    from concurrent.futures import as_completed
    done = 0
    with open(args.out, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.concurrency) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for f in as_completed(futs):     # write as each finishes
            fh.write(json.dumps(f.result()) + "\n")
            fh.flush()
            done += 1
            if done % 10 == 0:
                print("%d/%d" % (done, len(jobs)), flush=True)


# ------------------------------------------------------------------ score ---
def _match(gold, i, p, v):
    """Return (same_target, same_move): same element+param, and same move
    (exact for toggles/categorical; same direction and step within x2 for
    continuous/additive)."""
    if i != int(gold["i"]) or p != gold["p"]:
        return False, False
    kind = KIND.get(p, "cont")
    gv, go = float(gold["v"]), float(gold["v_old"])
    if kind in ("tog", "cat"):
        return True, abs(v - gv) < 1e-6 * max(1.0, abs(gv))
    dg, dp = gv - go, v - go
    if dg == 0 or dp == 0 or (dg > 0) != (dp > 0):
        return True, False
    ratio = abs(dp) / abs(dg)
    return True, 0.5 <= ratio <= 2.0


def _phi_of(row, i, p, v):
    domain = get_domain(row["variant"])
    st = domain.load(row["spec"])
    for a in row["history"]:
        apply_action(domain, st, a)
    if p not in domain.params or domain.get(st, i, p) is None:
        return None
    ev = PhiEvaluator(domain, row["spec"], offset_rule="infeasible")
    domain.set(st, i, p, v)
    return ev.score(st, seed0=stable_seed(row["problem_id"], "gp", i, p)).mean


def score(args):
    states = {json.loads(l)["sid"]: json.loads(l) for l in open(args.states, encoding="utf-8")}
    last = {}
    for l in open(args.runs, encoding="utf-8"):
        o = json.loads(l)
        k = (o["sid"], o["sample"])
        # a completed answer beats an error/truncated row for the same sample
        if k not in last or o.get("answer") or not last[k].get("answer"):
            last[k] = o
    runs = list(last.values())
    per = {}
    out_rows = []
    for o in runs:
        r = states[o["sid"]]
        ans = o.get("answer") or ""
        if o.get("finish") == "length" or "</think>" in ans or "<think>" in ans:
            ans = ""  # no completed answer
        act = _last_action(_domain(r["variant"]), ans)
        rec = {"sid": o["sid"], "variant": r["variant"], "sample": o["sample"], "mode": o["mode"],
               "gold_param": r["gold"]["p"], "parsed": act is not None, "tokens": o.get("tokens")}
        if act is not None:
            i, p, v = act
            tgt, mv = _match(r["gold"], i, p, v)
            phi = _phi_of(r, i, p, v) if args.phi else None
            phis = [x["phi"] for x in r["moves"]]
            rec.update(param=p, same_target=tgt, same_move=mv, phi=phi,
                       pct=(100.0 * sum(x < phi for x in phis) / len(phis)) if (phi is not None and phis) else None,
                       ge_gold=(phi is not None and phi >= r["gold_phi"] - 1e-9))
        out_rows.append(rec)
        per.setdefault(o["sid"], []).append(rec)
    with open(args.runs.replace(".jsonl", "_scored.jsonl"), "w", encoding="utf-8") as fh:
        for x in out_rows:
            fh.write(json.dumps(x) + "\n")
    print_summary(out_rows, per)


def print_summary(out_rows, per):
    by_v = {}
    for sid, recs in per.items():
        by_v.setdefault(recs[0]["variant"], []).append(recs)
    print("%-11s %5s %7s %9s %9s %10s %9s %8s" % ("variant", "states", "parsed", "target%", "move%",
                                                   "any_move%", "pct", ">=gold%"))
    tot = []
    for v in sorted(by_v):
        S = by_v[v]
        recs = [x for s in S for x in s]
        par = np.mean([x["parsed"] for x in recs])
        tg = np.mean([x.get("same_target", False) for x in recs])
        mv = np.mean([x.get("same_move", False) for x in recs])
        anym = np.mean([any(x.get("same_move", False) for x in s) for s in S])
        pcts = [x["pct"] for x in recs if x.get("pct") is not None]
        ge = [x.get("ge_gold", False) for x in recs]
        print("%-11s %5d %7.2f %9.2f %9.2f %10.2f %9s %8.2f" % (
            v, len(S), par, tg, mv, anym, "%.0f" % np.mean(pcts) if pcts else "-", np.mean(ge)))
        tot += S
    recs = [x for s in tot for x in s]
    print("%-11s %5d %7.2f %9.2f %9.2f %10.2f" % (
        "ALL", len(tot), np.mean([x["parsed"] for x in recs]),
        np.mean([x.get("same_target", False) for x in recs]),
        np.mean([x.get("same_move", False) for x in recs]),
        np.mean([any(x.get("same_move", False) for x in s) for s in tot])))


def compress(args):
    """Condense the model's own (matched) thinking into a short rationale and
    check it: same action as the model chose, numbers grounded in the
    observation, length. Input: own-mode runs + their _scored file."""
    states = {json.loads(l)["sid"]: json.loads(l) for l in open(args.states, encoding="utf-8")}
    own = {}
    for l in open(args.own, encoding="utf-8"):
        o = json.loads(l)
        own[(o["sid"], o["sample"])] = o
    scored = [json.loads(l) for l in open(args.own.replace(".jsonl", "_scored.jsonl"), encoding="utf-8")]
    pick = [x for x in scored if x.get("same_move")] if args.matched_only else \
        [x for x in scored if x.get("parsed")]
    jobs = [(states[x["sid"]], own[(x["sid"], x["sample"])]) for x in pick]

    def one(j):
        r, o = j
        try:
            out = call(args.endpoint, SYSTEM_COMPRESS, prompt_compress(r, o), 0.3, 1200, thinking=False)
        except Exception as e:
            out = {"answer": "", "error": str(e)}
        ans = out.get("answer") or ""
        dom = _domain(r["variant"])
        a_own = _last_action_text(dom, o.get("answer") or "")
        a_new = _last_action_text(dom, ans)
        reasoning = ans.split("<action>")[0].strip()
        exact = [float(t.replace(",", "")) for t in NUMBER.findall(r["obs"])]
        key = [float(v) for v in r["state_metrics"].values()]
        ok_g, bad = grounded(reasoning, exact, key)
        reasons = []
        if a_new is None or a_own is None or a_new != a_own:
            reasons.append("action")
        if not ok_g:
            reasons.append("ungrounded")
        return {"sid": r["sid"], "sample": o["sample"], "variant": r["variant"], "cot": ans,
                "reasons": reasons, "ungrounded": bad[:8], "pass": not reasons,
                "own_tokens": o.get("tokens")}

    with open(args.out, "w", encoding="utf-8") as fh, ThreadPoolExecutor(args.concurrency) as ex:
        res = list(ex.map(one, jobs))
        for x in res:
            fh.write(json.dumps(x) + "\n")
    from collections import Counter
    print("compressed %d  pass %.2f  reasons %s" % (
        len(res), np.mean([x["pass"] for x in res]) if res else 0,
        dict(Counter(rr for x in res for rr in x["reasons"]))))


CLAIMS_RE = re.compile(r"<claims>\s*(\{.*?\})\s*</claims>", re.S)
CANDS_RE = re.compile(r"<candidates>\s*(\[.*?\])\s*</candidates>", re.S)
_STATE_CACHE = {}


def _simulate(r, i, p, v):
    """Metrics after applying one move to the row's state (None if the move is invalid)."""
    d = get_domain(r["variant"])
    if r["sid"] not in _STATE_CACHE:
        st = d.load(r["spec"])
        for a in r["history"]:
            apply_action(d, st, a)
        _STATE_CACHE[r["sid"]] = st
    ch = d.clone(_STATE_CACHE[r["sid"]])
    try:
        d.set(ch, int(i), p, float(v))
        return d.evaluate(ch, seed=0)
    except Exception:
        return None


def _leaks(reasoning, r, exact, key):
    """Numbers in the reasoning that match an outcome or later-step value and nothing in the
    observation: the leak test used when arithmetic on observation numbers is allowed."""
    pool = []
    for v in (r.get("gold_outcome") or {}).values():
        try:
            pool.append(abs(float(v)))
        except (TypeError, ValueError):
            pass
    for s in r.get("post_trace") or []:
        pool += [abs(float(t.replace(",", ""))) for t in NUMBER.findall(s)]
    obs = [abs(x) for x in exact + key]
    out = []
    for tok in NUMBER.findall(reasoning):
        try:
            v = abs(float(tok.replace(",", "")))
        except ValueError:
            continue
        if v == int(v) and v <= 12:
            continue
        if any(abs(v - o) <= 0.005 * max(1.0, o) for o in obs):
            continue
        if any(abs(v - pv) <= 0.001 * max(1.0, pv) for pv in pool):
            out.append(tok)
    return out
PRED_RE = re.compile(r"\b(will|would|should|expect\w*|likely|so that|in order to|to (?:reduce|raise|lower|increase|decrease|relieve|restore|improve|fix|resolve|clear|boost|lift|cut|recover|bring))\b", re.I)


def _direction(a, b):
    a, b = float(a), float(b)
    if abs(b - a) <= 1e-9 * max(1e-12, abs(a), abs(b)):
        return "unchanged"
    return "up" if b > a else "down"


def verify_rat(args):
    """Deterministic checks on rationalized traces (option 2):
    action == gold; every number grounded in the observation (no leak from the
    outcome or later steps); every claimed effect agrees with the actual
    replayed outcome; claims parse. Stability = agreement across samples."""
    states = {json.loads(l)["sid"]: json.loads(l) for l in open(args.states, encoding="utf-8")}
    last = {}
    for l in open(args.runs, encoding="utf-8"):
        o = json.loads(l)
        k = (o["sid"], o["sample"])
        if k not in last or o.get("answer") or not last[k].get("answer"):
            last[k] = o
    runs = list(last.values())
    out, per = [], {}
    for o in runs:
        r = states[o["sid"]]
        ans = o.get("answer") or ""
        reasoning = ans.split("<action>")[0].strip()
        rec = {"sid": o["sid"], "variant": r["variant"], "sample": o["sample"], "reasons": []}
        # the chosen action is the last action outside the candidates block (rat3 lists the
        # alternatives after the action line)
        dom = _domain(r["variant"])
        act = _last_action(dom, CANDS_RE.sub("", ans))
        g = r["gold"]
        # the prompt shows the gold value at 6 significant digits (_fmtv); copying it is exact
        if act is None or not (act[0] == int(g["i"]) and act[1] == g["p"]
                               and (abs(act[2] - float(g["v"])) <= 1e-6 * max(1.0, abs(float(g["v"])))
                                    or _fmtv(act[2]) == _fmtv(g["v"]))):
            rec["reasons"].append("action")
        exact = [float(t.replace(",", "")) for t in NUMBER.findall(r["obs"])]
        key = [float(v) for v in r["state_metrics"].values()] + [float(g["v"]), float(g["v_old"]), float(g["i"])]
        ok, bad = grounded(reasoning, exact, key)
        if not ok:
            rec["ungrounded"] = bad[:8]
            if getattr(args, "lenient_grounding", False):
                # derived numbers allowed. Numbers that coincide with an outcome value are
                # recorded for the auditor; they gate only with --strict-leak, because in domains
                # whose effects are exact and linear (grid2) a correct derivation reproduces the
                # outcome value.
                lk = _leaks(reasoning, r, exact, key)
                if lk:
                    rec["leak"] = lk[:8]
                    if getattr(args, "strict_leak", False):
                        rec["reasons"].append("leak")
            else:
                rec["reasons"].append("ungrounded")
        cm = CLAIMS_RE.search(ans)
        effects = []
        if not cm:
            rec["reasons"].append("no_claims")
        else:
            try:
                effects = json.loads(cm.group(1)).get("effects") or []
            except ValueError:
                rec["reasons"].append("bad_claims")
        wrong = []
        if r.get("gold_outcome"):
            for e in effects:
                k = e.get("metric")
                if k not in r["state_metrics"] or k not in r["gold_outcome"]:
                    wrong.append((k, "unknown"))
                    continue
                actual = _direction(r["state_metrics"][k], r["gold_outcome"][k])
                if str(e.get("direction")).lower() != actual:
                    wrong.append((k, e.get("direction"), actual))
        if wrong:
            rec["reasons"].append("false_effect")
            rec["wrong"] = wrong
        if getattr(args, "simulate", False):
            # rat3: every alternative option is replayed in the simulator and its predicted
            # directions checked, so the comparison is verified, not just the chosen move
            cm2 = CANDS_RE.search(ans)
            cands = []
            if not cm2:
                rec["reasons"].append("no_candidates")
            else:
                try:
                    cands = json.loads(cm2.group(1))
                except ValueError:
                    rec["reasons"].append("bad_candidates")
                if cm2 and not rec["reasons"][-1:] == ["bad_candidates"] and not cands:
                    rec["reasons"].append("no_candidates")
            bad_c = []
            for c in cands if isinstance(cands, list) else []:
                ctext = str(c.get("move", "")) if isinstance(c, dict) else ""
                cact = _last_action(dom, ctext)
                if cact is None:
                    bad_c.append([str(c)[:60], "unparsable"])
                    continue
                ci, cp, cv = cact
                cname = _last_action_text(dom, ctext)
                if ci == int(g["i"]) and cp == g["p"] and _fmtv(cv) == _fmtv(g["v"]):
                    bad_c.append([cname, "is_gold"])
                    continue
                after = _simulate(r, ci, cp, cv)
                if after is None:
                    bad_c.append([cname, "invalid"])
                    continue
                for e in c.get("effects") or []:
                    k = e.get("metric") if isinstance(e, dict) else None
                    if k not in r["state_metrics"] or k not in after:
                        bad_c.append([cname, k, "unknown"])
                        continue
                    actual = _direction(r["state_metrics"][k], after[k])
                    if str(e.get("direction")).lower() != actual:
                        bad_c.append([cname, k, e.get("direction"), actual])
            if bad_c:
                rec["reasons"].append("false_candidate")
                rec["bad_candidates"] = bad_c[:8]
            rec["n_candidates"] = len(cands) if isinstance(cands, list) else 0
        rec["effects"] = sorted((e.get("metric"), str(e.get("direction"))) for e in effects
                                if isinstance(e, dict))
        rec["pass"] = not rec["reasons"]
        rec["words"] = len(reasoning.split())
        out.append(rec)
        per.setdefault(o["sid"], []).append(rec)
    with open(args.runs.replace(".jsonl", "_verified.jsonl"), "w", encoding="utf-8") as fh:
        for x in out:
            fh.write(json.dumps(x) + "\n")
    by_v = {}
    for sid, recs in per.items():
        by_v.setdefault(recs[0]["variant"], []).append(recs)
    from collections import Counter
    print("%-11s %6s %8s %12s %12s  failure reasons" % ("variant", "states", "pass%", "state_any%", "state_all%"))
    allrec = []
    for v in sorted(by_v):
        S = by_v[v]
        recs = [x for s in S for x in s]
        allrec += recs
        c = Counter(rr for x in recs for rr in x["reasons"])
        print("%-11s %6d %8.2f %12.2f %12.2f  %s" % (
            v, len(S), np.mean([x["pass"] for x in recs]),
            np.mean([any(x["pass"] for x in s) for s in S]),
            np.mean([all(x["pass"] for x in s) for s in S]), dict(c)))
    print("ALL pass %.2f" % np.mean([x["pass"] for x in allrec]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--data", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=20)
    a.add_argument("--variants", nargs="*", default=list(VARIANTS))
    b = sub.add_parser("run")
    b.add_argument("--states", required=True)
    b.add_argument("--mode", required=True)
    b.add_argument("--k", type=int, default=4)
    b.add_argument("--endpoint", default="http://127.0.0.1:8766/v1")
    b.add_argument("--out", required=True)
    b.add_argument("--concurrency", type=int, default=48)
    b.add_argument("--temperature", type=float, default=0.6)
    b.add_argument("--max-tokens", type=int, default=16000)
    b.add_argument("--no-think", action="store_true")
    b.add_argument("--variants", nargs="*")
    b.add_argument("--per-variant", type=int, default=0,
                   help="use only the first N states of each variant (0 = all)")
    c = sub.add_parser("score")
    c.add_argument("--states", required=True)
    c.add_argument("--runs", required=True)
    c.add_argument("--phi", action="store_true", help="also score each pick by Phi")
    d = sub.add_parser("verify")
    d.add_argument("--states", required=True)
    d.add_argument("--runs", required=True)
    d.add_argument("--simulate", action="store_true",
                   help="rat3: replay every listed alternative and check its predicted directions")
    d.add_argument("--lenient-grounding", action="store_true",
                   help="allow arithmetic on observation numbers; record numbers that match outcome values")
    d.add_argument("--strict-leak", action="store_true",
                   help="with --lenient-grounding: a number matching an outcome value fails the trace")
    e = sub.add_parser("compress")
    e.add_argument("--states", required=True)
    e.add_argument("--own", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--endpoint", default="http://127.0.0.1:8766/v1")
    e.add_argument("--concurrency", type=int, default=16)
    e.add_argument("--matched-only", action="store_true")
    args = ap.parse_args(argv)
    {"prepare": prepare, "run": run, "score": score, "verify": verify_rat,
     "compress": compress}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
