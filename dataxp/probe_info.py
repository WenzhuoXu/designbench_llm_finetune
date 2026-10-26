"""Information-sufficiency probe: does the observation let an agent choose well?

A strong model (never the model being tuned) sees ONLY the policy
observation for sampled repair-phase states and picks one action. Each pick is
scored by where its Phi lands among the legal moves at that state. A move type
whose picks score near the 50th percentile (random level) points to
information the observation does not give.

    # 1. sample states, write observations + the scored move set
    python -m dataxp.probe_info prepare --data <run dir> --variant frame3d2 \
        --n 10 --out <dir>
    # 2. an auditor writes <dir>/answers.json: [{"k": 0, "action": "SET_PARAM(3, r, 0.02)"}, ...]
    # 3. score
    python -m dataxp.probe_info score --out <dir>

The observation is what the policy sees at inference: domain/goal, describe(),
constraint status, the changes already made, and - new - the observed effect
of the previous change on every constrained metric and the objective. It never
contains the gold action.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from .domain import (Move, PhiEvaluator, action_form, apply_action, current_spelling,
                     speaks_verbs, stable_seed)
from .render_cot import iter_states, policy_observation


def observation(domain, spec, st, turn, history, variant):
    """Exactly the policy observation used for training prompts
    (render_cot.policy_observation), plus the action instruction: the verb
    list (or, under DATAXP_SPELLING=set_parameter, the SET_PARAM operation
    list) for a native-verb or verb-surface domain, the SET_PARAM form
    otherwise."""
    lines, _ = policy_observation(domain, spec, st, turn, history, variant)
    if speaks_verbs(domain):
        # no served text says "verb" under set_parameter (there is none)
        lead = ("with one of these verbs:" if current_spelling() == "multitool"
                else "performing one of these operations:")
        lines += ["", "Choose exactly ONE change, written as %s %s"
                  % (action_form(domain), lead), domain.grammar().rstrip()]
    else:
        lines += ["", "Choose exactly ONE change, written as SET_PARAM(<element index>, <parameter>, <value>)."]
    return "\n".join(lines)


def _move_seed(pid, domain, move: Move) -> int:
    """Per-move scoring seed: the historical (i, p) seed for SET_PARAM, the
    formatted move for a native verb."""
    if move.is_set_param:
        d = move.as_dict()
        return stable_seed(pid, "probe", d["i"], d["p"])
    return stable_seed(pid, "probe", domain.format_move(move))


def _row_move(m: dict) -> Move:
    """A scored-move row: {i, p, v, phi} (SET_PARAM) or {action, phi}."""
    return Move.from_dict(m["action"] if "action" in m else m)


def prepare(args):
    os.makedirs(args.out, exist_ok=True)
    # accept either an output run (<data>/<variant>/accepted/) or a scratch
    # run whose gates have not been applied (<data>/<variant>/*.jsonl)
    if not os.path.isdir(os.path.join(args.data, args.variant, "accepted")):
        import shutil
        src = os.path.join(args.data, args.variant)
        stage = os.path.join(args.out, "_stage", args.variant, "accepted")
        os.makedirs(stage, exist_ok=True)
        for f in ("problems.jsonl", "trajectories.jsonl"):
            shutil.copy(os.path.join(src, f), os.path.join(stage, f))
        args.data = os.path.join(args.out, "_stage")
    rows = []
    prev_metrics = {}
    seen = {}
    for domain, spec, st, turn, hist in iter_states(args.data, args.variant, True):
        pid = turn["problem_id"]
        seen[pid] = seen.get(pid, 0) + 1
        # one state per problem: its second repair turn, so there is a previous
        # change whose observed effect can be shown
        if seen[pid] != 2:
            prev_metrics[pid] = (turn["turn"], turn.get("state_metrics") or {})
            continue
        obs = observation(domain, spec, st, turn, hist, args.variant)
        ev = PhiEvaluator(domain, spec)
        rng = np.random.default_rng(stable_seed(pid, "probe", turn["turn"]))
        cands = domain.legal_moves(st, rng)
        scored = []
        for c in cands:
            child = domain.clone(st)
            domain.apply(child, c)
            s = ev.score(child, seed0=_move_seed(pid, domain, c))
            d = c.as_dict()
            if c.is_set_param:
                scored.append({"i": d["i"], "p": d["p"], "v": d["v"], "phi": s.mean})
            else:
                scored.append({"action": d, "phi": s.mean})
        k = len(rows)
        with open(os.path.join(args.out, "obs_%03d.txt" % k), "w", encoding="utf-8") as fh:
            fh.write(obs)
        rows.append({"k": k, "problem_id": pid, "turn": turn["turn"], "variant": args.variant,
                     "spelling": current_spelling() if speaks_verbs(domain) else "SET_PARAM",
                     "gold": turn["action"], "moves": scored, "spec": spec,
                     "history": [a for _, a in hist], "obs_chars": len(obs)})
        if len(rows) >= args.n:
            break
    with open(os.path.join(args.out, "states.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    print("wrote %d observations to %s (median %d chars)"
          % (len(rows), args.out, int(np.median([r["obs_chars"] for r in rows])) if rows else 0))


def score(args):
    from .domains import get_domain
    states = json.load(open(os.path.join(args.out, "states.json"), encoding="utf-8"))
    answers = {a["k"]: a["action"] for a in json.load(open(os.path.join(args.out, "answers.json"),
                                                           encoding="utf-8"))}
    out = []
    for r in states:
        domain = get_domain(r["variant"])
        spec = r["spec"]
        st = domain.load(spec)
        for a in r["history"]:
            apply_action(domain, st, a)
        phis = sorted(m["phi"] for m in r["moves"])
        gold = Move.from_dict(r["gold"])
        gold_phi = None
        for m in r["moves"]:
            if _row_move(m).matches(gold, rel=1e-9):
                gold_phi = m["phi"]
        gold_tag = gold.args["p"] if gold.is_set_param else gold.verb
        rec = {"k": r["k"], "problem_id": r["problem_id"], "gold_param": gold_tag}
        ans = answers.get(r["k"])
        mv = domain.parse_move(ans or "")
        if mv is None:
            rec.update(parsed=False, pct=0.0)
        else:
            if mv.is_set_param:
                i, p = mv.args["i"], mv.args["p"]
                ok, tag = p in domain.params and domain.get(st, i, p) is not None, p
            else:
                ok, tag = True, mv.verb
            child = domain.clone(st)
            if ok:
                try:
                    domain.apply(child, mv)
                except ValueError:
                    ok = False
            if not ok:
                rec.update(parsed=True, legal=False, pct=0.0, param=tag)
            else:
                ev = PhiEvaluator(domain, spec)
                phi = ev.score(child, seed0=_move_seed(r["problem_id"], domain, mv)).mean
                pct = 100.0 * sum(x < phi for x in phis) / max(len(phis), 1)
                rec.update(parsed=True, legal=True, param=tag, phi=phi, pct=pct,
                           gold_pct=(100.0 * sum(x < gold_phi for x in phis) / len(phis))
                           if gold_phi is not None else None)
        out.append(rec)
    with open(os.path.join(args.out, "scores.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    by = {}
    for rec in out:
        by.setdefault(rec.get("param", "?"), []).append(rec["pct"])
    print("states %d  mean percentile of picks %.1f (random ~50, gold ~%s)"
          % (len(out), np.mean([r["pct"] for r in out]),
             "%.0f" % np.mean([r["gold_pct"] for r in out if r.get("gold_pct") is not None])
             if any(r.get("gold_pct") is not None for r in out) else "n/a"))
    for p, v in sorted(by.items()):
        print("  picked %-14s n=%-3d mean pct %.1f" % (p, len(v), np.mean(v)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--data", required=True)
    a.add_argument("--variant", required=True)
    a.add_argument("--n", type=int, default=10)
    a.add_argument("--out", required=True)
    b = sub.add_parser("score")
    b.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    (prepare if args.cmd == "prepare" else score)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
