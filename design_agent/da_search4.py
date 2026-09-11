"""
Successive halving: screen wide and cheap, then roll out deep only what survives.

The current search evaluates every candidate to the full horizon. On the truss that is 32
candidates times up to twelve sizing passes each, and it is why the winning configuration costs
287 analyses against the heuristic's six. Most of those candidates are decided in the first
step -- a move that immediately worsens the governing margin almost never recovers by depth
twelve -- so the deep rollouts are spent overwhelmingly on candidates that were never going to
win.

So the turn is split into two stages. A wide pool is screened at depth one, which costs one
analysis per candidate. The best fraction are promoted and rolled to the full horizon, and the
argmax is taken over those. Everything else is unchanged: the same tool library, the same
potential, the same greedy pair on the anchor, the same do-nothing branch competing.

Two candidate pools per turn, screened and deep, are exposed as parameters so the same code can
be run at matched evaluation budget against the single-stage search rather than merely at
matched candidate count.
"""
import sys, math, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import da_domtools
from da_search import size_pass, potential
from da_search3 import apply_tool, propose, GENERIC


def rollout(dom, st, param, steps, budget=None, exp=3.0):
    cur = st
    for _ in range(max(0, steps)):
        if budget is not None and dom.calls >= budget:
            return dom.feasible(cur), cur
        if dom.feasible(cur):
            return True, cur
        nxt = size_pass(dom, cur, param, 1.05, exp)
        if nxt is cur:
            break
        cur = nxt
    return dom.feasible(cur), cur


def search_episode(dom, st, param, steps, rng, wide=64, keep=8, budget=None,
                   allowed=GENERIC):
    """Screen `wide` candidates at depth 1; roll the best `keep` to the horizon."""
    try:
        extra = dom.tools() or {}
    except Exception:
        extra = {}
    for turn in range(steps):
        if dom.feasible(st) or (budget is not None and dom.calls >= budget):
            break
        nxt = size_pass(dom, st, param)
        if nxt is not st:
            st = nxt
        if dom.feasible(st):
            break
        remaining = steps - turn - 1
        base_ok, _ = rollout(dom, st, param, remaining, budget)
        if base_ok:
            return 1
        best_v, best = potential(dom, st), None

        # --- stage one: one step each, cheap
        screened = []
        for kind, args in propose(dom, st, rng, wide, extra, allowed):
            if budget is not None and dom.calls >= budget:
                break
            cand = apply_tool(dom, st, param, kind, args, extra)
            if cand is None:
                continue
            ok, shallow = rollout(dom, cand, param, 1, budget)
            if ok:
                return 1
            screened.append((potential(dom, shallow), cand))
        if not screened:
            continue
        screened.sort(key=lambda x: -x[0])

        # --- stage two: the survivors go the full distance
        deep = []
        for _, cand in screened[:keep]:
            if budget is not None and dom.calls >= budget:
                break
            ok, rolled = rollout(dom, cand, param, remaining, budget)
            if ok:
                return 1
            v = potential(dom, rolled)
            deep.append((v, cand))
            if v > best_v + 1e-12:
                best_v, best = v, cand

        # --- greedy pair on the anchor, as before
        if deep and (budget is None or dom.calls < budget):
            deep.sort(key=lambda x: -x[0])
            anchor = deep[0][1]
            for kind, args in propose(dom, anchor, rng, min(8, keep), extra, allowed):
                if budget is not None and dom.calls >= budget:
                    break
                cand = apply_tool(dom, anchor, param, kind, args, extra)
                if cand is None:
                    continue
                ok, rolled = rollout(dom, cand, param, remaining, budget)
                if ok:
                    return 1
                v = potential(dom, rolled)
                if v > best_v + 1e-12:
                    best_v, best = v, cand
        if best is not None:
            st = best
    ok, _ = rollout(dom, st, param, 2, budget)
    return 1 if ok else 0
