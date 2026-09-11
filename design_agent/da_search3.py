"""
Take the last truss constant out of the portable core: let the search discover the exponent.

size_pass moves an element by (margin/m)**(1/exp), and exp has been fixed at 3.0 everywhere.
That number is the truss's own median elasticity. It was measured on trusses, hard-coded into
what is billed as a domain-portable library, and then carried unexamined into two domains it has
no reason to fit: the synthetic elements respond around 2.2, and pipe pressure responds as the
fifth power of diameter. The pipe domain only became solvable once this was worked around
locally, which is a sign the constant belongs to the domain and not to the framework.

So the exponent becomes a searched parameter like any other. A SIZE_PASS candidate carries its
own exponent, is rolled forward under that exponent, and when such a candidate is selected the
episode adopts it for subsequent rollouts. The search therefore converges on whatever response
its domain actually has, with no per-domain configuration and nothing above the interface
changed.

Everything else is da_search2 unchanged: generic tools plus whatever the domain registers,
rollout scoring, greedy pair on the anchor, do-nothing branch competing.
"""
import sys, math, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import da_domtools                       # registers tools() on Domain and TrussDomain
from da_search import size_pass, scale, trim, potential

GENERIC = ("SIZE_PASS", "SCALE", "TRIM")
EXP_RANGE = (1.2, 7.0)


def sample_args(dom, st, rng, kind):
    ne = dom.n_elements(st)
    if kind == "SIZE_PASS":
        return {"margin": rng.uniform(0.95, 1.35), "exp": rng.uniform(*EXP_RANGE)}
    if kind == "TRIM":
        return {"threshold": rng.uniform(1.2, 5.0), "factor": rng.uniform(0.75, 0.95)}
    k = rng.randint(1, max(1, min(4, ne)))
    return {"ids": rng.sample(range(ne), k), "factor": rng.uniform(0.75, 1.6)}


def apply_tool(dom, st, param, kind, args, extra):
    if kind in extra:
        try:
            return extra[kind][1](dom, st, args)
        except Exception:
            return None
    try:
        if kind == "SIZE_PASS":
            return size_pass(dom, st, param, args.get("margin", 1.05),
                             args.get("exp", 3.0))
        if kind == "SCALE":
            return scale(dom, st, param, args.get("ids", []), args.get("factor", 1.1))
        if kind == "TRIM":
            return trim(dom, st, param, args.get("threshold", 2.0),
                        args.get("factor", 0.85))
    except Exception:
        return None
    return None


def rollout(dom, st, param, steps, exp, budget=None):
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


def propose(dom, st, rng, n, extra, allowed=GENERIC):
    names = list(allowed) + list(extra)
    out = []
    for _ in range(n):
        kind = rng.choice(names)
        if kind in extra:
            try:
                args = extra[kind][0](dom, st, rng)
            except Exception:
                args = None
            if args is None:
                continue
            out.append((kind, args))
        else:
            out.append((kind, sample_args(dom, st, rng, kind)))
    return out


def search_episode(dom, st, param, steps, rng, nprop=16, budget=None,
                   allowed=GENERIC, exp0=3.0):
    """exp0 is only a starting guess; a selected SIZE_PASS replaces it for the episode."""
    try:
        extra = dom.tools() or {}
    except Exception:
        extra = {}
    exp = exp0
    for turn in range(steps):
        if dom.feasible(st) or (budget is not None and dom.calls >= budget):
            break
        nxt = size_pass(dom, st, param, 1.05, exp)
        if nxt is not st:
            st = nxt
        if dom.feasible(st):
            break
        remaining = steps - turn - 1
        # sweep the exponent in the continuation itself: this is where it decides the
        # trajectory, and adopting it only when a SIZE_PASS candidate wins outright meant
        # the parameter never moved off its initial value
        best_e, best_ev = exp, None
        for e in (1.5, 2.2, 3.0, 4.0, 5.0, 6.5):
            ok, rolled = rollout(dom, st, param, remaining, e, budget)
            if ok:
                return 1, e
            v = potential(dom, rolled)
            if best_ev is None or v > best_ev:
                best_ev, best_e = v, e
            if budget is not None and dom.calls >= budget:
                break
        exp = best_e
        best_v, best, best_exp, scored = potential(dom, st), None, exp, []
        for kind, args in propose(dom, st, rng, nprop, extra, allowed):
            if budget is not None and dom.calls >= budget:
                break
            cand = apply_tool(dom, st, param, kind, args, extra)
            if cand is None:
                continue
            e = args.get("exp", exp) if kind == "SIZE_PASS" else exp
            ok, rolled = rollout(dom, cand, param, remaining, e, budget)
            if ok:
                return 1, e
            v = potential(dom, rolled)
            scored.append((v, cand, e))
            if v > best_v + 1e-12:
                best_v, best, best_exp = v, cand, e
        if scored and (budget is None or dom.calls < budget):
            scored.sort(key=lambda x: -x[0])
            anchor, anchor_e = scored[0][1], scored[0][2]
            for kind, args in propose(dom, anchor, rng, min(8, nprop), extra, allowed):
                if budget is not None and dom.calls >= budget:
                    break
                cand = apply_tool(dom, anchor, param, kind, args, extra)
                if cand is None:
                    continue
                e = args.get("exp", anchor_e) if kind == "SIZE_PASS" else anchor_e
                ok, rolled = rollout(dom, cand, param, remaining, e, budget)
                if ok:
                    return 1, e
                v = potential(dom, rolled)
                if v > best_v + 1e-12:
                    best_v, best, best_exp = v, cand, e
        if best is not None:
            st, exp = best, best_exp
    ok, _ = rollout(dom, st, param, 2, exp, budget)
    return (1 if ok else 0), exp
