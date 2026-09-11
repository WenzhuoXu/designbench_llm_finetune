"""
The search, now drawing from the generic library PLUS whatever the domain registers.

Unchanged in structure from da_search: draw candidates, roll each forward under the sizing
rule, score by the potential, keep the best, try a greedy pair on the anchor, let a do-nothing
branch compete. The only difference is where candidates come from -- the generic three tools
plus dom.tools(), which the truss fills with topology and the synthetic domain leaves empty.

Nothing here knows what a member or a joint is. The domain does.
"""
import sys, math, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import da_domtools          # registers tools() on Domain and TrussDomain
from da_search import (size_pass, scale, trim, potential, rollout,
                       heuristic_episode, TOOLS as GENERIC, CLIP)


def sample_generic(dom, st, rng, kind):
    ne = dom.n_elements(st)
    if kind == "SIZE_PASS":
        return {"margin": rng.uniform(0.95, 1.35)}
    if kind == "TRIM":
        return {"threshold": rng.uniform(1.2, 5.0), "factor": rng.uniform(0.75, 0.95)}
    k = rng.randint(1, max(1, min(4, ne)))
    return {"ids": rng.sample(range(ne), k), "factor": rng.uniform(0.75, 1.6)}


def apply_generic(dom, st, param, kind, args):
    try:
        if kind == "SIZE_PASS":
            return size_pass(dom, st, param, args.get("margin", 1.05))
        if kind == "SCALE":
            return scale(dom, st, param, args.get("ids", []), args.get("factor", 1.1))
        if kind == "TRIM":
            return trim(dom, st, param, args.get("threshold", 2.0), args.get("factor", 0.85))
    except Exception:
        return None
    return None


def propose(dom, st, rng, n):
    extra = {}
    try:
        extra = dom.tools() or {}
    except Exception:
        extra = {}
    names = list(GENERIC) + list(extra)
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
            out.append((kind, sample_generic(dom, st, rng, kind)))
    return out


def apply_any(dom, st, param, kind, args):
    extra = {}
    try:
        extra = dom.tools() or {}
    except Exception:
        extra = {}
    if kind in extra:
        try:
            return extra[kind][1](dom, st, args)
        except Exception:
            return None
    return apply_generic(dom, st, param, kind, args)


def search_episode(dom, st, param, steps, rng, nprop=16):
    for turn in range(steps):
        if dom.feasible(st):
            return 1
        nxt = size_pass(dom, st, param)
        if nxt is not st:
            st = nxt
        if dom.feasible(st):
            return 1
        remaining = steps - turn - 1
        base_ok, _ = rollout(dom, st, param, remaining)
        if base_ok:
            return 1
        best_v, best = potential(dom, st), None
        scored = []
        for kind, args in propose(dom, st, rng, nprop):
            cand = apply_any(dom, st, param, kind, args)
            if cand is None:
                continue
            ok, rolled = rollout(dom, cand, param, remaining)
            if ok:
                return 1
            v = potential(dom, rolled)
            scored.append((v, cand))
            if v > best_v + 1e-12:
                best_v, best = v, cand
        if scored:
            scored.sort(key=lambda x: -x[0])
            anchor = scored[0][1]
            for kind, args in propose(dom, anchor, rng, min(8, nprop)):
                cand = apply_any(dom, anchor, param, kind, args)
                if cand is None:
                    continue
                ok, rolled = rollout(dom, cand, param, remaining)
                if ok:
                    return 1
                v = potential(dom, rolled)
                if v > best_v + 1e-12:
                    best_v, best = v, cand
        if best is not None:
            st = best
    ok, _ = rollout(dom, st, param, 2)
    return 1 if ok else 0
