"""
The search, written against the Domain interface only, so it ports by subclassing.

On trusses this architecture reaches 0.9256 against fully-stressed design's 0.4930 (discordant
186-0). Every line of that implementation, though, was written against trussme objects and the
DesignBench grammar. The portability claim -- a new domain is one subclass and nothing above it
changes -- has never actually been exercised.

This is the same architecture with the truss taken out of it. It touches only what da_domain
guarantees: n_elements, get, set, element_margins, budget_ratio, feasible, clone. The tool
library is written in those terms too:

  SIZE_PASS(margin)      the generic fully-stressed analogue: move every element towards its
                         requirement using the installed exponent, clipped
  SCALE(ids, factor)     scale the named elements
  TRIM(threshold, f)     shrink every element whose margin exceeds threshold

and the search is unchanged: draw candidates, roll each forward under SIZE_PASS, score by the
potential, keep the best, then try a greedy pair on the anchor. A do-nothing branch competes, so
the arm cannot fall below its own heuristic.

Nothing here mentions members, joints, buckling or mass. If the truss result was an artefact of
truss-specific engineering, this will not reproduce it.
"""
import sys, math, random, statistics
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import da_tools as T

INSTALLED_EXP = 3.0
CLIP = (0.7, 2.0)


# ---------------------------------------------------------------- tool library

def size_pass(dom, st, param, margin=1.05, exp=INSTALLED_EXP):
    """Generic fully-stressed analogue: every element moves towards its requirement."""
    em = dom.element_margins(st)
    if not em:
        return st
    nx = dom.clone(st)
    for i in range(dom.n_elements(st)):
        b = dom.get(st, i, param)
        if not b:
            continue
        m = em[i]
        if not m or m <= 0 or not math.isfinite(m):
            continue
        f = min(CLIP[1], max(CLIP[0], (margin / m) ** (1.0 / exp)))
        dom.set(nx, i, param, b * f)
    return nx


def scale(dom, st, param, ids, factor):
    nx = dom.clone(st)
    f = min(CLIP[1], max(CLIP[0], factor))
    for i in ids:
        if 0 <= i < dom.n_elements(st):
            b = dom.get(st, i, param)
            if b:
                dom.set(nx, i, param, b * f)
    return nx


def trim(dom, st, param, threshold, factor):
    em = dom.element_margins(st)
    if not em:
        return st
    nx = dom.clone(st)
    f = min(1.0, max(CLIP[0], factor))
    for i in range(dom.n_elements(st)):
        m = em[i]
        if m and math.isfinite(m) and m > threshold:
            b = dom.get(st, i, param)
            if b:
                dom.set(nx, i, param, b * f)
    return nx


TOOLS = ("SIZE_PASS", "SCALE", "TRIM")


def apply_tool(dom, st, param, name, args):
    try:
        if name == "SIZE_PASS":
            return size_pass(dom, st, param, args.get("margin", 1.05))
        if name == "SCALE":
            return scale(dom, st, param, args.get("ids", []), args.get("factor", 1.1))
        if name == "TRIM":
            return trim(dom, st, param, args.get("threshold", 2.0), args.get("factor", 0.85))
    except Exception:
        return None
    return None


def random_proposals(dom, st, rng, n):
    ne = dom.n_elements(st)
    out = []
    for _ in range(n):
        kind = rng.choice(TOOLS)
        if kind == "SIZE_PASS":
            out.append(("SIZE_PASS", {"margin": rng.uniform(0.95, 1.35)}))
        elif kind == "TRIM":
            out.append(("TRIM", {"threshold": rng.uniform(1.2, 5.0),
                                 "factor": rng.uniform(0.75, 0.95)}))
        else:
            k = rng.randint(1, max(1, min(4, ne)))
            out.append(("SCALE", {"ids": rng.sample(range(ne), k),
                                  "factor": rng.uniform(0.75, 1.6)}))
    return out


# ---------------------------------------------------------------- search

def potential(dom, st):
    em = dom.element_margins(st)
    if not em:
        return -1e9
    vals = [min(float(x), 50.0) for x in em if x and math.isfinite(x) and x > 0]
    if not vals:
        return -1e9
    strength = min(min(vals), 1.0)
    try:
        over = max(0.0, dom.budget_ratio(st) - 1.0)
    except Exception:
        over = 0.0
    return strength - 0.75 * over


def rollout(dom, st, param, steps, margin=1.05):
    cur = st
    for _ in range(max(0, steps)):
        if dom.feasible(cur):
            return True, cur
        nxt = size_pass(dom, cur, param, margin)
        if nxt is cur:
            break
        cur = nxt
    return dom.feasible(cur), cur


def heuristic_episode(dom, st, param, steps):
    ok, _ = rollout(dom, st, param, steps)
    return 1 if ok else 0


def search_episode(dom, st, param, steps, rng, nprop=16, propose=None):
    """propose(dom, st, rng, n) -> [(name, args)]; defaults to uniform random draws."""
    propose = propose or (lambda d, s, r, n: random_proposals(d, s, r, n))
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
        for nm, args in propose(dom, st, rng, nprop):
            cand = apply_tool(dom, st, param, nm, args)
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
            for _, other in scored[1:8]:
                # compose: re-apply the second move on top of the anchor
                for nm, args in propose(dom, anchor, rng, 1):
                    cand = apply_tool(dom, anchor, param, nm, args)
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
