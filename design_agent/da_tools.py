"""
Tool library, Phi_rho, and the inner optimiser.

Every function here touches ONLY the Domain interface, so the library is domain-portable by
construction. Nothing imports truss.

Design choices are pinned to measurements made on this project:
  - FIT_OP is ELEMENT-WISE, not one global exponent: measured elasticity median 2.32, below
    1.5 for 45.7% of elements, and the r^4 the policy asserts holds for 0.8%.
  - The fit only needs the ordering roughly right: rank-only assignment recovers 69% of the
    installed macro's benefit.
  - Phi uses a Kreisselmeier-Steinhauser aggregate, not a raw min, because a raw min is flat
    wherever a solve fails and gives an optimiser no gradient.
"""
from __future__ import annotations
import math


# ------------------------------------------------------------------ Phi

def phi_rho(dom, st, rho=8.0, w_budget=1.0, w_mass=0.05):
    """Smoothed potential. Higher is better. KS aggregate over performance margins so the
    surface stays differentiable where a hard min would be flat."""
    ms = dom.global_margins(st)
    if not ms:
        return -1e9
    # KS soft-min of (margin - 1), saturating at 0 once satisfied
    xs = [min(m - 1.0, 0.0) for m in ms]
    if rho and rho > 0:
        try:
            soft = -(1.0 / rho) * math.log(sum(math.exp(-rho * x) for x in xs))
        except (OverflowError, ValueError):
            soft = min(xs)
    else:
        soft = min(xs)
    b = dom.budget_ratio(st)
    over = min(0.0, 1.0 - b)
    return 5.0 * soft + 5.0 * w_budget * over - w_mass * b


# ------------------------------------------------------------------ tools

def worst_by_margin(dom, st, k):
    """Support selection: the k elements with the smallest own margin."""
    em = dom.element_margins(st)
    if em is None:
        return list(range(min(k, dom.n_elements(st))))
    return [i for _, i in sorted((m, i) for i, m in enumerate(em))][:k]


def deficit_set(dom, st):
    em = dom.element_margins(st)
    if em is None:
        return list(range(dom.n_elements(st)))
    return [i for i, m in enumerate(em) if m < 1.0]


def surplus_set(dom, st, thr=2.0):
    em = dom.element_margins(st)
    if em is None:
        return []
    return [i for i, m in enumerate(em) if m >= thr]


def PROBE(dom, st, elems, param, delta=0.10):
    """Perturb each element's own parameter by +-delta in log space and read the response of
    that element's OWN margin. Returns {elem: elasticity}. 2 evaluations per element."""
    em0 = dom.element_margins(st)
    if em0 is None:
        return {}
    out = {}
    for i in elems:
        base = dom.get(st, i, param)
        if not base or base <= 0 or not (0 < em0[i] < float("inf")):
            continue
        es = []
        for s in (+delta, -delta):
            nx = dom.clone(st)
            dom.set(nx, i, param, base * math.exp(s))
            em1 = dom.element_margins(nx)
            if em1 is None:
                continue
            a, b = em0[i], em1[i]
            if a > 0 and b > 0 and math.isfinite(a) and math.isfinite(b):
                es.append((math.log(b) - math.log(a)) / s)
        if es:
            out[i] = sum(es) / len(es)
    return out


def FIT_OP(elasticities, floor=1.0, default=3.0):
    """Turn probes into an element-wise local inverse. No functional form is assumed and no
    single exponent is shared -- that is the point."""
    # Elements whose own margin barely responds to their own parameter are NOT inverted:
    # 1/k explodes and the move slams the clip, spending budget on a member that cannot pay
    # it back. Measured: elasticity is below 1.5 for 45.7% of elements, and the budget is
    # what rejects candidates in 956 of 956 cases. Those elements fall back to `default`.
    e = {i: v for i, v in elasticities.items()
         if v is not None and math.isfinite(v) and v >= floor}

    def op(margin_now, i, target):
        """factor for element i to move its own margin from margin_now to target."""
        k = e.get(i, default)
        if margin_now is None or margin_now <= 0 or not math.isfinite(margin_now):
            return 1.0
        need = target / margin_now
        try:
            return need ** (1.0 / k)
        except (ValueError, ZeroDivisionError, OverflowError):
            return 1.0
    op.coverage = len(e)
    op.exponents = e
    return op


def INSTALLED_OP(exponent=3.0):
    """Control: one global exponent, the closed form's shape. Isolates whether fitting
    per-element buys anything over a single installed law."""
    def op(margin_now, i, target):
        if margin_now is None or margin_now <= 0 or not math.isfinite(margin_now):
            return 1.0
        try:
            return (target / margin_now) ** (1.0 / exponent)
        except (ValueError, ZeroDivisionError, OverflowError):
            return 1.0
    op.coverage = -1
    op.exponents = {}
    return op


def apply_op(dom, st, op, elems, param, target, clip=(0.7, 2.0)):
    """Two-sided: every element in `elems` moves toward `target`, which shrinks the ones
    already above it."""
    em = dom.element_margins(st)
    nx = dom.clone(st)
    for i in elems:
        base = dom.get(st, i, param)
        if base is None or base <= 0:
            continue
        f = op(em[i] if em else None, i, target)
        f = min(clip[1], max(clip[0], f))
        dom.set(nx, i, param, base * f)
    return nx


def TRIM(dom, st, param, thr, factor=0.85):
    nx = dom.clone(st)
    for i in surplus_set(dom, st, thr):
        b = dom.get(st, i, param)
        if b and b > 0:
            dom.set(nx, i, param, b * factor)
    return nx


def REALLOCATE(dom, st, op, param, target, clip=(0.7, 2.0)):
    """Mass-neutral: grow the deficit set, shrink the surplus set, then rescale the whole
    move so the budget ratio does not rise."""
    b0 = dom.budget_ratio(st)
    nx = apply_op(dom, st, op, list(range(dom.n_elements(st))), param, target, clip)
    b1 = dom.budget_ratio(nx)
    if b1 <= b0 or b1 <= 0:
        return nx
    scale = (b0 / b1) ** 0.5           # crude projection back onto the budget
    ny = dom.clone(st)
    for i in range(dom.n_elements(st)):
        a, b = dom.get(st, i, param), dom.get(nx, i, param)
        if a and b and a > 0:
            dom.set(ny, i, param, a * ((b / a) ** 1.0) * scale)
    return ny


# ------------------------------------------------------------------ inner optimiser

def inner_optimise(dom, st, build, slots, objective=phi_rho, budget=24):
    """Fill a program's numeric slots. `build(**kw) -> state`; `slots` maps a name to a
    (lo, hi) range searched in log space. Coordinate line search, which is enough for the
    1-2 dimensional slots the planner emits; swap for CMA-ES when dimension grows."""
    keys = list(slots)
    cur = {k: math.sqrt(slots[k][0] * slots[k][1]) for k in keys}
    best_x = build(**cur)
    best_v = objective(dom, best_x)
    per = max(3, budget // max(1, len(keys) * 2))
    for _ in range(2):
        for k in keys:
            lo, hi = slots[k]
            for j in range(per):
                t = lo * (hi / lo) ** (j / max(per - 1, 1))
                trial = dict(cur); trial[k] = t
                x = build(**trial)
                v = objective(dom, x)
                if v > best_v:
                    best_v, best_x, cur = v, x, trial
    return best_x, best_v, cur


def EFFICIENT_SUPPORT(dom, st, elasticities, k, param):
    """Pick the k elements with the best margin-per-unit-budget.

    A probe gives d log(margin) / d log(param) for each element. The budget cost of raising
    that parameter is roughly proportional to the element's current share of the resource,
    so the return on spend is elasticity / share. Under a binding budget you want to buy
    margin where it is cheap, which is a different question from where the margin is worst.
    """
    n = dom.n_elements(st)
    em = dom.element_margins(st) or [1.0] * n
    sizes = []
    for i in range(n):
        v = dom.get(st, i, param)
        sizes.append(v if v and v > 0 else 1.0)
    tot = sum(sizes) or 1.0
    score = []
    for i in range(n):
        el = elasticities.get(i)
        if el is None or not math.isfinite(el) or el <= 0:
            continue
        if em[i] >= 1.0:
            continue                      # already satisfied; not a place to spend
        share = sizes[i] / tot
        deficit = math.log(max(1.0 / max(em[i], 1e-6), 1.0))
        score.append((el / max(share, 1e-6) / max(deficit, 1e-6), i))
    score.sort(reverse=True)
    if not score:
        return worst_by_margin(dom, st, k)
    return [i for _, i in score[:k]]
