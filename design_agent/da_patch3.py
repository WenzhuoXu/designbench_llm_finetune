import pathlib

# ---- da_tools.py: elasticity-aware support selection + a safer FIT_OP floor ----
t = pathlib.Path("/ocean/projects/mch250030p/wxu7/llm_finetune/design_agent/da_tools.py")
s = t.read_text()

s = s.replace('def FIT_OP(elasticities, floor=0.15, default=1.0):',
              'def FIT_OP(elasticities, floor=1.0, default=3.0):')
s = s.replace("""    e = {i: v for i, v in elasticities.items()
         if v is not None and math.isfinite(v) and abs(v) >= floor}""",
              """    # Elements whose own margin barely responds to their own parameter are NOT inverted:
    # 1/k explodes and the move slams the clip, spending budget on a member that cannot pay
    # it back. Measured: elasticity is below 1.5 for 45.7% of elements, and the budget is
    # what rejects candidates in 956 of 956 cases. Those elements fall back to `default`.
    e = {i: v for i, v in elasticities.items()
         if v is not None and math.isfinite(v) and v >= floor}""")

s += '''

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
'''
t.write_text(s)

# ---- da_run.py: new arm using the efficient support ----
r = pathlib.Path("/ocean/projects/mch250030p/wxu7/llm_finetune/design_agent/da_run.py")
s = r.read_text()

s = s.replace('''def induced_op(dom, st, elems, delta=0.10):
    return T.FIT_OP(T.PROBE(dom, st, elems, "r", delta))''',
              '''def induced_op(dom, st, elems, delta=0.10):
    return T.FIT_OP(T.PROBE(dom, st, elems, "r", delta))


def step_efficient(dom, st, param, k=6):
    """Probe everything, spend on the elements where margin is cheapest per unit budget."""
    n = dom.n_elements(st)
    probes = T.PROBE(dom, st, list(range(n)), param)
    op = T.FIT_OP(probes)
    S = T.EFFICIENT_SUPPORT(dom, st, probes, k, param)
    keep = set(S) | set(T.surplus_set(dom, st, 2.0))
    build = lambda target: T.apply_op(dom, st, op, sorted(keep), param, target)
    x, v, _ = T.inner_optimise(dom, st, build, {"target": (1.0, 1.8)}, budget=18)
    return x if v > T.phi_rho(dom, st) else None''')

s = s.replace('''        elif arm == "induced_k5":
            nx = step_op(dom, st, param, induced_op, k=5)''',
              '''        elif arm == "induced_k5":
            nx = step_op(dom, st, param, induced_op, k=5)
        elif arm == "efficient":
            nx = step_efficient(dom, st, param)''')

s = s.replace('ARMS = ["base", "installed", "induced", "induced_k5"]',
              'ARMS = ["base", "installed", "induced", "efficient"]')
s = s.replace('"induced_k5": "PROBE + FIT_OP, support k=5       "',
              '"efficient": "PROBE -> spend where margin is cheapest"')
s = s.replace('''    for x, y in (("installed", "base"), ("induced", "base"),
                 ("induced", "installed"), ("induced_k5", "induced")):''',
              '''    for x, y in (("installed", "base"), ("induced", "base"),
                 ("induced", "installed"), ("efficient", "induced"),
                 ("efficient", "installed")):''')
r.write_text(s)
print("patched tools + run")
