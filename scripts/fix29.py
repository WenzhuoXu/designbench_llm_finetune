import io
p = "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts/probe29_synthdomain.py"
s = io.open(p, encoding="utf-8").read()

# 1. grammar: every emitted scaling is applied together, matching the model-free loop's
#    global reallocation. A single move per turn cannot fix n independent components.
old_sys = '''Each turn you are shown every component's current g_i, w_i and x_i, plus the total cost and the
budget. Propose {K} DIFFERENT candidate moves. Think briefly, then emit exactly {K} lines, each
of the form:

<action>SCALE(i, f)</action>

meaning multiply component i's size by factor f. Use 0.7 <= f <= 2.0. Only one candidate will
be executed, so make them genuinely different."""'''
new_sys = '''Each turn you are shown every component's current g_i, w_i and x_i, plus the total cost and the
budget. Think briefly, then emit one line per component you want to resize:

<action>SCALE(i, f)</action>

meaning multiply component i's size by factor f, with 0.7 <= f <= 2.0. EVERY line you emit is
applied, together, in the same turn. You may resize as many components as you like, including
all of them. Raising a component's size raises its g_i and its cost; lowering it frees budget.
You have a limited number of turns, so act on everything that needs it."""'''
assert old_sys in s
s = s.replace(old_sys, new_sys)

# 2. apply all parsed actions rather than scoring them as competing candidates
old_apply = """        best, bv = None, None
        for a, b in moves[:k]:
            try:
                nx = apply_action(dom, st, int(a), float(b))
            except Exception:
                continue
            m = dom.element_margins(nx)
            c = sum(nx["w"][i] * nx["x"][i] ** 2 for i in range(nx["n"]))
            v = min(m) - (0.5 if c > nx["B"] else 0.0)
            if bv is None or v > bv:
                best, bv = nx, v
        if best is None:
            history.append({"t": (text or "").strip()[:2000], "o": obs})
            continue
        st = best"""
new_apply = """        nx = dom.clone(st)
        applied = 0
        for a, b in moves:
            try:
                i, f = int(a), float(b)
            except Exception:
                continue
            if 0 <= i < dom.n_elements(nx) and 0.5 <= f <= 3.0:
                dom.set(nx, i, "x", nx["x"][i] * f)
                applied += 1
        if not applied:
            history.append({"t": (text or "").strip()[:2000], "o": obs})
            continue
        st = nx"""
assert old_apply in s
s = s.replace(old_apply, new_apply)

s = s.replace('st = dom.load({"seed": seed, "n": 12})', 'st = dom.load({"seed": seed, "n": 10})')
s = s.replace('ap.add_argument("--max-steps", type=int, default=4)',
              'ap.add_argument("--max-steps", type=int, default=6)')
s = s.replace('ap.add_argument("--k", type=int, default=6)',
              'ap.add_argument("--k", type=int, default=10)')
io.open(p, "w", encoding="utf-8").write(s)
print("patched")
