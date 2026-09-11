import pathlib
p = pathlib.Path("/ocean/projects/mch250030p/wxu7/llm_finetune/design_agent/da_run.py")
s = p.read_text()

# --- simulator-budget cap on every arm, so arms can be compared at matched cost ---
s = s.replace("def run_arm(spec, arm, max_steps=20):",
              "def run_arm(spec, arm, max_steps=20, sim_budget=None):")
s = s.replace("""    for _ in range(max_steps):
        if dom.feasible(st):
            return 1, dom.calls
        if arm == "base":""",
              """    for _ in range(max_steps):
        if dom.feasible(st):
            return 1, dom.calls
        if sim_budget and dom.calls >= sim_budget:
            break
        if arm == "base":""")

s = s.replace("def run_planner(spec, model, region, token, max_steps=20):",
              "def run_planner(spec, model, region, token, max_steps=20, sim_budget=None):")
s = s.replace("""    for _ in range(max_steps):
        if dom.feasible(st):
            return 1, dom.calls, n_llm""",
              """    for _ in range(max_steps):
        if dom.feasible(st):
            return 1, dom.calls, n_llm
        if sim_budget and dom.calls >= sim_budget:
            break""")

# --- workers carry the budget and a repeat index ---
s = s.replace("""def _w(t):
    spec, arm = t
    try:
        s, c = run_arm(spec, arm)
        return spec["problem_id"], arm, s, c
    except Exception:
        return spec["problem_id"], arm, None, 0""",
              """def _w(t):
    spec, arm, bud = t
    try:
        s, c = run_arm(spec, arm, sim_budget=bud)
        return spec["problem_id"], arm, s, c
    except Exception:
        return spec["problem_id"], arm, None, 0""")

s = s.replace("""def _wl(t):
    spec, model, region, token = t
    try:
        s, c, l = run_planner(spec, model, region, token)
        return spec["problem_id"], "planner", s, c, l""",
              """def _wl(t):
    spec, model, region, token, bud, rep = t
    try:
        s, c, l = run_planner(spec, model, region, token, sim_budget=bud)
        return spec["problem_id"], "planner", s, c, l""")

s = s.replace('    ap.add_argument("--llm", action="store_true")',
              '    ap.add_argument("--llm", action="store_true")\n'
              '    ap.add_argument("--reps", type=int, default=1, help="planner draws per problem")\n'
              '    ap.add_argument("--sim-budget", type=int, default=0, help="cap simulator calls per episode; 0 = uncapped")')

s = s.replace("        res = pool.map(_w, [(s, arm) for s in specs for arm in ARMS], chunksize=1)",
              "        res = pool.map(_w, [(s, arm, a.sim_budget or None) for s in specs for arm in ARMS], chunksize=1)")

s = s.replace("""            pres = list(ex.map(_wl, [(s, a.model, a.region, tok) for s in specs]))
        ps = {pid: s for pid, _, s, _, _ in pres if s is not None}""",
              """            pres = list(ex.map(_wl, [(s, a.model, a.region, tok, a.sim_budget or None, r)
                                     for r in range(a.reps) for s in specs]))
        agg = collections.defaultdict(list)
        for pid, _, s, _, _ in pres:
            if s is not None:
                agg[pid].append(s)
        # majority over reps, so the planner is scored at the same granularity as the
        # deterministic arms rather than on a single stochastic draw
        ps = {pid: int(sum(v) * 2 >= len(v)) for pid, v in agg.items()}
        if a.reps > 1:
            per = [statistics.mean(v[i] for v in agg.values()) for i in range(a.reps)]
            print("  planner per-draw rates: %s | spread %.1fpp"
                  % (", ".join("%.3f" % x for x in per), 100 * (max(per) - min(per))))""")

s = s.replace('print("  planner     solved %.3f   sim/ep %5.0f   llm/ep %4.1f   (%.0fs, n=%d)"',
              'print("  planner     solved %.3f   sim/ep %5.0f   llm/ep %4.1f   (%.0fs, n=%d)"')

p.write_text(s)
print("patched:", "sim_budget" in s, "reps" in s)
