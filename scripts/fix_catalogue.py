"""
Calibrate the catalogue domain so strength actually binds at the start.

check_catalogue found 120/120 instances already feasible before any move, which makes the
domain useless as a source of trajectories and made its search-vs-heuristic comparison
meaningless. The cause was a design error rather than a bug: demand comes from a share of one
pooled load, and at the uniform lightest assignment every member has capacity roughly eight
times its demand while mass is simultaneously at its minimum, so both constraints are satisfied
at the starting point.

The load is now solved for instead of drawn. At k=0 every section is identical, so

    share_i = (1/L_i^3) / sum_j (1/L_j^3)        demand_i = total_load * share_i * lever_i
    margin_i = Z_0 * f_y / demand_i

is linear in total_load, and total_load can be chosen to put the WORST margin at a target drawn
from U(0.35, 0.65). The start is then infeasible on strength by construction, while uniform
upgrading raises capacity by 1.345 per catalogue step against unchanged load shares, so a
feasible assignment still exists and the certified search still finds and prices it.

This leaves the property the domain exists for untouched: single upgrades remain
self-defeating 95% of the time, because raising one member's stiffness pulls load onto it.
Only coordinated moves help, which is exactly what a greedy per-element rule cannot do.

Also drops the duplicate `section` column, which repeated the parameter render_state prints.
"""
import io

p = "/ocean/projects/mch250030p/wxu7/llm_finetune/design_agent/da_catalogue.py"
s = io.open(p, encoding="utf-8").read()

old = '''        length = [r.uniform(2.0, 6.0) for _ in range(n)]
        lever = [r.uniform(0.25, 1.0) for _ in range(n)]
        total_load = r.uniform(60_000.0, 200_000.0)
        fy = 250e6'''

new = '''        length = [r.uniform(2.0, 6.0) for _ in range(n)]
        lever = [r.uniform(0.25, 1.0) for _ in range(n)]
        fy = 250e6

        # solve for the load that puts the worst margin below 1 at the lightest sections,
        # rather than drawing a load and hoping strength binds
        stiff0 = [1.0 / (L ** 3) for L in length]
        tot0 = sum(stiff0)
        cap0 = CATALOGUE[0]["Z"] * fy
        worst_unit = max((stiff0[i] / tot0) * lever[i] for i in range(n))
        target_margin = r.uniform(0.35, 0.65)
        total_load = cap0 / (worst_unit * target_margin)'''

assert old in s, "load body not found"
s = s.replace(old, new, 1)

old2 = '''    def columns(self, st):
        return [("section", [float(self._idx(st, i)) for i in range(st["n"])]),
                ("length", list(st["length"])),
                ("lever", list(st["lever"]))]'''
new2 = '''    def columns(self, st):
        # `section` is already printed as the parameter column by render_state
        return [("length", list(st["length"])),
                ("lever", list(st["lever"]))]'''
assert old2 in s, "columns body not found"
s = s.replace(old2, new2, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("catalogue domain recalibrated: load solved for, duplicate column dropped")
