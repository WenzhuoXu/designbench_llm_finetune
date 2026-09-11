"""
Make the multi-case domain discriminate the search, by changing its physics rather than a constant.

Three budget calibrations all returned search 1.0000 = envelope heuristic 1.0000, and the
harvest then showed the domain recording zero turns: the do-nothing continuation solves it
before the search chooses anything. The cause is not the budget. It is that enlarging a member
there is unambiguously good.

    stiffness  ~ x^2        load attracted to a member rises as x^2
    capacity   ~ x^3        its capacity rises as x^3

so capacity outruns attracted load and per-member envelope sizing converges monotonically. The
cross-case structure exists in the observation but never costs the heuristic anything.

Inverting the two exponents flips that:

    stiffness  ~ x^3
    capacity   ~ x^2

Now enlarging one member pulls load onto it faster than it gains capacity, so targeted repair
backfires -- the same mechanism that makes the catalogue domain hard, where a single upgrade
lowers its own margin 95.2% of the time. Uniform scaling still works, because scaling every
member by s leaves the load shares identical and multiplies every capacity by s^2, so feasible
designs still exist and the instance stays solvable. What breaks is the greedy per-member rule.

The domain's own SIZE_ENVELOPE exponent moves from 1/3 to 1/2 to match capacity ~ x^2, so the
heuristic is still the sensible rule for the new physics and the comparison stays fair.
"""
import io

p = "/ocean/projects/mch250030p/wxu7/llm_finetune/design_agent/da_cases.py"
s = io.open(p, encoding="utf-8").read()

old = '''        stiff = [st["x"][i] ** 2 for i in range(n)]'''
new = '''        # stiffness rises FASTER than capacity, so enlarging a member attracts load faster
        # than it gains strength and targeted repair backfires; uniform scaling still works
        stiff = [st["x"][i] ** 3 for i in range(n)]'''
assert old in s, "stiffness line not found"
s = s.replace(old, new, 1)

old2 = '''                capacity = st["cap_c"][i] * st["x"][i] ** 3'''
new2 = '''                capacity = st["cap_c"][i] * st["x"][i] ** 2'''
assert old2 in s, "capacity line not found"
s = s.replace(old2, new2, 1)

# both sizing tools invert capacity ~ x^2 now, not x^3
s = s.replace('newx.append(st["x"][i] * min(2.0, max(0.7, (want / m) ** (1.0 / 3.0))))',
              'newx.append(st["x"][i] * min(2.0, max(0.7, (want / m) ** (1.0 / 2.0))))')

# the budget certificate iterates the same rule, so it must use the same exponent
s = s.replace('self.set(probe, i, "x", probe["x"][i] * min(1.6, max(0.75, (1.0 / m) ** (1.0 / 3.0))))',
              'self.set(probe, i, "x", probe["x"][i] * min(1.6, max(0.75, (1.0 / m) ** (1.0 / 2.0))))')

io.open(p, "w", encoding="utf-8").write(s)
print("cases physics inverted: stiffness x^3, capacity x^2; sizing exponents updated to 1/2")
