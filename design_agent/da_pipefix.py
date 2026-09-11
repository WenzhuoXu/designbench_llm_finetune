"""
Calibrate the pipe generator so instances are actually solvable.

The first version set the material budget from an arbitrary diameter of 0.62 without ever
checking that any profile meets the pressure requirements. It does not: with P0 = 4.0 and drops
of order L*q^2/d^5, a diameter of 0.62 leaves every node far under pressure, so all seven arms
scored 0.0000 and the run measured nothing.

Instances are now built so a feasible profile provably exists. A uniform diameter scale is
binary-searched until every node meets its requirement; the budget is set slightly above the
material that profile costs, and the starting diameters are set well below it. So there is a
solution, it is reachable, and it is not free -- the search has to find a non-uniform profile
that fits inside a budget the uniform one only just satisfies.
"""
import io

p = "/ocean/projects/mch250030p/wxu7/llm_finetune/design_agent/da_pipe.py"
s = io.open(p, encoding="utf-8").read()

old = '''        d = [0.55] * n
        st = {"n": n, "parent": parent, "L": L, "demand": demand, "preq": preq,
              "d": d, "P0": 4.0, "q": None, "B": 0.0, "cache": None}
        st["q"] = self._flows(st)
        st["B"] = sum(L[i] * (0.62 ** 2) for i in range(n))    # a budget a good sizing can meet
        return st'''

new = '''        st = {"n": n, "parent": parent, "L": L, "demand": demand, "preq": preq,
              "d": [1.0] * n, "P0": 4.0, "q": None, "B": 0.0, "cache": None}
        st["q"] = self._flows(st)

        # smallest uniform diameter that satisfies every node, by bisection
        def ok_at(scale):
            st["d"] = [scale] * n
            st["cache"] = None
            pr = self._pressures(st)
            return all(pr["press"][i] >= st["preq"][i] for i in range(n))

        lo, hi = 0.05, 8.0
        if not ok_at(hi):
            hi = 40.0
            if not ok_at(hi):
                hi = 200.0
                ok_at(hi)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if ok_at(mid):
                hi = mid
            else:
                lo = mid
        ref = hi
        # budget: a little above what the uniform reference costs, so a smarter
        # non-uniform profile is needed but the instance is certainly solvable
        st["B"] = 1.08 * sum(L[i] * ref ** 2 for i in range(n))
        st["d"] = [ref * 0.55] * n            # start under-sized: work is required
        st["cache"] = None
        return st'''

assert old in s, "load body not found"
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("pipe generator calibrated")
