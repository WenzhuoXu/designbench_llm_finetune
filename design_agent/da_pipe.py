"""
A third domain: pipe network sizing. Coupled, no topology, physics-lite.

Two domains is a thin basis for a portability claim, and the two I have differ in the wrong
ways: the truss is coupled and registers topology tools, the synthetic domain is uncoupled and
registers none. Nothing separates "coupled" from "has topology to exploit".

This is a tree-shaped distribution network. Water leaves a source at pressure P0 and runs down
pipes to demand nodes; each pipe drops pressure by

    drop_i = L_i * q_i^2 / d_i^5

with the flow q_i fixed by the demands below it, and every node needs its pressure to stay at or
above its requirement. Enlarging a pipe therefore helps every node downstream of it and nobody
else -- real coupling along paths, with none of the truss's force redistribution and no topology
to change. The budget is material: sum of L_i * d_i^2.

Per-element attribution is the tightest node reachable through that pipe, which is what a
sizing rule would need and what the Domain interface asks for.

Third point for the claim that the search's margin tracks the slack its domain's heuristic
leaves: this one's heuristic can be made strong or weak by construction.
"""
import sys, math, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from da_domain import Domain


class PipeDomain(Domain):
    name = "pipe"
    params = ("d",)

    def __init__(self):
        self.calls = 0

    def load(self, spec):
        r = random.Random(spec["seed"])
        n = spec.get("n", 14)                       # pipes; node i+1 is fed by pipe i
        parent = [0] * n                            # parent[i] = feeding pipe index, -1 = source
        parent[0] = -1
        for i in range(1, n):
            parent[i] = r.randrange(0, i)           # random tree, shallow on average
        L = [math.exp(r.gauss(0.0, 0.35)) for _ in range(n)]
        demand = [max(0.15, r.lognormvariate(-0.4, 0.5)) for _ in range(n)]
        preq = [max(0.4, r.gauss(1.0, 0.25)) for _ in range(n)]
        st = {"n": n, "parent": parent, "L": L, "demand": demand, "preq": preq,
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
        st["dmax"] = 4.0 * ref

        # Price the budget off the LIGHTEST strength-feasible profile rather than a uniform
        # reference. Iterate the sizing rule at margin 1.00 to convergence; that design is
        # about as light as strength permits, so a pass at margin 1.05 oversizes and exceeds
        # the cap. Both constraints then bind, which a uniform reference at 1.08x never did.
        st["d"] = [ref] * n
        st["cache"] = None
        for _ in range(60):
            em = self.element_margins(st)
            if min(em) >= 0.999 and max(em) <= 1.06:
                break
            for i in range(n):
                m = em[i] if em[i] > 1e-9 else 1e-9
                st["d"][i] = min(st["dmax"], max(0.05,
                                                 st["d"][i] * min(1.5, max(0.7, (1.0 / m) ** 0.2))))
            st["cache"] = None
        st["B"] = 1.03 * sum(L[i] * st["d"][i] ** 2 for i in range(n))
        st["d"] = [ref * 0.78] * n            # start under-sized: work is required
        st["cache"] = None
        return st

    def _flows(self, st):
        """Flow in a pipe is the demand of everything it feeds."""
        n = st["n"]
        q = list(st["demand"])
        order = sorted(range(n), key=lambda i: -i)
        for i in order:
            p = st["parent"][i]
            if p >= 0:
                q[p] += q[i]
        return q

    def clone(self, st):
        c = dict(st)
        c["d"] = list(st["d"])
        c["cache"] = None
        return c

    def evaluate(self, st):
        if st["cache"] is None:
            self.calls += 1
            st["cache"] = self._pressures(st)
        return st["cache"]

    def _pressures(self, st):
        n = st["n"]
        drop = [st["L"][i] * (st["q"][i] ** 2) / max(st["d"][i], 1e-6) ** 5 for i in range(n)]
        press = [0.0] * n
        for i in range(n):
            p, acc = i, 0.0
            guard = 0
            while p >= 0 and guard <= n:
                acc += drop[p]
                p = st["parent"][p]
                guard += 1
            press[i] = st["P0"] - acc
        return {"press": press, "drop": drop}

    def n_elements(self, st):
        return st["n"]

    def get(self, st, i, p):
        return st["d"][i]

    def set(self, st, i, p, v):
        # the upper bound is set from the instance: a fixed 2.0 silently clipped every write
        # on networks whose required diameters are far larger, making them unsolvable
        st["d"][i] = min(st.get("dmax", 2.0), max(0.05, float(v)))
        st["cache"] = None

    def bounds(self, st, p):
        return (0.05, st.get("dmax", 2.0))

    def _node_margins(self, st):
        # floored strictly positive: a sizing rule that skips non-positive margins would
        # otherwise treat an over-drained network as a fixed point and never grow anything
        s = self.evaluate(st)
        return [max(s["press"][i] / st["preq"][i], 1e-3) for i in range(st["n"])]

    def element_margins(self, st):
        """Per pipe: the tightest node it feeds, itself included."""
        nm = self._node_margins(st)
        n = st["n"]
        out = [nm[i] for i in range(n)]
        for i in range(n - 1, -1, -1):
            p = st["parent"][i]
            if p >= 0 and out[i] < out[p]:
                out[p] = out[i]
        return out

    def global_margins(self, st):
        return [min(self._node_margins(st))]

    def budget_ratio(self, st):
        used = sum(st["L"][i] * st["d"][i] ** 2 for i in range(st["n"]))
        return used / st["B"] if st["B"] > 0 else 0.0

    def feasible(self, st):
        return min(self._node_margins(st)) >= 1.0 and self.budget_ratio(st) <= 1.0
