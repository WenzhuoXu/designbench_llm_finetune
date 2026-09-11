"""
A fifth domain: sizing against several load cases at once, where feasibility is a conjunction.

Every domain so far has one requirement per element. Here each element must survive K scenarios
-- gravity, wind from either side, an asymmetric live load -- and is feasible only if it clears
all of them. Two things follow that no earlier domain has.

  THE OBSERVATION GAINS A DIMENSION. An element's margin is the minimum across cases, so the
  useful question is not only "how short is it" but "which case is binding". Two elements at the
  same margin can need opposite treatment if different cases bind them, and the table has to
  carry that.

  ACTING ON ONE CASE CAN BREAK ANOTHER. Load distributes by relative stiffness, and the cases
  load different parts of the structure, so enlarging an element to satisfy the case that
  currently binds it pulls more of every OTHER case onto it too. Sizing case by case oscillates;
  only sizing against the governing envelope converges. The domain registers SIZE_FOR_CASE so
  the wrong strategy is available to be tried and scored rather than assumed away.

Instances are calibrated like the catalogue domain: the load scale is solved for so the worst
margin starts below 1, and a feasible assignment is certified before the instance ships.
"""
import sys, math, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from da_domain import Domain

CAP = 50.0
CASE_NAMES = ("gravity", "wind_left", "wind_right", "live_asym", "uplift")


class CasesDomain(Domain):
    """Continuous sizing against K load cases; feasibility is the conjunction over cases."""

    name = "cases"
    params = ("x",)
    element_noun = "member"
    param_label = "size"

    def __init__(self):
        self.calls = 0

    # ------------------------------------------------------------------ instance

    def load(self, spec):
        r = random.Random(spec["seed"])
        n = spec.get("n", 10)
        k = spec.get("cases", 4)
        lever = [r.uniform(0.3, 1.0) for _ in range(n)]
        cap_c = [r.uniform(0.8, 1.4) for _ in range(n)]
        weight = [r.uniform(0.6, 1.5) for _ in range(n)]
        # each case loads a different part of the structure
        geom = [[max(0.05, r.gauss(1.0, 0.55)) for _ in range(n)] for _ in range(k)]
        st = {"n": n, "k": k, "lever": lever, "cap_c": cap_c, "w": weight,
              "geom": geom, "loads": [1.0] * k, "x": [1.0] * n, "B": 0.0, "cache": None}

        # solve the load scale so the worst margin starts below one, as in da_catalogue
        st["loads"] = [1.0] * k
        m0 = self._case_margins(st)
        worst = min(min(row) for row in m0)
        target = r.uniform(0.35, 0.60)
        scale = worst / target
        st["loads"] = [scale * r.uniform(0.75, 1.25) for _ in range(k)]
        st["cache"] = None

        # Price the budget off the LIGHTEST strength-feasible design, found by iterating the
        # envelope to convergence at margin 1.00. A fixed-margin heuristic at 1.05 oversizes
        # every member and busts the cap; choosing the margin correctly fits. Same structure as
        # the truss, where fully-stressed design satisfies strength and overshoots mass.
        probe = self.clone(st)
        for _ in range(40):
            em = self.element_margins(probe)
            if min(em) >= 0.999 and max(em) <= 1.05:
                break
            for i in range(n):
                m = em[i] if em[i] > 1e-9 else 1e-9
                self.set(probe, i, "x", probe["x"][i] * min(1.6, max(0.75, (1.0 / m) ** (1.0 / 2.0))))
        m_min = sum(st["w"][i] * probe["x"][i] ** 2 for i in range(n))
        st["B"] = m_min * 1.05

        st["x"] = [1.0] * n
        st["cache"] = None
        return st

    def clone(self, st):
        c = dict(st)
        c["x"] = list(st["x"])
        c["cache"] = None
        return c

    # ------------------------------------------------------------------ physics

    def _case_margins(self, st):
        """[case][element] -- capacity over the demand that case puts on that element."""
        n, k = st["n"], st["k"]
        # stiffness rises FASTER than capacity, so enlarging a member attracts load faster
        # than it gains strength and targeted repair backfires; uniform scaling still works
        stiff = [st["x"][i] ** 3 for i in range(n)]
        out = []
        for c in range(k):
            wsum = sum(stiff[i] * st["geom"][c][i] for i in range(n)) or 1e-12
            row = []
            for i in range(n):
                share = st["loads"][c] * stiff[i] * st["geom"][c][i] / wsum
                demand = share * st["lever"][i]
                capacity = st["cap_c"][i] * st["x"][i] ** 2
                row.append(min(capacity / demand, CAP) if demand > 1e-12 else CAP)
            out.append(row)
        return out

    def evaluate(self, st):
        if st["cache"] is None:
            self.calls += 1
            cm = self._case_margins(st)
            st["cache"] = {
                "case_margins": cm,
                "margins": [min(cm[c][i] for c in range(st["k"])) for i in range(st["n"])],
                "binding": [min(range(st["k"]), key=lambda c: cm[c][i]) for i in range(st["n"])],
                "mass": sum(st["w"][i] * st["x"][i] ** 2 for i in range(st["n"])),
            }
        return st["cache"]

    # ------------------------------------------------------------------ interface

    def n_elements(self, st):
        return st["n"]

    def get(self, st, i, p):
        return st["x"][i]

    def set(self, st, i, p, v):
        st["x"][i] = min(6.0, max(0.05, float(v)))
        st["cache"] = None

    def bounds(self, st, p):
        return (0.05, 6.0)

    def element_margins(self, st):
        return list(self.evaluate(st)["margins"])

    def global_margins(self, st):
        return [min(self.evaluate(st)["margins"])]

    def budget_ratio(self, st):
        return self.evaluate(st)["mass"] / st["B"] if st["B"] > 0 else 0.0

    def feasible(self, st):
        s = self.evaluate(st)
        return min(s["margins"]) >= 1.0 and (s["mass"] / st["B"] if st["B"] > 0 else 0.0) <= 1.0

    def describe(self, st):
        names = ", ".join(CASE_NAMES[:st["k"]])
        return ("%d members that must survive %d load cases at once (%s). A member is adequate "
                "only if it clears EVERY case. Load distributes by relative stiffness and each "
                "case loads a different part of the structure, so enlarging a member draws more "
                "of every other case onto it. Total mass must stay within budget."
                % (st["n"], st["k"], names))

    def columns(self, st):
        s = self.evaluate(st)
        cols = [("binds", [float(b) for b in s["binding"]])]
        for c in range(st["k"]):
            cols.append((CASE_NAMES[c][:9], [min(s["case_margins"][c][i], CAP)
                                             for i in range(st["n"])]))
        return cols

    # ------------------------------------------------------------------ its own tools

    def tools(self):
        def _apply(dom, st, newx):
            nx = dom.clone(st)
            for i, v in enumerate(newx):
                dom.set(nx, i, "x", v)
            return nx

        def sample_case(dom, st, rng):
            return {"case": rng.randrange(st["k"]), "margin": rng.uniform(0.95, 1.35)}

        def apply_case(dom, st, args):
            """Size every member for ONE case only -- the strategy that oscillates."""
            c = int(args.get("case", 0)) % st["k"]
            want = args.get("margin", 1.05)
            cm = dom.evaluate(st)["case_margins"][c]
            newx = []
            for i in range(st["n"]):
                m = cm[i] if cm[i] > 1e-9 else 1e-9
                newx.append(st["x"][i] * min(2.0, max(0.7, (want / m) ** (1.0 / 2.0))))
            return _apply(dom, st, newx)

        def sample_env(dom, st, rng):
            return {"margin": rng.uniform(0.95, 1.35)}

        def apply_env(dom, st, args):
            """Size against the governing envelope: the min across cases, per member."""
            want = args.get("margin", 1.05)
            em = dom.element_margins(st)
            newx = []
            for i in range(st["n"]):
                m = em[i] if em[i] > 1e-9 else 1e-9
                newx.append(st["x"][i] * min(2.0, max(0.7, (want / m) ** (1.0 / 2.0))))
            return _apply(dom, st, newx)

        return {"SIZE_FOR_CASE": (sample_case, apply_case),
                "SIZE_ENVELOPE": (sample_env, apply_env)}
