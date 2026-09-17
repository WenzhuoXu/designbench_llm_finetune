"""
A fourth domain: discrete section selection with stiffness-attracted load. Non-monotone.

All three existing domains are continuous sizing problems with monotone response -- enlarge an
element and its own margin improves, always. That is why one sizing rule works in all of them,
and it makes a corpus built from them narrower than it looks. Two properties are missing:

  DISCRETE ACTIONS. Elements here choose a section from a catalogue of eight standard profiles.
  There is no continuous knob to scale, so SIZE_PASS as written has nothing to act on and the
  domain registers its own vocabulary instead: CATALOGUE_PASS, UPGRADE, DOWNGRADE, SET_SECTION.
  CATALOGUE_PASS is the discrete analogue of the sizing rule -- give each element the smallest
  section whose capacity clears margin times its demand -- which is worth having as evidence
  that the framework's central move generalises off the continuum.

  NON-MONOTONE RESPONSE. The elements are parallel load paths sharing one total load, and load
  distributes by relative stiffness. Upsizing an element raises its stiffness, which pulls MORE
  load onto it. Capacity grows as roughly the section modulus while attracted load grows as the
  second moment, so upsizing a member can lower its own margin and raise everyone else's. A
  greedy "strengthen the worst element" rule can therefore move backwards, which no other domain
  here punishes.

Instances are calibrated so a feasible assignment provably exists: a satisfying assignment is
searched for at generation time and the mass budget is set from it, so nothing unsolvable ships.
"""
import sys, math, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from da_domain import Domain

# eight profiles, deepening: area, second moment, section modulus, mass per length
# I grows faster than Z, which is what makes attracted load outrun capacity
CATALOGUE = []
for k in range(8):
    depth = 0.10 * (1.18 ** k)
    area = 0.004 * (1.14 ** k)
    I = area * depth * depth / 12.0 * 6.0
    Z = 2.0 * I / depth
    CATALOGUE.append({"name": "S%d" % k, "depth": depth, "area": area,
                      "I": I, "Z": Z, "mass": area * 7850.0})
NK = len(CATALOGUE)


class CatalogueDomain(Domain):
    """Parallel load paths choosing discrete sections; load follows stiffness."""

    name = "catalogue"
    params = ("k",)
    element_noun = "member"
    param_label = "section"

    def __init__(self):
        self.calls = 0

    # ------------------------------------------------------------------ instance

    def load(self, spec):
        r = random.Random(spec["seed"])
        n = spec.get("n", 10)
        length = [r.uniform(2.0, 6.0) for _ in range(n)]
        lever = [r.uniform(0.25, 1.0) for _ in range(n)]
        fy = 250e6

        # solve for the load that puts the worst margin below 1 at the lightest sections,
        # rather than drawing a load and hoping strength binds
        stiff0 = [1.0 / (L ** 3) for L in length]
        tot0 = sum(stiff0)
        cap0 = CATALOGUE[0]["Z"] * fy
        worst_unit = max((stiff0[i] / tot0) * lever[i] for i in range(n))
        target_margin = r.uniform(0.35, 0.65)
        total_load = cap0 / (worst_unit * target_margin)
        st = {"n": n, "length": length, "lever": lever, "total_load": total_load,
              "fy": fy, "k": [0.0] * n, "B": 0.0, "cache": None}

        # find a satisfying assignment so the instance is provably solvable, then set the
        # budget just above what it costs -- tight enough that the obvious answer of
        # "everything at the largest section" is usually too heavy
        best = None
        for _ in range(400):
            cand = [float(r.randrange(NK)) for _ in range(n)]
            st["k"], st["cache"] = cand, None
            if min(self._margins(st)) >= 1.0:
                m = self._mass(st)
                if best is None or m < best[0]:
                    best = (m, list(cand))
        if best is None:                      # fall back to the heaviest assignment
            st["k"] = [float(NK - 1)] * n
            st["cache"] = None
            best = (self._mass(st), list(st["k"]))
        st["B"] = best[0] * 1.06
        st["k"] = [0.0] * n                   # start at the lightest sections: infeasible
        st["cache"] = None
        return st

    def clone(self, st):
        c = dict(st)
        c["k"] = list(st["k"])
        c["cache"] = None
        return c

    # ------------------------------------------------------------------ physics

    def _idx(self, st, i):
        return max(0, min(NK - 1, int(round(st["k"][i]))))

    def _mass(self, st):
        return sum(CATALOGUE[self._idx(st, i)]["mass"] * st["length"][i]
                   for i in range(st["n"]))

    def _margins(self, st):
        n = st["n"]
        stiff = [CATALOGUE[self._idx(st, i)]["I"] / (st["length"][i] ** 3) for i in range(n)]
        tot = sum(stiff) or 1e-12
        out = []
        for i in range(n):
            share = st["total_load"] * stiff[i] / tot      # stiffer members attract load
            demand = share * st["lever"][i]
            cap = CATALOGUE[self._idx(st, i)]["Z"] * st["fy"]
            out.append(cap / demand if demand > 1e-9 else 50.0)
        return out

    def evaluate(self, st):
        if st["cache"] is None:
            self.calls += 1
            st["cache"] = {"margins": self._margins(st), "mass": self._mass(st)}
        return st["cache"]

    # ------------------------------------------------------------------ interface

    def n_elements(self, st):
        return st["n"]

    def get(self, st, i, p):
        return float(self._idx(st, i))

    def set(self, st, i, p, v):
        st["k"][i] = max(0.0, min(float(NK - 1), float(v)))
        st["cache"] = None

    def bounds(self, st, p):
        return (0.0, float(NK - 1))

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
        return ("%d parallel load paths carrying %.0f N in total, each choosing one of %d "
                "catalogue sections. Load distributes by relative stiffness, so enlarging a "
                "member draws more load onto it. Every member must reach margin 1.00 and total "
                "mass must stay within budget." % (st["n"], st["total_load"], NK))

    def columns(self, st):
        # `section` is already printed as the parameter column by render_state
        return [("length", list(st["length"])),
                ("lever", list(st["lever"]))]

    # ------------------------------------------------------------------ its own tools

    def tools(self):
        def _apply(dom, st, newk):
            nx = dom.clone(st)
            for i, v in enumerate(newk):
                dom.set(nx, i, "k", v)
            return nx

        def sample_pass(dom, st, rng):
            return {"margin": rng.uniform(0.95, 1.35)}

        def apply_pass(dom, st, args):
            """Discrete analogue of the sizing rule: smallest section that clears the margin."""
            want = args.get("margin", 1.05)
            n = st["n"]
            newk = list(st["k"])
            for i in range(n):
                for cand in range(NK):
                    trial = list(newk)
                    trial[i] = float(cand)
                    probe = dom.clone(st)
                    for j, v in enumerate(trial):
                        dom.set(probe, j, "k", v)
                    if probe["cache"] is None:
                        pass
                    if dom._margins(probe)[i] >= want:
                        newk[i] = float(cand)
                        break
                else:
                    newk[i] = float(NK - 1)
            return _apply(dom, st, newk)

        def sample_step(dom, st, rng):
            n = dom.n_elements(st)
            k = rng.randint(1, max(1, min(4, n)))
            return {"ids": rng.sample(range(n), k)}

        def apply_up(dom, st, args):
            newk = list(st["k"])
            for i in args.get("ids", []):
                if 0 <= i < len(newk):
                    newk[i] = min(float(NK - 1), newk[i] + 1.0)
            return _apply(dom, st, newk)

        def apply_down(dom, st, args):
            newk = list(st["k"])
            for i in args.get("ids", []):
                if 0 <= i < len(newk):
                    newk[i] = max(0.0, newk[i] - 1.0)
            return _apply(dom, st, newk)

        def sample_set(dom, st, rng):
            n = dom.n_elements(st)
            k = rng.randint(1, max(1, min(4, n)))
            return {"ids": rng.sample(range(n), k), "section": rng.randrange(NK)}

        def apply_set(dom, st, args):
            newk = list(st["k"])
            s = float(max(0, min(NK - 1, int(args.get("section", 0)))))
            for i in args.get("ids", []):
                if 0 <= i < len(newk):
                    newk[i] = s
            return _apply(dom, st, newk)

        return {"CATALOGUE_PASS": (sample_pass, apply_pass,
                                  "CATALOGUE_PASS(margin=<0.95-1.35>)"),
                "UPGRADE": (sample_step, apply_up,
                            "UPGRADE(ids=[<element ids>])"),
                "DOWNGRADE": (sample_step, apply_down,
                              "DOWNGRADE(ids=[<element ids>])"),
                "SET_SECTION": (sample_set, apply_set,
                                "SET_SECTION(ids=[<element ids>], section=<0-%d>)" % (NK - 1))}
