"""
Domain interface for the nested design planner.

Everything above this file -- tools, Phi, the inner optimiser, the planner -- touches ONLY
this interface. Swapping domains means writing one subclass; nothing else changes.

A domain that lacks per-element attribution returns None from element_margins(); tools that
need it degrade rather than crash, which is the battery case (constraint-level map, 9 named
parameters, no per-element margin vector).
"""
from __future__ import annotations
import copy, math, sys
from pathlib import Path

DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)


class Domain:
    """Minimal contract a design domain must satisfy."""

    name = "abstract"
    params: tuple = ()          # tunable per-element parameter names

    def load(self, spec): raise NotImplementedError
    def clone(self, st): raise NotImplementedError
    def evaluate(self, st) -> dict: raise NotImplementedError
    def n_elements(self, st) -> int: raise NotImplementedError
    def get(self, st, i, p): raise NotImplementedError
    def set(self, st, i, p, v): raise NotImplementedError
    def bounds(self, st, p): raise NotImplementedError

    def element_margins(self, st):
        """Per-element achieved/required, >=1 satisfies. None if the domain has no
        per-element attribution."""
        return None

    def global_margins(self, st):
        """Performance constraint margins, achieved/required, >=1 satisfies."""
        raise NotImplementedError

    def budget_ratio(self, st) -> float:
        """resource / limit, <=1 satisfies. inf if unbounded."""
        raise NotImplementedError

    def feasible(self, st) -> bool:
        return (min(self.global_margins(st)) >= 1.0) and (self.budget_ratio(st) <= 1.0)


# ---------------------------------------------------------------- truss

class TrussDomain(Domain):
    name = "truss"
    params = ("r",)

    def __init__(self):
        from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals
        self._an = _analyze_truss
        self._ld = _load_truss_and_goals
        self.calls = 0

    def load(self, spec):
        truss, goals = self._ld(spec)
        sp = ((spec.get("optimization") or {}).get("shape_params") or {})
        st = {"t": truss, "goals": goals, "spec": spec, "bounds": {}, "cache": None}
        for p in self.params:
            rb = sp.get(p) or {}
            st["bounds"][p] = (float(rb.get("min", 1e-9)), float(rb.get("max", 1e9)))
        self.evaluate(st)
        return st

    def clone(self, st):
        return {"t": copy.deepcopy(st["t"]), "goals": st["goals"], "spec": st["spec"],
                "bounds": st["bounds"], "cache": None}

    def evaluate(self, st) -> dict:
        if st["cache"] is None:
            self.calls += 1
            try:
                st["cache"] = dict(self._an(st["t"], st["goals"]))
            except Exception:
                st["cache"] = {"is_feasible": False, "mass": float("inf"),
                               "fos_buckling": 0.0, "fos_yielding": 0.0}
        return st["cache"]

    def n_elements(self, st): return len(st["t"].members)

    def _shape(self, st, i):
        return getattr(getattr(st["t"].members[i], "shape", None), "_params", None) or {}

    def get(self, st, i, p):
        v = self._shape(st, i).get(p)
        return float(v) if isinstance(v, (int, float)) else None

    def set(self, st, i, p, v):
        d = getattr(getattr(st["t"].members[i], "shape", None), "_params", None)
        if d is not None:
            lo, hi = self.bounds(st, p)
            d[p] = min(hi, max(lo, float(v)))
            st["cache"] = None

    def bounds(self, st, p): return st["bounds"].get(p, (1e-9, 1e9))

    def _req(self, st):
        g = st["goals"]
        return (float(g.get("minimum_fos_buckling", 1.5)),
                float(g.get("minimum_fos_yielding", 1.5)))

    def element_margins(self, st):
        self.evaluate(st)
        gb, gy = self._req(st)
        out = []
        for m in st["t"].members:
            def v(a):
                try:
                    x = float(getattr(m, a, float("inf")) or float("inf"))
                except Exception:
                    x = float("inf")
                return x if x == x else float("inf")
            out.append(min(v("fos_buckling") / gb, v("fos_yielding") / gy))
        return out

    def global_margins(self, st):
        s = self.evaluate(st)
        gb, gy = self._req(st)
        return [float(s.get("fos_buckling", 0.0) or 0.0) / gb,
                float(s.get("fos_yielding", 0.0) or 0.0) / gy]

    def budget_ratio(self, st) -> float:
        s = self.evaluate(st)
        B = float(st["goals"].get("maximum_mass", float("inf")))
        if not math.isfinite(B) or B <= 0:
            return 0.0
        return float(s.get("mass", float("inf")) or float("inf")) / B

    def feasible(self, st) -> bool:
        return bool(self.evaluate(st).get("is_feasible"))


REGISTRY = {"truss": TrussDomain}


def get_domain(name):
    if name not in REGISTRY:
        raise KeyError("unknown domain %r; have %s" % (name, sorted(REGISTRY)))
    return REGISTRY[name]()
