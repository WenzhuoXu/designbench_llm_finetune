"""
Give each domain its own vocabulary and columns, and give pipes a margin that stays informative.

check_serial showed A1 holding only in the weak sense: the layout was identical everywhere
because every domain rendered generically. All three said "element", showed a bare parameter
letter (r / x / d) and offered no domain content at all. The point of a shared format is that
the STRUCTURE is fixed while the CONTENT is the domain's own; without the second half a model
learns a schema but nothing about what it is designing.

It also showed the pipe domain rendering seven identical margins of 0.0010. That is the floor
added when accumulated pressure drops exceed the source pressure and press/preq goes negative.
It is correct as a feasibility test and useless as an observation: nothing distinguishes the
pipe that needs enlarging from the six that do not, so a trajectory teaches nothing.

The fix is a proper margin rather than a clamp. Feasibility at a node is

    P0 - sum(drops on its path)  >=  preq          equivalently   P0 >= preq + sum(drops)

so  margin = P0 / (preq + sum(drops))  is strictly positive everywhere, rises monotonically as
a pipe is enlarged, and is >= 1 exactly when the node is satisfied. Identical feasibility
semantics, no dead zone, and it ranks pipes sensibly while the network is badly under-sized.
"""
import sys, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from da_domain import Domain, TrussDomain
from da_synth import SynthDomain
from da_pipe import PipeDomain

CAP = 50.0


# ---------------------------------------------------------------- truss

TrussDomain.element_noun = "member"
TrussDomain.param_label = "radius"


def _truss_describe(self, st):
    n = self.n_elements(st)
    g = st.get("goals", {}) or {}
    return ("A planar truss of %d members. Every member must reach a factor of safety of %.1f "
            "against buckling and %.1f against yielding, and total mass must stay within %.1f."
            % (n, float(g.get("minimum_fos_buckling", 1.5)),
               float(g.get("minimum_fos_yielding", 1.5)),
               float(g.get("maximum_mass", float("nan")))))


def _truss_columns(self, st):
    """Buckling and yielding separately: the margin is their min, and which one binds matters."""
    gb = float((st.get("goals", {}) or {}).get("minimum_fos_buckling", 1.5))
    gy = float((st.get("goals", {}) or {}).get("minimum_fos_yielding", 1.5))
    self.evaluate(st)
    buck, yld = [], []
    for m in st["t"].members:
        def v(a):
            try:
                x = float(getattr(m, a, float("inf")) or float("inf"))
            except Exception:
                x = float("inf")
            return x if x == x else float("inf")
        buck.append(min(v("fos_buckling") / gb, CAP))
        yld.append(min(v("fos_yielding") / gy, CAP))
    return [("buckling", buck), ("yielding", yld)]


TrussDomain.describe = _truss_describe
TrussDomain.columns = _truss_columns


# ---------------------------------------------------------------- synthetic

SynthDomain.element_noun = "component"
SynthDomain.param_label = "size"


def _synth_describe(self, st):
    return ("A set of %d independent components. Each has a value that rises with its own size "
            "at a rate you are not told, and a cost weight; every value must reach 1.00 and "
            "total cost must stay within budget." % st["n"])


def _synth_columns(self, st):
    return [("weight", list(st["w"]))]


SynthDomain.describe = _synth_describe
SynthDomain.columns = _synth_columns


# ---------------------------------------------------------------- pipe

PipeDomain.element_noun = "pipe"
PipeDomain.param_label = "diameter"


def _pipe_describe(self, st):
    return ("A tree distribution network of %d pipes fed from a source at pressure %.1f. "
            "Every node must hold its required pressure, and total material must stay within "
            "budget. Enlarging a pipe raises pressure at every node downstream of it."
            % (st["n"], st["P0"]))


def _pipe_columns(self, st):
    return [("length", list(st["L"])), ("flow", list(st["q"]))]


def _pipe_node_margins(self, st):
    """P0 / (requirement + drops on the path): positive everywhere, >= 1 exactly when satisfied.

    The previous form was press/preq floored at 1e-3, which collapsed to a constant as soon as
    accumulated drops exceeded the source pressure and so could not rank pipes at all.
    """
    s = self.evaluate(st)
    out = []
    for i in range(st["n"]):
        drop_path = st["P0"] - s["press"][i]          # what the path has already consumed
        need = st["preq"][i] + max(drop_path, 0.0)
        out.append(st["P0"] / need if need > 1e-12 else CAP)
    return out


PipeDomain.describe = _pipe_describe
PipeDomain.columns = _pipe_columns
PipeDomain._node_margins = _pipe_node_margins
