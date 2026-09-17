"""
Domain-registered tools: the interface change that lets a domain contribute its own actions.

The generic search ports -- same code, one subclass -- and beats the synthetic domain's own
heuristic by 15.0pp (discordant 18-0). On trusses it manages only 0.5333 against the
truss-native implementation's 0.9256, and the reason is structural: the generic library is
SIZE_PASS, SCALE and TRIM, because add-member, remove-member and move-joint cannot be phrased
in terms of get/set on a parameter. Topology was carrying most of the truss result.

So the interface gains one method. A Domain may register extra tools:

    tools() -> {name: (sample(dom, st, rng) -> args or None,
                       apply(dom, st, args)  -> new state or None,
                       signature)}

where signature is the call as a model must write it, e.g.
"REMOVE_MEMBER(i=<member id>)". It is what the system prompt shows, so it lives
beside the apply function that reads those argument names rather than in the
prompt, where the two would drift apart. A two-element entry still works and is
advertised as NAME(...), which tells a model nothing about its arguments.

The base returns nothing, so every existing domain is unaffected and the search above the
interface does not change -- it simply asks the domain what else it can do. The truss registers
its three topology moves; the synthetic domain has no topology and registers none.

Appended here rather than edited into the class bodies so the existing interface stays intact
and this addition can be read on its own.
"""
import sys
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
_PROJECT = HERE.parent
_DESIGNBENCH = Path(os.environ.get("DESIGNBENCH_ROOT", _PROJECT.parent / "DesignBench"))
for p in (str(HERE), str(_PROJECT), str(_PROJECT / "scripts"), str(_DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import Domain, TrussDomain


def _base_tools(self):
    """Domains contribute nothing extra unless they say so."""
    return {}


Domain.tools = _base_tools


def _truss_tools(self):
    import api_grammar_2x2 as H

    def _rebuild(st, action):
        nt = H.apply_candidate(st["t"], action)
        if nt is None:
            return None
        return {"t": nt, "goals": st["goals"], "spec": st["spec"],
                "bounds": st["bounds"], "cache": None}

    def _njoints(st):
        try:
            return len(st["t"].joints)
        except Exception:
            return 0

    def _xy(st, j):
        try:
            c = st["t"].joints[j].coordinates
            return float(c[0]), float(c[1])
        except Exception:
            try:
                c = st["t"].joints[j].coords
                return float(c[0]), float(c[1])
            except Exception:
                return None

    def sample_add(dom, st, rng):
        nj = _njoints(st)
        if nj < 2:
            return None
        a, b = rng.sample(range(nj), 2)
        return {"j1": a, "j2": b}

    def apply_add(dom, st, args):
        return _rebuild(st, "ADD_MEMBER(%d, %d, 6061_T6_Aluminum, Pipe(r=0.030, t=0.004))"
                        % (args["j1"], args["j2"]))

    def sample_remove(dom, st, rng):
        n = dom.n_elements(st)
        if n <= 1:
            return None
        return {"i": rng.randrange(n)}

    def apply_remove(dom, st, args):
        return _rebuild(st, "REMOVE_MEMBER(%d)" % args["i"])

    def sample_move(dom, st, rng):
        nj = _njoints(st)
        if nj < 1:
            return None
        j = rng.randrange(nj)
        xy = _xy(st, j)
        if xy is None:
            return None
        return {"j": j, "x": xy[0] + rng.uniform(-0.4, 0.4),
                "y": xy[1] + rng.uniform(-0.4, 0.4)}

    def apply_move(dom, st, args):
        return _rebuild(st, "MOVE_JOINT(%d, [%.4f, %.4f, 0.0])"
                        % (args["j"], args["x"], args["y"]))

    return {"ADD_MEMBER": (sample_add, apply_add,
                          "ADD_MEMBER(j1=<joint id>, j2=<joint id>)"),
            "REMOVE_MEMBER": (sample_remove, apply_remove,
                              "REMOVE_MEMBER(i=<member id>)"),
            "MOVE_JOINT": (sample_move, apply_move,
                           "MOVE_JOINT(j=<joint id>, x=<new x>, y=<new y>)")}


TrussDomain.tools = _truss_tools
