"""
One serialisation for every domain, so a balanced corpus teaches one procedure.

The truss expert traces emit DesignBench grammar -- SCALE_PARAM(3, radius, 1.15). A pipe has no
radius and a material-selection domain has no continuous parameter at all, so trajectories from
different domains would arrive in different languages and a model trained on the mixture would
learn three formats rather than one design procedure. That defeats the point of balancing them.

So both halves of the conversation are made domain-agnostic here.

OBSERVATION. Every domain renders through render_state(): a problem line the domain supplies, an
element table keyed by margin, a budget line, and the worst element named. The structure is
identical everywhere; what differs is content the domain declares -- what it calls its elements,
what its sizing parameter is, and any extra columns worth showing. A truss says "member" and
shows radius; a pipe says "pipe" and shows length and flow; the layout does not move.

ACTION. The assistant emits TOOL CALLS, not the domain's native grammar:

    <tool>SIZE_PASS(margin=1.05)</tool>
    <tool>SCALE(ids=[3, 7], factor=1.20)</tool>
    <tool>TRIM(threshold=3.0, factor=0.85)</tool>
    <tool>ADD_MEMBER(j1=0, j2=5)</tool>          # only where the domain registers it

The first three exist in every domain; the rest come from Domain.tools(). Each domain translates
a tool call into whatever its executor wants, which is where truss grammar now lives -- below the
serialisation rather than in it.

Domains need not be modified: every hook has a default, and a domain that declares nothing still
renders correctly.
"""
import sys, math, json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

CAP = 50.0
TOOL_RE = re.compile(r"<tool>\s*([A-Z_]+)\s*\(([^)]*)\)\s*</tool>", re.I)


# ---------------------------------------------------------------- domain description hooks

def noun(dom):
    """What this domain calls one element. Domains may set `element_noun`."""
    return getattr(dom, "element_noun", None) or "element"


def param_name(dom):
    """Human name for the sizing parameter. Domains may set `param_label`."""
    lbl = getattr(dom, "param_label", None)
    if lbl:
        return lbl
    params = getattr(dom, "params", ()) or ()
    return params[0] if params else "size"


def describe(dom, st):
    """One-line problem statement. Domains may implement describe(st)."""
    fn = getattr(dom, "describe", None)
    if callable(fn):
        try:
            return str(fn(st))
        except Exception:
            pass
    return ("Size every %s so that all of them meet requirement, without exceeding the budget."
            % noun(dom))


def extra_columns(dom, st):
    """[(header, [values])] the domain wants shown. Domains may implement columns(st)."""
    fn = getattr(dom, "columns", None)
    if callable(fn):
        try:
            cols = fn(st)
            return cols if cols else []
        except Exception:
            return []
    return []


# ---------------------------------------------------------------- observation

def _fmt(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "     n/a"
    if not math.isfinite(v):
        return "     inf"
    if abs(v) >= 1e5:
        return "%9.1e" % v
    return "%9.4f" % v


def render_state(dom, st, header=True):
    """The same layout in every domain; only the content is domain-specific."""
    n = dom.n_elements(st)
    em = dom.element_margins(st) or []
    p = (getattr(dom, "params", ()) or ("x",))[0]
    cols = extra_columns(dom, st)
    lines = []
    if header:
        lines.append(describe(dom, st))
        lines.append("A %s meets requirement at margin 1.00 or above." % noun(dom))
        lines.append("")
    head = "  %-5s %10s %10s" % ("id", "margin", param_name(dom))
    for cname, _ in cols:
        head += " %10s" % cname[:10]
    lines.append("%s table:" % noun(dom).capitalize())
    lines.append(head)
    for i in range(n):
        m = em[i] if i < len(em) else None
        try:
            mv = min(float(m), CAP)
        except (TypeError, ValueError):
            mv = float("nan")
        row = "  %-5s %10s %10s" % ("E%d" % i, _fmt(mv), _fmt(dom.get(st, i, p)))
        for _, vals in cols:
            row += " %10s" % _fmt(vals[i] if i < len(vals) else None)
        lines.append(row)
    lines.append("")
    try:
        br = dom.budget_ratio(st)
        lines.append("budget: %.3f of the limit (%s)"
                     % (br, "within" if br <= 1.0 else "OVER"))
    except Exception:
        pass
    if em:
        j = min(range(len(em)), key=lambda i: em[i])
        lines.append("worst %s: E%d at margin %s" % (noun(dom), j, _fmt(min(em[j], CAP))))
    return "\n".join(lines)


# ---------------------------------------------------------------- action

def render_tool(name, args):
    """One canonical spelling, so the same move looks the same in every domain."""
    parts = []
    for k in sorted(args):
        v = args[k]
        if isinstance(v, (list, tuple)):
            parts.append("%s=[%s]" % (k, ", ".join(str(int(x)) for x in v)))
        elif isinstance(v, float):
            parts.append("%s=%.4g" % (k, v))
        else:
            parts.append("%s=%s" % (k, v))
    return "<tool>%s(%s)</tool>" % (name.upper(), ", ".join(parts))


def parse_tools(text):
    """Read tool calls back out of model or trace text. Returns [(NAME, {args})]."""
    out = []
    for name, argstr in TOOL_RE.findall(text or ""):
        args = {}
        for m in re.finditer(r"([A-Za-z_]+)\s*=\s*(\[[^\]]*\]|[-\d.eE+]+)", argstr):
            k, raw = m.group(1), m.group(2).strip()
            if raw.startswith("["):
                args[k] = [int(float(x)) for x in re.findall(r"-?\d+\.?\d*", raw)]
            else:
                try:
                    args[k] = float(raw)
                except ValueError:
                    continue
        out.append((name.upper(), args))
    return out


def tool_vocabulary(dom):
    """What this domain lets the model say: the shared three plus whatever it registers."""
    base = ["SIZE_PASS(margin=<0.95-1.35>)",
            "SCALE(ids=[<element ids>], factor=<0.70-2.00>)",
            "TRIM(threshold=<margin>, factor=<0.70-1.00>)"]
    try:
        reg = dom.tools() or {}
    except Exception:
        reg = {}
    # A domain that declares a signature gets it shown; one that does not is
    # advertised as NAME(...), which is honest but useless to a model.
    return base + [(reg[k][2] if len(reg[k]) > 2 else "%s(...)" % k) for k in sorted(reg)]
