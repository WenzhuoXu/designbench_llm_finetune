"""con1 answer formats (the target-format ablation axis, docs/consequence_round_con1.md §9).

Every format is rendered from the same simulated record (a move of transitions.py: verdict, refusal reason,
metrics_after, feasible, the verbatim result message) and parses back (score_pred.read_any) into the same facets
(regime, feasibility, violated constraints, per-metric direction and size, values where the format has them).

  verbatim   the harness's result message word for word (what a check returns)
  masked     verbatim text with fixed wording down-weighted in the loss (the owner: mask fixed wording in any verbatim
             level): informative = the after values (status block and EFFECT block), the VIOLATED markers, the feasible
             word, the refusal reason; fixed = the header, metric names, limits, before values, section titles
  facet      verdict first, then FEASIBLE AFTER, VIOLATED AFTER, and one line per metric: direction (up / down / same)
             and the size of the change relative to the current value in 1-2-5 bins (about factor-2 wide; no digits)
  delta      structured: verdict first, FEASIBLE AFTER, VIOLATED AFTER, one line per metric: the signed change at 3
             significant digits and the relative change in percent (3 significant digits)
  absolute   structured: verdict first, FEASIBLE AFTER, VIOLATED AFTER, one line per metric: before -> after at 4
             significant digits
  facetm     facet plus, before FEASIBLE AFTER, a MARGINS AFTER block: per constrained metric, the side of its limit after
             the move (satisfied / violated / at limit) and the size of its distance to the limit relative to the limit
             in the same 1-2-5 bins (no digits), so feasibility follows from the margins as it does from M's values
             (added 11-25 after round 1: F's regime 0.601 vs M's 0.673)
Every format's answer carries character spans of its informative content (render's second value); the loss weighs
those tokens 1 and the fixed wording FIXED_WEIGHT (0.05, not 0: the model must still learn to write the scaffold it
generates at test time; 0.05 removes ~95% of the scaffold's share of the loss).
"""
from __future__ import annotations

import math
import re

FORMATS = ("verbatim", "masked", "facet", "delta", "absolute", "facetm", "facetc", "dir")
NL = chr(10)
FIXED_WEIGHT = 0.05
_EDGES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


def _pct(x):
    return ("%g" % (100 * x))


BINS = [(_EDGES[0], "<0.1%")] + [(_EDGES[i + 1], "%s-%s%%" % (_pct(_EDGES[i]), _pct(_EDGES[i + 1])))
                                 for i in range(len(_EDGES) - 1)] + [(math.inf, ">1000%")]
BIN_NAMES = [b for _, b in BINS]
SAME_TOL = 1e-5            # relative: a change below 0.001% of the value is 'same' (the verbatim message prints 6 digits)

QUESTION_HEAD = ("\n\nQUESTION (no move is made now): if the move {move} were played at the current design, what would "
                 "the simulator return? ")
FOLLOW_HEAD = ("QUESTION (no move is made now): if the move {move} were played at the current design, what would the "
               "simulator return? ")
FORMAT_TEXT = {
    "verbatim": ("Answer with the simulator's result message exactly as a check of that move returns it: either the line "
                 "'CHECK RESULT for <move>: refused by the simulator: <reason>. The current design is unchanged.', or "
                 "the line 'CHECK RESULT for <move> (simulated on a copy; the current design is unchanged)' followed by "
                 "the block CONSTRAINT STATUS (after this move), the block EFFECT OF THIS MOVE (before -> after) with one "
                 "'metric  before -> after' line per metric, and the line 'Feasible after this move: yes.' or "
                 "'Feasible after this move: no.'"),
    "facet": ("Answer in this form: either 'VERDICT: refused' and 'REASON: <the simulator's reason>', or 'VERDICT: "
              "applied', 'FEASIBLE AFTER: yes|no', 'VIOLATED AFTER: <constraints violated after the move, or none>', "
              "then 'CHANGES:' and one line per metric: the metric, its direction (up, down or same) and the size of "
              "the change relative to its current value as one of " + ", ".join(BIN_NAMES) + " (or 'from 0' when the "
              "current value is 0)."),
    "delta": ("Answer in this form: either 'VERDICT: refused' and 'REASON: <the simulator's reason>', or 'VERDICT: "
              "applied', 'FEASIBLE AFTER: yes|no', 'VIOLATED AFTER: <constraints violated after the move, or none>', "
              "then 'CHANGES:' and one line per metric: the metric, the signed change (after minus before) to 3 "
              "significant digits and the relative change in percent to 3 significant digits (or 'from 0')."),
    "absolute": ("Answer in this form: either 'VERDICT: refused' and 'REASON: <the simulator's reason>', or 'VERDICT: "
                 "applied', 'FEASIBLE AFTER: yes|no', 'VIOLATED AFTER: <constraints violated after the move, or none>', "
                 "then 'METRICS (before -> after):' and one line per metric: the metric, its current value and its "
                 "value after the move, both to 4 significant digits."),
}
FORMAT_TEXT["masked"] = FORMAT_TEXT["verbatim"]
FORMAT_TEXT["facetm"] = (
    "Answer in this form: either 'VERDICT: refused' and 'REASON: <the simulator's reason>', or 'VERDICT: applied', then "
    "'MARGINS AFTER:' and one line per constrained metric: the metric, whether it satisfies its limit after the move "
    "(satisfied, violated or at limit) and the size of its distance to the limit relative to the limit as one of "
    + ", ".join(BIN_NAMES) + " (or 'limit 0' when the limit is 0), then 'FEASIBLE AFTER: yes|no', 'VIOLATED AFTER: "
    "<constraints violated after the move, or none>', then 'CHANGES:' and one line per metric: the metric, its "
    "direction (up, down or same) and the size of the change relative to its current value as one of "
    + ", ".join(BIN_NAMES) + " (or 'from 0' when the current value is 0).")
# facetc (11-26): facetm plus the move's chain through the design's own elements, before MARGINS AFTER: LOCAL = the
# moved element's internal-state quantities, REDISTRIBUTION = the CHAIN_K other elements whose state changes most
# (relative), then a line bounding the rest. Families with per-element state printed in the observation only; the
# others keep facetm (fmt_for).
CHAIN_FAMILIES = ("frame3d2", "grid2op", "circuit2")
CHAIN_K = 6
CHAIN_QTY = {"frame3d2": "member axial force N (kN, + tension)", "grid2op": "line loading rho",
             "circuit2": "the quantities of each element's RESULTS line"}
FORMAT_TEXT["facetc"] = (
    "Answer in this form: either 'VERDICT: refused' and 'REASON: <the simulator's reason>', or 'VERDICT: applied', then "
    "'LOCAL:' and one line per internal-state quantity of the moved element (the element, the quantity, its direction "
    "up, down or same and the size of its change relative to its current value as one of " + ", ".join(BIN_NAMES) +
    "; 'LOCAL: none' when the move sets no element with its own state), then 'REDISTRIBUTION:' and one line for each of "
    "the %d other elements whose state changes most in relative terms (the element, the quantity, direction and size "
    "as above, largest first), closed by 'all other elements: change at most <size>' (or 'all other elements: same'), "
    "then 'MARGINS AFTER:' and one line per constrained metric: the metric, whether it satisfies its limit after the "
    "move (satisfied, violated or at limit) and the size of its distance to the limit relative to the limit (or 'limit "
    "0' when the limit is 0), then 'FEASIBLE AFTER: yes|no', 'VIOLATED AFTER: <constraints violated after the move, or "
    "none>', then 'CHANGES:' and one line per metric: the metric, its direction (up, down or same) and the size of the "
    "change relative to its current value (or 'from 0' when the current value is 0)." % CHAIN_K)


# dir (11-27, parent's order for the base-model read): direction only, no size bin, no margin block.
FORMAT_TEXT["dir"] = (
    "Answer in this form: either 'VERDICT: refused' and 'REASON: <the simulator's reason>', or 'VERDICT: applied', "
    "'FEASIBLE AFTER: yes|no', 'VIOLATED AFTER: <constraints violated after the move, or none>', then 'CHANGES:' and "
    "one line per metric: the metric and its direction (up, down or same).")
NAMED_METRICS = " The metrics, one line each in this order: %s."
# 11-28 (parent's orders): the answer's terms defined in the question. ON BY DEFAULT since lib_v8 for every new arm and
# every builder call (question(..., defined=None) follows DEFINED_DEFAULT); an arm that must stay comparable with arm D
# (the effort arm, repeats, D-long, D-brief, delta) opts out explicitly: think_client --no-defined-terms, builders
# --no-defined-terms, or DEFINED_DEFAULT = False / question(..., defined=False). Every prompt before 11-28 was asked
# without them. 'same' is defined at 0.1% relative: a line whose true change is between the harness tolerance
# (SAME_TOL, 1e-5) and 0.1% is a 'subthreshold' line (make_line_flags.py) and is scored apart, so the headline truth
# is the same under either tolerance.
DEFINED_DEFAULT = True
SAME_TOL_DEFINED = 1e-3
DEFINE_FEASIBLE = " FEASIBLE AFTER is yes only when no constraint is violated after the move; otherwise it is no."
DEFINE_FEASIBLE_VERBATIM = (" The last line says yes only when no constraint is violated after the move; otherwise it "
                            "says no.")
DEFINE_SAME = (" A metric's direction is 'same' when it changes by less than 0.1% of its current value; otherwise it "
               "is up or down.")


def definitions(fmt: str) -> str:
    """The definitions appended to fmt's question when defined=True: feasible for every form, 'same' for the forms
    that ask for a direction word."""
    text = FORMAT_TEXT[fmt]
    out = DEFINE_FEASIBLE_VERBATIM if fmt in ("verbatim", "masked") else DEFINE_FEASIBLE
    if "(up, down or same)" in text or "up, down or same" in text:
        out += DEFINE_SAME
    return out


def fmt_for(fmt: str, variant: str) -> str:
    """The format a family is asked and trained in: facetc falls back to facetm outside CHAIN_FAMILIES."""
    return "facetm" if fmt == "facetc" and variant not in CHAIN_FAMILIES else fmt


def question(fmt: str, move: str, first: bool, metrics=None, defined=None) -> str:
    """The question text; with metrics (11-27, optional) the CHANGES lines are asked for exactly these metrics, in the
    given order ('one line per listed metric' + the list), so no metric is left to the reader's guess; the definitions
    of feasible and 'same' follow (definitions()) when defined is True, or when it is None and DEFINED_DEFAULT is True
    (the default since lib_v8)."""
    text = FORMAT_TEXT[fmt]
    if metrics:
        text = text.replace("one line per metric", "one line per listed metric") + NAMED_METRICS % ", ".join(metrics)
    if DEFINED_DEFAULT if defined is None else defined:
        text += definitions(fmt)
    return (QUESTION_HEAD if first else FOLLOW_HEAD).format(move=move) + text


# ---- per-element internal state (facetc): parsed from the observation text the harness renders ---------------------
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def parse_elements(variant: str, text: str) -> dict:
    """{element label: {quantity: value}} as the observation prints them (frame3d2: installed members' N; grid2op:
    lines' rho (0 when out), generators' p, storage power; circuit2: every key=value of each element's RESULTS
    line)."""
    out = {}
    if variant == "frame3d2":
        i = text.find("Installed members by place")
        j = text.find("Optional members", i)
        for line in text[i:j if j > i else len(text)].splitlines():
            m = re.match(r"^(m\d+) \d+-\d+ [^|]*\|[^|]*\|\s*(%s)" % _NUM, line)
            if m:
                out[m.group(1)] = {"N": float(m.group(2))}
    elif variant == "grid2op":
        for m in re.finditer(r"(?m)^  line (\d+): sub \d+ - sub \d+, (?:rho (%s)|OUT)" % _NUM, text):
            out["line %s" % m.group(1)] = {"rho": float(m.group(2)) if m.group(2) else 0.0}
        for m in re.finditer(r"(?m)^  (gen\d+) \w+ sub \d+: p (%s) MW" % _NUM, text):
            out[m.group(1)] = {"p": float(m.group(2))}
        for m in re.finditer(r"(?m)^(?:Storage units: |; )?(stor\d+) sub \d+: charge (%s) of [^,]*, power (%s) MW" % (_NUM, _NUM),
                             text.replace("; stor", "\nstor")):
            out[m.group(1)] = {"charge": float(m.group(2)), "power": float(m.group(3))}
    elif variant == "circuit2":
        i = text.find("RESULTS (")
        for line in text[i:].splitlines()[1:]:
            m = re.match(r"^  (\d) (\w+): (.*)$", line)
            if not m:
                if line.strip().startswith(("Other schemes", "Frequency response", "Sensitivity", "Headroom")):
                    break
                continue
            q = {}
            for k, v in re.findall(r"(?:(?<=^)|(?<=[\s,(]))([A-Za-z][\w()*/]*)\s*=\s*(%s)" % _NUM, m.group(3)):
                if k.count("(") == k.count(")"):            # 'Cc)' from 'gm1/(2 pi Cc)=...' is a formula tail
                    q.setdefault(k, float(v))
            out["%s %s" % (m.group(1), m.group(2))] = q
    return out


def moved_elements(variant: str, move: str, labels) -> list:
    """Labels of the elements a move sets (LOCAL); [] when it sets none with its own printed state."""
    m = re.match(r"\s*(\w+)\((.*)\)\s*$", move)
    if not m:
        return []
    verb, args = m.group(1), [a.strip() for a in re.split(r",(?![^\[]*\])", m.group(2))]
    ints = [int(a) for a in args if re.fullmatch(r"\d+", a)]
    want = []
    if variant == "frame3d2" and verb in ("SET_RADIUS", "SET_MATERIAL", "INSTALL_MEMBER", "REMOVE_MEMBER") and ints:
        want = ["m%d" % ints[0]]
    elif variant == "grid2op":
        if "LINE" in verb and ints:
            want = ["line %d" % ints[0]]
        elif verb == "REDISPATCH" and len(ints) >= 2:
            want = ["gen%d" % ints[0], "gen%d" % ints[1]]
        elif ("STORAGE" in verb or verb.startswith("STOR")) and ints:
            want = ["stor%d" % ints[0]]
        elif "CURTAIL" in verb and ints:
            want = ["gen%d" % ints[0]]
    elif variant == "circuit2" and ints:
        want = [lab for lab in labels if lab.split(" ", 1)[0] == str(ints[0])]
    return [w for w in want if w in labels]


def _nat(label: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", label)]


# absolute floors below which a per-element change is solver noise (zero-flow lines print rho ~1e-14 that varies
# between runs); circuit2 relies on SAME_TOL
CHAIN_FLOOR = {"grid2op": 1e-4, "frame3d2": 1e-3}


def _rel(b: float, a: float, floor: float = 0.0) -> float:
    if is_same(b, a) or abs(a - b) < floor:
        return 0.0
    return abs(a - b) / abs(b) if b != 0 else math.inf


def _dir_bin(b: float, a: float, floor: float = 0.0):
    if abs(a - b) < floor:
        return "same", "same"
    return direction(b, a), size_bin(b, a)


def chain_block(variant: str, move: str, before: dict, after: dict, k: int = CHAIN_K) -> dict:
    """The facetc chain of one applied move from the per-element state before and after (parse_elements):
    {local: [(label, qty, dir, bin)], redis: [(label, qty, dir, bin, rel)], rest: bin or 'same', ranked: [(label, rel)]}.
    Redistribution covers the other elements' main quantity (frame3d2 N, grid2op line rho, circuit2 per element the
    quantity with the largest relative change)."""
    labels = set(before) | set(after)
    moved = moved_elements(variant, move, labels)
    fl = CHAIN_FLOOR.get(variant, 0.0)
    local = []
    for lab in sorted(moved, key=_nat):
        b0, a0 = before.get(lab, {}), after.get(lab, {})
        for q in sorted(set(b0) | set(a0)):
            if q not in a0:
                local.append((lab, q, "removed", ""))
            elif q not in b0:
                local.append((lab, q, "added", ""))
            else:
                local.append((lab, q) + _dir_bin(b0[q], a0[q], fl))
    cand = []
    for lab in labels - set(moved):
        if variant == "grid2op" and not lab.startswith("line"):
            continue
        b0, a0 = before.get(lab), after.get(lab)
        if b0 is None or a0 is None:
            continue
        qs = ["N"] if variant == "frame3d2" else ["rho"] if variant == "grid2op" else sorted(set(b0) & set(a0))
        best = None
        for q in qs:
            if q in b0 and q in a0:
                r = _rel(b0[q], a0[q], fl)
                if best is None or r > best[0]:
                    best = (r, q)
        if best is not None:
            q = best[1]
            cand.append((lab, q) + _dir_bin(b0[q], a0[q], fl) + (best[0],))
    cand.sort(key=lambda x: (-x[4], _nat(x[0])))
    top = [c for c in cand if c[4] > 0][:k]
    rest = [c for c in cand[len(top):]]
    rest_bin = "same"
    if rest and rest[0][4] > 0:
        rest_bin = rest[0][3]
    return {"local": local, "redis": top, "rest": rest_bin, "ranked": [(c[0], c[1], c[2], c[4]) for c in cand]}


def parse_chain(text: str) -> dict:
    """The LOCAL and REDISTRIBUTION blocks of a facetc answer: {local: {(label, qty): dir}, redis: [(label, dir)]}."""
    out = {"local": {}, "redis": []}
    text = text or ""
    i, j, k = text.find("LOCAL:"), text.find("REDISTRIBUTION"), text.find("MARGINS AFTER:")
    if i < 0:
        return out
    loc = text[i:j if j > i else (k if k > i else len(text))]
    for m in re.finditer(r"(?m)^  (.+?) (\S+) (up|down|same|removed|added)\b", loc):
        out["local"][(m.group(1), m.group(2))] = m.group(3)
    if j >= 0:
        red = text[j:k if k > j else len(text)]
        for m in re.finditer(r"(?m)^  (?!all other)(.+?) (\S+) (up|down|same)\b", red):
            out["redis"].append((m.group(1), m.group(3)))
    return out


def chain_text(ch: dict):
    """(lines, informative spans relative to the joined text) of the LOCAL and REDISTRIBUTION blocks."""
    lines, spans, pos = [], [], 0

    def add(fixed, info=""):
        nonlocal pos
        if info:
            spans.append((pos + len(fixed), pos + len(fixed) + len(info)))
        lines.append(fixed + info)
        pos += len(fixed) + len(info) + 1

    if ch["local"]:
        add("LOCAL:")
        for lab, q, dr, bn in ch["local"]:
            add("  %s %s " % (lab, q), ("%s %s" % (dr, bn)).strip() if bn != "same" else "same")
    else:
        add("LOCAL: ", "none")
    if ch["redis"]:
        add("REDISTRIBUTION:")
        for lab, q, dr, bn, _ in ch["redis"]:
            add("  ", "%s %s %s" % (lab, q, ("%s %s" % (dr, bn)) if bn != "same" else "same"))
        add("  all other elements: ", "same" if ch["rest"] == "same" else "change at most %s" % ch["rest"])
    else:
        add("REDISTRIBUTION: ", "no other element changes")
    return lines, spans


def is_same(before: float, after: float) -> bool:
    return abs(after - before) <= max(SAME_TOL * abs(before), 1e-12)


def size_bin(before: float, after: float) -> str:
    d = after - before
    if is_same(before, after):
        return "same"
    if before == 0:
        return "from 0"
    r = abs(d) / abs(before)
    for hi, name in BINS:
        if r < hi:
            return name
    return BIN_NAMES[-1]


def direction(before: float, after: float) -> str:
    d = after - before
    if is_same(before, after):
        return "same"
    return "up" if d > 0 else "down"


def sig(x: float, n: int = 3) -> str:
    if x == 0 or not math.isfinite(x):
        return "0" if x == 0 else str(x)
    return "%.*g" % (n, x)


STATUS_VIOL = re.compile(r"^\s+([A-Za-z_][\w.\-]*)\s+\S+\s+limit .*VIOLATED", re.M)


def violated_after(result: str) -> list:
    """Constraints marked VIOLATED in the message's CONSTRAINT STATUS (after this move) block."""
    i = result.find("CONSTRAINT STATUS")
    j = result.find("EFFECT OF THIS MOVE")
    if i < 0:
        return []
    return STATUS_VIOL.findall(result[i:j if j > i else len(result)])


STATUS_LIMIT = re.compile(r"^\s+([A-Za-z_][\w.\-]*)\s+(\S+)\s+limit\s*(>=|<=)\s*(\S+)(.*)$")


def margins_after(result: str) -> list:
    """[(constraint, side, size)] from the message's CONSTRAINT STATUS (after this move) block, in its order: side
    'satisfied' / 'violated' (the harness's VIOLATED marker) / 'at limit'; size the bin of |value - limit| / |limit|,
    'limit 0' when the limit is 0, '' at the limit."""
    i = result.find("CONSTRAINT STATUS")
    j = result.find("EFFECT OF THIS MOVE")
    if i < 0:
        return []
    out = []
    for line in result[i:j if j > i else len(result)].splitlines()[1:]:
        m = STATUS_LIMIT.match(line)
        if not m:
            continue
        try:
            v, lim = float(m.group(2)), float(m.group(4))
        except ValueError:
            continue
        marg = (v - lim) if m.group(3) == ">=" else (lim - v)
        viol = "VIOLATED" in m.group(5)
        if not viol and abs(marg) <= max(SAME_TOL * abs(lim), 1e-12):
            out.append((m.group(1), "at limit", ""))
            continue
        side = "violated" if viol else "satisfied"
        if lim == 0:
            out.append((m.group(1), side, "limit 0"))
            continue
        r = abs(marg) / abs(lim)
        out.append((m.group(1), side, next((name for hi, name in BINS if r < hi), BIN_NAMES[-1])))
    return out


def render(fmt: str, move: dict, metrics0: dict):
    """(content, informative spans [(start, end)] or None). verbatim: None (every token weighs 1); every other
    format: the spans of its informative content (the rest is fixed wording, FIXED_WEIGHT in the loss)."""
    res = move["result"]
    if fmt == "verbatim":
        return res, None
    if fmt == "masked":
        return res, informative_spans(res)
    out, spans = [], []

    def add(fixed: str, info: str = "", tail: str = ""):
        start = sum(len(x) + 1 for x in out)
        out.append(fixed + info + tail)
        if info:
            spans.append((start + len(fixed), start + len(fixed) + len(info)))

    if move["verdict"] == "refused":
        reason = move.get("refused") or res.split("refused by the simulator:", 1)[-1].strip()
        add("VERDICT: ", "refused")
        add("REASON: ", reason)
        return NL.join(out), spans
    after = move["metrics_after"]
    viol = violated_after(res)
    add("VERDICT: ", "applied")
    if fmt == "facetm":
        add("MARGINS AFTER:")
        for k, side, size in margins_after(res):
            add("  %-22s " % k, ("%s %s" % (side, size)).strip())
    add("FEASIBLE AFTER: ", "yes" if move["feasible"] else "no")
    add("VIOLATED AFTER: ", ", ".join(viol) if viol else "none")
    add("METRICS (before -> after):" if fmt == "absolute" else "CHANGES:")
    for k in sorted(metrics0):
        if k not in after:
            continue
        b, a = float(metrics0[k]), float(after[k])
        name = "  %-22s " % k
        if fmt in ("facet", "facetm"):
            sb = size_bin(b, a)
            add(name, "same" if sb == "same" else "%s %s" % (direction(b, a), sb))
        elif fmt == "dir":
            add(name, direction(b, a))
        elif fmt == "absolute":
            add(name + "%s -> " % sig(b, 4), sig(a, 4))
        else:
            dr = direction(b, a)
            if dr == "same":
                add(name, "same")
            elif b == 0:
                add(name, "%s from 0" % (("+" if a > b else "") + sig(a - b)))
            else:
                add(name, "%s (%s%%)" % (("+" if a > b else "") + sig(a - b), ("+" if a > b else "") + sig(100.0 * (a - b) / abs(b))))
    return NL.join(out), spans


EFFECT_LINE = re.compile(r"^(\s+[A-Za-z_][\w.\-]*\s+\S+\s*->\s*)(\S+)\s*$")
STATUS_LINE = re.compile(r"^(\s+[A-Za-z_][\w.\-]*\s+)(\S+)(.*)$")


def informative_spans(res: str) -> list:
    """Character spans of the verbatim message that carry the prediction (masked format)."""
    spans = []
    if "refused by the simulator:" in res:
        i = res.index("refused by the simulator:") + len("refused by the simulator:")
        j = res.find(". The current design is unchanged.", i)
        spans.append((i, j if j > 0 else len(res)))
        k = res.find("refused by the simulator")
        spans.append((k, k + len("refused")))                 # the word that says refused
        return spans
    pos, sec = 0, None
    for line in res.splitlines(keepends=True):
        s = line.rstrip("\n")
        if s.startswith("CONSTRAINT STATUS"):
            sec = "status"
        elif s.startswith("EFFECT OF THIS MOVE"):
            sec = "effect"
        elif s.startswith("Feasible after this move:"):
            w = s.split(":", 1)[1]
            spans.append((pos + len(s) - len(w), pos + len(s)))
            sec = None
        elif s.startswith("FABRICATION"):
            spans.append((pos, pos + len(s)))
            sec = None
        elif s.startswith("CHECK RESULT"):
            sec = None
        elif sec == "effect":
            m = EFFECT_LINE.match(s)
            if m:
                spans.append((pos + m.start(2), pos + m.end(2)))
        elif sec == "status":
            m = STATUS_LINE.match(s)
            if m:
                spans.append((pos + m.start(2), pos + m.end(2)))
                v = s.find("VIOLATED")
                if v >= 0:
                    spans.append((pos + v, pos + len(s)))
        pos += len(line)
    return spans
