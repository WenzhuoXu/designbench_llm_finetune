"""con1 answer formats (the target-format ablation axis, docs/consequence_round_con1.md §9).

Every format is rendered from the same simulated record (a move of transitions.py: verdict, refusal reason,
metrics_after, feasible, the verbatim result message) and parses back (score_pred.read_any) into the same facets
(regime, feasibility, violated constraints, per-metric direction and size, values where the format has them).

  verbatim   the harness's result message word for word (what a check returns)
  masked     verbatim text; the loss only on informative tokens: the after values (status block and EFFECT block),
             the VIOLATED markers, the feasible word, the refusal reason. Not supervised: the header, metric names,
             limits, before values and section titles (all copyable from the prompt or the question)
  facet      verdict first, then FEASIBLE AFTER, VIOLATED AFTER, and one line per metric: direction (up / down / same)
             and the size of the change relative to the current value in half-decade bins (no digits of the value)
  delta      verdict first, then FEASIBLE AFTER, VIOLATED AFTER, and one line per metric: the signed change at 3
             significant digits and the relative change in percent (3 significant digits)
"""
from __future__ import annotations

import math
import re

FORMATS = ("verbatim", "masked", "facet", "delta")
BINS = [(0.001, "<0.1%"), (0.003, "0.1-0.3%"), (0.01, "0.3-1%"), (0.03, "1-3%"), (0.1, "3-10%"), (0.3, "10-30%"),
        (1.0, "30-100%"), (3.0, "100-300%"), (10.0, "300-1000%"), (math.inf, ">1000%")]
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
}
FORMAT_TEXT["masked"] = FORMAT_TEXT["verbatim"]


def question(fmt: str, move: str, first: bool) -> str:
    return (QUESTION_HEAD if first else FOLLOW_HEAD).format(move=move) + FORMAT_TEXT[fmt]


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


def render(fmt: str, move: dict, metrics0: dict):
    """(content, supervise_spans or None) of the answer for one simulated move."""
    res = move["result"]
    if fmt == "verbatim":
        return res, None
    if fmt == "masked":
        return res, informative_spans(res)
    if move["verdict"] == "refused":
        reason = move.get("refused") or res.split("refused by the simulator:", 1)[-1].strip()
        return "VERDICT: refused\nREASON: %s" % reason, None
    after = move["metrics_after"]
    viol = violated_after(res)
    lines = ["VERDICT: applied", "FEASIBLE AFTER: %s" % ("yes" if move["feasible"] else "no"),
             "VIOLATED AFTER: %s" % (", ".join(viol) if viol else "none"), "CHANGES:"]
    for k in sorted(metrics0):
        if k not in after:
            continue
        b, a = float(metrics0[k]), float(after[k])
        if fmt == "facet":
            sb = size_bin(b, a)
            lines.append("  %-22s %s" % (k, "same" if sb == "same" else "%s %s" % (direction(b, a), sb)))
        else:
            dr = direction(b, a)
            if dr == "same":
                lines.append("  %-22s same" % k)
            elif b == 0:
                lines.append("  %-22s %s from 0" % (k, ("+" if a > b else "") + sig(a - b)))
            else:
                lines.append("  %-22s %s (%s%%)" % (k, ("+" if a > b else "") + sig(a - b),
                                                    ("+" if a > b else "") + sig(100.0 * (a - b) / abs(b))))
    return "\n".join(lines), None


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
