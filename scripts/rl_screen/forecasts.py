"""Forecast lines of a reply's rationale, and whether the simulator bears them out.

A forecast-checked round asks the policy (at collection only) to compare two or three candidate
moves inside its rationale, one line each:

    FORECAST: VERB(arg, ...) -> refused | worse | better

refused  the simulator rejects the move and the design stays as it is
better   the move is applied and the potential rises (the rule of the gain screen)
worse    the move is applied and the potential does not rise

score_states.py parses every candidate, applies it on a clone of the state and records what the
simulator returned (`truth`); build_rows.py --require-forecasts keeps a reply only when `fault`
returns None. Nothing here reads the rationale beyond these lines.

Reading is strict. A line is read only in the one form above (a short label as in "FORECAST 2:"
and markdown marks are allowed): one move written as a single call, an arrow, one of the three
words, and none of the three words again after it (a hedge such as "worse or better", a second
forecast on the line). A line that says FORECAST and cannot be read this way is counted (`unread`),
and so is a line that says "forecast" in any other case together with an arrow or one of the three
words. A reply with an unread forecast line is not a row target: a forecast the check cannot see
may be wrong.
"""
import re

VERDICTS = ("refused", "worse", "better")
MAX_SCORED = 6          # candidates simulated per reply (a reply with more than three lines is not a row target anyway)
STRICT = re.compile(r"\bFORECAST\b")
LOOSE = re.compile(r"\bforecast\b", re.I)
HEAD = re.compile(r"^[ \t>*\-`#\d.)|]*FORECAST(?:[ \t]*[#(]?[A-Za-z0-9]{1,3}\)?)?\**[ \t]*:[ \t`*]*", re.I)
CALL = re.compile(r"[A-Za-z_][A-Za-z_0-9]*[ \t]*\(")
ARROW = re.compile(r"[ \t`*]*(?:->|=>|→|⇒)[ \t`*]*")
ANY_ARROW = re.compile(r"->|=>|→|⇒")
WORD = re.compile(r"\b(refused|worse|better)\b", re.I)
VERDICT = re.compile(r"(refused|worse|better)\b", re.I)


def call_end(text: str) -> int:
    """Index just past the balanced VERB(...) call that `text` starts with, or -1."""
    m = CALL.match(text)
    if not m:
        return -1
    depth = 1
    for i in range(m.end(), len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def read_line(line: str):
    """(move as written, verdict) of one forecast line in the form asked for, else None."""
    m = HEAD.match(line)
    if not m:
        return None
    rest = line[m.end():]
    end = call_end(rest)
    if end < 0:
        return None
    a = ARROW.match(rest, end)
    if not a:
        return None
    v = VERDICT.match(rest, a.end())
    if not v or WORD.search(rest, v.end()):
        return None
    return rest[:end], v.group(1).lower()


def read(reasoning: str):
    """([(move as written, verdict)] in the order written, number of forecast lines that cannot be read)."""
    lines, unread = [], 0
    for line in (reasoning or "").splitlines():
        if not LOOSE.search(line):
            continue
        got = read_line(line)
        if got is not None:
            lines.append(got)
        elif STRICT.search(line) or ANY_ARROW.search(line) or WORD.search(line):
            unread += 1
    return lines, unread


def forecast_lines(reasoning: str) -> list:
    return read(reasoning)[0]


def truth_of(applied: bool, phi, phi0: float) -> str:
    """What the simulator returned for a candidate: refused, or better / worse by the gain rule
    (the potential after the move is above the state's own by more than 1e-9)."""
    if not applied:
        return "refused"
    return "better" if phi > phi0 + 1e-9 else "worse"


def fault(fc: list, unread: int = 0, lo: int = 2, hi: int = 3):
    """None when the forecasts of a reply make it a row target, else the first reason it is not.
    fc: the reply's `forecasts` as score_states.py records them, one dict per line: parsed, text
    (the candidate as the domain formats it), said, truth, phi, noop, is_pick. unread: its
    forecast lines that could not be read."""
    fc = fc or []
    if unread:
        return "a forecast line cannot be read"
    if len(fc) < lo:
        return "fewer than %d forecast lines" % lo
    if len(fc) > hi:
        return "more than %d forecast lines" % hi
    if not all(c.get("parsed") for c in fc):
        return "a candidate is not a move"
    if len({c["text"] for c in fc}) < len(fc):
        return "a candidate is written twice"
    if any(c.get("noop") for c in fc):
        return "a candidate changes nothing"
    picks = [c for c in fc if c.get("is_pick")]
    if len(picks) != 1:
        return "the chosen move is not a candidate"
    if any(c["said"] != c["truth"] for c in fc):
        return "a forecast is wrong"
    p = picks[0]
    if p["truth"] != "better":
        return "the chosen move is not better"
    if any(c is not p and c["truth"] != "refused" and c["phi"] > p["phi"] + 1e-9 for c in fc):
        return "a rejected candidate does better"
    return None


def has_negative(fc: list) -> bool:
    """A rejected candidate the reply forecast, correctly, as refused or worse."""
    return any(not c.get("is_pick") and c.get("parsed") and c["said"] == c["truth"] and c["truth"] in ("refused", "worse")
               for c in fc or [])
