"""con1: read a predicted result message and score it against the simulator (docs/consequence_round_con1.md §3).

read_message(text, metrics0) -> {"kind": "refused" | "applied" | "unreadable", "metrics": predicted after-metrics}
    refused     the text contains 'refused by the simulator'
    applied     at least one metric line could be read: the after value of each EFFECT line ('name  before -> after'),
                else its CONSTRAINT STATUS value, else the state's own value (unchanged)
    unreadable  neither
verdict_of(pred, ctx) -> refused | better | worse | unreadable, with ctx = PhiContext of the state: the predicted Phi
    is the problem's potential of the predicted metrics (PhiEvaluator._phi_one; every family is deterministic) and
    the verdict uses the same rule as the truth (conseq_lib.State.verdict).
"""
from __future__ import annotations

import math
import re

EFFECT_HDR = "EFFECT OF THIS MOVE"
STATUS_HDR = "CONSTRAINT STATUS"
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan"
EFFECT_LINE = re.compile(r"^\s*([A-Za-z_][\w.\-]*)\s+(%s)\s*->\s*(%s)\s*$" % (NUM, NUM))
STATUS_LINE = re.compile(r"^\s*([A-Za-z_][\w.\-]*)\s+(%s)(?:\s|$)" % NUM)


def _f(x: str):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def read_message(text: str, metrics0: dict) -> dict:
    text = text or ""
    if "refused by the simulator" in text:
        return {"kind": "refused", "metrics": None}
    status, effect = {}, {}
    sec = None
    for line in text.splitlines():
        if line.startswith(STATUS_HDR):
            sec = "status"
            continue
        if line.startswith(EFFECT_HDR):
            sec = "effect"
            continue
        if not line.strip():
            continue
        if not line.startswith(" "):
            sec = None if not line.startswith(("CONSTRAINT", "EFFECT")) else sec
            continue
        if sec == "effect":
            m = EFFECT_LINE.match(line)
            if m and _f(m.group(3)) is not None:
                effect[m.group(1)] = _f(m.group(3))
        elif sec == "status":
            m = STATUS_LINE.match(line)
            if m and _f(m.group(2)) is not None:
                status[m.group(1)] = _f(m.group(2))
    if not status and not effect:
        return {"kind": "unreadable", "metrics": None}
    pred = dict(metrics0)
    pred.update({k: v for k, v in status.items() if k in pred})
    pred.update({k: v for k, v in effect.items() if k in pred})
    n_read = len(set(status) | set(effect))
    return {"kind": "applied", "metrics": pred, "n_read": n_read,
            "feasible_said": ("Feasible after this move: yes" in text) if "Feasible after this move:" in text else None}


class PhiContext:
    """What verdict_of needs about one state: the problem's evaluator, Phi and pixels of the state."""

    def __init__(self, ev, variant: str, phi0: float, bad0: int, opening: bool = True):
        self.ev, self.variant, self.phi0, self.bad0, self.opening = ev, variant, float(phi0), int(bad0), opening

    def phi(self, metrics: dict) -> float:
        return float(self.ev._phi_one(metrics))

    def verdict(self, phi: float, viol: float) -> str:
        if self.opening and self.variant == "photonics":
            bad = int(round(viol or 0.0))
            return "better" if (bad < self.bad0 or (bad == self.bad0 and phi > self.phi0 + 1e-9)) else "worse"
        return "better" if phi > self.phi0 + 1e-9 else "worse"


def verdict_of(pred: dict, ctx: PhiContext) -> tuple:
    """(verdict, predicted phi or None)."""
    if pred["kind"] != "applied":
        return pred["kind"], None
    m = pred["metrics"]
    try:
        phi = ctx.phi(m)
    except Exception:                                      # noqa: BLE001
        return "unreadable", None
    viol = float(m.get("min_feature_violations") or 0.0) if ctx.variant == "photonics" else 0.0
    return ctx.verdict(phi, viol), phi


def rank_value(verdict: str, phi, viol, variant: str):
    """A sortable predicted (or true) quality of a move at a state: refused / unreadable last."""
    if verdict in ("refused", "unreadable") or phi is None:
        return (-1, 0.0, 0.0)
    if variant == "photonics":
        return (0, -float(viol or 0.0), float(phi))
    return (0, 0.0, float(phi))


def metric_scores(pred_metrics: dict, true_after: dict, before: dict, rel_tol: float = 1e-6) -> dict:
    """N: per metric with a true change, is the predicted change's sign right; and is the prediction closer to the
    truth than 'no change'."""
    n = sign_ok = closer = 0
    for k, t in true_after.items():
        b = before.get(k)
        if b is None or k not in pred_metrics:
            continue
        tol = rel_tol * max(1.0, abs(b))
        dt = t - b
        if abs(dt) <= tol:
            continue
        n += 1
        dp = pred_metrics[k] - b
        sp = 0 if abs(dp) <= tol else (1 if dp > 0 else -1)
        sign_ok += sp == (1 if dt > 0 else -1)
        closer += abs(pred_metrics[k] - t) < abs(b - t)
    return {"n": n, "sign_ok": sign_ok, "closer": closer}
