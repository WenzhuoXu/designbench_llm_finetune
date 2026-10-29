"""Probe environment of the physics-CoT stream (docs/cot_stream_validation.md §2).

A StreamState wraps one datav4 state record (rows/v2/states_phi.jsonl). `check(text)` simulates a
change on a copy and returns what the next observation would show differently; nothing is applied.
`hset()` is the held-out probe set H(s) and `queries(h)` the quantities asked for each H move.
"""
from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional

import numpy as np

from ..domain import Move, stable_seed
from ..domains import get_domain
from ..ladder import _binding_constraint
from ..ownrows.common import evaluator, rebuild_state, try_apply
from ..render_cot import _limits, policy_observation

ARG_RE = re.compile(r"^\s*([A-Z_]+)\((\-?\d+)")
F3_QTY = ("N_kN", "fos_b")
N_LOWEST = 3


def element_of_text(variant: str, text: Optional[str]) -> Optional[str]:
    """Element a move acts on, from its text: frame3d2 'mN' / 'JN', other families 'e<first int>'."""
    m = ARG_RE.match(text or "")
    if not m:
        return None
    verb, n = m.group(1), int(m.group(2))
    if variant == "frame3d2":
        return ("J%d" if verb == "MOVE_JOINT" else "m%d") % n
    return "e%d" % n


def _is_header(line: str) -> bool:
    s = line.strip()
    return bool(s) and not line.startswith(" ") and (s.endswith(":") or s.isupper() or "(" in s[:40])


class StreamState:
    def __init__(self, rec: dict):
        self.r = rec
        self.sid = rec["sid"]
        self.v = rec["variant"]
        self.d = get_domain(self.v)
        self.spec = rec["spec"]
        self.st = rebuild_state(self.d, self.spec, rec["history"])
        self.ev = evaluator(self.d, self.spec, rec.get("offset_rule") or "infeasible")
        self.m0 = dict(rec["state_metrics"])
        self.lim = _limits(self.spec.get("goals") or {}, self.m0)
        self.nom0 = self._nom(self.st)
        self._base = self._render(self.st, self.m0, rec.get("binding_constraint"))
        self._cache: Dict[str, dict] = {}

    # -- rendering ---------------------------------------------------------------------------
    def _render(self, st, metrics, binding="auto") -> List[str]:
        if binding == "auto":
            binding = _binding_constraint(self.ev.program, metrics)
        turn = {"state_metrics": metrics, "binding_constraint": binding}
        lines, _ = policy_observation(self.d, self.spec, st, turn, [], self.v)
        text = "\n".join(lines)
        head, _, rest = text.partition("\n\n")   # drop the DOMAIN / GOAL head
        return rest.split("\n")

    @staticmethod
    def _status_block(lines: List[str]) -> List[str]:
        for i, l in enumerate(lines):
            if l.startswith("CONSTRAINT STATUS"):
                j = i + 1
                while j < len(lines) and lines[j].strip():
                    j += 1
                return lines[i:j]
        return []

    def _diff(self, child: List[str]) -> List[str]:
        """Lines of `child` that differ from the current observation, each with its block header
        and nearest sub-header; '  ...' marks omitted unchanged lines. The CONSTRAINT STATUS block
        is excluded here (it is appended in full)."""
        base = self._base
        sm = difflib.SequenceMatcher(a=base, b=child, autojunk=False)
        changed = set()
        for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
            if tag in ("replace", "insert"):
                changed.update(range(j1, j2))
        status = self._status_block(child)
        if status:
            s0 = child.index(status[0])
            changed -= set(range(s0, s0 + len(status)))
        keep = set(changed)
        for j in sorted(changed):
            # block header = first line after the previous blank line; sub-header = nearest header
            k = j
            while k > 0 and child[k - 1].strip():
                k -= 1
            keep.add(k)
            for h in range(j - 1, k, -1):
                if _is_header(child[h]):
                    keep.add(h)
                    break
        out, last = [], None
        for j in sorted(keep):
            if last is not None and j > last + 1:
                skipped = [child[x] for x in range(last + 1, j)]
                out.append("" if not any(s.strip() for s in skipped) else "  ...")
            out.append(child[j])
            last = j
        return out

    def _nom(self, st):
        if self.v != "frame3d2":
            return None
        nom = self.d._nominal(st)
        if nom.get("mech"):
            return {"mech": True}
        return {"mech": False, "N": [float(x) for x in nom["N"]], "fos_b": [float(x) for x in nom["fos_b"]],
                "fos_y": [float(x) for x in nom["fos_y"]]}

    # -- the probe ---------------------------------------------------------------------------
    def check(self, text: str) -> dict:
        key = (text or "").strip()
        if key in self._cache:
            return self._cache[key]
        mv = self.d.parse_move(key)
        if mv is None:
            out = {"ok": False, "err": "unparsed", "text": key}
        else:
            child, ap, info = try_apply(self.d, self.st, mv)
            if child is None:
                out = {"ok": False, "err": info.get("err"), "text": key}
            else:
                met = {k: float(v) for k, v in self.d.evaluate(self.d.clone(child), seed=0).items()}
                lines = self._render(child, met)
                out = {"ok": True, "text": key, "applied": self.d.format_move(ap),
                       "noop": bool(info.get("noop")), "clamped": bool(info.get("clamped")),
                       "metrics": met, "nom": self._nom(child),
                       "diff": self._diff(lines), "status": self._status_block(lines)}
        self._cache[key] = out
        return out

    def result_message(self, res: dict, n_used: int, cap: int) -> str:
        left = cap - n_used
        if not res["ok"]:
            return ("CHECK RESULT for %s: not a legal change for this design (%s); nothing was simulated. "
                    "Checks left for this step: %d." % (res["text"], res["err"], left))
        head = "CHECK RESULT for %s (simulated; the current design is unchanged)." % res["applied"]
        if res["clamped"]:
            head += " The value was set to the nearest allowed value."
        if res["noop"]:
            head += " This change leaves the design as it is."
        body = ["Lines of the observation that this change would alter (unchanged lines are left out; "
                "'...' marks a gap):"] + res["diff"] + [""] + [
                    res["status"][0].replace("(current design)", "(after this change)")] + res["status"][1:]
        return head + "\n" + "\n".join(body) + "\n\nChecks left for this step: %d." % left

    # -- held-out probe set ------------------------------------------------------------------
    def hset(self) -> List[str]:
        moves = self.r.get("moves") or []
        by = sorted(moves, key=lambda m: -float(m["phi"]))
        el = lambda t: element_of_text(self.v, t)  # noqa: E731
        H = [self.r["gold_text"]]
        used = {el(H[0])}
        nxt = next((m["text"] for m in by if el(m["text"]) not in used), None)
        if nxt:
            H.append(nxt)
            used.add(el(nxt))
        third = None
        w = (self.r.get("pilot") or {}).get("weakest") if self.v == "frame3d2" else None
        if w and w != "unparsed" and w not in used:
            third = next((m["text"] for m in by if el(m["text"]) == w), None)
        if third is None:
            third = next((m["text"] for m in by if el(m["text"]) not in used), None)
        if third:
            H.append(third)
            used.add(el(third))
        pool = [m["text"] for m in moves if m["text"] not in H]
        if pool:
            rng = np.random.default_rng(stable_seed(self.sid, "stream_h4"))
            H.append(pool[int(rng.integers(len(pool)))])
        return H

    def lowest_members(self) -> List[int]:
        if not self.nom0 or self.nom0.get("mech"):
            return []
        fy, fb = np.array(self.nom0["fos_y"]), np.array(self.nom0["fos_b"])
        gov = np.where(np.isfinite(fy), np.minimum(fy, np.where(np.isfinite(fb), fb, np.inf)), np.inf)
        idx = [int(i) for i in np.argsort(gov) if np.isfinite(gov[i])]
        return idx[:N_LOWEST]

    def queries(self, h_text: str) -> dict:
        q = {"metrics": sorted(self.m0)}
        if self.v == "frame3d2":
            mem = list(self.lowest_members())
            e = element_of_text(self.v, h_text)
            if e and e.startswith("m") and int(e[1:]) not in mem:
                mem.append(int(e[1:]))
            q["elements"] = {"m%d" % i: list(F3_QTY) for i in mem}
        return q

    def truth(self, res: dict, q: dict) -> dict:
        """Simulated values of the queried quantities for a checked move: {key: (y, current)},
        key 'metric' or 'm15.N_kN'; members absent or fos_b in tension give None."""
        out = {}
        for k in q["metrics"]:
            out[k] = (res["metrics"].get(k), self.m0.get(k))
        for el, qs in (q.get("elements") or {}).items():
            i = int(el[1:])
            for qq in qs:
                y = cur = None
                if res.get("nom") and not res["nom"].get("mech") and i < len(res["nom"]["N"]):
                    if qq == "N_kN":
                        y, cur = res["nom"]["N"][i] / 1000.0, self.nom0["N"][i] / 1000.0
                    else:
                        y, cur = res["nom"][qq][i], self.nom0[qq][i]
                if y is not None and not np.isfinite(y):
                    y = None
                if cur is not None and not np.isfinite(cur):
                    cur = None
                out["%s.%s" % (el, qq)] = (y, cur)
        return out

    def scale(self, key: str, y: float, cur: Optional[float]) -> float:
        if "." in key:
            q = key.split(".", 1)[1]
            if q.startswith("fos"):
                return max(abs(y), 1.5)
            return max(abs(y), abs(cur or 0.0), 1e-9)
        lim = self.lim.get(key)
        if lim and lim[1] != 0:
            return max(abs(y), abs(lim[1]))
        return max(abs(y), abs(cur or 0.0), 1e-9)

    def same_move(self, a: str, b: str) -> bool:
        ma, mb = self.d.parse_move(a or ""), self.d.parse_move(b or "")
        if ma is None or mb is None:
            return (a or "").replace(" ", "") == (b or "").replace(" ", "")
        return Move.matches(ma, mb, rel=1e-2)
