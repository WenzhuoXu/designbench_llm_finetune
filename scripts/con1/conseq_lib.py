"""con1 (consequence round): the simulator's result message for a move at a state, exactly as the harness shows it.

Everything prompt- or result-facing goes through the benchmark harness's own functions (dataxp_policy_eval_v3rc.py,
built by scripts/rl/harness_v3r/make_v3rc.py, sha1 9c82418581a0): observation_native + open_rules_prompt +
reject_note for the prompt the policy is served at a state, open_apply / check_text for what a check of a move
returns. Nothing here re-implements a renderer:

  Episode replay   Replay.from_episode(...) walks a recorded episode turn by turn with the harness's own state
                   bookkeeping (_run_episode: labelled history, prev_metrics, per-turn scoring seed, refused-move
                   note), so each turn's prompt can be compared byte for byte with the one the harness recorded.
  Result message   result_of(state, move) = the v3rc check of that move at that state: CHECK RESULT header,
                   CONSTRAINT STATUS (after this move), EFFECT OF THIS MOVE (before -> after), the fabrication line
                   (photonics), 'Feasible after this move: yes|no.'; or the refusal line. Native variants go through
                   check_text itself. A verb-surface variant (SET_PARAM underneath, verb spelling) has no checks in
                   the harness (--checks is native-only); its message is check_text's construction on the harness's
                   verb-surface apply path (check_move, domain.set clamping, history entry (element_label, applied)),
                   i.e. the message v3rc would show if it allowed checks there.
  Verdict          refused | better | worse: better when Phi after the move is above Phi of the state by more than
                   1e-9 (forecasts.truth_of, the scr2 gate's rule); photonics under the open rules ranks by
                   (fewer rule-breaking pixels, then Phi) (open_rules.open_gain), as every round since 11-22 does.

All variants are deterministic (no domain sets stochastic = True), so the scoring seeds only matter for exact
reproduction of the harness's records, and they are reproduced anyway (stable_seed(pid, 'eval', 's', t + 1)).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V3RC_SHA1 = "9c82418581a045d6655648315bfcddd2112765c8"

QUESTION = ("\n\nQUESTION (no move is made now): if the move {move} were played at the current design, what would the "
            "simulator return? Answer with the simulator's result message exactly as a check of that move returns it: "
            "either the line 'CHECK RESULT for <move>: refused by the simulator: <reason>. The current design is "
            "unchanged.', or the line 'CHECK RESULT for <move> (simulated on a copy; the current design is unchanged)' "
            "followed by the block CONSTRAINT STATUS (after this move), the block EFFECT OF THIS MOVE (before -> after) "
            "with one 'metric  before -> after' line per metric, and the line 'Feasible after this move: yes.' or "
            "'Feasible after this move: no.'")
FOLLOW_UP = ("QUESTION (no move is made now): if the move {move} were played at the current design, what would the "
             "simulator return? Answer with the simulator's result message exactly as a check of that move returns it.")


def load_harness(path: str):
    """Import the v3rc harness file as a module (an assemble_dataxp_sft.py must sit next to it)."""
    import hashlib
    raw = open(path, "rb").read()
    h = hashlib.sha1(raw).hexdigest()
    if h != V3RC_SHA1:
        raise SystemExit("%s: sha1 %s, expected the benchmark's v3rc %s" % (path, h, V3RC_SHA1))
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location("dataxp_policy_eval_v3rc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Env:
    """The harness module and the dataxp modules it imports, loaded once per process."""

    def __init__(self, code: str, harness: str):
        self.H = load_harness(harness)
        self.D, self.L, self.R, self.get_domain = self.H.import_dataxp(code)
        self.C = self.H.import_common()
        self._domains = {}

    def domain(self, variant: str):
        if variant not in self._domains:
            self._domains[variant] = self.get_domain(variant)
        return self._domains[variant]


class State:
    """One turn's state of an episode under the harness's bookkeeping (what _run_episode holds at the top of a turn).

    Fields: variant, spec, pid, t (turn index), st (live design), cur (PhiScore of st), metrics, binding, history
    (labelled, as the harness keeps it), prev_metrics, rejected (moves refused at this design: (text, why)),
    applied (serialised applied moves so far), native, spoken."""

    def __init__(self, env: Env, variant: str, spec: dict, opening: bool = True):
        H, D = env.H, env.D
        self.env, self.variant, self.spec, self.pid = env, variant, spec, spec["problem_id"]
        self.domain = d = env.domain(variant)
        self.native = H.is_native(d)
        self.spoken = H.speaks_verbs(D, d)
        if not self.spoken:
            raise ValueError("%s is not verb-spelled: con1 covers the 11 benchmark families only" % variant)
        self.opening = opening
        self.ev = D.PhiEvaluator(d, spec)
        self.st = d.load(spec)
        self.cur = self.ev.score(self.st, seed0=D.stable_seed(self.pid, "eval", "s", 0))
        self.t = 0
        self.history, self.applied, self.rejected = [], [], []
        self.prev_metrics = None
        self.native_prompt = "served"

    # -- what the policy is shown -------------------------------------------------------------------------------
    @property
    def metrics(self) -> dict:
        return dict(self.cur.agg_metrics)

    @property
    def binding(self):
        return self.env.L._binding_constraint(self.ev.program, self.metrics)

    def prompt(self) -> tuple:
        """(system, user) as _run_episode builds them for the llm policy at this turn (open rules, refusal note)."""
        H = self.env.H
        system, user = H.observation_native(self.env.C, self.domain, self.spec, self.st, self.metrics, self.binding,
                                            self.history, self.prev_metrics, self.variant, self.native_prompt)
        if self.opening:
            system, user = H.open_rules_prompt(self.variant, system, user, self.metrics)
        if self.rejected:                     # noting is on under --open-rules for the llm policy
            user = user + H.reject_note(self.rejected)
        return system, user

    # -- moves ----------------------------------------------------------------------------------------------------
    def parse(self, text: str):
        """The move as the harness reads it from '<action>text</action>' (None when unreadable)."""
        mv, _ = self.env.C.parse_reply(self.domain, "<action>%s</action>" % text)
        return mv

    def fmt(self, mv) -> str:
        if not self.native:
            d = mv.as_dict() if hasattr(mv, "as_dict") else mv
            mv = self.env.D.Move("SET_PARAM", {"i": int(d["i"]), "p": d["p"], "v": float(d["v"])})
        return self.domain.format_move(mv)

    def _verb_surface_apply(self, mv):
        """(child, applied dict, None) or (None, None, why): the harness's verb-surface apply path."""
        H, d = self.env.H, self.domain
        m = mv.as_dict() if hasattr(mv, "as_dict") else dict(mv)
        m = {"i": int(m["i"]), "p": m["p"], "v": float(m["v"])}
        ok, why, v_old = H.check_move(d, self.st, m)
        if not ok:
            return None, None, why
        child = d.clone(self.st)
        d.set(child, m["i"], m["p"], m["v"])
        v_new = float(d.get(child, m["i"], m["p"]))
        return child, {"i": m["i"], "p": m["p"], "v": v_new, "v_old": v_old}, None

    def result_of(self, mv) -> dict:
        """What a v3rc check of `mv` at this state returns: {move, refused | (phi, feasible, viol, metrics_after),
        verdict, result} (result = the message text, word for word)."""
        H, D, d = self.env.H, self.env.D, self.domain
        out = {"move": self.fmt(mv)}
        seed = D.stable_seed(self.pid, "eval", "s", self.t + 1)
        if self.native:
            child = d.clone(self.st)
            try:
                H.open_apply(d, self.variant, child, mv, self.opening)
            except Exception as e:                       # noqa: BLE001
                out.update(refused=str(e)[:300], verdict="refused",
                           result="CHECK RESULT for %s: refused by the simulator: %s. The current design is unchanged."
                                  % (d.format_move(mv), str(e)[:300]))
                return out
            sc = self.ev.score(child, seed0=seed)
            text = H.check_text(self.env.C, self.env.R, self.env.L, d, self.spec, child, sc, mv, self.history,
                                self.metrics, self.variant, self.native_prompt, self.ev)
        else:
            child, applied, why = self._verb_surface_apply(mv)
            if child is None:
                out.update(refused=str(why), verdict="refused",
                           result="CHECK RESULT for %s: refused by the simulator: %s. The current design is unchanged."
                                  % (out["move"], why))
                return out
            sc = self.ev.score(child, seed0=seed)
            text = self._check_text_vs(child, sc, applied, out["move"])
        viol = float(sc.agg_metrics.get("min_feature_violations") or 0.0) if self.variant == "photonics" else 0.0
        out.update(phi=float(sc.mean), feasible=bool(sc.feasible), viol=viol,
                   metrics_after={k: float(v) for k, v in sc.agg_metrics.items()},
                   verdict=self.verdict(float(sc.mean), viol), result=text)
        return out

    def _check_text_vs(self, child, sc, applied, mvtext: str) -> str:
        """check_text's construction for a verb-surface move: the observation after the move (history entry as the
        harness appends it on that path), cut to CONSTRAINT STATUS and the move's EFFECT, plus the feasible line."""
        H, d = self.env.H, self.domain
        cm = {k: float(v) for k, v in sc.agg_metrics.items()}
        h2 = list(self.history) + [(d.element_label(child, applied["i"]), applied)]
        _, u2 = H.observation_native(self.env.C, d, self.spec, child, cm,
                                     self.env.L._binding_constraint(self.ev.program, cm), h2, self.metrics,
                                     self.variant, self.native_prompt)
        i, j = u2.find("CONSTRAINT STATUS"), u2.find("CHANGES ALREADY MADE")
        k, e = u2.find("OBSERVED EFFECT OF THE LAST CHANGE"), u2.find("YOUR MOVE")
        if e < 0:
            e = len(u2)
        if not (0 <= i < j <= k < e):
            raise RuntimeError("check_text_vs: observation sections not found in order (%d %d %d %d)" % (i, j, k, e))
        status = u2[i:j].strip().replace("CONSTRAINT STATUS (current design)", "CONSTRAINT STATUS (after this move)")
        effect = u2[k:e].strip().replace("OBSERVED EFFECT OF THE LAST CHANGE", "EFFECT OF THIS MOVE")
        lines = ["CHECK RESULT for %s (simulated on a copy; the current design is unchanged)" % mvtext, status, effect,
                 "Feasible after this move: %s." % ("yes" if sc.feasible else "no")]
        return "\n\n".join(lines)

    def bad_now(self) -> int:
        return int(round(float(self.metrics.get("min_feature_violations") or 0.0))) if self.variant == "photonics" else 0

    def verdict(self, phi: float, viol: float) -> str:
        phi0 = float(self.cur.mean)
        if self.opening and self.variant == "photonics":
            bad, bad0 = int(round(viol)), self.bad_now()
            return "better" if (bad < bad0 or (bad == bad0 and phi > phi0 + 1e-9)) else "worse"
        return "better" if phi > phi0 + 1e-9 else "worse"

    def phi_of_metrics(self, metrics: dict) -> float:
        """Phi of a (predicted) metrics dict under this problem's program (deterministic domains: one replicate)."""
        return float(self.ev._phi_one(metrics))

    # -- moving on --------------------------------------------------------------------------------------------------
    def commit(self, mv, answer_text: str | None = None):
        """Apply `mv` as the harness does at the end of a turn (refused: the turn is spent and the note grows; applied:
        history, prev_metrics, rescoring). Returns the fail reason or None. answer_text: the model's answer, from which
        the refusal note takes the move as written (action_text)."""
        H, D, d = self.env.H, self.env.D, self.domain
        why = None
        reply = {"answer": answer_text if answer_text is not None else "<action>%s</action>" % self.fmt(mv)}
        if self.native:
            ad = mv.as_dict()
            child = d.clone(self.st)
            try:
                H.open_apply(d, self.variant, child, mv, self.opening)
            except Exception as e:                       # noqa: BLE001
                why = "illegal:%s" % str(e)[:80]
                H.note_refusal(self.rejected, reply, mv, str(e)[:300])
            else:
                del self.rejected[:]
                self.history.append((self.env.R._label_for(d, child, ad), ad))
                self.prev_metrics = self.metrics
                self.st = child
                self.cur = self.ev.score(self.st, seed0=D.stable_seed(self.pid, "eval", "s", self.t + 1))
                self.applied.append(ad)
        else:
            m = mv.as_dict() if hasattr(mv, "as_dict") else dict(mv)
            m = {"i": int(m["i"]), "p": m["p"], "v": float(m["v"])}
            ok, why, v_old = H.check_move(d, self.st, m)
            if not ok:
                H.note_refusal(self.rejected, reply, m, why)
            else:
                del self.rejected[:]
                child = d.clone(self.st)
                d.set(child, m["i"], m["p"], m["v"])
                v_new = float(d.get(child, m["i"], m["p"]))
                applied = {"i": m["i"], "p": m["p"], "v": v_new, "v_old": v_old}
                self.history.append((d.element_label(child, m["i"]), applied))
                self.prev_metrics = self.metrics
                self.st = child
                self.cur = self.ev.score(self.st, seed0=D.stable_seed(self.pid, "eval", "s", self.t + 1))
                self.applied.append(applied)
        self.t += 1
        return why

    def skip_turn(self):
        """A turn with no parsable move: the harness spends it (no note, no change)."""
        self.t += 1

    def legal(self, k: int | None = None, max_candidates=None) -> list:
        """The domain's enumerated legal set at this design (the scoring tools' rng: legal_rng(pid, k))."""
        rng = self.env.C.legal_rng(self.pid, len(self.applied) if k is None else k)
        return list(self.domain.legal_moves(self.st, rng, max_candidates=max_candidates)
                    if max_candidates else self.domain.legal_moves(self.st, rng))

    def snapshot(self) -> dict:
        """A serialisable description that rebuilds this state (replay_actions)."""
        return {"variant": self.variant, "problem_id": self.pid, "t": self.t, "applied": list(self.applied),
                "turn_log": list(getattr(self, "turn_log", []))}


def replay_turns(env: Env, variant: str, spec: dict, turns: list, opening: bool = True, upto: int | None = None):
    """Walk a recorded episode's turns (each {t, action/move, fail, answer}) with the harness's bookkeeping, yielding
    (turn record, State at the top of that turn) before applying the turn's committed move."""
    s = State(env, variant, spec, opening)
    s.turn_log = []
    for rec in turns:
        if upto is not None and rec["t"] >= upto:
            break
        if int(rec["t"]) != s.t:
            raise RuntimeError("%s: turn index %s, replay at %d" % (spec["problem_id"], rec["t"], s.t))
        yield rec, s
        mv = rec.get("action") or rec.get("move")
        fail = rec.get("fail")
        if mv is None:
            s.turn_log.append(("skip", None, None))
            s.skip_turn()
            continue
        if s.native:
            mvo = env.D.Move.from_dict(mv)
        else:
            mvo = {"i": int(mv["i"]), "p": mv["p"], "v": float(mv["v"])}
        s.turn_log.append(("move", mv, rec.get("answer")))
        why = s.commit(mvo, rec.get("answer"))
        if (why is None) != (fail is None):
            raise RuntimeError("%s t%s: replay %s, recorded fail %r" % (spec["problem_id"], rec["t"], why, fail))


def rebuild(env: Env, snap: dict, spec: dict, opening: bool = True) -> State:
    """The State a snapshot describes (turn_log replayed: refusals and skipped turns included)."""
    s = State(env, snap["variant"], spec, opening)
    s.turn_log = []
    for kind, mv, answer in snap.get("turn_log") or []:
        if kind == "skip":
            s.turn_log.append((kind, None, None))
            s.skip_turn()
            continue
        mvo = env.D.Move.from_dict(mv) if s.native else {"i": int(mv["i"]), "p": mv["p"], "v": float(mv["v"])}
        s.turn_log.append((kind, mv, answer))
        s.commit(mvo, answer)
    if s.t != int(snap["t"]):
        raise RuntimeError("rebuild: t %d != %s" % (s.t, snap["t"]))
    return s


def jdump(o) -> str:
    def default(x):
        try:
            import numpy as np
            if isinstance(x, np.integer):
                return int(x)
            if isinstance(x, np.floating):
                return float(x)
            if isinstance(x, np.ndarray):
                return x.tolist()
        except Exception:                                 # noqa: BLE001
            pass
        return str(x)
    return json.dumps(o, default=default, ensure_ascii=False)


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:                                     # noqa: BLE001
        return False
