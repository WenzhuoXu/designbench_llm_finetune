"""Per-domain smoke test: contract compliance + a tiny end-to-end episode.

    python -m dataxp.smoke seir seir_stoch
    python -m dataxp.smoke all
"""

from __future__ import annotations

import math
import sys

import numpy as np

from .domain import Move, PhiEvaluator, enumerate_candidates, stable_seed
from .domains import get_domain, list_variants
from .ladder import run_episode


def run_smoke(variant: str) -> bool:
    print("=== smoke: %s ===" % variant, flush=True)
    try:
        domain = get_domain(variant)
        for idx in range(3):
            rng = np.random.default_rng(stable_seed("smoke", variant, idx))
            spec = domain.make_problem(idx, rng)
            spec.setdefault("problem_id", "%s_smoke_%d" % (variant, idx))
            assert isinstance(spec.get("goals"), dict) and spec["goals"], \
                "spec.goals missing/empty"
            assert spec.get("family"), "spec.family missing"
            import json
            json.dumps(spec)  # must be JSON-serialisable

            program = domain.program(spec)
            assert program.n_constraints >= 1, "no constraints parsed from goals"
            assert program.objective_key, "no objective set"
            assert program.validity_bounds, "validity envelope missing (D9)"
            ref = domain.reference(spec)
            assert ref is None or (math.isfinite(float(ref)) and float(ref) > 0), \
                "bad reference"

            st = domain.load(spec)
            if domain.native_moves:
                # a native-verb domain may have no element/param addressing
                try:
                    n_el = int(domain.n_elements(st))
                except NotImplementedError:
                    n_el = -1
            else:
                n_el = domain.n_elements(st)
                assert n_el >= 1, "no elements"
            m1 = domain.evaluate(domain.clone(st), seed=7)
            m2 = domain.evaluate(domain.clone(st), seed=7)
            for k, v in m1.items():
                assert isinstance(v, (int, float)), "metric %s not numeric" % k
                v2 = m2.get(k)
                same = (v == v2) or (isinstance(v, float) and isinstance(v2, float)
                                     and math.isnan(v) and math.isnan(v2))
                assert same, "evaluate not deterministic in (st, seed): %s" % k
            if domain.stochastic:
                m3 = domain.evaluate(domain.clone(st), seed=8)
                assert any(m3.get(k) != m1.get(k) for k in m1), \
                    "stochastic variant does not vary with seed"

            # the move path: legal moves exist, format/parse and as_dict/
            # from_dict round-trip, grammar() is written, applying a move
            # changes the state
            rng2 = np.random.default_rng(1)
            moves = domain.legal_moves(st, rng2)
            assert len(moves) >= 4, "too few legal moves (%d)" % len(moves)
            assert isinstance(moves[0], Move), "legal_moves must return Move objects"
            assert domain.grammar().strip(), "grammar() is empty"
            for m in moves[:8]:
                txt = domain.format_move(m)
                back = domain.parse_move("<action>%s</action>" % txt)
                assert back is not None and back.matches(m, rel=1e-5), \
                    "format/parse round trip failed: %s -> %r" % (txt, back)
                assert Move.from_dict(m.as_dict()).matches(m, rel=0.0), \
                    "as_dict/from_dict round trip failed: %r" % (m,)
            child = domain.clone(st)
            domain.apply(child, moves[0])
            mc = domain.evaluate(domain.clone(child), seed=7)
            assert any(mc.get(k) != m1.get(k) for k in m1), \
                "legal move did not change any metric"
            if not domain.native_moves:
                # Candidate path == Move path, dict for dict (v1/v2 bit-for-bit)
                cands = enumerate_candidates(domain, st, np.random.default_rng(1))
                assert [c.as_dict() for c in cands] == [m.as_dict() for m in moves], \
                    "legal_moves differs from enumerate_candidates"
                c = cands[0]
                child = domain.clone(st)
                domain.set(child, c.i, c.p, c.v)
                assert abs(domain.get(child, c.i, c.p) - c.v) < 1e-9, "set/get mismatch"

            # short episode end-to-end
            ev = PhiEvaluator(domain, spec)
            res = run_episode(domain, spec, ev, "phi_select", turns=4, record=True)
            assert res.sim_calls > 0
            print("  problem %d: %d elements, %d constraints, %d cands, "
                  "phi episode ok (feasible_ever=%s, degenerate=%s)"
                  % (idx, n_el, program.n_constraints, len(moves),
                     res.feasible_ever, res.degenerate), flush=True)
        print("PASS %s" % variant, flush=True)
        return True
    except Exception:
        import traceback
        traceback.print_exc()
        print("FAIL %s" % variant, flush=True)
        return False


def main(argv) -> int:
    names = argv or ["all"]
    if names == ["all"]:
        names = list_variants()
    ok = all([run_smoke(v) for v in names])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
