#!/usr/bin/env python
"""Independent validation of generated truss problems.

The generator agent was stopped before reporting its own validation, so nothing
about these files is taken on trust. Checks, in order:

  1. loads and FEA returns finite mass / FOS / deflection
  2. the INITIAL design is INFEASIBLE (else there is nothing to solve)
  3. NOT a degenerate mechanism -- the existing set contains ~11 problems that
     report ~1e13-1e14 m deflection with huge FOS and are structurally
     meaningless; those must not be reproduced
  4. metadata sanity: optimal_mass present and positive, maximum_mass = 1.1x it

Solvability is measured separately by search_ladder.py on the same directory, so
the generated and original sets go through identical policies.

CPU only.
"""
from __future__ import annotations
import argparse, json, sys, statistics as st
from pathlib import Path
from collections import Counter

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals  # noqa: E402
from llm_finetune.training.rl.posterior.potential import program_from_truss_spec  # noqa: E402

# A real truss under these loads deflects on the order of millimetres to
# centimetres. Anything past this is a mechanism, not a structure.
DEFLECTION_SANE_M = 1.0

# Degeneracy MUST be read from the RAW simulator, not from _analyze_truss.
# _sanitize_fea_state clamps an infinite deflection to exactly 1.0, which slips
# past a `> 1.0` threshold -- that bug let all 11 known-degenerate originals pass
# an earlier version of this file. Verified against the 11 problems the split
# file marks degenerate: raw deflection is inf and raw fos_buckling is 0.
def _raw_state(spec, goals):
    from validation.truss_executor import analyze_truss, load_truss_from_problem
    return analyze_truss(load_truss_from_problem(spec), goals)


def _is_degenerate(raw):
    import math
    d = raw.get("deflection")
    if d is None or not math.isfinite(d) or abs(d) > DEFLECTION_SANE_M:
        return True
    fb, fy = raw.get("fos_buckling"), raw.get("fos_yielding")
    for v in (fb, fy):
        if v is None or not math.isfinite(v):
            return True
    # A mechanism carries no load: zero buckling capacity with finite mass.
    return fb == 0 and fy == 0


def check(path):
    try:
        spec = json.load(open(path))
    except Exception as e:
        return "unparseable_json", None
    if not isinstance(spec, dict) or "topology" not in spec:
        return "not_a_problem", None
    try:
        truss, goals = _load_truss_and_goals(spec)
        state = _analyze_truss(truss, goals)
    except Exception:
        return "fea_raised", None
    vals = [state.get(k) for k in ("mass", "fos_buckling", "fos_yielding", "deflection")]
    if any(v is None or v != v or abs(v) == float("inf") for v in vals):
        return "nonfinite_fea", state
    try:
        raw = _raw_state(spec, goals)
    except Exception:
        return "raw_fea_raised", state
    if _is_degenerate(raw):
        return "degenerate_mechanism", state
    program = program_from_truss_spec(spec, initial_mass=state.get("mass"))
    if program.is_feasible(state):
        return "already_feasible", state
    md = spec.get("_metadata") or {}
    om = md.get("optimal_mass")
    mm = (spec.get("goals") or {}).get("maximum_mass")
    if not om or om <= 0:
        return "bad_optimal_mass", state
    if not mm or abs(mm / om - 1.1) > 1e-6:
        return "bad_mass_ratio", state
    return "PASS", state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pattern", default="*.json")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    files = sorted(str(p) for p in Path(a.dir).glob(a.pattern))
    print(f"validating {len(files)} files in {a.dir}", flush=True)

    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        out = pool.map(check, files, chunksize=4)

    reasons = Counter(r for r, _ in out)
    npass = reasons.get("PASS", 0)
    print(f"\n  PASS {npass}/{len(files)} = {npass/max(len(files),1):.3f}")
    for r, c in reasons.most_common():
        if r != "PASS":
            print(f"  reject {r:24s} {c}")
    ok = [s for r, s in out if r == "PASS" and s]
    if ok:
        for k in ("mass", "fos_buckling", "fos_yielding", "deflection"):
            v = [s[k] for s in ok if isinstance(s.get(k), (int, float))]
            if v:
                print(f"  {k:14s} median={st.median(v):12.4f}  min={min(v):10.4f}  max={max(v):12.4f}")


if __name__ == "__main__":
    main()
