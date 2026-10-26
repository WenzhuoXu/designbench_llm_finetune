"""Generation job for one domain-variant: generate problems, run the ladder,
apply the three gates, harvest pairs + trajectories, write the accepted
dataset and a full report.

    python -m dataxp.run_generation --variant seir --n 130 \
        --tmp /trace/scratch/biosimmlab/wxu2/dataxp_tmp \
        --out /trace/home/wxu2/dataxp_out [--quick] [--workers 8]

All intermediate files live under --tmp/<variant>/ (scratch); only the final
accepted dataset + report are copied to --out/<variant>/.

Appending problems to an existing corpus (--start-index K, default 0):

    python -m dataxp.run_generation --variant grid2op --start-index 130 --n 130 ...

generates the problems of indices K..K+n-1, each exactly as a run with
--n K+n would generate it (rng = default_rng(stable_seed(variant, seed, i)),
problem_id "<variant>_<i:04d>"), so a later batch continues the same index
stream instead of repeating it. Some domains keep a per-process registry of
the problems already built and re-draw a start that repeats one of them
(grid2op and photonics: a module-level _BUILT, see their make_problem
comments); for those, problem i depends on problems 0..i-1 having been built
first in the same process. With K > 0 the run therefore first builds indices
0..K-1 (make_problem only: not evaluated, not written) for every domain in
PREBUILD_VARIANTS or whose module carries such a registry, so the new problems
are the ones a single run of K+n would have produced and cannot duplicate the
earlier ones (grid2op, checked 2026-10-24: without the pre-build, problem 0132
would repeat an earlier problem). engibench and molopt are in PREBUILD_VARIANTS
although their make_problem reads no registry: engibench's photonics2d
problems still depend on the process history at round-off level (cached
ceviche objects; without the pre-build 2 of the 6 specs at 130-135 differ in
the last printed digits), so the pre-build makes the new problems bit-equal to
a single run's; molopt is pre-built for the same reason (held out, not
measured). It costs one make_problem per earlier index (~100 s for 130 on
grid2op, engibench or photonics).
The per-problem caps (phi_d2 on the first 40, red-team on the first 12; 3 and
2 with --quick) count positions in THIS run, not problem indices. K = 0 runs
exactly the code path of before (byte-identical files; report.json gains the
keys start_index / prebuilt only when K > 0).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from typing import Optional

import numpy as np

from .domain import PhiEvaluator, stable_seed
from .domains import get_domain, list_variants
from .gates import data_quality_gate, distribution_variance_gate, phi_measurable_gate
from .harvest import harvest_pairs, harvest_trajectory, write_jsonl
from .ladder import redteam_envelope, run_episode

LADDER_POLICIES = ("random", "greedy", "blind", "phi_select")

# Pre-built (registry only) before the first problem when --start-index > 0.
PREBUILD_VARIANTS = ("grid2op", "photonics", "engibench", "molopt")

_WORKER = {}


def _has_problem_registry(domain) -> bool:
    """True when the domain's module keeps a per-process registry of the
    problems built so far (the module-level _BUILT of grid2op / photonics)."""
    mod = sys.modules.get(type(domain).__module__)
    return mod is not None and hasattr(mod, "_BUILT")


def _make_spec(domain, variant: str, seed: int, i: int) -> dict:
    """Problem i of the variant's index stream: the same rng and id for any
    run that reaches index i."""
    rng = np.random.default_rng(stable_seed(variant, seed, i))
    spec = domain.make_problem(i, rng)
    spec["problem_id"] = "%s_%04d" % (variant, i)
    spec["domain_variant"] = variant
    return spec


def build_specs(domain, variant: str, seed: int, start: int, n: int):
    """(specs of indices start..start+n-1, prebuilt record or None). With
    start > 0, problems 0..start-1 are built first (and discarded) for the
    domains that need it (module docstring)."""
    prebuilt = None
    if start > 0:
        registry = _has_problem_registry(domain)
        if registry or variant in PREBUILD_VARIANTS:
            tp = time.time()
            for i in range(start):
                _make_spec(domain, variant, seed, i)   # registry only
            prebuilt = {"indices": [0, start - 1], "module_registry": registry,
                        "seconds": round(time.time() - tp, 1)}
            print("[%s] pre-built problems 0..%d (registry only, %s, %.1f s)"
                  % (variant, start - 1,
                     "module registry" if registry else "no module registry: no-op today",
                     time.time() - tp), flush=True)
        else:
            prebuilt = {"indices": None, "module_registry": False, "seconds": 0.0}
        print("[%s] problem indices %d..%d" % (variant, start, start + n - 1), flush=True)
    specs = [_make_spec(domain, variant, seed, i) for i in range(start, start + n)]
    return specs, prebuilt


def _init_worker(variant: str, turns: int, d2_cap: int, redteam_cap: int,
                 offset_rule: str = "infeasible"):
    _WORKER["domain"] = get_domain(variant)
    _WORKER["offset_rule"] = offset_rule
    _WORKER["turns"] = turns
    _WORKER["d2_cap"] = d2_cap
    _WORKER["redteam_cap"] = redteam_cap


def _run_problem(payload):
    index, spec = payload
    domain = _WORKER["domain"]
    turns = _WORKER["turns"]
    out = {"index": index, "problem_id": spec.get("problem_id"), "error": None}
    try:
        ev = PhiEvaluator(domain, spec, offset_rule=_WORKER["offset_rule"])
        agg0 = ev.initial_agg
        # RAW relative violation: ev.program has already been scaled by
        # with_violation_scale (dividing by this problem's own initial
        # violation), so the normalised value is ~1.0 for every problem by
        # construction and carries no difficulty signal for G1.
        out["initial_violation"] = ev.program.total_violation(
            agg0, tau=0.0, normalised=False)
        out["initially_infeasible"] = not ev.program.is_feasible(agg0)
        obj_key = ev.program.objective_key
        out["initial_objective"] = (float(agg0.get(obj_key))
                                    if obj_key and agg0.get(obj_key) is not None
                                    else None)
        out["feasibility_offset"] = ev.feasibility_offset
        out["violation_ref"] = ev.program.violation_ref

        episodes = {}
        for policy in LADDER_POLICIES:
            ev_p = PhiEvaluator(domain, spec, offset_rule=_WORKER["offset_rule"])
            res = run_episode(domain, spec, ev_p, policy, turns=turns,
                              record=(policy == "phi_select"))
            episodes[policy] = res
        if index < _WORKER["d2_cap"]:
            ev_p = PhiEvaluator(domain, spec, offset_rule=_WORKER["offset_rule"])
            episodes["phi_d2"] = run_episode(domain, spec, ev_p, "phi_d2",
                                             turns=turns)
        out["ladder"] = {p: r.summary() for p, r in episodes.items()}

        sel = episodes["phi_select"]
        out["measurable_frac"] = sel.measurable_frac
        out["degenerate"] = sel.degenerate
        out["hint_match_frac"] = sel.hint_match_frac

        # time-stepping domains: exact duplicate pairs by (problem, step,
        # chosen, rejected) are dropped and counted (n_dropped_duplicate)
        pairs, pair_stats = harvest_pairs(
            sel, dedup_time_key=(domain.time_key
                                 if getattr(domain, "advances_time", False) else None))
        traj, replay_ok = harvest_trajectory(domain, spec, ev, sel)
        out["pairs"] = pairs
        out["pair_stats"] = pair_stats
        out["trajectory"] = traj
        out["replay_ok"] = replay_ok

        if index < _WORKER["redteam_cap"]:
            out["redteam"] = redteam_envelope(domain, spec, turns=turns)
    except Exception:
        out["error"] = traceback.format_exc()
    return out


def _aggregate_ladder(results, policy):
    rows = [r["ladder"][policy] for r in results
            if r.get("ladder") and policy in r["ladder"]]
    if not rows:
        return None
    feasible = sum(1 for r in rows if r["feasible_ever"]) / len(rows)
    ratios = sorted(r["obj_over_ref"] for r in rows
                    if r.get("obj_over_ref") is not None
                    and math.isfinite(r["obj_over_ref"]))
    return {
        "n": len(rows),
        "feasible_rate": feasible,
        "median_obj_over_ref": ratios[len(ratios) // 2] if ratios else None,
        "mean_sim_calls": sum(r["sim_calls"] for r in rows) / len(rows),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list_variants())
    ap.add_argument("--n", type=int, default=130)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--tmp", required=True, help="scratch dir for temp storage")
    ap.add_argument("--out", required=True, help="final dataset + report dir")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--quick", action="store_true",
                    help="6 problems, 8 turns: end-to-end validation only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset-rule", default="infeasible",
                    choices=("infeasible", "raw", "legacy_scaled"),
                    help="feasibility-offset rule; run1 used legacy_scaled")
    ap.add_argument("--start-index", type=int, default=0,
                    help="first problem index (default 0). K > 0 generates indices K..K+n-1 "
                         "as a run of --n K+n would, pre-building 0..K-1 (registry only) for "
                         "domains whose make_problem depends on the problems built before it")
    args = ap.parse_args(argv)
    if args.start_index < 0:
        ap.error("--start-index must be >= 0")

    n = 6 if args.quick else args.n
    turns = 8 if args.quick else args.turns
    d2_cap = 3 if args.quick else 40
    redteam_cap = 2 if args.quick else 12

    tmp_dir = os.path.join(args.tmp, args.variant)
    out_dir = os.path.join(args.out, args.variant)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    domain = get_domain(args.variant)
    print("[%s] generating %d problems (stochastic=%s)" %
          (args.variant, n, domain.stochastic), flush=True)

    start = args.start_index
    specs, prebuilt = build_specs(domain, args.variant, args.seed, start, n)
    with open(os.path.join(tmp_dir, "problems.jsonl"), "w") as fh:
        write_jsonl(fh, specs)

    payloads = list(enumerate(specs))
    results = []
    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(args.variant, turns, d2_cap, redteam_cap,
                                args.offset_rule)) as pool:
            for k, res in enumerate(pool.imap_unordered(_run_problem, payloads)):
                results.append(res)
                _progress(tmp_dir, k + 1, n, t0)
    else:
        _init_worker(args.variant, turns, d2_cap, redteam_cap, args.offset_rule)
        for k, payload in enumerate(payloads):
            results.append(_run_problem(payload))
            _progress(tmp_dir, k + 1, n, t0)
    results.sort(key=lambda r: r["index"])

    errors = [r for r in results if r.get("error")]
    ok = [r for r in results if not r.get("error")]
    for e in errors:
        print("[%s] ERROR on %s:\n%s" % (args.variant, e["problem_id"], e["error"]),
              flush=True)

    # ── gates ────────────────────────────────────────────────────────────────
    g1 = distribution_variance_gate(
        [specs[r["index"]] for r in ok],
        [r.get("initial_violation") for r in ok],
        [r.get("initial_objective") for r in ok],
        [bool(r.get("initially_infeasible")) for r in ok],
    )
    # Stochastic variants are judged against explicitly different G2/G3 bars,
    # because the quantities mean different things under noise:
    #   * deterministic: an unmeasurable state means Phi is FLAT across the
    #     candidates - a genuinely degenerate state (the truss-mechanism case).
    #   * stochastic: it means the replicate budget could not resolve the top
    #     candidate from a typical one. That state yields no preference pair,
    #     but it corrupts nothing, and the remaining states are still clean
    #     data. The gate's job here is to catch a domain where Phi is GLOBALLY
    #     unusable, not to demand that noise be absent.
    # Likewise the pair-yield floor: the noise-band filter is DESIGNED to
    # reject tied pairs, so a low yield under noise is the filter working. The
    # floor catches mis-calibration (NaNs, envelope failures), not noise.
    g2_state, g2_prob = (0.35, 0.70) if domain.stochastic else (0.60, 0.80)
    pair_yield_floor = 0.15 if domain.stochastic else 0.70
    g2 = phi_measurable_gate(ok, min_state_frac=g2_state,
                             min_problem_frac=g2_prob)
    pair_stats = {"n_emitted": 0, "n_rejected_gap": 0,
                  "n_rejected_invalid": 0, "n_rejected_nan": 0}
    if getattr(domain, "advances_time", False):
        pair_stats["n_dropped_duplicate"] = 0      # time-stepping domains only
    included = set(g2["included_problems"])
    all_pairs, all_trajs = [], []
    n_replay_ok = n_replay_failed = n_traj = 0
    for r in ok:
        if r["problem_id"] not in included:
            continue
        for k in pair_stats:
            pair_stats[k] += r["pair_stats"].get(k, 0)
        all_pairs.extend(r["pairs"])
        if r.get("trajectory") is not None:
            all_trajs.append(r["trajectory"])
            n_traj += 1
            n_replay_ok += 1
        elif not r.get("replay_ok", True):
            # A replay FAILURE is a trajectory whose seeded re-execution did
            # not reproduce. A problem that was already feasible at its initial
            # state took no actions and so has nothing to replay - counting
            # that as a failure failed G3 on frame3d_stoch (3 such problems)
            # while the recorded replay_ok was False for exactly zero of 130.
            n_traj += 1
            n_replay_failed += 1
    g3 = data_quality_gate(pair_stats, {
        "n_traj": n_traj, "n_replay_ok": n_replay_ok,
        "n_replay_failed": n_replay_failed, "n_nan": 0,
    }, min_pair_yield=pair_yield_floor)

    ladder_table = {p: _aggregate_ladder(ok, p)
                    for p in LADDER_POLICIES + ("phi_d2",)}
    coupling = [r["hint_match_frac"] for r in ok
                if r.get("hint_match_frac") is not None]
    redteams = [r["redteam"] for r in ok if r.get("redteam")]

    all_passed = g1["passed"] and g2["passed"] and g3["passed"]
    if "n_dropped_duplicate" in pair_stats:
        n_dup = pair_stats["n_dropped_duplicate"]
        print("[%s] time-stepping domain: %d exact duplicate pair%s dropped (%.1f%% of %d)"
              % (args.variant, n_dup, "" if n_dup == 1 else "s",
                 100.0 * n_dup / max(1, n_dup + pair_stats["n_emitted"]),
                 n_dup + pair_stats["n_emitted"]), flush=True)
    report = {
        "variant": args.variant,
        "offset_rule": args.offset_rule,
        "stochastic": domain.stochastic,
        "n_problems": n,
        "n_errors": len(errors),
        "quick": args.quick,
        "gates": {"G1": g1, "G2": g2, "G3": g3},
        "gate_thresholds": {
            "G2_min_state_frac": g2_state,
            "G2_min_problem_frac": g2_prob,
            "G3_min_pair_yield": pair_yield_floor,
            "stochastic_bars_applied": bool(domain.stochastic),
        },
        "all_gates_passed": all_passed,
        "ladder": ladder_table,
        "uplift_phi_vs_greedy": (
            ladder_table["phi_select"]["feasible_rate"]
            - ladder_table["greedy"]["feasible_rate"]
            if ladder_table.get("phi_select") and ladder_table.get("greedy")
            else None),
        "coupling_feedback_sufficiency": (
            sum(coupling) / len(coupling) if coupling else None),
        "reference_class": domain.reference_class,
        "envelope_redteam": {
            "n_run": len(redteams),
            "n_exploits_caught": sum(1 for r in redteams if r["exploit_caught"]),
            "details": redteams,
        },
        "dataset": {
            "n_problems_included": len(included),
            "n_pairs": len(all_pairs),
            "n_trajectories": len(all_trajs),
        },
        "wall_seconds": time.time() - t0,
    }
    if start > 0:          # absent at the default, so a K = 0 report is unchanged
        report["start_index"] = start
        report["problem_indices"] = [start, start + n - 1]
        report["prebuilt"] = prebuilt

    # everything to scratch first, accepted dataset + report to out
    with open(os.path.join(tmp_dir, "pairs.jsonl"), "w") as fh:
        write_jsonl(fh, all_pairs)
    with open(os.path.join(tmp_dir, "trajectories.jsonl"), "w") as fh:
        write_jsonl(fh, all_trajs)
    with open(os.path.join(tmp_dir, "per_problem.jsonl"), "w") as fh:
        write_jsonl(fh, [{k: v for k, v in r.items()
                          if k not in ("pairs", "trajectory")} for r in results])
    with open(os.path.join(tmp_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    shutil.copy(os.path.join(tmp_dir, "report.json"),
                os.path.join(out_dir, "report.json"))
    if all_passed:
        accepted = os.path.join(out_dir, "accepted")
        os.makedirs(accepted, exist_ok=True)
        with open(os.path.join(accepted, "problems.jsonl"), "w") as fh:
            write_jsonl(fh, [s for s in specs if s["problem_id"] in included])
        shutil.copy(os.path.join(tmp_dir, "pairs.jsonl"),
                    os.path.join(accepted, "pairs.jsonl"))
        shutil.copy(os.path.join(tmp_dir, "trajectories.jsonl"),
                    os.path.join(accepted, "trajectories.jsonl"))
        print("[%s] ALL GATES PASSED: %d problems, %d pairs, %d trajectories -> %s"
              % (args.variant, len(included), len(all_pairs), len(all_trajs),
                 accepted), flush=True)
    else:
        failed = [g for g, r in (("G1", g1), ("G2", g2), ("G3", g3))
                  if not r["passed"]]
        print("[%s] GATES FAILED (%s): data NOT included; see report.json"
              % (args.variant, ",".join(failed)), flush=True)

    print("[%s] done in %.1f s" % (args.variant, time.time() - t0), flush=True)
    return 0 if not errors else (0 if len(errors) < max(1, n // 10) else 1)


def _progress(tmp_dir: str, done: int, total: int, t0: float) -> None:
    print("[progress] %d/%d problems (%.1f s elapsed)"
          % (done, total, time.time() - t0), flush=True)
    try:
        with open(os.path.join(tmp_dir, "progress.json"), "w") as fh:
            json.dump({"done": done, "total": total,
                       "elapsed_s": time.time() - t0}, fh)
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
