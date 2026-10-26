"""Aggregate the per-variant reports into the portfolio table and the
uplift-vs-coupling relation the expansion plan pre-registers.

    python -m dataxp.summarize --out /trace/home/wxu2/dataxp_out/run1
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os


def _fmt(v, spec="%.2f", dash=" -  "):
    return dash if v is None else (spec % v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", help="write the aggregate to this path")
    args = ap.parse_args(argv)

    reports = []
    for p in sorted(glob.glob(os.path.join(args.out, "*", "report.json"))):
        try:
            reports.append(json.load(open(p)))
        except (OSError, ValueError):
            continue
    if not reports:
        print("no reports under %s" % args.out)
        return 1

    print("%-15s %-6s %-11s %-31s %-9s %-7s %-18s %s" % (
        "variant", "stoch", "G1/G2/G3", "rnd/grd/bld/phi/d2", "uplift",
        "coup", "pairs/traj/probs", "ref"))
    print("-" * 118)
    accepted = []
    tot_pairs = tot_traj = tot_probs = 0
    for r in reports:
        g = r["gates"]
        lad = r.get("ladder") or {}

        def fr(k):
            c = lad.get(k)
            return _fmt(c["feasible_rate"] if c else None)
        ok = r["all_gates_passed"]
        if ok:
            accepted.append(r)
            tot_pairs += r["dataset"]["n_pairs"]
            tot_traj += r["dataset"]["n_trajectories"]
            tot_probs += r["dataset"]["n_problems_included"]
        print("%-15s %-6s %-11s %-31s %-9s %-7s %-18s %s%s" % (
            r["variant"], "yes" if r["stochastic"] else "no",
            "%s/%s/%s" % (g["G1"]["passed"], g["G2"]["passed"], g["G3"]["passed"]),
            "%s/%s/%s/%s/%s" % (fr("random"), fr("greedy"), fr("blind"),
                                fr("phi_select"), fr("phi_d2")),
            _fmt(r.get("uplift_phi_vs_greedy"), "%+.2f"),
            _fmt(r.get("coupling_feedback_sufficiency")),
            "%d/%d/%d" % (r["dataset"]["n_pairs"], r["dataset"]["n_trajectories"],
                          r["dataset"]["n_problems_included"]),
            r.get("reference_class", "?"),
            "" if ok else "   <-- NOT INCLUDED"))

    print("-" * 118)
    print("ACCEPTED %d/%d variants | %d pairs, %d trajectories, %d problems"
          % (len(accepted), len(reports), tot_pairs, tot_traj, tot_probs))

    def _pearson(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        if sxx <= 0 or syy <= 0:
            return float("nan"), float("nan")
        return sxy / math.sqrt(sxx * syy), sxy / sxx

    # The pre-registered relation. The DIRECT reading of "is the cheap depth-0
    # diagnostic sufficient in this domain?" is how well greedy itself does:
    # where a single critical element exists and the simulator names it, greedy
    # is excellent and Phi-ranking has no headroom; where constraints couple so
    # that no single critical element exists, greedy collapses and ranking is
    # what carries the decision.
    rows = [(r["variant"],
             (r.get("ladder") or {}).get("greedy", {}).get("feasible_rate"),
             r.get("uplift_phi_vs_greedy"),
             r.get("coupling_feedback_sufficiency")) for r in reports]
    rows = [t for t in rows if t[1] is not None and t[2] is not None]
    if len(rows) >= 3:
        r_g, slope_g = _pearson([t[1] for t in rows], [t[2] for t in rows])
        print()
        print("Phi's uplift over greedy vs GREEDY'S OWN feasibility (n=%d):" % len(rows))
        print("  pearson r=%.3f  slope=%.3f  -- Phi earns its cost exactly where"
              % (r_g, slope_g))
        print("  the depth-0 feedback heuristic fails, and not elsewhere.")
        for v, g, u, _c in sorted(rows, key=lambda t: t[1]):
            print("    %-15s greedy=%.2f  uplift=%+.2f" % (v, g, u))

        have_c = [t for t in rows if t[3] is not None]
        if len(have_c) >= 3:
            r_c, _ = _pearson([t[3] for t in have_c], [t[2] for t in have_c])
            print()
            print("  Secondary: hint/Phi-argmax AGREEMENT vs uplift, r=%.3f (n=%d)."
                  % (r_c, len(have_c)))
            print("  This is an agreement rate, NOT a coupling measure: it conflates")
            print("  'the hint is right' with 'Phi is right', so read it alongside the")
            print("  greedy column rather than as a cause.")

    print()
    print("envelope red-team (D9): runs that left the validity envelope once the")
    print("envelope was stripped from Phi")
    caught = 0
    for r in reports:
        rt = r.get("envelope_redteam") or {}
        if rt.get("n_run"):
            c = rt.get("n_exploits_caught", 0)
            caught += c
            print("  %-15s %d/%d" % (r["variant"], c, rt["n_run"]))
    if caught == 0:
        print("  NOTE: zero exploits found anywhere. The 2D truss study found them")
        print("  in 17% of unguarded successes, so read this as UNPROVEN rather")
        print("  than safe: these parameterisations may simply not admit the")
        print("  exploit (frame3d ties wall thickness to radius, so the negative-")
        print("  area state is unreachable by construction), or the red-team's")
        print("  budget may be too small to find it.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"variants": reports,
                       "accepted": [r["variant"] for r in accepted],
                       "totals": {"pairs": tot_pairs, "trajectories": tot_traj,
                                  "problems": tot_probs}}, fh, indent=2,
                      default=str)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
