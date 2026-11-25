"""con1 benchmark readouts beyond feasible@20 (docs/consequence_round_con1.md §4 rule 4 and §5 I2).

    python bench_stats.py --arm con1 FILE [FILE ...] --arm pref1 FILE [FILE ...] [--boot 2000]

Per arm and family group: refused share of turns, share of applied moves that raise Phi, turns per episode;
I2 recovery: after a surprise (the committed move refused, or applied and not raising Phi), the next turn's move is
applied and raises Phi (episodes that have a next turn); FORECAST lines (scripts/rl_screen/forecasts.py) in the
rationale that name the committed move: share borne out (refused / worse / better against the turn's outcome).
se by bootstrap over problems; differences between the first two arms with paired resampling of problems.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl_screen"))
from forecasts import read as read_forecasts  # noqa: E402

GROUPS = {"all(11)": None, "native(3)": {"engibench", "grid2op", "photonics"},
          "replay(5)": {"circuit2", "frame3d2", "grid2", "seir2", "traffic2"},
          "heldout_family(3)": {"inventory2", "molopt", "thermal2"}}


def norm(s):
    return "".join((s or "").split())


def episode_counts(ep):
    c = collections.Counter()
    phi = ep.get("phi_start")
    outcomes = []
    for t in ep.get("turns") or []:
        c["turns"] += 1
        if t.get("fail"):
            out = "refused"
            c["refused"] += 1
        else:
            up = phi is not None and t["phi_after"] > phi + 1e-9
            out = "better" if up else "worse"
            c["applied"] += 1
            c["phi_up"] += up
            phi = t["phi_after"]
        outcomes.append(out)
        lines, _ = read_forecasts(t.get("reasoning") or "")
        mv = (t.get("answer") or "")
        mv = mv.rsplit("<action>", 1)[-1].split("</action>", 1)[0] if "<action>" in mv else ""
        if lines:
            c["turns_with_forecasts"] += 1
            for text, said in lines:
                if norm(text) == norm(mv):
                    c["forecast_own"] += 1
                    c["forecast_own_right"] += said == out
                    break
    for a, b in zip(outcomes, outcomes[1:]):
        if a != "better":
            c["surprise_next"] += 1
            c["recovered"] += b == "better"
    return c


def load(paths):
    per = collections.defaultdict(collections.Counter)    # (variant, problem) -> counts summed over runs
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            ep = r.get("episode") or r
            if ep.get("crashed") or ep.get("feasible_start"):
                continue
            per[(ep["variant"], ep["problem_id"])].update(episode_counts(ep))
            per[(ep["variant"], ep["problem_id"])]["episodes"] += 1
    return per


RATIOS = [("refused share", "refused", "turns"), ("phi-up of applied", "phi_up", "applied"),
          ("turns per episode", "turns", "episodes"), ("I2 recovery", "recovered", "surprise_next"),
          ("forecast lines: turns with", "turns_with_forecasts", "turns"),
          ("forecast of own move right", "forecast_own_right", "forecast_own")]


def ratio(per, keys, num, den):
    n = sum(per[k][num] for k in keys if k in per)
    d = sum(per[k][den] for k in keys if k in per)
    return n / d if d else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", action="append", required=True, help="NAME FILE [FILE ...]")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    arms = [(x[0], load(x[1:])) for x in a.arm]
    rng = np.random.default_rng(a.seed)
    for gname, fams in GROUPS.items():
        keys = sorted(set.intersection(*[{k for k in per if fams is None or k[0] in fams} for _, per in arms]))
        if not keys:
            continue
        print("== %s (%d problems in every arm)" % (gname, len(keys)))
        boots = [rng.integers(len(keys), size=len(keys)) for _ in range(a.boot)]
        for lab, num, den in RATIOS:
            vals = []
            for name, per in arms:
                v = ratio(per, keys, num, den)
                bs = [ratio(per, [keys[i] for i in b], num, den) for b in boots[:400]]
                bs = [x for x in bs if x == x]
                vals.append("%s %.3f +- %.3f" % (name, v, float(np.std(bs, ddof=1)) if len(bs) > 1 else float("nan")))
            line = "  %-28s %s" % (lab, " | ".join(vals))
            if len(arms) >= 2:
                (n0, p0), (n1, p1) = arms[0], arms[1]
                d = ratio(p0, keys, num, den) - ratio(p1, keys, num, den)
                bs = []
                for b in boots:
                    kk = [keys[i] for i in b]
                    x = ratio(p0, kk, num, den) - ratio(p1, kk, num, den)
                    if x == x:
                        bs.append(x)
                se = float(np.std(bs, ddof=1)) if len(bs) > 1 else float("nan")
                line += " | %s-%s %+.3f +- %.3f" % (n0, n1, d, se)
            print(line)


if __name__ == "__main__":
    main()
