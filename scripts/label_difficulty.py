#!/usr/bin/env python
"""Attach a MEASURED difficulty label to generated problems.

The label is NOT a function of the generation knobs -- a stratum is only the
generator's intent, and intent and outcome were measured to disagree.  It is a
function of which fixed policies actually solved the problem, under the same
20-step budget the ladder reports:

    easy   greedy_critical solves it                (classical sizing is enough)
    medium greedy fails, lookahead_v2_d1 solves it  (depth-1 search adds the value)
    hard   d1 fails, lookahead_v2_d2 solves it      (needs multi-step lookahead)
    open   no policy in the ladder solves it        (a feasible design provably
           exists -- V4 holds a certificate -- but none of these policies find it)

Writes ``_metadata.difficulty`` and ``_metadata.policy_solved`` in place, and
prints per-stratum and per-difficulty counts.

CPU only.  Usage:
    python scripts/label_difficulty.py --problems <dir> --runs <runs.jsonl>
    python scripts/label_difficulty.py ... --report-only     # no writes
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ORDER = ("greedy_critical", "lookahead_v2_d1", "lookahead_v2_d2")


def classify(solved: dict[str, bool]) -> str:
    if solved.get("greedy_critical"):
        return "easy"
    if solved.get("lookahead_v2_d1"):
        return "medium"
    if solved.get("lookahead_v2_d2"):
        return "hard"
    return "open"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--drop-open", action="store_true",
                    help="move problems no policy solves into <dir>/open/")
    a = ap.parse_args()

    solved: dict[str, dict[str, bool]] = defaultdict(dict)
    for line in open(a.runs):
        r = json.loads(line)
        solved[r["problem_id"]][r["policy"]] = bool(r.get("feasible"))

    files = sorted(Path(a.problems).glob("*.json"))
    by_stratum: dict[str, Counter] = defaultdict(Counter)
    by_diff: Counter = Counter()
    pol_by_stratum: dict[str, Counter] = defaultdict(Counter)
    n_by_stratum: Counter = Counter()
    written = 0
    opened: list[Path] = []

    for p in files:
        try:
            spec = json.load(open(p))
        except Exception:
            continue
        if not isinstance(spec, dict) or "topology" not in spec:
            continue
        pid = spec.get("problem_id", p.stem)
        s = solved.get(pid)
        if s is None:
            continue
        label = classify(s)
        stratum = (spec.get("_metadata") or {}).get("stratum", "?")
        by_stratum[stratum][label] += 1
        n_by_stratum[stratum] += 1
        by_diff[label] += 1
        for pol in ORDER:
            if s.get(pol):
                pol_by_stratum[stratum][pol] += 1
        if label == "open":
            opened.append(p)
        if not a.report_only:
            spec.setdefault("_metadata", {})["difficulty"] = label
            spec["_metadata"]["policy_solved"] = {k: bool(s.get(k, False)) for k in ORDER}
            with open(p, "w") as fh:
                json.dump(spec, fh, indent=2)
            written += 1

    total = sum(by_diff.values())
    print(f"\nlabelled {total} problems (wrote {written})")
    print("difficulty counts:", dict(by_diff))
    print(f"\n{'stratum':<10} {'n':>5} " + " ".join(f"{p:>17}" for p in ORDER)
          + "   " + " ".join(f"{d:>7}" for d in ("easy", "medium", "hard", "open")))
    for stratum in sorted(n_by_stratum):
        n = n_by_stratum[stratum]
        rates = " ".join(f"{pol_by_stratum[stratum][p]/n:>17.3f}" for p in ORDER)
        counts = " ".join(f"{by_stratum[stratum][d]:>7d}" for d in ("easy", "medium", "hard", "open"))
        print(f"{stratum:<10} {n:>5} {rates}   {counts}")
    n = total
    if n:
        overall = Counter()
        for stratum in n_by_stratum:
            for p in ORDER:
                overall[p] += pol_by_stratum[stratum][p]
        print(f"{'ALL':<10} {n:>5} " + " ".join(f"{overall[p]/n:>17.3f}" for p in ORDER)
              + "   " + " ".join(f"{by_diff[d]:>7d}" for d in ("easy", "medium", "hard", "open")))

    if a.drop_open and not a.report_only and opened:
        dest = Path(a.problems) / "open"
        dest.mkdir(exist_ok=True)
        for p in opened:
            p.rename(dest / p.name)
        print(f"\nmoved {len(opened)} unsolved-by-any-policy problems to {dest}")


if __name__ == "__main__":
    main()
