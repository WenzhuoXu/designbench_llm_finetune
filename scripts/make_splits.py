#!/usr/bin/env python
"""Build a reproducible train/eval split over a design-problem set.

Until now GRPO trained on ``DesignBench/data/problems`` and evaluation read the
SAME directory, so every reported "held-out" number was measured on problems the
policy had trained on. This script produces a fixed, stratified, seeded split and
writes it to ``data/splits/<name>.json`` so training and evaluation can both
point at it.

Stratification keeps train and eval matched on the three things that actually
predict difficulty here: which constraint family the problem belongs to, whether
the simulator is degenerate at the initial state, and the initial violation
magnitude. Domain-agnostic: it reads the problem's own goals through
DesignProgram, so the same script splits a battery problem set.

    python scripts/make_splits.py --name truss_v1 --eval-frac 0.35
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm_finetune.training.rl.posterior.potential import program_from_truss_spec  # noqa: E402


def initial_state(spec: dict) -> dict:
    from validation.truss_executor import analyze_truss, load_truss_from_problem
    truss = load_truss_from_problem(spec)
    return analyze_truss(truss, spec.get("goals", {}) or {})


def describe(spec: dict) -> dict:
    st = initial_state(spec)
    prog = program_from_truss_spec(spec, initial_mass=st.get("mass"))
    mass = float(st.get("mass", 0.0) or 0.0)
    defl = float(st.get("deflection", 0.0) or 0.0)
    degenerate = (not math.isfinite(mass)) or mass <= 0.0 or (not math.isfinite(defl))
    family = "mass" if prog.limit_for("mass") is not None else (
        "deflection" if prog.limit_for("deflection") is not None else "fos_only")
    return {
        "problem_id": spec.get("problem_id"),
        "family": family,
        "degenerate": bool(degenerate),
        "initial_violation": float(prog.total_violation(st, tau=0.05)),
        "initial_feasible": bool(st.get("is_feasible", False)),
        "n_members": len((spec.get("topology") or {}).get("members", [])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(DESIGNBENCH / "data/problems"))
    ap.add_argument("--name", default="truss_v1")
    ap.add_argument("--eval-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--out-dir", default=str(PROJECT / "data/splits"))
    args = ap.parse_args()

    records = []
    seen_ids: set[str] = set()
    for f in sorted(Path(args.problems).glob("*.json")):
        spec = json.load(open(f))
        if not (isinstance(spec, dict) and "topology" in spec):
            continue
        spec.setdefault("problem_id", f.stem)
        rec = describe(spec)
        pid = rec["problem_id"]
        if pid in seen_ids:   # duplicate problem_id across files -> keep one
            print(f"  note: duplicate problem_id {pid!r} in {f.name}, skipped")
            continue
        seen_ids.add(pid)
        records.append(rec)

    # Stratum = family x degenerate x initial-violation tercile.
    viols = sorted(r["initial_violation"] for r in records if math.isfinite(r["initial_violation"]))
    q1 = viols[len(viols) // 3] if viols else 0.0
    q2 = viols[2 * len(viols) // 3] if viols else 0.0

    def tercile(v):
        if not math.isfinite(v):
            return "inf"
        return "lo" if v <= q1 else ("mid" if v <= q2 else "hi")

    strata = defaultdict(list)
    for r in records:
        strata[(r["family"], r["degenerate"], tercile(r["initial_violation"]))].append(r)

    # Within each stratum, order by initial violation and deal cards alternately
    # so the two sides match on the continuous covariate too, not just on the
    # stratum labels. The seeded offset keeps it reproducible but unbiased.
    import random
    rng = random.Random(args.seed)
    train, evl = [], []
    period = max(int(round(1.0 / max(args.eval_frac, 1e-6))), 2)
    for key in sorted(strata, key=str):
        group = sorted(strata[key], key=lambda r: (r["initial_violation"], r["problem_id"]))
        offset = rng.randrange(period)
        for i, rec in enumerate(group):
            (evl if (i + offset) % period == 0 else train).append(rec)
        # a stratum of one still has to land somewhere; keep it in train
        if len(group) == 1 and group[0] in evl:
            evl.remove(group[0])
            train.append(group[0])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": args.name,
        "seed": args.seed,
        "eval_frac": args.eval_frac,
        "source": str(args.problems),
        "n_total": len(records),
        "train": sorted(r["problem_id"] for r in train),
        "eval": sorted(r["problem_id"] for r in evl),
        "records": {r["problem_id"]: r for r in records},
    }
    path = out_dir / f"{args.name}.json"
    json.dump(payload, open(path, "w"), indent=2)

    def summarise(rows, label):
        fam = defaultdict(int)
        deg = sum(1 for r in rows if r["degenerate"])
        for r in rows:
            fam[r["family"]] += 1
        mv = [r["initial_violation"] for r in rows if math.isfinite(r["initial_violation"])]
        print(f"  {label:<6} n={len(rows):>3}  families={dict(fam)}  degenerate={deg}  "
              f"mean_init_violation={(sum(mv)/len(mv) if mv else float('nan')):.3f}")

    print(f"wrote {path}  (total {len(records)})")
    summarise(train, "train")
    summarise(evl, "eval")
    overlap = set(payload["train"]) & set(payload["eval"])
    print(f"  overlap: {len(overlap)}  (must be 0)")


if __name__ == "__main__":
    main()
