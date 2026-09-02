#!/usr/bin/env python
"""Scan randomly perturbed starting cells, to get independent battery problems.

`battery_scan.py` ships 4 hand-designed variants (std/dense/thin/coarse). Crossed
with a handful of currents that caps out around 40 scans, and since one problem is
built per scan, the battery experiments top out near n=37 -- an MDE of ~18-25pp,
which cannot resolve the +11pp depth effect measured on truss.

Densely sampling CURRENT would not fix it: 8.0A and 8.5A on the same starting cell
are near-duplicates, and treating them as independent is the pseudo-replication that
already inflated one p-value in this project by 10x.

A randomly perturbed starting cell IS a different design task: different electrode
thicknesses, porosities, particle radii and active-material fractions, hence a
different feasible region and a different binding constraint. Each is drawn
log-uniformly inside HARD_BOUNDS, so the cells are spread over the whole design box
rather than clustered near the baseline.

Writes scan_rnd<NNN>_<I>A.jsonl, which make_battery_problems_grid.py already consumes.
"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path

sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts")
import battery_scan as BS
from battery_env import ALIASES, HARD_BOUNDS


def random_variant(rng, frac=0.55):
    """A starting cell drawn inside HARD_BOUNDS, log-uniform per parameter.

    `frac` keeps the draw off the very edges of the box: a cell pinned at a bound
    has no headroom in that direction and tends to produce a problem that is either
    trivially infeasible or unreachable.
    """
    out = {}
    for alias, full in ALIASES.items():
        lo, hi = HARD_BOUNDS[full]
        if lo <= 0 or hi <= 0:
            v = rng.uniform(lo, hi)
        else:
            llo, lhi = math.log(lo), math.log(hi)
            mid, half = 0.5 * (llo + lhi), 0.5 * (lhi - llo) * frac
            v = math.exp(rng.uniform(mid - half, mid + half))
        out[full] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-variants", type=int, default=30)
    ap.add_argument("--currents", default="8,12,16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="first variant index")
    ap.add_argument("--only", default="", help="comma-separated 'rndNNN:current' to run")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    variants = {}
    for i in range(a.start + a.n_variants):
        v = random_variant(rng)
        if i >= a.start:
            variants["rnd%03d" % i] = v
    BS.VARIANTS.update(variants)

    todo = []
    if a.only:
        for tok in a.only.split(","):
            name, cur = tok.split(":")
            todo.append((name, float(cur)))
    else:
        for name in variants:
            for c in a.currents.split(","):
                todo.append((name, float(c)))

    res = Path("/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder")
    done = 0
    for name, cur in todo:
        out = res / ("scan_%s_%dA.jsonl" % (name, int(cur)))
        if out.exists() and out.stat().st_size > 0:
            continue
        try:
            BS.main(cur, name)
            done += 1
        except Exception as e:
            print("FAILED %s %gA: %s" % (name, cur, type(e).__name__), flush=True)
    print("scanned %d new (variant,current) pairs" % done, flush=True)


if __name__ == "__main__":
    main()
