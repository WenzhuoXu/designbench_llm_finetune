#!/usr/bin/env python
"""Turn the per-domain corpus into the SFT scaling-curve tiers.

Two held-out sets, testing different things:

  dev_seen       instances the model never trained on, in domains it did train on
                 -- ordinary generalisation.
  dev_catalogue  the entire catalogue domain, absent from every training tier
                 -- whether the distilled policy transfers to a design problem
                 whose action vocabulary it has never been supervised on.

Tiers are nested: train_1k is a prefix of train_10k is a prefix of train_all,
so a difference between two points on the curve is data quantity and nothing
else. Within a tier the turn budget is split evenly across training domains,
and trajectories are taken whole -- a truncated trajectory would teach the
model to stop before the design is feasible.
"""
import argparse, json, random, zlib
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
HELDOUT_DOMAIN = "catalogue"
DEV_FRAC = 0.05


def seed_of(s):
    return zlib.crc32(s.encode()) & 0xFFFFFFFF


def load(src):
    by_dom = defaultdict(list)
    for f in sorted(Path(src).rglob("sft.jsonl")):
        for line in f.open():
            r = json.loads(line)
            by_dom[r["domain"]].append(r)
    return by_dom


def write(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return sum(r["turns"] for r in rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(PROJECT / "data/corpus_v2"))
    ap.add_argument("--out", default=str(PROJECT / "data/tiers"))
    ap.add_argument("--tiers", default="1000,10000,100000,0")  # 0 = everything
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    by_dom = load(a.src)
    print("loaded: " + ", ".join("%s %d traj / %d turns"
          % (d, len(v), sum(r["turns"] for r in v)) for d, v in sorted(by_dom.items())))

    # Split by INSTANCE, not by trajectory: one instance yields one trajectory
    # here, but keeping the split instance-keyed means it stays honest if the
    # generator is ever run with several rollouts per problem.
    train_dom, dev_rows = {}, []
    for d, rows in sorted(by_dom.items()):
        if d == HELDOUT_DOMAIN:
            continue
        rng = random.Random(seed_of("split" + d))
        rows = sorted(rows, key=lambda r: r["instance"])
        rng.shuffle(rows)
        cut = max(1, int(len(rows) * DEV_FRAC))
        dev_rows += rows[:cut]
        train_dom[d] = rows[cut:]
    dev_cat = sorted(by_dom.get(HELDOUT_DOMAIN, []), key=lambda r: r["instance"])

    n_dev = write(out / "dev_seen.jsonl", dev_rows)
    n_cat = write(out / "dev_catalogue.jsonl", dev_cat)
    print("  dev_seen       %5d traj / %6d turns  (%s)"
          % (len(dev_rows), n_dev, ",".join(sorted(train_dom))))
    print("  dev_catalogue  %5d traj / %6d turns  (held-out domain)"
          % (len(dev_cat), n_cat))

    avail = sum(sum(r["turns"] for r in v) for v in train_dom.values())
    for spec in a.tiers.split(","):
        budget = int(spec)
        name = "all" if budget == 0 else ("%dk" % (budget // 1000) if budget >= 1000 else str(budget))
        target = avail if budget == 0 else budget
        if budget and budget > avail:
            print("  train_%-5s skipped: %d turns requested, %d available" % (name, budget, avail))
            continue
        # Every domain contributes the same number of turns, so the scarcest
        # training domain -- truss, the only one with finite supply -- sets the
        # per-domain budget rather than skewing the mix.
        scarcest = min(sum(r["turns"] for r in pool) for pool in train_dom.values())
        per = min(target / len(train_dom), scarcest)
        rows, got = [], Counter()
        for d, pool in sorted(train_dom.items()):
            for r in pool:                      # pool order is fixed, so tiers nest
                if got[d] >= per:
                    break
                rows.append(r); got[d] += r["turns"]
        tot = write(out / ("train_%s.jsonl" % name), rows)
        print("  train_%-5s %6d traj / %7d turns  | %s"
              % (name, len(rows), tot,
                 ", ".join("%s %.2f" % (d, got[d] / tot) for d in sorted(got))))
