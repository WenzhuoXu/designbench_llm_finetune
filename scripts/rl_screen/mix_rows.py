"""Step 5 of a screened supervised round: the training and dev files of the round.

    train = the new screened rows (minus the dev problems) + a replay sample of the rows that trained
            the starting adapter, drawn uniformly at random (so the replay keeps that file's own mix
            of families), --replay-ratio replay rows per new row
    dev   = every new row of a random --dev-frac of the new rows' problems, per family (a problem's
            rows go to dev or train together), at most --dev-max rows (evenly over the families)

Nothing else is dropped. The manifest (printed, and <out-train>.manifest.json) counts rows per
family and source.

    python mix_rows.py --new rows_gain.jsonl [more.jsonl ...] --replay train_dx38mtR.jsonl \\
        --out-train train.jsonl --out-dev dev.jsonl [--replay-ratio 0.5] [--dev-frac 0.05] [--dev-max 240] [--seed 0]
"""
import argparse
import collections
import json
import random
import sys


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def split_dev(rows: list, frac: float, dev_max: int, seed: int):
    """(train rows, dev rows): whole problems go to dev, a share `frac` of each family's problems."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["domain"]][r["instance"]].append(r)
    dev_inst = set()
    for fam in sorted(by):
        inst = sorted(by[fam])
        random.Random("%d/dev/%s" % (seed, fam)).shuffle(inst)
        n = int(round(len(inst) * frac)) if frac > 0 else 0
        dev_inst |= {(fam, i) for i in inst[:max(n, 1 if frac > 0 and len(inst) > 1 else 0)]}
    train = [r for r in rows if (r["domain"], r["instance"]) not in dev_inst]
    dev = [r for r in rows if (r["domain"], r["instance"]) in dev_inst]
    if dev_max and len(dev) > dev_max:
        fams = sorted({r["domain"] for r in dev})
        per = max(1, dev_max // len(fams))
        keep = []
        for fam in fams:
            fr = [r for r in dev if r["domain"] == fam]
            random.Random("%d/devcut/%s" % (seed, fam)).shuffle(fr)
            keep.extend(fr[:per])
        dev = keep                       # the dev rows cut here are not trained on either
    return train, dev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-dev", required=True)
    ap.add_argument("--replay-ratio", type=float, default=0.5)
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--dev-max", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    new, seen, n_dup = [], set(), 0
    for p in a.new:
        for r in read(p):
            key = r.get("sid") or json.dumps(r, sort_keys=True)
            if key in seen:                  # the same state in two files: once
                n_dup += 1
                continue
            seen.add(key)
            new.append(r)
    train_new, dev = split_dev(new, a.dev_frac, a.dev_max, a.seed)
    # a problem with rows in dev gives the training file nothing, replay rows included
    dev_problems = {(r["domain"], r["instance"]) for r in new} - {(r["domain"], r["instance"]) for r in train_new}
    replay_file = read(a.replay)
    replay_all = [r for r in replay_file if (r["domain"], r["instance"]) not in dev_problems]
    n_rep = min(len(replay_all), int(round(len(train_new) * a.replay_ratio)))
    idx = list(range(len(replay_all)))
    random.Random("%d/replay" % a.seed).shuffle(idx)
    replay = [replay_all[i] for i in sorted(idx[:n_rep])]
    train = train_new + replay
    with open(a.out_train, "w", encoding="utf-8") as fo:
        for r in train:
            fo.write(json.dumps(r) + "\n")
    with open(a.out_dev, "w", encoding="utf-8") as fo:
        for r in dev:
            fo.write(json.dumps(r) + "\n")
    fams = sorted({r["domain"] for r in train + dev + replay_all})
    count = lambda rows: collections.Counter(r["domain"] for r in rows)          # noqa: E731
    man = {"new rows": len(new), "new rows in train": len(train_new), "dev rows": len(dev),
           "new rows in neither (dev problems beyond --dev-max)": len(new) - len(train_new) - len(dev),
           "new rows given twice (kept once)": n_dup,
           "replay rows": len(replay), "replay file rows": len(replay_file),
           "replay file rows on dev problems (never drawn)": len(replay_file) - len(replay_all), "train rows": len(train),
           "replay_ratio": a.replay_ratio, "dev_frac": a.dev_frac, "seed": a.seed,
           "per family": {f: {"new in train": count(train_new)[f], "dev": count(dev)[f], "replay": count(replay)[f],
                              "train": count(train)[f]} for f in fams}}
    json.dump(man, open(a.out_train + ".manifest.json", "w", encoding="utf-8"), indent=1)
    print("%-12s %14s %8s %8s %8s" % ("family", "new in train", "dev", "replay", "train"))
    for f in fams:
        p = man["per family"][f]
        print("%-12s %14d %8d %8d %8d" % (f, p["new in train"], p["dev"], p["replay"], p["train"]))
    print("%-12s %14d %8d %8d %8d" % ("all", len(train_new), len(dev), len(replay), len(train)))
    print("new rows held out of both files: %d; given twice: %d; replay rows on dev problems left out: %d" % (
        man["new rows in neither (dev problems beyond --dev-max)"], n_dup, len(replay_file) - len(replay_all)))
    print("MIX_ROWS_DONE %s" % a.out_train)
    return 0


if __name__ == "__main__":
    sys.exit(main())
