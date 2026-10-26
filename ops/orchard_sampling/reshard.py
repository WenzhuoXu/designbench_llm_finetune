"""Seed each shard's sample file with the complete records found in any OTHER sample file of
the domain (the smoke/ stage, another job's subdir, an earlier sharding, a merged
samples.jsonl), so that dataxp.ownrows.sample, which resumes on its own output file only,
skips them. Makes resumption independent of the GPU count and lets two jobs share a domain.

    python reshard.py <rowdir> <N> [<subdir>]      # shard files are <rowdir>/<subdir>/samples.shard<i>.jsonl
"""
import glob
import json
import os
import sys


def complete(o):
    return bool(o.get("answer") or o.get("thinking")) and o.get("finish") != "length" and not o.get("error")


def main():
    rowdir, n = sys.argv[1], int(sys.argv[2])
    sub = sys.argv[3] if len(sys.argv) > 3 else ""
    outdir = os.path.join(rowdir, sub) if sub else rowdir
    os.makedirs(outdir, exist_ok=True)
    sid_shard = {}
    for i in range(n):
        with open(os.path.join(rowdir, "states.shard%d.jsonl" % i), encoding="utf-8") as fh:
            for l in fh:
                if l.strip():
                    sid_shard[json.loads(l)["sid"]] = i
    targets = [os.path.abspath(os.path.join(outdir, "samples.shard%d.jsonl" % i)) for i in range(n)]
    files = sorted(set(os.path.abspath(f) for f in
                       glob.glob(os.path.join(rowdir, "samples.shard*.jsonl"))
                       + glob.glob(os.path.join(rowdir, "*", "samples.shard*.jsonl"))
                       + glob.glob(os.path.join(rowdir, "samples.jsonl"))))
    have = [set() for _ in range(n)]
    pool = {}
    for f in files:
        own = targets.index(f) if f in targets else None
        with open(f, encoding="utf-8") as fh:
            for l in fh:
                try:
                    o = json.loads(l)
                except ValueError:
                    continue
                k = (o["sid"], int(o["sample"]))
                i = sid_shard.get(o["sid"])
                if i is None:
                    continue
                if own is not None and own == i:
                    have[i].add(k)          # already in its own shard file (complete or not; sample.py decides)
                elif complete(o):
                    pool.setdefault(k, o)
    added = 0
    for i in range(n):
        with open(targets[i], "a", encoding="utf-8") as fh:
            for k, o in pool.items():
                if sid_shard[k[0]] == i and k not in have[i]:
                    fh.write(json.dumps(o) + "\n")
                    have[i].add(k)
                    added += 1
    print("reshard %s (%s) -> %d shards: %d records seeded from %d files" % (
        rowdir, sub or ".", n, added, len(files)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
