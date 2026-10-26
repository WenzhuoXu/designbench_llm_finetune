"""Concatenate sampler shard files into one samples.jsonl with one line per (sid, sample).
A complete record (an answer, finish != "length", no error) beats an incomplete one; among
equals the last seen wins. Unreadable (cut) lines are skipped and counted.

    python merge_shards.py --out <samples.jsonl> <samples.shard0.jsonl> ...
"""
import argparse
import json
import sys


def complete(o):
    return bool(o.get("answer") or o.get("thinking")) and o.get("finish") != "length" and not o.get("error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    recs, order, bad, dup = {}, [], 0, 0
    for f in a.files:
        with open(f, encoding="utf-8") as fh:
            for l in fh:
                try:
                    o = json.loads(l)
                except ValueError:
                    bad += 1
                    continue
                k = (o["sid"], int(o["sample"]))
                if k in recs:
                    dup += 1
                    if complete(recs[k]) and not complete(o):
                        continue
                else:
                    order.append(k)
                recs[k] = o
    with open(a.out, "w", encoding="utf-8") as fh:
        for k in order:
            fh.write(json.dumps(recs[k]) + "\n")
    n_complete = sum(complete(recs[k]) for k in order)
    print("merged %d files -> %s: %d records (%d complete, %d duplicates dropped, %d unreadable lines)" % (
        len(a.files), a.out, len(order), n_complete, dup, bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
