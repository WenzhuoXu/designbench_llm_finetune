"""Split a states.jsonl into N shards by line index mod N, written next to it as
states.shard<i>.jsonl. Deterministic, so a requeued job rebuilds the same shards and
each sampler resumes on its own append-only output.

    python shard.py <states.jsonl> <N> <outdir>
"""
import os
import sys

src, n, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(src, encoding="utf-8") as fh:
    lines = [l for l in fh if l.strip()]
os.makedirs(outdir, exist_ok=True)
for i in range(n):
    with open(os.path.join(outdir, "states.shard%d.jsonl" % i), "w", encoding="utf-8") as fh:
        fh.writelines(lines[i::n])
print("sharded %s: %d states -> %d shards of %s" % (src, len(lines), n,
      "/".join(str(len(lines[i::n])) for i in range(n))), flush=True)
