"""cover_upto.py <states.jsonl> <samples.jsonl> <D> [k=8] [--cut OUT]: prints `complete_below_D touched_at_or_above_D`
(states of file index < D with all k replies; states of index >= D with any reply). With --cut OUT, writes the states
of index < D plus the states of index >= D that have replies to OUT (file order) and prints the counts."""
import json
import os
import sys

args = sys.argv[1:]
cut = None
if "--cut" in args:
    i = args.index("--cut")
    cut = args[i + 1]
    del args[i:i + 2]
states_p, samples_p, d = args[0], args[1], int(args[2])
k = int(args[3]) if len(args) > 3 else 8
lines = [l for l in open(states_p) if l.strip()]
sids = [json.loads(l)["sid"] for l in lines]
have = {}
try:
    with open(samples_p, "rb") as f:
        for raw in f:
            if not raw.endswith(b"\n"):
                continue
            try:
                o = json.loads(raw)
            except ValueError:
                continue
            if o.get("error"):
                continue
            have.setdefault(o["sid"], set()).add(int(o["sample"]))
except FileNotFoundError:
    pass
below = sum(1 for s in sids[:d] if len(have.get(s, ())) >= k)
above = sum(1 for s in sids[d:] if s in have)
if cut:
    keep = [l for i, l in enumerate(lines) if i < d or sids[i] in have]
    with open(cut + ".tmp", "w") as fo:
        fo.writelines(l if l.endswith("\n") else l + "\n" for l in keep)
    os.replace(cut + ".tmp", cut)
    print("cut: kept %d of %d states (%d below D, %d at or above D with replies)" % (len(keep), len(lines), d, above))
else:
    print(below, above)
