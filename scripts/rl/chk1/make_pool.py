"""chk1 collection pool: the photonics / grid2op / engibench problems of train_p2_datav4.jsonl (the training pool;
the benchmark problems are not in it), interleaved round-robin across the three families so any prefix covers all
three; pool_fwd.jsonl in that order, pool_rev.jsonl back to front (for a second machine)."""
import json

W = "/storage/wxu/rl_open"
by = {"photonics": [], "grid2op": [], "engibench": []}
for line in open(W + "/rl/problems/train_p2_datav4.jsonl"):
    r = json.loads(line)
    if r["variant"] in by:
        by[r["variant"]].append(line if line.endswith("\n") else line + "\n")
bench = set()
for f in ("problems_v3_native.jsonl", "problems_v2.jsonl"):
    for line in open(W + "/exclude/" + f):
        r = json.loads(line)
        bench.add((r["variant"], r["spec"]["problem_id"]))
out = []
n = max(len(v) for v in by.values())
for i in range(n):
    for v in ("engibench", "grid2op", "photonics"):
        if i < len(by[v]):
            out.append(by[v][i])
clash = sum(1 for l in out if (json.loads(l)["variant"], json.loads(l)["spec"]["problem_id"]) in bench)
assert clash == 0, "%d pool problems are benchmark problems" % clash
open(W + "/chk1/pool_fwd.jsonl", "w").writelines(out)
open(W + "/chk1/pool_rev.jsonl", "w").writelines(out[::-1])
print({k: len(v) for k, v in by.items()}, "total", len(out), "benchmark clashes", clash)
