import json, sys
pool, orch, drc = sys.argv[1], sys.argv[2], sys.argv[3]
order = [json.loads(l)["spec"]["problem_id"] for l in open(pool) if l.strip() and json.loads(l)["variant"] in ("grid2op", "engibench")]
order = [p for p in order if p.startswith("grid2op")] + [p for p in order if p.startswith("engibench")]
o = set(json.loads(l)["key"].split("/")[-1] for l in open(orch) if l.endswith("\n") and l.strip())
d = set(open(drc).read().split())
miss = [p for p in order if p not in o and p not in d]
print("pool", len(order), "orchard", len(o), "drc", len(d), "both", len(o & d), "missing", len(miss))
print("missing:", " ".join(miss))
