"""frontier.py <states.jsonl> <samples.jsonl> [k=8] (deployed on drc as /storage/wxu/rl_open/frontier.py): drc's progress
through its states file in file order: the number of states with all k replies, the index of the first state not yet
complete, the highest index with any reply, and the number of states with some but not all replies (in flight)."""
import json
import sys

states_p, samples_p = sys.argv[1], sys.argv[2]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
sids = [json.loads(l)["sid"] for l in open(states_p) if l.strip()]
have = {}
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
done = [len(have.get(s, ())) >= k for s in sids]
first_open = next((i for i, d in enumerate(done) if not d), len(sids))
touched = [i for i, s in enumerate(sids) if s in have]
partial = sum(1 for s in sids if 0 < len(have.get(s, ())) < k)
print("complete %d first_incomplete %d max_touched %d partial %d" % (sum(done), first_open, max(touched) if touched else -1, partial))
