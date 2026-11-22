"""open_selfcheck.py <episodes.jsonl> <states.jsonl> <scored.jsonl> <out samples.jsonl> [--write-samples]
The scorer under the open rules (score_states.py on states from open_states.py) must give the policy's own move at
every state the outcome the harness gave it in the episode: refused there <=> refused by the scorer, and when applied
the same potential (photonics, grid2op and engibench are deterministic simulators).

--write-samples: writes, for every state, the episode's own reply as sample 0 (the samples file score_states.py
reads). Without it: compares scored.jsonl with the episodes and prints the agreement per family."""
import collections
import json
import sys

ep_path, st_path, sc_path, sm_path = sys.argv[1:5]
write = "--write-samples" in sys.argv[5:]
states = [json.loads(x) for x in open(st_path, encoding="utf-8") if x.strip()]
by_turn = {}
for line in open(ep_path, encoding="utf-8"):
    r = json.loads(line)
    for t in r["episode"].get("turns") or []:
        by_turn[(r["key"], int(t["t"]))] = t
first = {}
for s in states:                                        # the turn of the episode that first served this state
    first.setdefault(s["sid"], by_turn[(s["source"], s["turn"])])
if write:
    with open(sm_path, "w", encoding="utf-8") as fo:
        for s in states:
            t = first[s["sid"]]
            fo.write(json.dumps({"sid": s["sid"], "sample": 0, "thinking": t.get("reasoning") or "", "answer": t.get("answer") or "",
                                 "finish": t.get("finish") or "stop", "sha1": "own"}) + "\n")
    print("samples written", len(states))
    sys.exit(0)
c = collections.defaultdict(collections.Counter)
shown = 0
for line in open(sc_path, encoding="utf-8"):
    x = json.loads(line)
    if x.get("state"):
        if x.get("error"):
            c[x["variant"]]["state errors"] += 1
        continue
    t = first[x["sid"]]
    fam = x["variant"]
    ref_h = str(t.get("fail") or "").startswith("illegal")
    ref_s = bool(x.get("parsed")) and not x.get("legal")
    c[fam]["own moves"] += 1
    if t.get("fail") and not ref_h:
        c[fam]["harness: not a move (%s)" % str(t["fail"])[:20]] += 1
        continue
    if ref_h != ref_s:
        c[fam]["DISAGREE refused/applied"] += 1
        if shown < 6:
            shown += 1
            print("  disagree:", x["sid"], "| harness:", str(t.get("fail"))[:120], "| scorer:", x.get("reason"), x.get("phi"))
        continue
    if ref_h:
        c[fam]["agree: refused"] += 1
        c[fam]["  of which an undone blackout"] += "would end the episode" in str(t.get("fail"))
        continue
    d = abs(float(x["phi"]) - float(t["phi_after"])) / max(1.0, abs(float(t["phi_after"])))
    if d <= 1e-6:
        c[fam]["agree: applied, same potential"] += 1
    else:
        c[fam]["DISAGREE potential"] += 1
        if shown < 6:
            shown += 1
            print("  potential differs:", x["sid"], x["phi"], t["phi_after"])
for fam in sorted(c):
    print(fam, dict(sorted(c[fam].items())))
bad = sum(v for f in c for k, v in c[f].items() if k.startswith("DISAGREE") or k == "state errors")
print("OPEN_SELFCHECK_%s" % ("OK" if bad == 0 else "FAILED (%d)" % bad))
