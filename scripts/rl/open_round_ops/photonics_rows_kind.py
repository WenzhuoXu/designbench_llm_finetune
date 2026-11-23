"""photonics_rows_kind.py <rows.jsonl> <scored.jsonl>: what each photonics gain row's picked reply did, from the scored
records (rule-breaking pixels before/after, Phi before/after, feasible after). Prints the row keys of the first row too."""
import collections
import json
import sys

rows_p, scored_p = sys.argv[1], sys.argv[2]
state, reply = {}, {}
for line in open(scored_p):
    try:
        o = json.loads(line)
    except ValueError:
        continue
    if o.get("state"):
        state[o["sid"]] = o
    else:
        reply[(o["sid"], int(o["sample"]))] = o
first = True
kind = collections.Counter()
miss = 0
for line in open(rows_p):
    r = json.loads(line)
    if first:
        print("row keys:", sorted(r.keys()))
        print("state record keys:", sorted(next(iter(state.values())).keys()))
        print("reply record keys:", sorted(next(iter(reply.values())).keys()))
        first = False
    sid = r.get("sid")
    smp = r.get("sample", r.get("rationale_sample"))
    rep = reply.get((sid, int(smp))) if smp is not None else None
    st = state.get(sid)
    if rep is None or st is None:
        miss += 1
        continue
    b0, b1 = st.get("bad_before"), rep.get("bad_after")
    if b0 is None or b1 is None:
        kind["no pixel counts"] += 1
    elif b1 < b0:
        kind["repair: fewer rule-breaking pixels (%s)" % ("now 0" if b1 == 0 else "still >0")] += 1
    elif b1 == b0 and b0 == 0:
        kind["clean design, Phi up"] += 1
    elif b1 == b0:
        kind["same rule-breaking pixels (>0), Phi up"] += 1
    else:
        kind["MORE rule-breaking pixels"] += 1
print("rows without a scored record:", miss)
for k, v in kind.most_common():
    print("%-55s %6d" % (k, v))
