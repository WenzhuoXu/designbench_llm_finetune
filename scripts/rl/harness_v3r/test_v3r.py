"""dataxp_policy_eval_v3r.py against stored episodes, with the model replaced by the stored replies.

1. Flag off: every picked episode is reproduced (the same fail and potential after every turn), in every family.
2. --open-rules:
   - a family the open rules do not change: the same outcomes, and the prompt after a refused move carries the note;
   - grid2op, an episode a move ended (blackout): the turns before it are the same; that move is refused with
     "would end the episode", the grid is as it was (potential unchanged), the next prompt names it, and every
     prompt carries the rule line;
   - photonics, an episode with a move refused by the fabrication rule: the turns before it are the same; that move
     is applied, min_feature_violations > 0 after it, the design is not feasible, the next prompt says so, and the
     prompt's rule sentence is the new one.
After the first changed turn the stored replies no longer answer the new state, so nothing later is compared.

usage: python test_v3r.py <v3r script> <request.json> <episodes.jsonl> <harness dir>
(run with the simulator env's python, PYTHONPATH = the repo, DX_ORIG_SCRIPTS / DATAXP_SPELLING set)
"""
import collections
import json
import sys

from llm_finetune.rl_dataxp import harness as HG

v3r, req_path, ep_path, hdir = sys.argv[1:5]
req = json.load(open(req_path))
specs = {e["key"]: e for e in req["episodes"]}
ARGV = ["--code", hdir + "/dataxp_code_v3", "--native-prompt", "served", "--instruction-file",
        hdir + "/configs/task_instruction_V2.txt", "--policy", "llm", "--model", "x", "--base-url", "http://x",
        "--temperature", "0", "--out", "/dev/null"]


class Fake:
    def __init__(self, turns):
        self.turns, self.users = turns, []

    def __call__(self, system, user):
        k = len(self.users)
        self.users.append(user)
        t = self.turns[min(k, len(self.turns) - 1)]
        return {"answer": t.get("answer") or "", "reasoning": "", "think_closed": True, "finish": "stop", "completion_tokens": 1}


def play(flags, r):
    H = HG.load_harness(v3r)
    turns = r["episode"]["turns"]
    a = HG.harness_args(H, ARGV + ["--turns", str(len(turns))] + flags)
    mods = H.import_dataxp(a.code)
    domain = mods[3](r["variant"])
    fake = Fake(turns)
    ep = H.run_episode(mods, r["variant"], specs[r["key"]]["spec"], a, fake, HG.instruction_for_variant(H, a, mods[0], domain))
    return ep, fake.users


def refused(t):
    return str(t.get("fail") or "").startswith("illegal")


def same_turn(g, s):
    return g.get("fail") == s.get("fail") and abs(g["phi_after"] - s["phi_after"]) <= 1e-9 * max(1.0, abs(s["phi_after"]))


def blackout_turn(ep):
    for k, t in enumerate(ep["turns"]):
        if not t.get("fail") and (t.get("metrics_after") or {}).get("blackout") == 1.0:
            return k
    return None


def feature_turn(ep):
    for k, t in enumerate(ep["turns"]):
        if "minimum feature rule" in str(t.get("fail") or ""):
            return k
    return None


recs = [json.loads(line) for line in open(ep_path, encoding="utf-8")]
recs = [r for r in recs if not r["episode"].get("feasible_start") and r["key"] in specs]
whole = lambda r: all(len(x.get("answer") or "") < 600 for x in r["episode"]["turns"])      # noqa: E731
picked = {}
for r in recs:
    v = r["variant"]
    ill = [refused(x) for x in r["episode"]["turns"]]
    if v not in picked and whole(r) and any(ill) and not all(ill) and v not in ("grid2op", "photonics"):
        picked[v] = r
for r in recs:                                   # plain episodes for the families with no refusal at all
    if r["variant"] not in picked and r["variant"] not in ("grid2op", "photonics") and whole(r):
        picked[r["variant"]] = r
g_ep = next((r for r in recs if r["variant"] == "grid2op" and whole(r) and blackout_turn(r["episode"]) not in (None, 0)), None)
p_ep = next((r for r in recs if r["variant"] == "photonics" and whole(r) and feature_turn(r["episode"]) not in (None, 0)), None)
bad = 0


def report(name, ok, extra=""):
    global bad
    bad += not ok
    print("%-44s %s %s" % (name, "ok" if ok else "MISMATCH", extra), flush=True)


# 1. flag off: reproduced
for fam, r in sorted(list(picked.items()) + [("grid2op", g_ep), ("photonics", p_ep)]):
    if r is None:
        report("%s: flag off (no episode found)" % fam, False)
        continue
    ep, users = play([], r)
    st = r["episode"]["turns"]
    ok = len(ep["turns"]) == len(st) and all(same_turn(g, s) for g, s in zip(ep["turns"], st)) and not any(
        "REFUSED MOVES" in u or "RULE IN THIS RUN" in u or "FABRICATION:" in u for u in users)
    report("%s %s: flag off" % (fam, r["key"]), ok, "turns %d, refused %d" % (len(st), sum(refused(s) for s in st)))

# 2a. open rules on a family they do not change
for fam, r in sorted(picked.items()):
    ep, users = play(["--open-rules"], r)
    st = r["episode"]["turns"]
    ok = ep.get("open_rules") is True and len(ep["turns"]) == len(st) and all(same_turn(g, s) for g, s in zip(ep["turns"], st))
    for k in range(1, len(users)):
        ok = ok and (("REFUSED MOVES" in users[k]) == refused(st[k - 1]))
    report("%s: --open-rules, outcomes unchanged" % fam, ok, "notes %d" % sum("REFUSED MOVES" in u for u in users))

# 2b. grid2op
if g_ep is not None:
    k = blackout_turn(g_ep["episode"])
    st = g_ep["episode"]["turns"]
    ep, users = play(["--open-rules"], g_ep)
    got = ep["turns"]
    before = st[k - 1]["phi_after"]
    ok = all(same_turn(got[i], st[i]) for i in range(k))
    ok = ok and refused(got[k]) and "would end the episode" in str(got[k].get("fail")) and abs(got[k]["phi_after"] - before) <= 1e-9 * max(1.0, abs(before))
    ok = ok and (got[k].get("metrics_after") or {}).get("blackout") == 0.0
    ok = ok and len(users) > k + 1 and "REFUSED MOVES" in users[k + 1] and "would end the episode" in users[k + 1]
    ok = ok and all("RULE IN THIS RUN" in u for u in users)
    report("grid2op %s: blackout at turn %d undone" % (g_ep["key"], k), ok, "fail: %s" % str(got[k].get("fail"))[:110])
    if len(users) > k + 1:
        print("   note:", users[k + 1].split("REFUSED MOVES")[-1][:420].replace("\n", " | "))

# 2c. photonics
if p_ep is not None:
    k = feature_turn(p_ep["episode"])
    st = p_ep["episode"]["turns"]
    ep, users = play(["--open-rules"], p_ep)
    got = ep["turns"]
    m = got[k].get("metrics_after") or {}
    ok = all(same_turn(got[i], st[i]) for i in range(k))
    ok = ok and not got[k].get("fail") and float(m.get("min_feature_violations", 0)) > 0 and not got[k]["feasible_after"]
    ok = ok and len(users) > k + 1 and "FABRICATION:" in users[k + 1]
    ok = ok and all("is still applied; min_feature_violations" in u and "is rejected and the design stays" not in u for u in users)
    report("photonics %s: fabrication refusal at turn %d applied" % (p_ep["key"], k), ok,
           "min_feature_violations %s, feasible %s" % (m.get("min_feature_violations"), got[k]["feasible_after"]))
    if len(users) > k + 1:
        print("   prompt end:", users[k + 1][-300:].replace("\n", " | "))

print("V3R_TEST_%s" % ("OK" if bad == 0 else "FAILED"))
