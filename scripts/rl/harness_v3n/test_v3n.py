"""dataxp_policy_eval_v3n.py against stored episodes, with the model replaced by the stored replies.
Without --reject-note the variant reproduces the stored episode (same refusals, same potential after
every turn) and serves the same prompt again after a refused move. With it, the same replies give
the same outcomes, and the prompt after a refused move is the old prompt plus the note; after an
applied move the note is gone.

usage: python test_v3n.py <v3n script> <orig script> <request.json> <rollouts step jsonl> <harness dir>
(run with the simulator env's python, PYTHONPATH = the repo, DX_ORIG_SCRIPTS / DATAXP_SPELLING set)
"""
import collections
import json
import sys

from llm_finetune.rl_dataxp import harness as HG

v3n, orig, req_path, step_path, hdir = sys.argv[1:6]
req = json.load(open(req_path))
specs = {e["key"]: e for e in req["episodes"]}
recs = [json.loads(line) for line in open(step_path, encoding="utf-8")]
ARGV = ["--code", hdir + "/dataxp_code_v3", "--native-prompt", "served", "--instruction-file",
        hdir + "/configs/task_instruction_V2.txt", "--policy", "llm", "--model", "x", "--base-url", "http://x",
        "--temperature", "0", "--out", "/dev/null"]


class Fake:
    def __init__(self, turns):
        self.turns, self.users, self.tokens = turns, [], collections.Counter()

    def __call__(self, system, user):
        k = len(self.users)
        self.users.append(user)
        return {"answer": self.turns[k]["answer"], "reasoning": "", "think_closed": True, "finish": "stop",
                "completion_tokens": 1}


def play(script, flags, r):
    H = HG.load_harness(script)
    turns = r["episode"]["turns"]
    a = HG.harness_args(H, ARGV + ["--turns", str(len(turns))] + flags)
    mods = H.import_dataxp(a.code)
    domain = mods[3](r["variant"])
    fake = Fake(turns)
    ep = H.run_episode(mods, r["variant"], specs[r["key"]]["spec"], a, fake, HG.instruction_for_variant(H, a, mods[0], domain))
    return ep, fake.users


def refused(t):
    return str(t.get("fail") or "").startswith("illegal")


picked = {}
for r in recs:
    t = r["episode"].get("turns") or []
    ill = [refused(x) for x in t]
    # a refused move, then another refused move, then an applied one, all answers short enough to be whole
    if r["variant"] not in picked and any(ill) and not all(ill) and ill.index(False) > 0 or (
            r["variant"] not in picked and any(ill[k] and not ill[k + 1] for k in range(len(ill) - 1))):
        if all(len(x.get("answer") or "") < 600 and "episode has ended" not in str(x.get("fail")) for x in t):
            picked[r["variant"]] = r
bad = 0
for fam, r in sorted(picked.items()):
    stored = r["episode"]["turns"]
    for name, script, flags in (("orig", orig, []), ("v3n, flag off", v3n, []), ("v3n, --reject-note", v3n, ["--reject-note"])):
        ep, users = play(script, flags, r)
        got = ep["turns"]
        same = len(got) == len(stored) and all(
            (g.get("fail") == s.get("fail")) and abs(g["phi_after"] - s["phi_after"]) <= 1e-9 * max(1.0, abs(s["phi_after"]))
            for g, s in zip(got, stored))
        notes = ["REFUSED MOVES" in u for u in users]
        if flags:
            ok = same and ep.get("reject_note") is True
            for k in range(1, len(users)):
                prev_ref = refused(stored[k - 1])
                ok = ok and (notes[k] == prev_ref)
                if prev_ref:
                    base = users[k].split("\n\nREFUSED MOVES")[0]
                    ok = ok and base == users0[k] and got[k].get("reject_note", 0) >= 1
            ok = ok and not notes[0]
        else:
            ok = same and not any(notes) and all(users[k] == users[k - 1] for k in range(1, len(users)) if refused(stored[k - 1]))
            users0 = users
        bad += not ok
        print("%-10s %-20s %s  turns %d, refused %d, prompts with a note %d" % (
            fam, name, "ok" if ok else "MISMATCH", len(got), sum(refused(s) for s in stored), sum(notes)))
        if flags:
            k = max(range(len(users)), key=lambda i: users[i].count("\n- ") if notes[i] else -1)
            print("   the note at turn %d:%s" % (k, users[k].split("\n\nREFUSED MOVES")[1].rstrip().replace("\n", "\n      ")
                                               if notes[k] else " none"))
            print("   (note shown as it follows 'REFUSED MOVES')")
print("families tested: %s" % sorted(picked))
print("V3N_TEST_%s" % ("OK" if bad == 0 and picked else "FAILED"))
