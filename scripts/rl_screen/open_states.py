"""Step 1 of a round under the open rules: the states of a policy's own episodes collected with
dataxp_policy_eval_v3r.py --open-rules --record-prompts, each with the exact prompt the policy was served there.

Under the open rules the prompt carries what the simulator returned: the moves refused at the current design and
why (a grid2op move that would have ended the episode included), and on photonics the count of pixels that break
the fabrication rule. So a state here is the design (the problem plus the moves applied so far) together with the
prompt served at it: the same design is one state per distinct prompt (before and after a refusal, say). No
replay is needed: the prompt is the recorded one, the design is rebuilt from the applied moves when scored
(score_states.py --open-rules).

    python open_states.py --episodes FILE [FILE ...] --out states.jsonl [--exclude-specs problems_eval.jsonl ...]
        [--variants photonics grid2op engibench]

Each FILE is an evaluation record file (rollout eval) with its request beside it (<FILE>.request.json); the
request must hold --open-rules and --record-prompts. Episodes on an excluded problem stop the run.
"""
import argparse
import collections
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_states import canon, request_of, state_id  # noqa: E402


def open_sid(variant: str, pid: str, history: list, system: str, user: str) -> str:
    return "%s~%s" % (state_id(variant, pid, history), hashlib.sha1((system + "\x00" + user).encode()).hexdigest()[:8])


def episode_states(rec: dict, spec: dict, spelling) -> list:
    """The state rows of one episode, in turn order (a state per turn, duplicates included)."""
    ep = rec["episode"]
    rows, moves, phi = [], [], float(ep["phi_start"])
    for t in ep.get("turns") or []:
        if "user" not in t or "system" not in t:
            raise ValueError("%s: turn %s has no recorded prompt (collect with --record-prompts)" % (rec["key"], t.get("t")))
        applied = t.get("fail") is None and t.get("action") is not None
        user = t["user"]
        rows.append({
            "sid": open_sid(rec["variant"], spec["problem_id"], moves, t["system"], user),
            "variant": rec["variant"], "problem_id": spec["problem_id"], "k": len(moves), "turn": int(t["t"]), "split": "train",
            "native": bool(ep.get("native")), "spelling": spelling, "spec": spec, "history": list(moves),
            "system": t["system"], "observation": user, "instruction": "",
            "instruction_sha1": ep.get("instruction_sha1"), "binding_constraint": t.get("binding"),
            "phi_before": phi, "n_prior_changes": len(moves), "source": rec["key"], "source_turns": 1,
            "source_refused": int(str(t.get("fail") or "").startswith("illegal")),
            "own_action": t["action"] if applied else None,
            "open_rules": True, "note_refused": int(t.get("reject_note") or 0),
            "note_fabrication": "\nFABRICATION: " in user, "note_blackout": "would end the episode" in user,
        })
        if applied:
            moves.append(t["action"])
            phi = float(t["phi_after"])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-specs", nargs="*", default=[])
    ap.add_argument("--variants", nargs="*", default=None)
    a = ap.parse_args(argv)
    banned = set()
    for path in a.exclude_specs:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                banned.add(canon(r.get("spec", r)))
    cnt = collections.Counter()
    seen = set()
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fo:
        for path in a.episodes:
            req = json.load(open(request_of(path)))
            hargv = req["harness_argv"]
            if "--open-rules" not in hargv or "--record-prompts" not in hargv:
                raise SystemExit("%s was not collected with --open-rules --record-prompts: %s" % (path, hargv))
            spelling = (req.get("env") or {}).get("DATAXP_SPELLING") or os.environ.get("DATAXP_SPELLING")
            specs = {e["key"]: e["spec"] for e in req["episodes"]}
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                if a.variants and rec["variant"] not in a.variants:
                    continue
                cnt["episodes"] += 1
                ep = rec["episode"]
                if ep.get("crashed") or rec.get("rl_error") or not ep.get("turns"):
                    cnt["episodes without turns (crashed, or feasible at the start)"] += 1
                    continue
                spec = specs[rec["key"]]
                if canon(spec) in banned:
                    raise SystemExit("%s: %s is an excluded (evaluation) problem" % (path, spec["problem_id"]))
                cnt["episodes used"] += 1
                cnt["episodes feasible at 20 turns"] += bool(ep.get("feasible@20"))
                for r in episode_states(rec, spec, spelling):
                    cnt["turns"] += 1
                    if r["sid"] in seen:
                        cnt["turns at a state already written"] += 1
                        continue
                    seen.add(r["sid"])
                    fo.write(json.dumps(r) + "\n")
                    v = r["variant"]
                    cnt["states, %s" % v] += 1
                    cnt["states with refused moves in the prompt, %s" % v] += r["note_refused"] > 0
                    cnt["states with an undone blackout in the prompt, %s" % v] += r["note_blackout"]
                    cnt["states with fabrication violations in the prompt, %s" % v] += r["note_fabrication"]
    os.replace(tmp, a.out)
    cnt["distinct states written"] = len(seen)
    for k in sorted(cnt):
        print("%-62s %d" % (k, cnt[k]))
    print("OPEN_STATES_DONE %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
