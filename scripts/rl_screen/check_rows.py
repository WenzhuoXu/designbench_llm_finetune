"""chk1 rows: training rows for the check-before-commit protocol (harness v3rc --checks C) from collection episodes
(v3rc --checks C --check-actions --record-prompts: dx38mtR at T=0 on the training pool, each new move it proposes
checked in the simulator before it commits; naming a checked move again commits it).

A turn of such an episode is one conversation: system; user + CHECKS rule; per check, the model's reply (its last
tag renamed <check>, as the harness sent it) and the simulator's result + budget line; the committing reply.
A turn is kept when
  1. every reply of the turn finished (finish == "stop"; an unfinished or unclosed reply ends the turn early),
  2. the committed move is one the turn checked (its <action> text equals a check's text, whitespace aside, and the
     commit's outcome equals that check's: the check is scored with the commit's seed) and it was applied,
  3. its outcome is the best of the turn's applied checks (key: feasible, fewer feature-rule pixels (photonics),
     higher Phi; ties allowed),
  4. it improves on the design the turn started from (same key),
  5. no check of the turn is unreadable, every check reply ends with a <check> tag, and (grid2op, where time advances
     every turn) the commit is not DO_NOTHING() unless it makes the design feasible: the null move's Phi change is
     load drift, not a gain (review 11-24).
A kept turn gives one row per reply: messages = the conversation up to that reply (earlier replies as the answer text
the harness sent, without their thinking) + the reply as the target {reasoning_content: its thinking, content: its
answer}. The rows' prompts are the prompts the model was served.

usage: python check_rows.py --episodes FILE [FILE ...] --out rows.jsonl [--report report.txt]
           [--verify-requests requests.jsonl --keep-all]   (plumbing test: every row prompt must be a logged request)
"""
import argparse
import collections
import json
import re
import sys

TAG_LAST = re.compile(r"<(action|check)>\s*(.*?)\s*</\1>", re.S)
VIOL_RE = re.compile(r"minimum feature rule at (\d+) pixels")


def check_rule(c):
    """The CHECKS paragraph of dataxp_policy_eval_v3rc.py (sha1 9c82418581a0), verbatim."""
    return ("\n\nCHECKS: before you commit a move you may check up to %d candidate moves in the simulator. To check "
            "a move, end your reply with <check>VERB(arg, ...)</check> instead of the <action> line: the simulator "
            "applies it to a copy of the current design and shows you the result; the design itself stays as it is, "
            "and you then reply again. Each check costs one simulator call. Commit a move with the <action> line as "
            "usual whenever you are ready; after %d checks you must commit.\n" % (c, c))


def tail(left):
    return ("You may check %d more move%s, or commit one with the <action> line." % (left, "" if left == 1 else "s")
            if left > 0 else "No checks left: commit a move now with the <action> line.")


def last_tag(answer):
    m = TAG_LAST.findall(answer or "")
    return (m[-1][0], m[-1][1]) if m else (None, None)


def squash(s):
    return re.sub(r"\s+", "", s or "")


def key(feasible, viol, phi):
    return (bool(feasible), -float(viol or 0.0), float(phi))


def viol_after(variant, t):
    if variant != "photonics":
        return 0.0
    return float((t.get("metrics_after") or {}).get("min_feature_violations") or 0.0)


def start_viol(variant, user):
    """Feature-rule pixels of the start design (photonics), read from the turn-0 prompt; 0 when it names none."""
    if variant != "photonics":
        return 0.0
    m = VIOL_RE.findall(user or "")
    return float(m[0]) if m else 0.0


def conversation(t, C):
    """[(prefix messages, target message)] of a turn: one entry per reply, the commit last."""
    msgs = [{"role": "system", "content": t["system"]}, {"role": "user", "content": t["user"] + check_rule(C)}]
    out = []
    checks = t.get("checks") or []
    for i, x in enumerate(checks):
        out.append((list(msgs), {"role": "assistant", "reasoning_content": x.get("reasoning", ""),
                                 "content": x.get("answer", "")}))
        msgs += [{"role": "assistant", "content": x.get("answer", "")},
                 {"role": "user", "content": x["result"] + "\n\n" + tail(C - (i + 1))}]
    out.append((list(msgs), {"role": "assistant", "reasoning_content": t.get("reasoning", ""),
                             "content": t.get("answer", "")}))
    return out


def judge(ep, t, before):
    """(None, commit check index) when the turn is kept, else (reason, None)."""
    v = ep["variant"]
    checks = t.get("checks") or []
    if not checks:
        return "no check", None
    if t.get("check_as_action"):
        return "commit forced from a check tag", None
    if any(x.get("finish") != "stop" for x in checks) or t.get("finish") != "stop":
        return "a reply did not finish", None
    if t.get("fail"):
        return "commit refused or unread", None
    if any(x.get("parse") for x in checks):
        return "a check could not be read", None        # its reply would teach an unreadable move
    if any(last_tag(x.get("answer"))[0] != "check" for x in checks):
        return "a check reply does not end with a check tag", None
    kind, text = last_tag(t.get("answer"))
    if kind != "action":
        return "commit has no action tag", None
    if v == "grid2op" and squash(text) == "DO_NOTHING()" and not t.get("feasible_after"):
        # time advances in grid2op: the null move's Phi change is load drift, not a gain the move made
        return "commit is DO_NOTHING and the design stays infeasible", None
    idx = [i for i, x in enumerate(checks) if "phi" in x and squash(x.get("text")) == squash(text)]
    if not idx:
        return "committed move not among the applied checks", None
    i = idx[-1]
    x = checks[i]
    ck = key(x["feasible"], x.get("viol"), x["phi"])
    cm = key(t.get("feasible_after"), viol_after(v, t), t.get("phi_after"))
    if ck != cm:
        return "commit outcome differs from its check", None
    best = max(key(y["feasible"], y.get("viol"), y["phi"]) for y in checks if "phi" in y)
    if cm < best:
        return "a checked move was better", None
    if not cm > before:
        return "commit does not improve the design", None
    return None, i


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--verify-requests", default=None)
    ap.add_argument("--keep-all", action="store_true", help="plumbing test only: rows of every turn with checks")
    ap.add_argument("--tier", default="chk1")
    ap.add_argument("--teacher", default="dx38mtR-own-check-collect")
    a = ap.parse_args(argv)
    c = collections.defaultdict(collections.Counter)
    nck = collections.defaultdict(collections.Counter)
    seen_ep, rows = set(), []
    for path in a.episodes:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            ep = r.get("episode") or r
            v = ep.get("variant", "?")
            k_ep = r.get("key") or (v, ep.get("problem_id"))   # rollout key (variant, problem, run member)
            k_ep = json.dumps(k_ep, sort_keys=True)
            if k_ep in seen_ep:
                c[v]["duplicate episode (kept the first)"] += 1
                continue
            seen_ep.add(k_ep)
            # calls_turns_mismatch: the RL wrapper wants one model call per turn; a checked turn makes several
            if r.get("rl_error") not in (None, "calls_turns_mismatch") or ep.get("crashed"):
                c[v]["episode crashed"] += 1
                continue
            if ep.get("feasible_start"):
                c[v]["episode starts feasible"] += 1
                continue
            c[v]["episodes"] += 1
            turns = ep.get("turns") or []
            if not turns:
                continue
            before = key(False, start_viol(v, turns[0].get("user")), ep["phi_start"])
            for t in turns:
                c[v]["turns"] += 1
                C = int(t.get("checks_max") or 0)
                if C <= 0 or "system" not in t:
                    c[v]["turn without checks_max or recorded prompt"] += 1
                    why, i = "no checks_max", None
                else:
                    why, i = judge(ep, t, before)
                    if a.keep_all and why not in ("no check",):
                        why, i = None, None             # plumbing test: every turn's prompts, whatever its outcome
                c[v]["drop: " + why if why else "kept"] += 1
                if why is None:
                    nck[v][len(t["checks"])] += 1
                    for k, (prefix, target) in enumerate(conversation(t, C)):
                        rows.append({
                            "domain": v, "instance": ep["problem_id"], "turn": int(t["t"]), "reply": k,
                            "n_checks": len(t["checks"]), "commit_check": i, "checks_max": C,
                            "kind": "commit" if k == len(t["checks"]) else "check",
                            "messages": prefix + [target], "tier": a.tier, "teacher": a.teacher,
                            "sid": "%s|%s|%s" % (v, ep["problem_id"], t["t"]),
                            "feasible_after": bool(t.get("feasible_after")), "phi_after": t.get("phi_after"),
                            "words": len((target.get("reasoning_content") or "").split()), "source": path})
                if not t.get("fail") and t.get("phi_after") is not None:
                    before = key(t.get("feasible_after"), viol_after(v, t), t["phi_after"])
    with open(a.out, "w", encoding="utf-8") as fo:
        for row in rows:
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = []
    for v in sorted(c):
        lines.append("%s: %s" % (v, ", ".join("%s %d" % kv for kv in sorted(c[v].items()))))
        lines.append("   kept turns by number of checks: %s" % dict(sorted(nck[v].items())))
    byk = collections.Counter((r["domain"], r["kind"]) for r in rows)
    lines.append("rows %d: %s" % (len(rows), dict(sorted(byk.items()))))
    if rows:
        w = sorted(r["words"] for r in rows)
        lines.append("reasoning words per row: median %d, p90 %d, max %d" % (w[len(w) // 2], w[int(len(w) * 0.9)], w[-1]))
    if a.verify_requests:
        logged = set()
        for line in open(a.verify_requests, encoding="utf-8"):
            q = json.loads(line)
            logged.add(json.dumps(q["messages"], sort_keys=True))
        bad = sum(1 for r in rows if json.dumps(r["messages"][:-1], sort_keys=True) not in logged)
        lines.append("verify: %d of %d row prompts are not a logged request" % (bad, len(rows)))
    txt = "\n".join(lines)
    print(txt)
    if a.report:
        open(a.report, "w", encoding="utf-8").write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
