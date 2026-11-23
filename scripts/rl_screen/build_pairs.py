"""Preference pairs for a preference round (llm_finetune/rl_dataxp/dpo_rows.py): per state, one reply that the
simulator bore out and one that it did not, both the policy's own replies sampled at that state.

    chosen    the reply build_rows.py picks under the same settings (--min-pct, --max-words, gain required):
              applied, raises the potential, not a no-op, the highest percentile among the state's legal moves
    rejected  one reply drawn (seeded by the state id) among the state's other replies that parsed and that the
              simulator refused (an illegal move: under the open rules the grid stays as it was and the turn is
              spent), or applied without raising the potential, or applied as a no-op; its move differs from the
              chosen move; the same text checks as the chosen reply (a rationale without think tags, at most
              --max-words words, the reply that was scored, not cut at the length limit)

Both replies are written as generated: reasoning_content = the think block, content = the answer, which must be
exactly one <action>...</action> (so the pair contrasts two of the model's own outputs; build_rows writes the
domain's formatting of the applied move instead, which a refused move does not have).

    python build_pairs.py --states states.jsonl --samples samples.jsonl --scored scored.jsonl --out pairs.jsonl \\
        [--min-pct 0] [--max-words 1000] [--tier pref1] [--seed 0]

The report prints, per family, how many states give a pair and why the others do not, and what the rejected
replies are (refused / no gain / no-op).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_rows as BR  # noqa: E402

ACTION_ONLY = re.compile(r"^\s*<action>[^<>]*</action>\s*$")


def reply_text_ok(x: dict, got, max_words: int):
    """None when the reply's text can stand in a pair, else why not. got: (thinking, sha1, prompt tag, answer)."""
    if got is None:
        return "no_text"
    reason, sha1, _tag, answer = got
    reason = reason.strip()
    if x.get("reply_sha1") != sha1:
        return "reply_changed_after_scoring"
    if not reason:
        return "no_rationale"
    if "<think>" in reason or "</think>" in reason:
        return "think_tag_in_rationale"
    if max_words and len(reason.split()) > max_words:
        return "too_long"
    if not ACTION_ONLY.match(answer or ""):
        return "answer_not_one_action"
    return None


def bad_kind(x: dict):
    """refused / no_gain / noop for a reply the simulator did not bear out, None for one it did (or that did not
    parse: no move to contrast)."""
    if not x.get("parsed"):
        return None
    if not x.get("ok"):
        return "refused" if str(x.get("reason") or "").startswith("illegal") else None
    if x.get("noop"):
        return "noop"
    if not x.get("gain"):
        return "no_gain"
    return None


def move_of(x: dict) -> str:
    return BR.norm_move(x.get("text") or x.get("pick_text") or "")


def assistant(got) -> dict:
    return {"role": "assistant", "reasoning_content": got[0].strip(), "content": got[3].strip()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pct", type=float, default=0.0)
    ap.add_argument("--max-words", type=int, default=1000)
    ap.add_argument("--tier", default="pref1")
    ap.add_argument("--teacher", default="dx38mtR-own-think-sim-pairs")
    ap.add_argument("--sample-prompt", default="served")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    states = [json.loads(line) for line in open(a.states, encoding="utf-8") if line.strip()]
    answers, answers_br = {}, {}
    for line in open(a.samples, encoding="utf-8"):
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if not o.get("error") and o.get("answer") and o.get("finish") != "length":
            key = (o["sid"], int(o["sample"]))
            answers[key] = (o.get("thinking") or "", o.get("sha1"), o.get("prompt"), o.get("answer") or "")
            answers_br[key] = answers[key][:3]
    scored = collections.defaultdict(list)
    state_rec = {}
    for line in open(a.scored, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("state"):
            state_rec[r["sid"]] = r
        else:
            scored[r["sid"]].append(r)

    fams = sorted({s["variant"] for s in states})
    per = {f: collections.Counter() for f in fams}
    n = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for s in states:
            c = per[s["variant"]]
            c["states"] += 1
            sc = scored.get(s["sid"], [])
            if s["sid"] not in state_rec or state_rec[s["sid"]].get("error"):
                c["not scored"] += 1
                continue
            c["scored"] += 1
            shown = BR.norm_move(s["system"] + s["observation"] + s["instruction"])
            x, _reason, why = BR.pick(sc, answers_br, a.min_pct, a.max_words, True, False, shown, False, a.sample_prompt)
            if x is None:
                c["no pair: no chosen reply (" + why + ")"] += 1
                continue
            gx = answers.get((x["sid"], int(x["sample"])))
            w = reply_text_ok(x, gx, a.max_words)
            if w is not None:
                c["no pair: chosen reply text " + w] += 1
                continue
            cands, cwhy = [], collections.Counter()
            for y in sorted(sc, key=lambda r: int(r["sample"])):
                kind = bad_kind(y)
                if kind is None:
                    continue
                if move_of(y) == move_of(x):
                    cwhy["same move as chosen"] += 1
                    continue
                gy = answers.get((y["sid"], int(y["sample"])))
                if a.sample_prompt and gy and gy[2] and gy[2] != a.sample_prompt:
                    cwhy["sampled_under_another_prompt"] += 1
                    continue
                wy = reply_text_ok(y, gy, a.max_words)
                if wy is not None:
                    cwhy[wy] += 1
                    continue
                cands.append((y, gy, kind))
            if not cands:
                top = cwhy.most_common(1)[0][0] if cwhy else "every other reply gained or did not parse"
                c["no pair: no rejected reply (mostly %s)" % top] += 1
                continue
            rng = random.Random(int(hashlib.sha1(("%d|%s" % (a.seed, s["sid"])).encode()).hexdigest()[:12], 16))
            y, gy, kind = cands[rng.randrange(len(cands))]
            base = BR.make_row_base(s, x, gx[0].strip(), a.tier, a.teacher, a.sample_prompt)
            pair = {k: v for k, v in base.items() if k != "messages"}
            pair.update({
                "messages": base["messages"][:2],
                "chosen": assistant(gx),
                "rejected": assistant(gy),
                "chosen_sample": int(x["sample"]), "rejected_sample": int(y["sample"]),
                "rejected_kind": kind, "rejected_reason": str(y.get("reason") or "")[:200] or None,
                "rejected_pct": y.get("pct"), "rejected_phi": y.get("phi"),
                "rejected_words": len(gy[0].split()), "n_rejected_candidates": len(cands),
                "chosen_move": x.get("text") or x.get("pick_text"), "rejected_move": y.get("text") or y.get("pick_text"),
            })
            fo.write(json.dumps(pair) + "\n")
            n += 1
            c["pairs"] += 1
            c["pairs: rejected " + kind] += 1
            c["pairs: chosen feasible after the move"] += bool(x.get("feasible_after"))
    keys = []
    for f in fams:
        for k in per[f]:
            if k not in keys:
                keys.append(k)
    print("pairs: chosen = build_rows pick (min pct %g, max words %s, gain required); rejected = a seeded draw among "
          "refused / no-gain / no-op replies with another move" % (a.min_pct, a.max_words or "none"))
    print("%-72s %s %9s" % ("", " ".join("%9s" % f[:9] for f in fams), "all"))
    for k in sorted(keys, key=lambda k: (not k.startswith(("states", "scored", "not scored")), k)):
        print("%-72s %s %9d" % (k[:72], " ".join("%9d" % per[f][k] for f in fams), sum(per[f][k] for f in fams)))
    print("BUILD_PAIRS_DONE %d pairs -> %s" % (n, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
