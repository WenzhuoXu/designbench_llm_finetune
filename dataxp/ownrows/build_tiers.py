"""Step 5: build the own / own_act tiers and their manifest.

Port of up_tools/build_own.py. Per state: among the replies that parsed,
applied, and carry a non-empty written reasoning of at most --max-words words
without a <think> tag, take the one with the highest Phi percentile (ties:
lowest sample index). Keep the state if that pick is at or above --min-pct
(90, the 90th percentile of the enumerated legal moves) or is the teacher's
move (same_move). A state with no such reply is dropped from BOTH arms, so the
two tiers are matched row for row.

  <tag>      assistant: reasoning_content = the model's own written reasoning,
             content = <action>format_move(pick)</action>
  <tag>_act  assistant: reasoning_content = "", same content

Rows are in the DX-26 / sftprobe tier format consumed by scripts/train_sft.py
with data.target_fn=assistant_after_think: messages = [system, user,
assistant{content, reasoning_content}], plus the metadata those tiers carry
(domain, instance, turn, turns, binding_constraint, n_prior_changes,
instruction_sha1, spelling, action, clamped, tier, teacher) and the selection readouts
(rationale_sample, own_pct, own_same_move, own_phi, gold_action, gold_phi).

Writes <out>/{train_,dev_in_}{<tag>,<tag>_act}.jsonl and <out>/manifest_<tag>.json.

    python -m dataxp.ownrows.build_tiers --states <dir>/states_phi.jsonl \
        --samples <dir>/samples.jsonl --scored <dir>/scored.jsonl --out <dir>/tiers \
        [--min-pct 90 --max-words 320 --tag own]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

from .common import read_jsonl, reasoning_of, served_messages

TEACHER = "qwen3.8-27b-own-nothink-phi-selected"


def pick_best(scored_for_sid, answers, min_pct, max_words, allow_noop):
    """(best record, reasoning, drop reason). best is None when the state is dropped."""
    best = None
    why = collections.Counter()
    for x in sorted(scored_for_sid, key=lambda r: int(r["sample"])):
        k = int(x["sample"])
        if not x.get("ok") or x.get("pct") is None:
            why["not_scored:" + str(x.get("reason") or "no_pct")] += 1
            continue
        if x.get("noop") and not allow_noop:
            why["noop"] += 1
            continue
        reason = reasoning_of(answers.get((x["sid"], k), ""))
        if not reason:
            why["no_reasoning"] += 1
            continue
        if len(reason.split()) > max_words:
            why["too_long"] += 1
            continue
        if "<think>" in reason or "</think>" in reason:
            why["think_leak"] += 1
            continue
        cand = (float(x["pct"]), -k, x, reason)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        return None, None, "no_usable_sample" if not why else "no_usable_sample(" + ",".join(
            "%s=%d" % kv for kv in sorted(why.items())) + ")"
    pct, _, x, reason = best
    if not (pct >= min_pct or x.get("same_move")):
        return None, None, "below_threshold"
    return x, reason, None


def make_row(state, x, reason, arm, tag, k):
    msgs = served_messages(state)
    msgs.append({"role": "assistant",
                 "reasoning_content": reason if arm == tag else "",
                 "content": "<action>%s</action>" % x["text"]})
    return {
        "domain": state["variant"], "instance": state["problem_id"], "turn": int(state["turn"]),
        "turns": 1, "binding_constraint": state.get("binding_constraint"),
        "n_prior_changes": int(state.get("n_prior_changes", len(state.get("history", [])))),
        "messages": msgs,
        "instruction_sha1": state["instruction_sha1"],
        "spelling": state.get("spelling"),
        "action": x["action"], "clamped": bool(x.get("clamped")),
        "tier": "ownrows_" + arm, "teacher": TEACHER,
        "rationale_sample": k, "sample_mode": x.get("mode"), "sample_prompt": x.get("prompt"),
        "own_pct": x["pct"], "own_same_move": bool(x.get("same_move")), "own_phi": x.get("phi"),
        "gold_action": state["gold"], "gold_phi": x.get("gold_phi"), "gold_pct": x.get("gold_pct"),
        "n_candidates": x.get("n_candidates"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pct", type=float, default=90.0)
    ap.add_argument("--max-words", type=int, default=320,
                    help="reasoning word cap (build_own default; own_cands used 900)")
    ap.add_argument("--tag", default="own", help="arms <tag> and <tag>_act")
    ap.add_argument("--allow-noop", action="store_true",
                    help="keep SET_PARAM picks that leave the value unchanged")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    states = read_jsonl(a.states)
    answers = {}
    with open(a.samples, encoding="utf-8") as fh:
        for l in fh:
            try:
                o = json.loads(l)
            except ValueError:
                continue
            if o.get("answer") and o.get("finish") != "length":
                answers[(o["sid"], int(o["sample"]))] = o["answer"]
    scored = collections.defaultdict(list)
    for r in read_jsonl(a.scored):
        scored[r["sid"]].append(r)

    report = {"min_pct": a.min_pct, "max_words": a.max_words, "tag": a.tag}
    arms = (a.tag, a.tag + "_act")
    for split, prefix in (("train", "train_"), ("dev", "dev_in_")):
        rows = [s for s in states if s.get("split", "train") == split]
        kept, dropped = [], collections.Counter()
        per = collections.defaultdict(lambda: [0, 0])
        pcts, same, words = [], [], []
        for s in rows:
            per[s["variant"]][1] += 1
            x, reason, why = pick_best(scored.get(s["sid"], []), answers, a.min_pct,
                                       a.max_words, a.allow_noop)
            if x is None:
                dropped[why.split("(")[0]] += 1
                continue
            per[s["variant"]][0] += 1
            pcts.append(float(x["pct"]))
            same.append(bool(x.get("same_move")))
            words.append(len(reason.split()))
            kept.append((s, x, reason))
        for arm in arms:
            path = os.path.join(a.out, "%s%s.jsonl" % (prefix, arm))
            with open(path, "w", encoding="utf-8") as fh:
                for s, x, reason in kept:
                    fh.write(json.dumps(make_row(s, x, reason, arm, a.tag, int(x["sample"]))) + "\n")
        words.sort()
        report[split] = {
            "states": len(rows), "kept": len(kept), "per_domain": dict(per),
            "dropped": dict(dropped),
            "mean_pct": sum(pcts) / max(len(pcts), 1),
            "same_move_share": sum(same) / max(len(same), 1),
            "reasoning_words_median": words[len(words) // 2] if words else None,
            "files": {arm: os.path.join(a.out, "%s%s.jsonl" % (prefix, arm)) for arm in arms},
        }
    with open(os.path.join(a.out, "manifest_%s.json" % a.tag), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
