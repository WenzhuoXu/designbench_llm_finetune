"""Step 4 of a screened supervised round: one training row per state whose best sampled reply
passes the screen, in the format of the dx38 rows (dataxp.ownrows.build_tiers.make_row, read by the
assistant_after_think target): messages = [system, user, assistant{reasoning_content, content}],
content = <action>the applied move, formatted by the domain</action>.

The screen, per state, over the replies that parsed, applied, are not a no-op and carry a rationale
without a think tag:
    eligible   the reply raises the potential (--no-require-gain drops this), and its rationale has
               at most --max-words words (0 = no cap, the default; the dx38 rows were cut at 320)
    pick       the eligible reply with the highest percentile among the state's legal moves
               (ties: lowest sample index), as build_tiers picks
    keep       the pick is at or above --min-pct (90, as the dx38 rows)
The report prints, per family, what each condition costs, and how many rows other settings of the
screen would keep, so no condition is applied unseen.

    python build_rows.py --states states.jsonl --samples samples.jsonl --scored scored.jsonl \\
        --out rows.jsonl [--min-pct 90] [--max-words 0] [--no-require-gain] [--tier scr1] [--teacher NAME]
"""
import argparse
import collections
import json
import sys


def reasoning_of(answer: str) -> str:
    """dataxp.ownrows.common.reasoning_of: the reply before its last <action>."""
    a = answer or ""
    i = a.rfind("<action>")
    return (a if i < 0 else a[:i]).strip()


def usable(x: dict, reason: str) -> str | None:
    """None when the reply can be a row target, else why not."""
    if not x.get("ok") or x.get("pct") is None:
        return "not_applied:" + str(x.get("reason") or "no_pct").split(":")[0]
    if x.get("noop"):
        return "noop"
    if not reason:
        return "no_rationale"
    if "<think>" in reason or "</think>" in reason:
        return "think_tag_in_rationale"
    return None


def pick(scored: list, answers: dict, min_pct: float, max_words: int, require_gain: bool):
    """(reply record, rationale, None) or (None, None, why the state gives no row)."""
    why = collections.Counter()
    best = None
    for x in sorted(scored, key=lambda r: int(r["sample"])):
        reason = reasoning_of(answers.get((x["sid"], int(x["sample"])), ""))
        w = usable(x, reason)
        if w is None and require_gain and not x.get("gain"):
            w = "no_gain"
        if w is None and max_words and len(reason.split()) > max_words:
            w = "too_long"
        if w is not None:
            why[w] += 1
            continue
        cand = (float(x["pct"]), -int(x["sample"]), x, reason)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        top = why.most_common(1)[0][0] if why else "no_reply"
        return None, None, "no_eligible_reply (mostly %s)" % top
    if best[0] < min_pct:
        return None, None, "below_min_pct"
    return best[2], best[3], None


def make_row(state: dict, x: dict, reason: str, tier: str, teacher: str) -> dict:
    return {
        "domain": state["variant"], "instance": state["problem_id"], "turn": int(state["turn"]), "turns": 1,
        "binding_constraint": state.get("binding_constraint"), "n_prior_changes": int(state["n_prior_changes"]),
        "messages": [{"role": "system", "content": state["system"]},
                     {"role": "user", "content": state["observation"] + state["instruction"]},
                     {"role": "assistant", "reasoning_content": reason, "content": "<action>%s</action>" % x["text"]}],
        "instruction_sha1": state.get("instruction_sha1"), "spelling": state.get("spelling"),
        "action": x["action"], "clamped": bool(x.get("clamped")), "tier": tier, "teacher": teacher,
        "rationale_sample": int(x["sample"]), "sample_mode": "own_think", "sample_prompt": "served",
        "own_pct": x["pct"], "own_phi": x["phi"], "n_candidates": x.get("n_candidates"),
        "sid": state["sid"], "phi_before": state.get("phi_before"), "feasible_after": bool(x.get("feasible_after")),
        "same_as_own": bool(x.get("same_as_own")), "source": state.get("source"), "words": len(reason.split()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pct", type=float, default=90.0)
    ap.add_argument("--max-words", type=int, default=0)
    ap.add_argument("--no-require-gain", action="store_true")
    ap.add_argument("--tier", default="scr1")
    ap.add_argument("--teacher", default="dx38mtR-own-think-phi-screened")
    a = ap.parse_args(argv)

    states = [json.loads(line) for line in open(a.states, encoding="utf-8") if line.strip()]
    answers = {}
    for line in open(a.samples, encoding="utf-8"):
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("answer") and o.get("finish") != "length":
            answers[(o["sid"], int(o["sample"]))] = o["answer"]
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
    alt = {f: collections.Counter() for f in fams}
    settings = (("pct>=%g, gain, no word cap" % a.min_pct, a.min_pct, 0, True),
                ("pct>=%g, gain, <=320 words" % a.min_pct, a.min_pct, 320, True),
                ("pct>=%g only" % a.min_pct, a.min_pct, 0, False),
                ("gain only (best of k)", 0.0, 0, True),
                ("pct>=50, gain", 50.0, 0, True))
    words, pcts = collections.defaultdict(list), collections.defaultdict(list)
    n_rows = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for s in states:
            c = per[s["variant"]]
            c["states"] += 1
            sc = scored.get(s["sid"], [])
            if s["sid"] not in state_rec or state_rec[s["sid"]].get("error"):
                c["not scored"] += 1
                continue
            c["scored"] += 1
            c["replies"] += len(sc)
            c["replies applied"] += sum(1 for x in sc if x.get("ok"))
            c["replies applied with gain"] += sum(1 for x in sc if x.get("ok") and x.get("gain"))
            c["states with an applied reply"] += any(x.get("ok") for x in sc)
            c["states with a gaining reply"] += any(x.get("ok") and x.get("gain") for x in sc)
            for name, mp, mw, rg in settings:
                alt[s["variant"]][name] += pick(sc, answers, mp, mw, rg)[0] is not None
            x, reason, why = pick(sc, answers, a.min_pct, a.max_words, not a.no_require_gain)
            if x is None:
                c["dropped: " + why] += 1
                continue
            row = make_row(s, x, reason, a.tier, a.teacher)
            fo.write(json.dumps(row) + "\n")
            n_rows += 1
            c["rows"] += 1
            c["rows: the policy's own move at this state"] += row["same_as_own"]
            c["rows: design feasible after the move"] += row["feasible_after"]
            c["rows: rationale over 320 words"] += row["words"] > 320
            words[s["variant"]].append(row["words"])
            pcts[s["variant"]].append(row["own_pct"])
    keys = []
    for f in fams:
        for k in per[f]:
            if k not in keys:
                keys.append(k)
    print("screen: min pct %g, max words %s, gain required %s" % (a.min_pct, a.max_words or "none", not a.no_require_gain))
    print("%-52s %s %9s" % ("", " ".join("%9s" % f[:9] for f in fams), "all"))
    for k in keys:
        print("%-52s %s %9d" % (k, " ".join("%9d" % per[f][k] for f in fams), sum(per[f][k] for f in fams)))
    print("%-52s %s" % ("rationale words, median", " ".join("%9d" % (sorted(words[f])[len(words[f]) // 2] if words[f] else 0) for f in fams)))
    print("%-52s %s" % ("percentile of the pick, mean", " ".join("%9.1f" % (sum(pcts[f]) / len(pcts[f]) if pcts[f] else 0) for f in fams)))
    print("\nrows other screens would keep:")
    for name, *_ in settings:
        print("%-52s %s %9d" % (name, " ".join("%9d" % alt[f][name] for f in fams), sum(alt[f][name] for f in fams)))
    print("BUILD_ROWS_DONE %d rows -> %s" % (n_rows, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
