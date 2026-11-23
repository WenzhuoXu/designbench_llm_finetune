"""Step 4 of a screened supervised round: one training row per state whose best sampled reply
passes the screen, in the format of the dx38 rows (dataxp.ownrows.build_tiers.make_row, read by the
assistant_after_think target): messages = [system, user, assistant{reasoning_content, content}],
content = <action>the applied move, formatted by the domain</action>.

The screen, per state, over the replies that parsed, applied, are not a no-op and carry a rationale
without a think tag:
    eligible   the reply raises the potential (--no-require-gain drops this), its rationale has
               at most --max-words words (0 = no cap, the default; the dx38 rows were cut at 320),
               and, with --require-consistent, the last move its rationale writes out is its own
               move (a rationale that settles on FILL_RECT(1, 2, 3, 4) and then emits another move
               is not a target; a rationale in prose is not judged)
    pick       the eligible reply with the highest percentile among the state's legal moves
               (ties: lowest sample index), as build_tiers picks
    keep       the pick is at or above --min-pct (90, as the dx38 rows)
The report prints, per family, what each condition costs, and how many rows other settings of the
screen would keep, so no condition is applied unseen.

    python build_rows.py --states states.jsonl --samples samples.jsonl --scored scored.jsonl \\
        --out rows.jsonl [--min-pct 90] [--max-words 0] [--no-require-gain] [--require-consistent]
        [--require-forecasts --sample-prompt served+forecast_v1] [--tier scr1] [--teacher NAME]

--require-forecasts (a forecast-checked round, forecasts.py): the replies were sampled with a
collection-only instruction to compare two or three candidate moves in the rationale, one
"FORECAST: MOVE -> refused | worse | better" line each. A reply is eligible only when it has two
or three such lines, every candidate is a move of the domain and changes the design, its own move
is one of them, the simulator returned what every line says, and no rejected candidate does better
than its own move. Among a state's eligible replies, one that rejects a candidate it rightly
forecast as refused or worse goes first. The row keeps the state's served prompt: the instruction
is not in it. The report adds what each of these conditions costs and every forecast against the
simulator.
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecasts  # noqa: E402


def usable(x: dict, reason: str, sha1=None) -> str | None:
    """None when the reply can be a row target, else why not. sha1: of the reply now in the samples
    file; a reply that is not the one that was scored (the state was sampled again) is not used."""
    if not x.get("ok") or x.get("pct") is None:
        return "not_applied:" + str(x.get("reason") or "no_pct").split(":")[0]
    if x.get("reply_sha1") != sha1:
        return "reply_changed_after_scoring"
    if x.get("noop"):
        return "noop"
    if not reason:
        return "no_rationale"
    if "<think>" in reason or "</think>" in reason:
        return "think_tag_in_rationale"
    return None


MOVE = re.compile(r"\b([A-Z][A-Z_]{2,})\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)")


def norm_move(text: str) -> str:
    return re.sub(r"[\s\"']+", "", text or "").upper()


def ends_on_another_move(reason: str, x: dict, shown: str = "") -> bool:
    """True when the last move the rationale writes out (VERB(arguments)) is not the reply's move.
    Not counted as written moves: a move the prompt itself lists (`shown` = the prompt, through
    norm_move: the history, a signature), and an abbreviated one (VERB(1, ...)). A rationale in
    prose, with no move written out, is not judged (False)."""
    own = {norm_move(x.get("text")), norm_move(x.get("pick_text"))} - {""}
    last = None
    for verb, args in MOVE.findall(reason):
        m = norm_move("%s(%s)" % (verb, args))
        if "..." in m or "…" in m:
            continue
        if m in own or m not in shown:
            last = m
    return last is not None and last not in own


def pick(scored: list, replies: dict, min_pct: float, max_words: int, require_gain: bool,
         consistent: bool = False, shown: str = "", forecast: bool = False, sample_prompt=None):
    """(reply record, rationale, None) or (None, None, why the state gives no row). replies:
    (sid, sample) -> (rationale = the think block, sha1 of the reply). consistent: a reply whose
    rationale ends on another move than its own (ends_on_another_move) is not eligible. forecast:
    a reply is eligible only when the simulator bears out every FORECAST line of its rationale
    (forecasts.fault); among the eligible, one that rejects a candidate it rightly forecast as
    refused or worse goes before one whose candidates are all better; --min-pct is then a
    condition on every reply, not on the winner. sample_prompt: a reply recorded as sampled under
    another prompt is not used."""
    why = collections.Counter()
    best = None
    for x in sorted(scored, key=lambda r: int(r["sample"])):
        got = replies.get((x["sid"], int(x["sample"])), ("", None))
        reason, sha1, tag = got[0].strip(), got[1], (got[2] if len(got) > 2 else None)
        w = usable(x, reason, sha1)
        if w is None and sample_prompt and tag and tag != sample_prompt:
            w = "sampled_under_another_prompt"
        if w is None and require_gain and not x.get("gain"):
            w = "no_gain"
        if w is None and max_words and len(reason.split()) > max_words:
            w = "too_long"
        if w is None and consistent and ends_on_another_move(reason, x, shown):
            w = "rationale_ends_on_another_move"
        if w is None and forecast:
            f = reply_fault(x, reason)
            if f is not None:
                w = "forecasts: " + f
            elif float(x["pct"]) < min_pct:
                w = "below_min_pct"
        if w is not None:
            why[w] += 1
            continue
        # open_rank (photonics under the open rules, score_states.py): fewer rule-breaking pixels, then Phi; it breaks
        # ties of the percentile (the same for every reply of a state elsewhere)
        cand = (int(forecast and forecasts.has_negative(x.get("forecasts"))), float(x["pct"]), tuple(x.get("open_rank") or ()),
                -int(x["sample"]), x, reason)
        if best is None or cand[:4] > best[:4]:
            best = cand
    if best is None:
        top = why.most_common(1)[0][0] if why else "no_reply"
        return None, None, "no_eligible_reply (mostly %s)" % top
    if best[1] < min_pct:
        return None, None, "below_min_pct"
    return best[4], best[5], None


def reply_fault(x: dict, reason: str):
    """None when the forecasts of a scored reply make it a row target (forecasts.fault), else why
    not. The rationale is read again: lines that read differently now than when the reply was
    scored are not trusted. A move the domain changed on apply (a clamped value) is not a target
    either: the row's action would not be the candidate the rationale chose."""
    fc = x.get("forecasts") or []
    if x.get("forecast_error"):
        return "the forecast check raised an error"
    lines, unread = forecasts.read(reason)
    if len(lines) <= forecasts.MAX_SCORED and [(c.get("move"), c.get("said")) for c in fc] != lines:
        return "its lines read differently now than when scored"
    f = forecasts.fault(fc, unread)
    if f is None and x.get("text") != x.get("pick_text"):
        f = "the move made was changed on apply"
    return f


def forecast_funnel(sc: list, replies: dict, c: collections.Counter, table: collections.Counter) -> None:
    """What the forecast screen costs, reply by reply (c), and every parsed forecast against the
    simulator (table: (said, truth) -> lines)."""
    for x in sc:
        fc = x.get("forecasts") or []
        for f in fc:
            if f.get("parsed"):
                table[(f["said"], f["truth"])] += 1
        if not (fc or x.get("forecast_unread") or x.get("forecast_error")):
            c["fc reply: no forecast line"] += 1
            continue
        c["fc replies with a forecast line"] += 1
        fault = reply_fault(x, replies.get((x["sid"], int(x["sample"])), ("", None))[0].strip())
        if fault is None and not (x.get("ok") and x.get("gain")):
            fault = "the reply's own move is not applied with a gain"
        c["fc reply: " + (fault or "every forecast right, chosen move best (eligible)")] += 1


def make_row(state: dict, x: dict, reason: str, tier: str, teacher: str, sample_prompt: str = "served") -> dict:
    extra = {}
    if x.get("forecasts"):
        extra = {"forecasts": [[f["move"], f["said"]] for f in x["forecasts"]],
                 "has_negative_forecast": forecasts.has_negative(x["forecasts"])}
    return dict(make_row_base(state, x, reason, tier, teacher, sample_prompt), **extra)


def make_row_base(state: dict, x: dict, reason: str, tier: str, teacher: str, sample_prompt: str = "served") -> dict:
    return {
        "domain": state["variant"], "instance": state["problem_id"], "turn": int(state["turn"]), "turns": 1,
        "binding_constraint": state.get("binding_constraint"), "n_prior_changes": int(state["n_prior_changes"]),
        "messages": [{"role": "system", "content": state["system"]},
                     {"role": "user", "content": state["observation"] + state["instruction"]},
                     {"role": "assistant", "reasoning_content": reason, "content": "<action>%s</action>" % x["text"]}],
        "instruction_sha1": state.get("instruction_sha1"), "spelling": state.get("spelling"),
        "action": x["action"], "clamped": bool(x.get("clamped")), "tier": tier, "teacher": teacher,
        "rationale_sample": int(x["sample"]), "sample_mode": "own_think", "sample_prompt": sample_prompt,
        "own_pct": x["pct"], "own_phi": x["phi"], "n_candidates": x.get("n_candidates"),
        "sid": state["sid"], "phi_before": state.get("phi_before"), "feasible_after": bool(x.get("feasible_after")),
        "same_as_own": bool(x.get("same_as_own")), "source": state.get("source"), "words": len(reason.split()),
        "action_tag_in_rationale": "<action>" in reason,
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
    ap.add_argument("--require-consistent", action="store_true",
                    help="a reply whose rationale ends on another move than its own is not eligible")
    ap.add_argument("--require-forecasts", action="store_true",
                    help="a reply is eligible only when the simulator bears out every FORECAST line of its rationale (forecasts.py)")
    ap.add_argument("--sample-prompt", default="served", help="recorded in every row: the prompt the replies were sampled under")
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
        if not o.get("error") and o.get("answer") and o.get("finish") != "length":
            answers[(o["sid"], int(o["sample"]))] = (o.get("thinking") or "", o.get("sha1"), o.get("prompt"))
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
    settings = [("pct>=%g, gain, no word cap" % a.min_pct, a.min_pct, 0, True, False, False),
                ("pct>=%g, gain, <=320 words" % a.min_pct, a.min_pct, 320, True, False, False),
                ("pct>=%g, gain, <=1000 words" % a.min_pct, a.min_pct, 1000, True, False, False),
                ("pct>=%g, gain, rationale ends on its own move" % a.min_pct, a.min_pct, 0, True, True, False),
                ("pct>=%g only" % a.min_pct, a.min_pct, 0, False, False, False),
                ("gain only (best of k)", 0.0, 0, True, False, False),
                ("pct>=50, gain", 50.0, 0, True, False, False)]
    if a.require_forecasts:
        settings += [("forecasts right, gain, no word cap", 0.0, 0, True, False, True),
                     ("forecasts right, gain, <=1000 words", 0.0, 1000, True, False, True),
                     ("forecasts right, gain, <=320 words", 0.0, 320, True, False, True),
                     ("forecasts right, gain, pct>=90", 90.0, 0, True, False, True)]
    words, pcts = collections.defaultdict(list), collections.defaultdict(list)
    fc_table = {f: collections.Counter() for f in fams}
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
            shown = norm_move(s["system"] + s["observation"] + s["instruction"])
            if a.require_forecasts:
                forecast_funnel(sc, answers, c, fc_table[s["variant"]])
            for name, mp, mw, rg, cons, fc in settings:
                alt[s["variant"]][name] += pick(sc, answers, mp, mw, rg, cons, shown, fc, a.sample_prompt)[0] is not None
            x, reason, why = pick(sc, answers, a.min_pct, a.max_words, not a.no_require_gain, a.require_consistent, shown,
                                  a.require_forecasts, a.sample_prompt)
            if x is None:
                c["dropped: " + why] += 1
                continue
            row = make_row(s, x, reason, a.tier, a.teacher, a.sample_prompt)
            c["rows: the rationale ends on another move"] += ends_on_another_move(reason, x, shown)
            if a.require_forecasts:
                c["rows: a rejected candidate rightly forecast as refused or worse"] += bool(row.get("has_negative_forecast"))
                c["rows: forecast lines"] += len(row.get("forecasts") or [])
            fo.write(json.dumps(row) + "\n")
            n_rows += 1
            c["rows"] += 1
            c["rows: the policy's own move at this state"] += row["same_as_own"]
            c["rows: design feasible after the move"] += row["feasible_after"]
            c["rows: rationale over 320 words"] += row["words"] > 320
            c["rows: an <action> line drafted inside the rationale"] += row["action_tag_in_rationale"]
            words[s["variant"]].append(row["words"])
            pcts[s["variant"]].append(row["own_pct"])
    keys = []
    for f in fams:
        for k in per[f]:
            if k not in keys:
                keys.append(k)
    print("screen: min pct %g, max words %s, gain required %s, rationale must end on its own move %s, forecasts must be right %s" % (
        a.min_pct, a.max_words or "none", not a.no_require_gain, a.require_consistent, a.require_forecasts))
    print("%-52s %s %9s" % ("", " ".join("%9s" % f[:9] for f in fams), "all"))
    for k in keys:
        print("%-52s %s %9d" % (k, " ".join("%9d" % per[f][k] for f in fams), sum(per[f][k] for f in fams)))
    print("%-52s %s" % ("rationale words, median", " ".join("%9d" % (sorted(words[f])[len(words[f]) // 2] if words[f] else 0) for f in fams)))
    print("%-52s %s" % ("percentile of the pick, mean", " ".join("%9.1f" % (sum(pcts[f]) / len(pcts[f]) if pcts[f] else 0) for f in fams)))
    print("\nrows other screens would keep:")
    for name, *_ in settings:
        print("%-52s %s %9d" % (name, " ".join("%9d" % alt[f][name] for f in fams), sum(alt[f][name] for f in fams)))
    if a.require_forecasts:
        print("\nforecast lines against the simulator (every candidate that is a move; said -> returned):")
        for said in forecasts.VERDICTS:
            for truth in forecasts.VERDICTS:
                print("%-52s %s %9d" % ("said %s, simulator %s" % (said, truth),
                                        " ".join("%9d" % fc_table[f][(said, truth)] for f in fams),
                                        sum(fc_table[f][(said, truth)] for f in fams)))
        right = {f: sum(n for (s_, t_), n in fc_table[f].items() if s_ == t_) for f in fams}
        total = {f: sum(fc_table[f].values()) for f in fams}
        print("%-52s %s %9.3f" % ("share of forecasts right",
                                  " ".join("%9.3f" % (right[f] / total[f] if total[f] else 0.0) for f in fams),
                                  sum(right.values()) / max(1, sum(total.values()))))
    print("BUILD_ROWS_DONE %d rows -> %s" % (n_rows, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
