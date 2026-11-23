"""The screened supervised round's data steps (scripts/rl_screen/): the parts that need neither a
simulator nor a model."""
import importlib.util
import json
import os

import pytest

HERE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "rl_screen")


def load(name):
    spec = importlib.util.spec_from_file_location("rl_screen_" + name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PS, SS, BR, MX = load("policy_states"), load("sample_states"), load("build_rows"), load("mix_rows")


def test_applied_moves_and_ended_states():
    turns = [{"fail": "illegal:X breaks a rule", "action": {"verb": "X"}},          # refused at the start state
             {"fail": None, "action": {"verb": "A"}},                               # applied
             {"fail": "no_action"},
             {"fail": None, "action": {"verb": "B"}},                               # applied: ends the episode
             {"fail": "illegal:the episode has ended (power flow diverged)", "action": {"verb": "C"}}]
    moves, before, ended = PS.applied_moves(turns)
    assert moves == [{"verb": "A"}, {"verb": "B"}]
    assert before == [0, 0, 1, 1, 2]
    assert ended == {2}


def test_state_id_names_the_problem_and_the_moves():
    a = PS.state_id("fam", "p1", [{"verb": "A", "args": {"x": 1}}])
    assert a == PS.state_id("fam", "p1", [{"args": {"x": 1}, "verb": "A"}])        # key order does not matter
    assert a != PS.state_id("fam", "p1", [{"verb": "A", "args": {"x": 2}}])
    assert a != PS.state_id("fam", "p2", [{"verb": "A", "args": {"x": 1}}])
    assert a.startswith("fam|p1|1-") and PS.state_id("fam", "p1", []).startswith("fam|p1|0-")


def test_request_file_beside_a_rollout_file(tmp_path):
    for name in ("step_00003.compact.jsonl", "step_00004.jsonl", "x.episodes.jsonl"):
        (tmp_path / name).write_text("")
    (tmp_path / "step_00003.request.json").write_text("{}")
    (tmp_path / "step_00004.request.json").write_text("{}")
    (tmp_path / "x.episodes.jsonl.request.json").write_text("{}")
    assert PS.request_of(str(tmp_path / "step_00003.compact.jsonl")).endswith("step_00003.request.json")
    assert PS.request_of(str(tmp_path / "step_00004.jsonl")).endswith("step_00004.request.json")
    assert PS.request_of(str(tmp_path / "x.episodes.jsonl")).endswith("x.episodes.jsonl.request.json")
    (tmp_path / "lonely.jsonl").write_text("")
    with pytest.raises(SystemExit):
        PS.request_of(str(tmp_path / "lonely.jsonl"))


def test_canon_ignores_the_problem_name():
    assert PS.canon({"problem_id": "a", "index": 3, "x": [1, 2]}) == PS.canon({"problem_id": "b", "x": [1, 2]})
    assert PS.canon({"x": [1, 2]}) != PS.canon({"x": [1, 3]})


def test_split_reply_reads_both_server_forms():
    assert SS.split_reply({"content": "\n\n<action>A</action>", "reasoning_content": " why \n"}) == ("why", "<action>A</action>", True)
    assert SS.split_reply({"content": "<think>why\n</think>\n\n<action>A</action>"}) == ("why", "<action>A</action>", True)
    assert SS.split_reply({"content": "still thinking"})[2] is False               # the think block never closed


def _reply(content, reasoning=None, finish="stop", cached=None):
    msg = {"content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    usage = {"completion_tokens": 30, "prompt_tokens": 100}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"usage": usage, "choices": [{"index": 0, "finish_reason": finish, "message": msg}]}


def test_record_of_one_reply():
    r = SS.record_of("s", 3, _reply("\n\n<action>B</action>", " r1 <action>draft</action>\n", cached=96))
    assert (r["sid"], r["sample"], r["thinking"], r["answer"]) == ("s", 3, "r1 <action>draft</action>", "<action>B</action>")
    assert r["tokens"] == 30 and r["cached_tokens"] == 96 and r["sha1"] == SS.reply_sha1(r["thinking"], r["answer"])
    cut = SS.record_of("s", 0, _reply("", "cut", finish="length"))
    assert cut["answer"] == "" and cut["thinking"] == "" and cut["finish"] == "length" and cut["sha1"] is None
    assert SS.record_of("s", 0, _reply("no closing tag"))["finish"] == "length"     # the think block never closed


class _Args:
    servers, model, temperature, top_p, top_k, seed, timeout, attempts, retry_wait, k = ["u0", "u1"], "m", 0.8, 1.0, 0, 0, 1.0, 2, 0.0, 4


def test_a_state_asks_its_first_reply_alone_and_retries_with_another_seed():
    state = {"sid": "s", "system": "S", "observation": "U", "instruction": ""}
    calls, lock_order = [], []

    def post(url, body, timeout):
        calls.append((url, body["seed"]))
        if len(calls) == 1:
            lock_order.append("first")
        if body["seed"] == SS.seed_of("s", 2, 0, 0):                # sample 2 fails on its first try
            raise OSError("boom")
        return _reply("<action>A</action>", "why")

    recs = SS.draw_state(state, [0, 1, 2, 3], 0, _Args, post)
    assert [r["sample"] for r in recs] == [0, 1, 2, 3] and not any(r.get("error") for r in recs)
    assert calls[0] == ("u0", SS.seed_of("s", 0, 0, 0)) and lock_order == ["first"]
    assert ("u1", SS.seed_of("s", 2, 1, 0)) in calls                # the retry: next server, another seed
    assert len({s for _, s in calls}) == 5                          # every request has its own seed

    def dead(url, body, timeout):
        raise OSError("gone")
    assert SS.draw(state, 1, 0, _Args, dead) == {"sid": "s", "sample": 1, "error": "OSError: gone"}


def test_resume_reads_replies_and_closes_a_cut_line(tmp_path):
    p = tmp_path / "samples.jsonl"
    p.write_text(json.dumps({"sid": "a", "sample": 0, "answer": "x"}) + "\n" + json.dumps({"sid": "a", "sample": 1, "error": "e"}) + "\n"
                 + '{"sid": "b", "sample": 0, "ans')            # a killed run's last line
    assert SS.done_replies(str(p)) == {("a", 0)}
    with SS.open_append(str(p)) as fo:
        fo.write(json.dumps({"sid": "b", "sample": 0, "answer": "y"}) + "\n")
    assert SS.done_replies(str(p)) == {("a", 0), ("b", 0)}
    assert SS.done_replies(str(tmp_path / "none.jsonl")) == set()


def _scored(sample, pct, gain=True, ok=True, noop=False, reason=None):
    return {"sid": "s", "sample": sample, "ok": ok, "pct": pct if ok else None, "gain": gain, "noop": noop,
            "reason": reason, "text": "M(%d)" % sample, "action": {"verb": "M"}, "phi": float(pct or 0),
            "reply_sha1": "h%d" % sample}


def test_pick_is_the_best_eligible_reply():
    ans = {("s", k): ("why %d" % k, "h%d" % k) for k in range(5)}
    sc = [_scored(0, 95, gain=False), _scored(1, 80), _scored(2, 80), _scored(3, 0, ok=False, reason="illegal:x"), _scored(4, 99, noop=True)]
    x, reason, why = BR.pick(sc, ans, 0.0, 0, True)
    assert x["sample"] == 1 and reason == "why 1" and why is None                  # ties: lowest sample index
    assert BR.pick(sc, ans, 90.0, 0, True)[2] == "below_min_pct"
    assert BR.pick(sc, ans, 90.0, 0, False)[0]["sample"] == 0                      # without the gain condition
    assert BR.pick(sc, ans, 0.0, 1, True)[2].startswith("no_eligible_reply (mostly too_long")
    assert BR.pick([_scored(0, 0, ok=False, reason="illegal:x")], ans, 0.0, 0, True)[2] == "no_eligible_reply (mostly not_applied:illegal)"
    assert BR.pick([], ans, 0.0, 0, True)[2] == "no_eligible_reply (mostly no_reply)"
    leak = dict(ans)
    leak[("s", 1)] = ("a <think> tag", "h1")
    assert BR.pick(sc, leak, 0.0, 0, True)[0]["sample"] == 2
    # the state was sampled again after it was scored: the reply now in the file is not the scored one
    again = dict(ans)
    again[("s", 1)] = ("another rationale", "other")
    assert BR.pick(sc, again, 0.0, 0, True)[0]["sample"] == 2
    assert BR.pick([_scored(1, 80)], again, 0.0, 0, True)[2] == "no_eligible_reply (mostly reply_changed_after_scoring)"


def test_a_rationale_that_ends_on_another_move():
    x = {"text": "FILL_RECT(1, 2, 3, 4)", "pick_text": "FILL_RECT(1,2,3,4)"}
    end = BR.ends_on_another_move
    assert not end("Thicken the left strut by one cell.", x)                               # prose: not judged
    assert not end("ERODE(0, 1) hurt. So FILL_RECT(1, 2, 3, 4) it is.", x)
    assert not end('I will fill_rect it: FILL_RECT( 1,2 ,3, 4 )', x)                       # spacing does not matter
    assert end("FILL_RECT(1, 2, 3, 4) is safe, but ERODE(0, 1) gains more.", x)
    assert end("Best: FILL_RECT(1, 2, 3, 9).", x)                                          # same verb, other arguments
    shown = BR.norm_move("history: ERODE(0, 1)  signature: FILL_RECT(x0, y0, x1, y1)")
    assert not end("FILL_RECT(1, 2, 3, 4), undoing the earlier ERODE(0, 1).", x, shown)    # a move the prompt lists
    assert not end("FILL_RECT(1, 2, 3, 4), i.e. FILL_RECT(x0, y0, x1, y1).", x, shown)
    assert not end("FILL_RECT(1, 2, 3, 4) rather than SPLIT_SUB(1, ...).", x)              # abbreviated
    clamped = {"text": "SCALE(2, 0.5)", "pick_text": "SCALE(2, 0.1)"}                      # the move as written counts too
    assert not end("So SCALE(2, 0.1).", clamped) and not end("So SCALE(2, 0.5).", clamped)
    # in pick: the reply is passed over, and the next best one is taken
    ans = {("s", 1): ("So MOVE(7).", "h1"), ("s", 2): ("So MOVE(2).", "h2")}
    sc = [dict(_scored(1, 90), text="MOVE(1)"), dict(_scored(2, 80), text="MOVE(2)")]
    assert BR.pick(sc, ans, 0.0, 0, True)[0]["sample"] == 1
    assert BR.pick(sc, ans, 0.0, 0, True, True)[0]["sample"] == 2
    assert BR.pick(sc[:1], ans, 0.0, 0, True, True)[2] == "no_eligible_reply (mostly rationale_ends_on_another_move)"


def test_make_row_is_a_dx38_row():
    state = {"variant": "fam", "problem_id": "p", "turn": 2, "n_prior_changes": 2, "system": "S", "observation": "U",
             "instruction": "", "sid": "fam|p|2-ab", "phi_before": -3.0, "source": "k"}
    row = BR.make_row(state, dict(_scored(1, 97.5), feasible_after=True), "why", "scr1_gain", "teacher")
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert row["messages"][2] == {"role": "assistant", "reasoning_content": "why", "content": "<action>M(1)</action>"}
    assert (row["domain"], row["instance"], row["turn"], row["turns"], row["tier"]) == ("fam", "p", 2, 1, "scr1_gain")
    assert row["own_pct"] == 97.5 and row["feasible_after"] is True and row["words"] == 1


def test_mix_keeps_problems_whole_and_counts_everything(tmp_path):
    new = [{"domain": "f%d" % (i % 2), "instance": "p%02d" % (i // 3), "messages": [], "tier": "new", "i": i} for i in range(120)]
    rep = [{"domain": "g", "instance": "q%d" % i, "messages": [], "tier": "old"} for i in range(500)]
    (tmp_path / "new.jsonl").write_text("".join(json.dumps(r) + "\n" for r in new))
    (tmp_path / "rep.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rep))
    args = ["--new", str(tmp_path / "new.jsonl"), "--replay", str(tmp_path / "rep.jsonl"), "--out-train", str(tmp_path / "train.jsonl"),
            "--out-dev", str(tmp_path / "dev.jsonl"), "--replay-ratio", "0.5", "--dev-frac", "0.1", "--dev-max", "0"]
    assert MX.main(args) == 0
    train = [json.loads(line) for line in open(tmp_path / "train.jsonl")]
    dev = [json.loads(line) for line in open(tmp_path / "dev.jsonl")]
    tn = [r for r in train if r["tier"] == "new"]
    assert len(tn) + len(dev) == 120 and len(dev) > 0
    assert not {(r["domain"], r["instance"]) for r in tn} & {(r["domain"], r["instance"]) for r in dev}
    assert sum(r["tier"] == "old" for r in train) == round(len(tn) * 0.5)
    man = json.load(open(str(tmp_path / "train.jsonl") + ".manifest.json"))
    assert man["train rows"] == len(train) and man["dev rows"] == len(dev)
    first = open(tmp_path / "train.jsonl").read()
    assert MX.main(args) == 0 and open(tmp_path / "train.jsonl").read() == first   # deterministic


def test_mix_drops_repeats_and_replay_rows_of_dev_problems(tmp_path):
    new = [{"domain": "f", "instance": "p%02d" % (i // 2), "messages": [], "tier": "new", "sid": "s%d" % i} for i in range(40)]
    rep = [{"domain": "f", "instance": "p%02d" % i, "messages": [], "tier": "old"} for i in range(20)]
    for name, rows in (("a.jsonl", new), ("b.jsonl", new[:10]), ("rep.jsonl", rep)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert MX.main(["--new", str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl"), "--replay", str(tmp_path / "rep.jsonl"),
                    "--out-train", str(tmp_path / "train.jsonl"), "--out-dev", str(tmp_path / "dev.jsonl"),
                    "--replay-ratio", "10", "--dev-frac", "0.25", "--dev-max", "0"]) == 0
    train = [json.loads(line) for line in open(tmp_path / "train.jsonl")]
    dev = [json.loads(line) for line in open(tmp_path / "dev.jsonl")]
    assert len([r for r in train if r["tier"] == "new"]) + len(dev) == 40           # the 10 repeated rows count once
    dev_p = {r["instance"] for r in dev}
    assert len(dev_p) == 5 and not dev_p & {r["instance"] for r in train}           # replay rows of dev problems too
    assert sum(r["tier"] == "old" for r in train) == 15
    man = json.load(open(str(tmp_path / "train.jsonl") + ".manifest.json"))
    assert man["new rows given twice (kept once)"] == 10 and man["replay file rows on dev problems (never drawn)"] == 5


def _square(x):
    if x == 3:
        os._exit(1)                                                     # a simulator that takes its process down
    if x == 5:
        raise ValueError("bad job")
    return x * x


def test_a_dying_worker_fails_its_own_job_only():
    CS = load("crashsafe")
    got = {i: (res, err) for i, res, err in CS.run_jobs(_square, list(range(8)), 3)}
    assert sorted(got) == list(range(8))
    assert got[3] == (None, CS.DIED) and got[5][0] is None and "bad job" in got[5][1]
    assert all(got[i] == (i * i, None) for i in (0, 1, 2, 4, 6, 7))


# ---- a forecast-checked round (forecasts.py)
FC = load("forecasts")


def test_forecast_lines_are_read_in_the_forms_a_reply_writes_them():
    think = ("The left strut is thin.\n"
             "FORECAST: FILL_RECT(1, 2, 3, 4) -> better\n"
             "- **FORECAST:** `ERODE(0, 1)` -> **worse** (it thins the strut further)\n"
             "2. Forecast: SPLIT_SUB(54) => Refused.\n"
             "FORECAST: SHIFT_FEATURE(3, 1, 0) \u2192 better\n"
             "The forecast: prose with -> better is not a line.\n"
             "FORECAST: MOVE(1) -> unsure\n"
             "So FILL_RECT(1, 2, 3, 4).")
    lines, unread = FC.read(think)
    assert lines == [("FILL_RECT(1, 2, 3, 4)", "better"), ("ERODE(0, 1)", "worse"),
                     ("SPLIT_SUB(54)", "refused"), ("SHIFT_FEATURE(3, 1, 0)", "better")]
    assert unread == 2                     # the prose line with an arrow, and the line with an unknown word
    assert FC.forecast_lines("") == [] and FC.read(None) == ([], 0)
    # a forecast line the check cannot read is counted, never skipped
    for bad in ("FORECAST: A(1) -> worse or better, hard to say", "FORECAST: A(1) -> better; FORECAST: B(2) -> better",
                "FORECAST: A(1) -> (better)", "FORECAST: instead of A(1) I could B(2) -> worse", "FORECAST: A(1) -> not better",
                "FORECAST: A(1)", "I forecast that B(2) -> better", "FORECAST (alternative): A(1) -> worse",
                "FORECAST: A(1) -> better -> worse", "FORECAST: A(1 -> better"):
        assert FC.read(bad) == ([], 1), bad
    assert FC.read("The forecast horizon is three steps.") == ([], 0)              # prose with no arrow and no verdict word
    assert FC.read("FORECAST 2: CONNECT(a->b, 3) -> worse (load 1.4 -> 1.1)") == ([("CONNECT(a->b, 3)", "worse")], 0)
    assert FC.read("FORECAST: A(B(2), 3) => refused") == ([("A(B(2), 3)", "refused")], 0)
    assert FC.truth_of(False, None, -3.0) == "refused"
    assert FC.truth_of(True, -2.0, -3.0) == "better" and FC.truth_of(True, -3.0, -3.0) == "worse"


def _fc(text, said, truth, phi=None, pick=False, parsed=True, noop=False):
    return {"move": text, "text": text, "said": said, "truth": truth, "phi": phi, "is_pick": pick, "parsed": parsed, "noop": noop}


def _lines(fc):
    return "\n".join("FORECAST: %s -> %s" % (c["move"], c["said"]) for c in fc)


def test_a_reply_is_a_target_only_when_every_forecast_is_borne_out():
    good = [_fc("A(1)", "better", "better", -1.0, pick=True), _fc("B(2)", "refused", "refused"), _fc("C(3)", "worse", "worse", -5.0)]
    assert FC.fault(good) is None and FC.has_negative(good)
    all_better = [_fc("A(1)", "better", "better", -1.0, pick=True), _fc("C(3)", "better", "better", -2.0)]
    assert FC.fault(all_better) is None and not FC.has_negative(all_better)
    assert FC.fault(good[:1]) == "fewer than 2 forecast lines" and FC.fault(None) == "fewer than 2 forecast lines"
    assert FC.fault(good + [_fc("D(4)", "worse", "worse", -9.0)]) == "more than 3 forecast lines"
    assert FC.fault([good[0], _fc("?", "worse", None, parsed=False)]) == "a candidate is not a move"
    assert FC.fault([good[0], dict(good[0], is_pick=False)]) == "a candidate is written twice"
    assert FC.fault([good[0], _fc("N(0)", "worse", "worse", -3.0, noop=True)]) == "a candidate changes nothing"
    assert FC.fault([dict(good[0], is_pick=False), good[1]]) == "the chosen move is not a candidate"
    assert FC.fault([good[0], _fc("B(2)", "worse", "refused")]) == "a forecast is wrong"
    assert FC.fault([_fc("A(1)", "worse", "worse", -4.0, pick=True), good[1]]) == "the chosen move is not better"
    assert FC.fault([good[0], _fc("C(3)", "better", "better", -0.5)]) == "a rejected candidate does better"
    # a wrong forecast about a rejected candidate does not count as a negative example
    assert not FC.has_negative([good[0], _fc("B(2)", "worse", "refused")])
    assert FC.fault(good, unread=1) == "a forecast line cannot be read"


def test_pick_under_the_forecast_screen():
    right = [_fc("M(1)", "better", "better", 90.0, pick=True), _fc("X(9)", "better", "better", 10.0)]
    with_no = [_fc("M(2)", "better", "better", 70.0, pick=True), _fc("X(8)", "refused", "refused")]
    wrong = [_fc("M(3)", "better", "better", 99.0, pick=True), _fc("X(7)", "worse", "better", 50.0)]
    ans = {("s", 0): ("why 0", "h0"), ("s", 1): (_lines(right), "h1"), ("s", 2): (_lines(with_no), "h2"), ("s", 3): (_lines(wrong), "h3")}
    sc = [dict(_scored(0, 99), pick_text="M(0)"), dict(_scored(1, 90), forecasts=right, pick_text="M(1)"),
          dict(_scored(2, 70), forecasts=with_no, pick_text="M(2)"), dict(_scored(3, 99), forecasts=wrong, pick_text="M(3)")]
    assert BR.pick(sc, ans, 0.0, 0, True)[0]["sample"] == 0                                # the old screen: highest percentile
    x, reason, why = BR.pick(sc, ans, 0.0, 0, True, forecast=True)
    assert x["sample"] == 2 and why is None            # a rightly rejected candidate goes before a higher percentile
    assert BR.pick(sc[:2], ans, 0.0, 0, True, forecast=True)[0]["sample"] == 1
    assert BR.pick([sc[0]], ans, 0.0, 0, True, forecast=True)[2] == "no_eligible_reply (mostly forecasts: fewer than 2 forecast lines)"
    assert BR.pick([sc[3]], ans, 0.0, 0, True, forecast=True)[2] == "no_eligible_reply (mostly forecasts: a forecast is wrong)"
    row = BR.make_row({"variant": "fam", "problem_id": "p", "turn": 0, "n_prior_changes": 0, "system": "S", "observation": "U",
                       "instruction": "", "sid": "s"}, sc[2], "why 2", "scr2_fc", "t", "served+forecast_v1")
    assert row["forecasts"] == [["M(2)", "better"], ["X(8)", "refused"]] and row["has_negative_forecast"] is True
    assert row["sample_prompt"] == "served+forecast_v1" and row["messages"][1]["content"] == "U"
    assert "forecasts" not in BR.make_row({"variant": "fam", "problem_id": "p", "turn": 0, "n_prior_changes": 0, "system": "S",
                                           "observation": "U", "instruction": "", "sid": "s"}, sc[0], "why 0", "t", "t")


def test_the_extra_instruction_is_in_the_request_and_not_in_the_row():
    state = {"sid": "s", "system": "S", "observation": "U", "instruction": "I"}

    class A(_Args):
        extra, prompt_tag = "\n\nBEFORE YOU DECIDE", "served+forecast_v1"
    assert SS.request_body(state, 0, 0, A)["messages"][1]["content"] == "UI\n\nBEFORE YOU DECIDE"
    assert SS.request_body(state, 0, 0, _Args)["messages"][1]["content"] == "UI"
    rec = SS.draw(state, 0, 0, A, lambda url, body, timeout: _reply("<action>A</action>", "why"))
    assert rec["prompt"] == "served+forecast_v1"
    assert SS.draw(state, 0, 0, _Args, lambda url, body, timeout: _reply("<action>A</action>", "why"))["prompt"] == "served"


def test_build_rows_with_forecasts_end_to_end(tmp_path, capsys):
    state = {"variant": "fam", "problem_id": "p", "turn": 0, "n_prior_changes": 0, "system": "S", "observation": "U",
             "instruction": "", "sid": "s"}
    think = "FORECAST: M(1) -> better\nFORECAST: X(8) -> refused\nSo M(1)."
    samples = [{"sid": "s", "sample": 0, "thinking": "no lines", "answer": "<action>M(0)</action>", "sha1": "h0"},
               {"sid": "s", "sample": 1, "thinking": think, "answer": "<action>M(1)</action>", "sha1": "h1"}]
    scored = [dict(_scored(0, 99)),
              dict(_scored(1, 60), pick_text="M(1)",
                   forecasts=[_fc("M(1)", "better", "better", 60.0, pick=True), _fc("X(8)", "refused", "refused")]),
              {"sid": "s", "variant": "fam", "state": True, "n_candidates": 3, "phi_before": 0.0}]
    files = {}
    for name, recs in (("states", [state]), ("samples", samples), ("scored", scored)):
        files[name] = str(tmp_path / (name + ".jsonl"))
        with open(files[name], "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in recs))
    out = str(tmp_path / "rows.jsonl")
    base = ["--states", files["states"], "--samples", files["samples"], "--scored", files["scored"], "--out", out, "--min-pct", "0"]
    assert BR.main(base) == 0
    assert json.loads(open(out).read())["rationale_sample"] == 0                           # without the forecast screen
    assert BR.main(base + ["--require-forecasts", "--sample-prompt", "served+forecast_v1"]) == 0
    row = json.loads(open(out).read())
    assert row["rationale_sample"] == 1 and row["messages"][2]["reasoning_content"] == think
    assert row["messages"][1]["content"] == "U" and row["has_negative_forecast"] is True
    rep = capsys.readouterr().out
    assert "said refused, simulator refused" in rep and "share of forecasts right" in rep
    assert "fc reply: no forecast line" in rep and "forecasts right, gain, no word cap" in rep


def test_the_forecast_screen_reads_the_rationale_again():
    right = [_fc("M(1)", "better", "better", 90.0, pick=True), _fc("X(9)", "refused", "refused")]
    x = dict(_scored(1, 90), forecasts=right, pick_text="M(1)")
    assert BR.reply_fault(x, _lines(right)) is None
    assert BR.reply_fault(x, _lines(right) + "\nFORECAST: Y(2) -> (worse)") == "a forecast line cannot be read"
    assert BR.reply_fault(x, _lines(right[:1])) == "its lines read differently now than when scored"
    assert BR.reply_fault(dict(x, forecast_error="OverflowError: x"), _lines(right)) == "the forecast check raised an error"
    assert BR.reply_fault(dict(x, text="M(0.5)"), _lines(right)) == "the move made was changed on apply"
    # in pick: --min-pct is a condition on every reply under the forecast screen, not on the winner
    with_no = dict(_scored(1, 50), forecasts=right, pick_text="M(1)")
    plain = [_fc("M(2)", "better", "better", 95.0, pick=True), _fc("X(8)", "better", "better", 1.0)]
    all_better = dict(_scored(2, 95), forecasts=plain, pick_text="M(2)")
    ans = {("s", 1): (_lines(right), "h1", "served+forecast_v1"), ("s", 2): (_lines(plain), "h2", "served+forecast_v1")}
    assert BR.pick([with_no, all_better], ans, 0.0, 0, True, forecast=True)[0]["sample"] == 1
    assert BR.pick([with_no, all_better], ans, 90.0, 0, True, forecast=True)[0]["sample"] == 2
    # a reply recorded as sampled under another prompt is not used
    assert BR.pick([with_no, all_better], ans, 0.0, 0, True, forecast=True, sample_prompt="served+forecast_v1")[0]["sample"] == 1
    assert BR.pick([with_no], ans, 0.0, 0, True, forecast=True, sample_prompt="served")[2] == "no_eligible_reply (mostly sampled_under_another_prompt)"


def test_sampling_does_not_continue_a_file_sampled_under_another_prompt(tmp_path):
    states = tmp_path / "states.jsonl"
    states.write_text(json.dumps({"sid": "s", "variant": "fam", "system": "S", "observation": "U", "instruction": ""}) + "\n")
    out = tmp_path / "samples.jsonl"
    out.write_text(json.dumps({"sid": "s", "sample": 0, "answer": "x", "prompt": "served"}) + "\n")
    extra = tmp_path / "extra.txt"
    extra.write_text("\n\nBEFORE YOU DECIDE\n")
    base = ["--states", str(states), "--out", str(out), "--servers", "http://127.0.0.1:9", "--model", "m", "--k", "1"]
    assert SS.main(base + ["--extra-instruction-file", str(extra), "--prompt-tag", "served+forecast_v1"]) == 2
    assert SS.prompts_in(str(out)) == {("served", None)}
    assert SS.main(base) == 0                                                      # the same prompt: nothing left to ask for
    rec = SS.record_of("s", 1, _reply("<action>A</action>", "why"), "served+forecast_v1", "abc")
    assert rec["prompt_sha1"] == "abc" and "prompt_sha1" not in SS.record_of("s", 1, _reply("<action>A</action>", "why"))


class _CueArgs(_Args):
    extra, prompt_tag, prompt_sha1 = "", "served+forecast_v1+cue", "abc"
    cue, cue_min_lines, cue_max, end_id, ctx = "FORECAST:", 2, 3, 99, 1000
    cues = [("\n\nFORECAST:", [20, 21]), ("\nFORECAST:", [22, 21])]


def _server(script):
    """A stand-in for the three routes cued sampling uses; script: the completions to return, in order."""
    calls = []

    def post_to(url, path, body, timeout):
        calls.append((path, body))
        if path == "/chat/completions/render":
            return {"token_ids": [1, 2, 3]}
        text, ids, finish = script[sum(1 for p, _ in calls if p == "/completions") - 1]
        return {"choices": [{"text": text, "token_ids": ids, "finish_reason": finish}], "usage": {"completion_tokens": len(ids), "prompt_tokens": len(body["prompt"])}}
    return post_to, calls


def test_cued_sampling_continues_the_policys_own_rationale():
    state = {"sid": "s", "system": "S", "observation": "U", "instruction": ""}
    post_to, calls = _server([("The strut is thin.</think>", [10, 11, 99], "stop"),       # the rationale: no forecast line
                              (" A(1) -> better\n", [30, 31], "stop"),                     # after the first cue: one line
                              (" B(2) -> refused", [40], "stop"),                          # after the second cue: two lines
                              ("\n\n<action>A(1)</action><|im_end|>", [50, 51], "stop")])  # the think block closed: the action
    rec = SS.cued_reply(state, 0, 0, _CueArgs, "u0", post_to)
    assert rec["thinking"] == "The strut is thin.\n\nFORECAST: A(1) -> better\n\nFORECAST: B(2) -> refused"
    assert rec["answer"] == "<action>A(1)</action>" and rec["cues"] == 2 and rec["finish"] == "stop"
    assert FC.read(rec["thinking"]) == ([("A(1)", "better"), ("B(2)", "refused")], 0)
    assert rec["tokens"] == 8 and rec["prompt_tokens"] == 3 and rec["sha1"] == SS.reply_sha1(rec["thinking"], rec["answer"])
    assert (rec["prompt"], rec["prompt_sha1"], rec["mode"]) == ("served+forecast_v1+cue", "abc", "own_think_cued")
    comp = [b for p, b in calls if p == "/completions"]
    assert comp[0]["prompt"] == [1, 2, 3] and comp[0]["stop_token_ids"] == [99] and comp[0]["max_tokens"] == 1000 - 3 - 8
    assert comp[1]["prompt"] == [1, 2, 3, 10, 11, 20, 21]                 # the closing tag is not kept; the first cue follows
    assert comp[2]["prompt"] == [1, 2, 3, 10, 11, 20, 21, 30, 31, 22, 21]
    assert comp[3]["prompt"] == [1, 2, 3, 10, 11, 20, 21, 30, 31, 22, 21, 40, 99] and "stop_token_ids" not in comp[3]
    assert len({b["seed"] for b in comp}) == 4 and calls[0][1]["messages"][1]["content"] == "U"

    # a rationale that already has its lines gets no cue
    own = "Thin strut.\nFORECAST: A(1) -> better\nFORECAST: B(2) -> worse\n"
    post_to, calls = _server([(own, [10, 11, 12], "stop"), ("<action>A(1)</action>", [50], "stop")])
    rec = SS.cued_reply(state, 1, 0, _CueArgs, "u0", post_to)
    assert rec["cues"] == 0 and rec["thinking"] == own.strip() and len(calls) == 3
    assert [b for p, b in calls if p == "/completions"][1]["prompt"] == [1, 2, 3, 10, 11, 12, 99]

    # an unread forecast line counts towards the lines: the reply is not cued again for it
    post_to, calls = _server([("Thin.\nFORECAST: A(1) -> better\nFORECAST: B(2) -> maybe", [10], "stop"), ("<action>A(1)</action>", [50], "stop")])
    assert SS.cued_reply(state, 2, 0, _CueArgs, "u0", post_to)["cues"] == 0

    # cued lines the check cannot read count too (the reply is not a row target, so no further cue)
    post_to, calls = _server([("Thin.", [10], "stop"), (" not a line", [30], "stop"), (" still none", [31], "stop"), ("<action>A(1)</action>", [50], "stop")])
    rec = SS.cued_reply(state, 3, 0, _CueArgs, "u0", post_to)
    assert rec["cues"] == 2 and FC.read(rec["thinking"]) == ([], 2)

    # never more than --cue-max cues; a reply cut at the length limit has no answer
    class One(_CueArgs):
        cue_max = 1
    post_to, calls = _server([("Thin.", [10], "stop"), (" A(1) -> better", [30], "stop"), ("<action>A(1)</action>", [50], "stop")])
    rec = SS.cued_reply(state, 3, 0, One, "u0", post_to)
    assert rec["cues"] == 1 and len([1 for p, _ in calls if p == "/completions"]) == 3 and len(FC.read(rec["thinking"])[0]) == 1
    post_to, calls = _server([("Thin.", [10], "length")])
    rec = SS.cued_reply(state, 4, 0, _CueArgs, "u0", post_to)
    assert rec["finish"] == "length" and rec["thinking"] == "" and rec["answer"] == "" and rec["sha1"] is None
    # through draw: the cued path is taken, and a failing server gives an error record
    post_to, calls = _server([(own, [10], "stop"), ("<action>A(1)</action>", [50], "stop")])
    assert SS.draw(state, 5, 0, _CueArgs, None, post_to)["cues"] == 0

    def dead(url, path, body, timeout):
        raise OSError("gone")
    assert SS.draw(state, 6, 0, _CueArgs, None, dead) == {"sid": "s", "sample": 6, "error": "OSError: gone"}

    # a state's prompt is rendered once for all its replies
    class Two(_CueArgs):
        cue_min_lines = 0
    post_to, calls = _server([("Thin.", [10], "stop"), ("<action>A(1)</action>", [50], "stop")] * 3)
    recs = SS.draw_state(state, [0, 1, 2], 0, Two, None, post_to)
    assert [r["sample"] for r in recs] == [0, 1, 2] and all(r["answer"] == "<action>A(1)</action>" for r in recs)
    assert sum(1 for p, _ in calls if p == "/chat/completions/render") == 1
    assert all(b["prompt"][:3] == [1, 2, 3] for p, b in calls if p == "/completions")


def test_kv_budget_limits_the_states_in_flight_on_a_server():
    import threading
    import time
    b = SS.Budget(100)
    assert b.acquire(60) == 60 and b.used == 60
    got = []

    def later(name, cost):
        got.append((name, b.acquire(cost)))
    big = threading.Thread(target=later, args=("big", 500))      # more than the whole budget: counted as all of it
    big.start()
    time.sleep(0.1)
    small = threading.Thread(target=later, args=("small", 10))   # would fit now, but arrived after the big one
    small.start()
    time.sleep(0.1)
    assert got == [] and b.used == 60
    b.release(60)
    big.join(5)
    time.sleep(0.1)
    assert got == [("big", 100)] and b.used == 100               # the big one alone, the small one still waits
    b.release(100)
    small.join(5)
    assert got == [("big", 100), ("small", 10)] and b.used == 10
    b.release(10)
    assert b.used == 0 and b.queue == []

    # draw_state takes the prompt once plus --reply-tokens per missing reply, on its own server, and gives it back
    class Two(_CueArgs):
        cue_min_lines = 0
        servers = ["u0", "u1"]
        reply_tokens = 7
        budgets = [SS.Budget(1000), SS.Budget(1000)]
    seen = []
    state = {"sid": "s", "system": "S", "observation": "U", "instruction": ""}
    inner, calls = _server([("Thin.", [10], "stop"), ("<action>A(1)</action>", [50], "stop")] * 3)

    def post_to(url, path, body, timeout):
        if path == "/completions":
            seen.append((Two.budgets[0].used, Two.budgets[1].used))
        return inner(url, path, body, timeout)
    recs = SS.draw_state(state, [0, 1, 2], 1, Two, None, post_to)          # j = 1: the second server
    assert [r["sample"] for r in recs] == [0, 1, 2] and not any(r.get("error") for r in recs)
    assert set(seen) == {(0, 3 + 3 * 7)}                                   # 3 prompt tokens + 3 replies of 7
    assert (Two.budgets[0].used, Two.budgets[1].used) == (0, 0)

    class Heavy(Two):                                                      # --prompt-weight: the prompt counted twice
        prompt_weight = 2.0
    seen.clear()
    inner, calls = _server([("Thin.", [10], "stop"), ("<action>A(1)</action>", [50], "stop")] * 3)
    SS.draw_state(state, [0, 1, 2], 1, Heavy, None, post_to)
    assert set(seen) == {(0, 2 * 3 + 3 * 7)}

    # an error in a reply still gives the budget back; without a rendered prompt nothing is taken
    def dead(url, path, body, timeout):
        raise OSError("gone")
    recs = SS.draw_state(state, [0], 1, Two, None, dead)
    assert recs[0]["error"] and (Two.budgets[0].used, Two.budgets[1].used) == (0, 0)


# ---------------------------------------------------------------- open rules (open_rules.py, open_states.py, open_replay.py)
OR = load("open_rules")


def test_open_rule_texts_are_the_harness_texts():
    src = open(os.path.join(os.path.dirname(HERE), "rl", "harness_v3r", "dataxp_policy_eval_v3r.py"), encoding="utf-8").read()
    ns = {}
    for name in ("PHOTONICS_RULE_OLD", "PHOTONICS_RULE_NEW", "GRID2OP_RULE"):
        start = src.index(name + " = ")
        end = src.index("\n\n", start)
        exec(src[start:end], ns)                                                  # noqa: S102
        assert ns[name] == getattr(OR, name), name


class _Dom:
    """Stand-in domain: a state is {"x": int, "cell": {...}}; a move {"d": int} adds d; d == 0 is refused; the strict
    apply also refuses d > 5 (a 'fabrication rule'); x >= 100 ends the episode (a 'blackout')."""
    def clone(self, st):
        return {"x": st["x"], "cell": dict(st["cell"])}

    def _apply_unchecked(self, st, mv):
        if mv["d"] == 0:
            raise ValueError("changes nothing")
        st["x"] += mv["d"]
        if st["x"] >= 100:
            st["cell"] = {"done": True, "reason": "power flow diverged"}

    def apply(self, st, mv):
        if mv["d"] > 5:
            raise ValueError("breaks the rule")
        self._apply_unchecked(st, mv)

    def format_move(self, mv):
        return "ADD(%d)" % mv["d"]

    def load(self, spec):
        return {"x": spec["x0"], "cell": {"done": False}}


def test_open_rules_apply():
    d = _Dom()
    st = {"x": 0, "cell": {"done": False}}
    strict = lambda dd, s, m: OR.try_apply_open(dd, "other", s, m, None)          # noqa: E731
    assert strict(d, st, {"d": 7})[0] is None and "breaks the rule" in strict(d, st, {"d": 7})[2]["err"]
    child, mv, info = OR.try_apply_open(d, "photonics", st, {"d": 7}, None)        # photonics: applied
    assert child["x"] == 7 and st["x"] == 0 and info["err"] is None
    big = {"x": 98, "cell": {"done": False}}
    child, mv, info = OR.try_apply_open(d, "grid2op", big, {"d": 3}, None)         # the episode would end: refused
    assert child is None and "would end the episode (power flow diverged)" in info["err"] and big["x"] == 98
    assert OR.try_apply_open(d, "photonics", st, {"d": 0}, None)[0] is None        # changes nothing: still refused

    class SP:
        is_set_param = True
    assert OR.try_apply_open(d, "grid2", st, SP(), lambda *a: "strict path") == "strict path"
    applied = []
    st2 = OR.rebuild_open(d, "photonics", {"x0": 1}, [{"d": 9}, {"d": 2}], lambda dd, s, a: applied.append(a), lambda a: a)
    assert st2["x"] == 12 and applied == []
    OR.rebuild_open(d, "grid2op", {"x0": 1}, [{"d": 2}], lambda dd, s, a: applied.append(a), lambda a: a)
    assert applied == [{"d": 2}]


def test_open_rank_puts_the_fabrication_rule_first():
    assert OR.bad_pixels("grid2op", {"anything": 1}) == 0
    # a move that breaks the rule at a clean design is not a gain, whatever Phi does (Phi is capped for invalid designs)
    assert not OR.open_gain(bad=3, phi=-19.99, bad0=0, phi0=-22.77)
    assert OR.open_gain(bad=0, phi=-22.77, bad0=3, phi0=-19.99)                     # a full repair is
    assert OR.open_gain(bad=1, phi=-30.0, bad0=3, phi0=-19.99)                      # a partial repair is
    assert OR.open_gain(bad=2, phi=-19.0, bad0=2, phi0=-19.99)                      # as many pixels, higher Phi
    assert not OR.open_gain(bad=2, phi=-19.99, bad0=2, phi0=-19.99)
    assert OR.open_rank(0, -30.0) > OR.open_rank(1, -1.0) > OR.open_rank(1, -2.0)


def test_pick_breaks_percentile_ties_by_open_rank():
    ans = {("s", k): ("why %d" % k, "h%d" % k) for k in range(3)}
    sc = [dict(_scored(0, 100), open_rank=[-2, -10.0]), dict(_scored(1, 100), open_rank=[0, -25.0]),
          dict(_scored(2, 100), open_rank=[-1, -5.0])]
    assert BR.pick(sc, ans, 0.0, 0, True)[0]["sample"] == 1                           # the full repair
    assert BR.pick([_scored(0, 80), _scored(1, 80)], ans, 0.0, 0, True)[0]["sample"] == 0   # elsewhere as before


def test_open_prompt():
    s, u = OR.open_prompt("photonics", "sys", "rule. " + OR.PHOTONICS_RULE_OLD + " more")
    assert u == "rule. " + OR.PHOTONICS_RULE_NEW + " more" and s == "sys"
    with pytest.raises(ValueError):
        OR.open_prompt("photonics", "sys", "no rule here")
    assert OR.open_prompt("grid2op", "s", "u") == ("s", "u" + OR.GRID2OP_RULE)
    with pytest.raises(ValueError):
        OR.open_prompt("grid2op", "s", "u" + OR.GRID2OP_RULE)
    assert OR.open_prompt("circuit2", "s", "u") == ("s", "u")


def test_open_states_from_recorded_prompts(tmp_path):
    OS = load("open_states")
    spec = {"problem_id": "p1", "x": 1}
    turns = [{"t": 0, "system": "S", "user": "U0" + OR.GRID2OP_RULE, "binding": "b", "fail": None, "action": {"verb": "A", "args": {}},
              "phi_after": -5.0},
             {"t": 1, "system": "S", "user": "U1", "fail": "illegal:A would end the episode (x)", "action": {"verb": "B", "args": {}},
              "phi_after": -5.0},
             {"t": 2, "system": "S", "user": "U1\n\nREFUSED MOVES ...would end the episode...", "reject_note": 1, "fail": None,
              "action": {"verb": "C", "args": {}}, "phi_after": -1.0},
             {"t": 3, "system": "S", "user": "U3\n\nFABRICATION: the current design breaks", "fail": None, "action": {"verb": "D", "args": {}},
              "phi_after": 0.0}]
    rec = {"key": "m/r1/p1", "variant": "photonics", "episode": {"phi_start": -9.0, "turns": turns, "native": True, "feasible@20": True}}
    ep = tmp_path / "e.episodes.jsonl"
    ep.write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n")
    req = {"harness_argv": ["--code", "c", "--open-rules", "--record-prompts"], "env": {}, "episodes": [{"key": "m/r1/p1", "spec": spec}]}
    (tmp_path / "e.episodes.jsonl.request.json").write_text(json.dumps(req))
    out = tmp_path / "states.jsonl"
    assert OS.main(["--episodes", str(ep), "--out", str(out)]) == 0
    rows = [json.loads(x) for x in open(out)]
    assert [r["turn"] for r in rows] == [0, 1, 2, 3]                                  # the second copy adds nothing
    assert [len(r["history"]) for r in rows] == [0, 1, 1, 2] and rows[3]["history"][1] == {"verb": "C", "args": {}}
    assert [r["phi_before"] for r in rows] == [-9.0, -5.0, -5.0, -1.0]
    assert rows[1]["sid"] != rows[2]["sid"] and rows[1]["sid"].split("~")[0] == rows[2]["sid"].split("~")[0]
    assert [r["own_action"] is not None for r in rows] == [True, False, True, True]
    assert rows[2]["note_refused"] == 1 and rows[2]["note_blackout"] and rows[3]["note_fabrication"] and all(r["open_rules"] for r in rows)
    assert not rows[0]["note_blackout"]                                               # the rule line is not a note
    assert rows[1]["observation"] == "U1" and rows[1]["instruction"] == ""
    bad = dict(req, harness_argv=["--code", "c"])
    (tmp_path / "e.episodes.jsonl.request.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        OS.main(["--episodes", str(ep), "--out", str(out)])


def test_open_replay(tmp_path):
    RP = load("open_replay")
    rows = [{"domain": "photonics", "sid": "a", "messages": [{"role": "system", "content": "s"},
                                                            {"role": "user", "content": "x " + OR.PHOTONICS_RULE_OLD},
                                                            {"role": "assistant", "content": "<action>A()</action>"}]},
            {"domain": "grid2op", "sid": "b", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                                          {"role": "assistant", "content": "c"}]},
            {"domain": "seir2", "sid": "c", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                                        {"role": "assistant", "content": "c"}]}]
    src = tmp_path / "rep.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out.jsonl"
    assert RP.main(["--in", str(src), "--out", str(out)]) == 0
    got = [json.loads(x) for x in open(out)]
    assert got[0]["messages"][1]["content"] == "x " + OR.PHOTONICS_RULE_NEW and got[0]["prompt_rules"] == "open"
    assert got[1]["messages"][1]["content"] == "u" + OR.GRID2OP_RULE
    assert got[2] == rows[2] and [g["messages"][2] for g in got] == [r["messages"][2] for r in rows]
