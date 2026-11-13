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
