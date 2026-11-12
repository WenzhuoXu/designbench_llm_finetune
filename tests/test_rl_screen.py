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


def test_records_of_one_request():
    out = {"usage": {"completion_tokens": 30, "prompt_tokens": 100},
           "choices": [{"index": 1, "finish_reason": "stop", "message": {"content": "<action>B</action>", "reasoning_content": "r1"}},
                       {"index": 0, "finish_reason": "length", "message": {"content": "", "reasoning_content": "cut"}},
                       {"index": 2, "finish_reason": "stop", "message": {"content": "no closing tag"}}]}
    recs = SS.records_of("s", out, 3)
    assert [r["sample"] for r in recs] == [0, 1, 2]
    assert recs[0]["answer"] == "" and recs[0]["finish"] == "length" and recs[0]["tokens"] == 30
    assert recs[1]["answer"] == "r1\n\n<action>B</action>" and recs[1]["tokens"] is None
    assert recs[2]["answer"] == "" and recs[2]["finish"] == "length"               # unclosed think block
    assert BR.reasoning_of(recs[1]["answer"]) == "r1"
    with pytest.raises(ValueError):
        SS.records_of("s", out, 8)


def _scored(sample, pct, gain=True, ok=True, noop=False, reason=None):
    return {"sid": "s", "sample": sample, "ok": ok, "pct": pct if ok else None, "gain": gain, "noop": noop,
            "reason": reason, "text": "M(%d)" % sample, "action": {"verb": "M"}, "phi": float(pct or 0)}


def test_pick_is_the_best_eligible_reply():
    ans = {("s", k): "why %d\n\n<action>M(%d)</action>" % (k, k) for k in range(5)}
    sc = [_scored(0, 95, gain=False), _scored(1, 80), _scored(2, 80), _scored(3, 0, ok=False, reason="illegal:x"), _scored(4, 99, noop=True)]
    x, reason, why = BR.pick(sc, ans, 0.0, 0, True)
    assert x["sample"] == 1 and reason == "why 1" and why is None                  # ties: lowest sample index
    assert BR.pick(sc, ans, 90.0, 0, True)[2] == "below_min_pct"
    assert BR.pick(sc, ans, 90.0, 0, False)[0]["sample"] == 0                      # without the gain condition
    assert BR.pick(sc, ans, 0.0, 1, True)[2].startswith("no_eligible_reply (mostly too_long")
    assert BR.pick([_scored(0, 0, ok=False, reason="illegal:x")], ans, 0.0, 0, True)[2] == "no_eligible_reply (mostly not_applied:illegal)"
    assert BR.pick([], ans, 0.0, 0, True)[2] == "no_eligible_reply (mostly no_reply)"
    leak = dict(ans)
    leak[("s", 1)] = "a <think> tag\n\n<action>M(1)</action>"
    assert BR.pick(sc, leak, 0.0, 0, True)[0]["sample"] == 2


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
