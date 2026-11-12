"""Supervised rounds on rows (llm_finetune/rl_dataxp/sft_rows.py): the row order and schedule, the
supervised text, and on the tiny Qwen3_5 model of the trainer tests a whole run and its resume."""
import importlib.util
import json
import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")

from llm_finetune.rl_dataxp import sft_rows as SR  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
try:
    from test_rl_dataxp_trainer import make_cfg, problems_file, tiny_model  # noqa: E402,F401
    HAVE_MODEL = True
except BaseException:                                           # the trainer tests skip themselves here
    HAVE_MODEL = False


def test_row_order_is_a_shuffle_per_epoch():
    o = SR.row_order(10, 2.5, seed=3)
    assert len(o) == 25
    assert sorted(o[:10]) == list(range(10)) and sorted(o[10:20]) == list(range(10))
    assert o[:10] != o[10:20] and o == SR.row_order(10, 2.5, seed=3) and o != SR.row_order(10, 2.5, seed=4)
    assert len(SR.row_order(7, 1.0, 0)) == 7 and len(SR.row_order(7, 0.5, 0)) == 4


def test_lr_factor_warms_up_then_follows_a_cosine():
    steps = 100
    f = [SR.lr_factor(n, steps, 0.03) for n in range(steps)]
    assert f[0] == pytest.approx(0.25) and f[1] == pytest.approx(0.5) and f[2] == pytest.approx(0.75)
    assert f[3] == pytest.approx(1.0) and max(f) == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(f[3:], f[4:])) and 0 < f[-1] < 0.01
    assert f[3 + 48] == pytest.approx(0.5 * (1 + math.cos(math.pi * 48 / 97)))
    assert SR.lr_factor(0, 10, 0.0) == 1.0


def test_completion_text_is_the_probe_reference_text():
    p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "rl_probe", "gold_rows_to_run.py")
    spec = importlib.util.spec_from_file_location("gold_rows_to_run", p)
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    for m in ({"reasoning_content": " a rationale\n", "content": "\n\n<action>X(1)</action> "},
              {"reasoning_content": "", "content": "<action>Y()</action>"}):
        assert SR.completion_text(m) == G.completion_text(m)
    assert SR.completion_text({"reasoning_content": "r", "content": "<action>Z</action>"}) == "r\n</think>\n\n<action>Z</action><|im_end|>"


class _Tok:
    """Stands in for the tokenizer: the rows of these tests carry their token ids."""

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()


def _rows(tmp_path, n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = [{"p": torch.randint(5, 60, (12 + i,), generator=g).tolist(), "c": torch.randint(5, 60, (6 + i % 3,), generator=g).tolist(),
             "domain": "fam%d" % (i % 2), "tier": "t"} for i in range(n)]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _cfg(tiny_model, run_dir, problems_file, rows, **sft):  # noqa: F811
    cfg = make_cfg(tiny_model, run_dir, problems_file)
    cfg["sft"] = dict({"rows_file": rows, "dev_file": rows, "epochs": 3.0, "rows_per_step": 2, "lr": 2e-2,
                       "save_every": 1, "warmup_ratio": 0.0}, **sft)
    return cfg


@pytest.fixture
def patched(monkeypatch):
    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer", _Tok)
    monkeypatch.setattr(SR, "row_sample", lambda r, tok: {"prompt_ids": r["p"], "completion_ids": r["c"],
                                                         "domain": r["domain"], "tier": r["tier"]})


@pytest.mark.skipif(not HAVE_MODEL, reason="needs the tiny Qwen3_5 model of the trainer tests")
def test_a_run_lowers_the_loss_and_writes_the_final_adapter(tiny_model, tmp_path, problems_file, patched):  # noqa: F811
    rows = _rows(tmp_path)
    hist = SR.train(_cfg(tiny_model, tmp_path / "run", problems_file, rows), log=lambda s: None)
    assert sorted(hist) == list(range(9))                      # 6 rows x 3 epochs / 2 per step
    start = json.load(open(tmp_path / "run" / "dev_start.json"))["dev_loss"]
    assert hist[8]["dev_loss"] < start - 0.05
    assert all(m["rows"] == 2 for m in hist.values()) and hist[0]["by_tier"] == {"t": 2}
    assert os.path.exists(tmp_path / "run" / "final" / "adapter_model.safetensors")
    assert json.load(open(tmp_path / "run" / "DONE"))["steps"] == 9
    # a finished run resumes to nothing more: same history, no further step
    again = SR.train(_cfg(tiny_model, tmp_path / "run", problems_file, rows), log=lambda s: None)
    assert sorted(again) == list(range(9))


@pytest.mark.skipif(not HAVE_MODEL, reason="needs the tiny Qwen3_5 model of the trainer tests")
def test_resume_matches_an_uninterrupted_run(tiny_model, tmp_path, problems_file, patched, monkeypatch):  # noqa: F811
    rows = _rows(tmp_path)
    SR.train(_cfg(tiny_model, tmp_path / "full", problems_file, rows), log=lambda s: None)
    real, calls = SR.train_step, []

    def dies(*a, **k):
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError("killed")
        return real(*a, **k)

    monkeypatch.setattr(SR, "train_step", dies)
    with pytest.raises(RuntimeError, match="killed"):
        SR.train(_cfg(tiny_model, tmp_path / "part", problems_file, rows), log=lambda s: None)
    monkeypatch.setattr(SR, "train_step", real)
    SR.train(_cfg(tiny_model, tmp_path / "part", problems_file, rows), log=lambda s: None)
    a = json.load(open(tmp_path / "full" / "ckpt" / "step_00009" / "state.json"))["checksum"]
    b = json.load(open(tmp_path / "part" / "ckpt" / "step_00009" / "state.json"))["checksum"]
    assert a == pytest.approx(b, rel=1e-9)


@pytest.mark.skipif(not HAVE_MODEL, reason="needs the tiny Qwen3_5 model of the trainer tests")
def test_the_loss_temperature_reaches_the_log_probabilities(tiny_model, tmp_path, problems_file):  # noqa: F811
    from test_rl_dataxp_trainer import policy_of
    from llm_finetune.rl_dataxp import trainer as TR
    pol = policy_of(make_cfg(tiny_model, tmp_path, problems_file))
    dist = TR.Dist(1)
    smp = [{"prompt_ids": [5, 6, 7, 8, 9, 10, 11, 12], "completion_ids": [13, 14, 15, 16]}]
    at_1 = SR.score_rows(pol, smp, dist, 1.0)["dev_loss"]
    at_half = SR.score_rows(pol, smp, dist, 0.5)["dev_loss"]
    with torch.no_grad():
        lp, _ = pol.logprobs(smp[0]["prompt_ids"], smp[0]["completion_ids"], 0.5)
    assert at_half == pytest.approx(float(-lp.mean()), rel=1e-5) and at_half != pytest.approx(at_1, rel=1e-3)
    opt = torch.optim.SGD(pol.params, lr=0.0)
    assert SR.train_step(pol, smp, dist, opt, 1.0, 0.5)["loss"] == pytest.approx(at_half, rel=1e-4)


@pytest.mark.skipif(not HAVE_MODEL, reason="needs the tiny Qwen3_5 model of the trainer tests")
def test_resume_refuses_other_rows(tiny_model, tmp_path, problems_file, patched):  # noqa: F811
    SR.train(_cfg(tiny_model, tmp_path / "run", problems_file, _rows(tmp_path), epochs=1.0), log=lambda s: None)
    os.remove(tmp_path / "run" / "DONE")
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(SystemExit, match="fingerprint"):
        SR.train(_cfg(tiny_model, tmp_path / "run", problems_file, _rows(other, seed=1), epochs=1.0), log=lambda s: None)
