"""scripts/rl_probe/gold_rows_to_run.py: SFT rows as a run directory for the gradient probe."""
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("gold_rows_to_run", os.path.join(ROOT, "scripts", "rl_probe", "gold_rows_to_run.py"))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


class Tok:
    """One token per character; the generation prompt opens a think block, as Qwen3.x does."""

    def __init__(self, opener="<think>\n"):
        self.opener = opener

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        text = "".join("[%s]%s" % (m["role"], m["content"]) for m in messages) + "[assistant]" + self.opener
        return [ord(c) for c in text]

    def __call__(self, text, add_special_tokens=False):
        class R:
            input_ids = [ord(c) for c in text]
        return R

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def row(fam, inst, turn, rs=0, reasoning=" because \n", content=" <action>X(1)</action> "):
    return {"domain": fam, "instance": inst, "turn": turn, "rationale_sample": rs, "tier": "t", "teacher": "q",
            "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U%s" % turn, "supervise": True},
                         {"role": "assistant", "reasoning_content": reasoning, "content": content}]}


def test_pick_rows_spreads_instances_and_is_deterministic():
    index = {"b": {"b%02d" % i: [100 + i] for i in range(10)}, "a": {"a0": [0, 1, 2], "a1": [3]}}
    got = G.pick_rows(index, 4)
    assert [g[0] for g in got] == ["a", "a", "b", "b", "b", "b"]          # families in order, a has only 2 instances
    assert [g[1] for g in got[2:]] == ["b00", "b02", "b05", "b07"]          # evenly spaced over the sorted ids
    assert got[0][1] == "a0" and got[0][2] in (0, 1, 2) and got[1] == ("a", "a1", 3)
    assert G.pick_rows(index, 4) == got
    assert len({g[1] for g in got}) == len(got)                             # one row per instance


def test_row_record_is_the_served_prompt_and_the_trained_completion():
    tok = Tok()
    rec = G.row_record(row("fam", "fam_0001", 3, rs=2), tok, 7)
    call = rec["calls"][0]
    assert tok.decode(call["prompt_token_ids"]) == "[system]S[user]U3[assistant]<think>\n"
    assert tok.decode(call["token_ids"]) == "because\n</think>\n\n<action>X(1)</action><|im_end|>"
    assert rec["group"] == "fam_0001" and rec["variant"] == "fam" and rec["key"] == "gold/fam_0001/0007_t3_r2"
    assert rec["row"]["turn"] == 3 and rec["row"]["rationale_sample"] == 2
    empty = G.row_record(row("fam", "fam_0002", 0, reasoning=None), tok, 0)
    assert tok.decode(empty["calls"][0]["token_ids"]) == "\n</think>\n\n<action>X(1)</action><|im_end|>"
    with pytest.raises(ValueError):
        G.row_record(row("fam", "fam_0003", 0), Tok(opener=""), 0)          # a template that opens no think block
    bad = row("fam", "fam_0004", 0)
    bad["messages"] = bad["messages"][:-1]
    with pytest.raises(ValueError):
        G.row_record(bad, tok, 0)


def test_fresh_lora_is_the_default_initialisation(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    like = tmp_path / "like"
    like.mkdir()
    sd = {"m.q.lora_A.weight": torch.full((4, 100), 3.0, dtype=torch.bfloat16),
          "m.q.lora_B.weight": torch.full((50, 4), 2.0, dtype=torch.bfloat16)}
    st.save_file(sd, str(like / "adapter_model.safetensors"))
    (like / "adapter_config.json").write_text('{"r": 4}')
    G.fresh_lora(str(like), str(tmp_path / "new"), seed=0)
    new = st.load_file(str(tmp_path / "new" / "adapter_model.safetensors"))
    a, b = new["m.q.lora_A.weight"], new["m.q.lora_B.weight"]
    assert a.dtype == torch.bfloat16 and a.shape == (4, 100) and b.shape == (50, 4)
    assert float(b.float().abs().max()) == 0.0
    assert 0.05 < float(a.float().abs().max()) <= 0.1 + 1e-3 and abs(float(a.float().mean())) < 0.02   # +-1/sqrt(100)
    assert (tmp_path / "new" / "adapter_config.json").read_text() == '{"r": 4}'
    G.fresh_lora(str(like), str(tmp_path / "again"), seed=0)
    assert torch.equal(st.load_file(str(tmp_path / "again" / "adapter_model.safetensors"))["m.q.lora_A.weight"], a)
    st.save_file({"m.other.weight": torch.zeros(2, 2)}, str(like / "adapter_model.safetensors"))
    with pytest.raises(SystemExit):
        G.fresh_lora(str(like), str(tmp_path / "bad"), seed=0)


def test_main_writes_a_run_directory_the_probe_reads(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: Tok()))
    rows = [row("fa", "fa_%d" % i, t) for i in range(5) for t in range(2)] + [row("fb", "fb_0", 0)]
    src = tmp_path / "rows.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    run = tmp_path / "run"
    argv = ["--rows", str(src), "--tokenizer", "none", "--per-family", "3", "--run-dir", str(run), "--adapter", str(adapter)]
    try:
        assert G.main(argv) == 0
    except OSError as e:                                                   # no symlink privilege (Windows)
        pytest.skip("cannot link here: %s" % e)
    recs = [json.loads(line) for line in open(run / "rollouts" / "step_00000.jsonl", encoding="utf-8")]
    assert [r["variant"] for r in recs] == ["fa", "fa", "fa", "fb"] and len({r["group"] for r in recs}) == 4
    assert os.path.exists(run / "ckpt" / "step_00000" / "adapter" / "adapter_model.safetensors")
    assert G.main(argv) == 0                                               # a rerun changes nothing
    assert [json.loads(line)["key"] for line in open(run / "rollouts" / "step_00000.jsonl", encoding="utf-8")] == [r["key"] for r in recs]
    pytest.importorskip("torch")
    from llm_finetune.rl_dataxp.grad_probe import nearest_adapter, probe_samples
    samples = probe_samples(recs, 0)
    assert len(samples) == 4 and all(s["turn"] == 0 and s["prompt_ids"] and s["completion_ids"] for s in samples)
    assert nearest_adapter(str(run), 0)[0] == 0
    with pytest.raises(SystemExit):
        G.main(argv + ["--fresh-lora"])
