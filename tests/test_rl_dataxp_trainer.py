"""dataxp GRPO trainer on a tiny Qwen3_5ForConditionalGeneration (the real architecture: linear-
attention + full-attention layers behind the multimodal wrapper), CPU, float32.

Needs torch, transformers (>= the version with Qwen3_5), peft, safetensors, omegaconf.
"""
import copy
import json
import os
import random
import re

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")
if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
    pytest.skip("transformers without Qwen3_5", allow_module_level=True)

import yaml  # noqa: E402

from llm_finetune.rl_dataxp import trainer as TR  # noqa: E402
from llm_finetune.rl_dataxp.adapters import concat_adapters  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGEX = yaml.safe_load(open(os.path.join(REPO, "configs", "model", "qwen38_27b.yaml")))["lora"]["target_modules"]
VOCAB = 128


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    text = dict(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
                num_key_value_heads=1, head_dim=16, linear_key_head_dim=8, linear_value_head_dim=8,
                linear_num_key_heads=2, linear_num_value_heads=4,
                layer_types=["linear_attention"] * 3 + ["full_attention"], max_position_embeddings=256)
    vis = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32, num_position_embeddings=16)
    cfg = Qwen3_5Config(text_config=text, vision_config=vis, architectures=["Qwen3_5ForConditionalGeneration"])
    torch.manual_seed(0)
    m = Qwen3_5ForConditionalGeneration(cfg)
    d = tmp_path_factory.mktemp("tiny")
    m.save_pretrained(str(d))
    return str(d)


@pytest.fixture
def problems_file(tmp_path):
    p = tmp_path / "problems.jsonl"
    with open(p, "w") as f:
        for v in ("va", "vb"):
            for k in range(3):
                f.write(json.dumps({"variant": v, "spec": {"problem_id": "%s_%d" % (v, k)}}) + "\n")
    return str(p)


def make_cfg(tiny_model, run_dir, problems_file, **over):
    cfg = TR.load_config(None, [])
    cfg.update(run_dir=str(run_dir), seed=3)
    cfg["model"].update(path=tiny_model, torch_dtype="float32", attn_implementation="sdpa")
    cfg["policy"]["lora"].update(r=4, lora_alpha=8, target_modules=REGEX)
    cfg["rollout"].update(problems_file=problems_file, problems_per_step=2, group_size=4, horizon=3,
                          temperature=1.0, servers=["http://fake/v1"])
    cfg["grpo"].update(steps=3, lr=5e-3, warmup_steps=0, kl_coef=0.05)
    cfg["guards"].update(enabled=False)
    for k, v in over.items():
        TR.apply_overrides(cfg, ["%s=%s" % (k, json.dumps(v))])
    return cfg


class FakeBackend:
    """Deterministic episodes: token ids, turn count and success are functions of the episode key."""

    def __init__(self):
        self.adapters = []

    def collect(self, step, adapter_dir, episodes):
        assert os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")), adapter_dir
        self.adapters.append(adapter_dir)
        return self.load(step, episodes)

    def load(self, step, episodes):
        out = []
        for e in episodes:
            r = random.Random(e["key"])
            n = r.randint(1, 3)
            ok = r.random() < 0.5
            calls = [{"prompt_token_ids": [r.randrange(1, VOCAB) for _ in range(r.randint(5, 10))],
                      "token_ids": [r.randrange(1, VOCAB) for _ in range(r.randint(2, 5))],
                      "logprobs": None, "finish": "stop"} for _ in range(n)]
            ep = {"first_feasible_turn": n if ok else None, "feasible_start": False,
                  "turns": [{"fail": None, "finish": "stop"}] * n}
            out.append({"key": e["key"], "group": e["group"], "member": e["member"], "variant": e["variant"],
                        "episode": ep, "calls": calls})
        return out


def policy_of(cfg):
    return TR.Policy(cfg, torch.device("cpu"))


def test_regex_reaches_linear_attention_and_logprob_paths(tiny_model, tmp_path, problems_file):
    pol = policy_of(make_cfg(tiny_model, tmp_path, problems_file))
    names = [n for n, _ in pol.trainable]
    assert any("linear_attn.in_proj_qkv" in n for n in names) and any("self_attn.q_proj" in n for n in names)
    assert not any("visual" in n for n in names)
    prompt, comp = [3, 5, 7, 9, 11], [13, 17, 19]
    lp, raw = pol.logprobs(prompt, comp, 0.7, raw=True)
    with torch.no_grad():                                          # without logits_to_keep: same numbers
        logits = pol.model(input_ids=torch.tensor([prompt + comp]), use_cache=False).logits[0, 4:7]
        ref = torch.log_softmax(logits / 0.7, -1).gather(-1, torch.tensor(comp)[:, None]).squeeze(-1)
        ref_raw = torch.log_softmax(logits, -1).gather(-1, torch.tensor(comp)[:, None]).squeeze(-1)
    assert torch.allclose(lp, ref, atol=1e-5) and torch.allclose(raw, ref_raw, atol=1e-5)
    # a fresh adapter (B = 0) is the base: the KL reference agrees with the policy exactly
    assert torch.allclose(pol.ref_logprobs(prompt, comp, 0.7), lp.detach(), atol=1e-6)


def test_update_moves_logprobs_with_the_advantage(tiny_model, tmp_path, problems_file):
    cfg = make_cfg(tiny_model, tmp_path, problems_file, **{"grpo.kl_coef": 0.0, "grpo.lr": 2e-2})
    pol = policy_of(cfg)
    dist = TR.Dist(10)
    opt = torch.optim.AdamW(pol.params, lr=2e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: 1.0)
    prompt = [3, 5, 7, 9]
    good, bad = [11, 12, 13], [21, 22, 23]
    samples = [{"prompt_ids": prompt, "completion_ids": good, "advantage": 1.0, "vllm_logprobs": None},
               {"prompt_ids": prompt, "completion_ids": bad, "advantage": -1.0, "vllm_logprobs": None}]
    with torch.no_grad():
        g0, b0 = pol.logprobs(prompt, good, 1.0)[0].sum(), pol.logprobs(prompt, bad, 1.0)[0].sum()
    for _ in range(3):
        st = TR.train_on_samples(pol, samples, dist, cfg, opt, sched)
    with torch.no_grad():
        g1, b1 = pol.logprobs(prompt, good, 1.0)[0].sum(), pol.logprobs(prompt, bad, 1.0)[0].sum()
    assert g1 > g0 and b1 < b0
    assert st["tokens"] == 6 and st["clip_frac"] == 0.0 and st["grad_norm"] > 0


def test_kl_reference_base_vs_init(tiny_model, tmp_path, problems_file):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration
    base = Qwen3_5ForConditionalGeneration.from_pretrained(tiny_model)
    pm = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=REGEX, init_lora_weights=False))
    init = tmp_path / "init_adapter"
    pm.save_pretrained(str(init))                                   # a nonzero adapter: "S"
    prompt, comp = [3, 5, 7, 9], [11, 12]
    for mode, zero in (("init", True), ("base", False)):
        cfg = make_cfg(tiny_model, tmp_path / mode, problems_file,
                       **{"policy.init_adapter": str(init), "grpo.kl_reference": mode})
        pol = policy_of(cfg)
        lp, _ = pol.logprobs(prompt, comp, 1.0)
        ref = pol.ref_logprobs(prompt, comp, 1.0)
        d = (ref - lp.detach()).abs().max().item()
        assert (d < 1e-6) == zero, (mode, d)
        names = [n for n, p in pol.model.named_parameters() if p.requires_grad]
        assert names and all(".default." in n for n in names)        # the ref copy never trains


def test_train_resume_matches_an_uninterrupted_run(tiny_model, tmp_path, problems_file):
    full = make_cfg(tiny_model, tmp_path / "full", problems_file)
    TR.train(full, backend=FakeBackend(), log=lambda s: None)
    part = make_cfg(tiny_model, tmp_path / "part", problems_file, **{"grpo.steps": 2})
    TR.train(part, backend=FakeBackend(), log=lambda s: None)
    os.makedirs(os.path.join(tmp_path, "part", "ckpt", "step_00003.tmp"))   # a checkpoint cut short
    assert TR.latest_ckpt(str(tmp_path / "part")) == 2
    be = FakeBackend()
    TR.train(make_cfg(tiny_model, tmp_path / "part", problems_file), backend=be, log=lambda s: None)
    assert [os.path.basename(os.path.dirname(a)) for a in be.adapters] == ["step_00002"]   # only the missing step
    from safetensors.torch import load_file
    a = load_file(str(tmp_path / "full" / "ckpt" / "step_00003" / "adapter" / "adapter_model.safetensors"))
    b = load_file(str(tmp_path / "part" / "ckpt" / "step_00003" / "adapter" / "adapter_model.safetensors"))
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    ma, mb = TR.read_metrics(str(tmp_path / "full")), TR.read_metrics(str(tmp_path / "part"))
    assert ma[2]["pg"] == mb[2]["pg"] and ma[2]["kl"] == mb[2]["kl"]
    assert os.path.exists(tmp_path / "full" / "DONE")
    # a finished run returns at once
    assert TR.train(make_cfg(tiny_model, tmp_path / "full", problems_file), backend=FakeBackend(), log=lambda s: None)


def test_resume_refuses_a_changed_config(tiny_model, tmp_path, problems_file):
    TR.train(make_cfg(tiny_model, tmp_path, problems_file, **{"grpo.steps": 1}), backend=FakeBackend(), log=lambda s: None)
    with pytest.raises(SystemExit, match="fingerprint"):
        TR.train(make_cfg(tiny_model, tmp_path, problems_file, **{"grpo.steps": 2, "grpo.lr": 1e-3}),
                 backend=FakeBackend(), log=lambda s: None)


def test_guard_stops_a_stalled_run(tiny_model, tmp_path, problems_file):
    cfg = make_cfg(tiny_model, tmp_path, problems_file, **{
        "grpo.lr": 0.0, "grpo.steps": 10, "guards.enabled": True, "guards.window": 2, "guards.stall_steps": 4})
    with pytest.raises(TR.GuardStop, match="not moved"):
        TR.train(cfg, backend=FakeBackend(), log=lambda s: None)
    assert json.load(open(tmp_path / "STOPPED.json"))["step"] == 3
    with pytest.raises(SystemExit, match="stopped earlier"):
        TR.train(cfg, backend=FakeBackend(), log=lambda s: None)


def test_checkpoint_pruning(tiny_model, tmp_path, problems_file):
    cfg = make_cfg(tiny_model, tmp_path, problems_file, **{
        "grpo.steps": 5, "checkpoint.keep_last": 2, "checkpoint.keep_adapter_every": 2, "checkpoint.keep_adapter_steps": []})
    TR.train(cfg, backend=FakeBackend(), log=lambda s: None)
    root = tmp_path / "ckpt"
    have = {d: sorted(os.listdir(root / d)) for d in os.listdir(root)}
    assert sorted(have) == ["step_00000", "step_00002", "step_00004", "step_00005"]
    assert have["step_00000"] == have["step_00002"] == ["COMPLETE", "adapter", "state.json"]
    assert "optimizer.pt" in have["step_00004"] and "optimizer.pt" in have["step_00005"]


def test_concat_adapters_is_exact(tiny_model, tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration
    paths, specs = [], [(4, 8, 1.0), (2, 2, 0.5)]
    for k, (r, alpha, _) in enumerate(specs):
        torch.manual_seed(10 + k)
        pm = get_peft_model(Qwen3_5ForConditionalGeneration.from_pretrained(tiny_model),
                            LoraConfig(r=r, lora_alpha=alpha, target_modules=REGEX, init_lora_weights=False))
        p = tmp_path / ("a%d" % k)
        pm.save_pretrained(str(p))
        paths.append(str(p))
    out = tmp_path / "sum"
    prov = concat_adapters(paths, [w for _, _, w in specs], str(out))
    assert prov["rank"] == 6
    ids = torch.randint(0, VOCAB, (1, 12), generator=torch.Generator().manual_seed(1))
    merged = PeftModel.from_pretrained(Qwen3_5ForConditionalGeneration.from_pretrained(tiny_model), str(out)).eval()
    expect = Qwen3_5ForConditionalGeneration.from_pretrained(tiny_model).eval()
    from safetensors.torch import load_file
    mods = dict(expect.named_modules())
    for p, (r, alpha, w) in zip(paths, specs):
        sd = load_file(os.path.join(p, "adapter_model.safetensors"))
        for k in sd:
            if k.endswith(".lora_A.weight"):
                name = re.sub(r"^base_model\.model\.", "", k[: -len(".lora_A.weight")])
                with torch.no_grad():
                    mods[name].weight += w * alpha / r * (sd[k.replace("lora_A", "lora_B")] @ sd[k])
    with torch.no_grad():
        a = merged(input_ids=ids, use_cache=False).logits
        b = expect(input_ids=ids, use_cache=False).logits
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
    with torch.no_grad(), merged.disable_adapter():                # disabled: the base model again
        c = merged(input_ids=ids, use_cache=False).logits
    base = Qwen3_5ForConditionalGeneration.from_pretrained(tiny_model).eval()
    with torch.no_grad():
        assert torch.allclose(c, base(input_ids=ids, use_cache=False).logits, atol=1e-6)


def _dist_worker(rank, world, port, cfg):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    TR.train(cfg, backend=FakeBackend(), log=lambda s: None)


def test_two_ranks_match_one(tiny_model, tmp_path, problems_file):
    """Token-balanced sharding + gradient all-reduce over 2 processes (gloo) gives the update a single
    process computes, up to float summation order."""
    import socket
    import torch.multiprocessing as mp
    one = make_cfg(tiny_model, tmp_path / "one", problems_file, **{"grpo.steps": 2})
    TR.train(one, backend=FakeBackend(), log=lambda s: None)
    two = make_cfg(tiny_model, tmp_path / "two", problems_file, **{"grpo.steps": 2})
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    mp.spawn(_dist_worker, args=(2, port, two), nprocs=2, join=True)
    from safetensors.torch import load_file
    a = load_file(str(tmp_path / "one" / "ckpt" / "step_00002" / "adapter" / "adapter_model.safetensors"))
    b = load_file(str(tmp_path / "two" / "ckpt" / "step_00002" / "adapter" / "adapter_model.safetensors"))
    assert a.keys() == b.keys()
    worst = max((a[k] - b[k]).abs().max().item() for k in a)
    assert worst < 1e-5, worst
    m1, m2 = TR.read_metrics(str(tmp_path / "one")), TR.read_metrics(str(tmp_path / "two"))
    assert m1[1]["tokens"] == m2[1]["tokens"] and m1[1]["pg"] == pytest.approx(m2[1]["pg"], rel=1e-5)
