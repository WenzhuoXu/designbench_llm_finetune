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


def _decoder_layers(pol):
    return [m for m in pol.trunk.modules() if hasattr(m, "block_type") and hasattr(m, "_gradient_checkpointing_func")]


def _grads(pol, prompt, comp):
    for p in pol.params:
        p.grad = None
    lp, _ = pol.logprobs(prompt, comp, 0.9)
    (lp.sum() * 1.0).backward()
    return lp.detach(), [p.grad.detach().clone() if p.grad is not None else None for p in pol.params]


def test_chunked_logprobs_and_offloaded_checkpoints_are_exact(tiny_model, tmp_path, problems_file):
    """Long-sequence paths: the loss over the completion in chunks of 2 tokens, every layer's
    checkpointed input offloaded (threshold 1 token), each layer's MLP and token-mixer LoRA
    projections run over the sequence in checkpointed chunks of 3 tokens, and each layer itself run
    in segments (4 tokens: 4+4+3; 5 tokens: 5+6, the 1-token tail joined; 2 tokens with the
    3-token chunks inside), give the plain path's log-probs and grads."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration
    init = tmp_path / "init_adapter"
    torch.manual_seed(5)
    get_peft_model(Qwen3_5ForConditionalGeneration.from_pretrained(tiny_model),
                   LoraConfig(r=4, lora_alpha=8, target_modules=REGEX, init_lora_weights=False)).save_pretrained(str(init))
    prompt, comp = [3, 5, 7, 9, 11, 13], [17, 19, 23, 29, 31]
    whole = {"grpo.logit_chunk": 2, "grpo.offload_min_tokens": 1, "grpo.layer_segment": None}
    runs, long_calls = {}, {}
    for name, over in (("plain", {"grpo.logit_chunk": 0, "grpo.offload_min_tokens": None}),
                       ("chunked", {"grpo.logit_chunk": 2, "grpo.offload_min_tokens": None}),
                       ("offload", whole),
                       ("seqchunk", dict(whole, **{"grpo.seq_chunk": 3})),
                       ("segment", dict(whole, **{"grpo.layer_segment": 4, "grpo.seq_chunk": None})),
                       ("segtail", dict(whole, **{"grpo.layer_segment": 5, "grpo.seq_chunk": None})),
                       ("segseq", dict(whole, **{"grpo.layer_segment": 2, "grpo.seq_chunk": 3}))):
        cfg = make_cfg(tiny_model, tmp_path / name, problems_file, **dict(over, **{"policy.init_adapter": str(init)}))
        pol = policy_of(cfg)
        assert bool(pol.long_path) == name.startswith("seg"), (name, pol.long_path)
        if pol.long_path:                  # the start-up check: every layer segmented, same hidden states and grads
            assert pol.long_path["layers"] == len(_decoder_layers(pol)) and pol.long_path["rel_err"] < 1e-4, pol.long_path
            assert set(pol.long_path["grad_err"]) == {"linear_attention", "full_attention"}, pol.long_path
            assert max(pol.long_path["grad_err"].values()) < 1e-4 and pol.long_path["trunk_grad_rel"] < 1e-3, pol.long_path
            assert all(p.grad is None for p in pol.params) and pol._ckpt_state["long_calls"] == 0
        if name == "seqchunk":
            calls = []             # what the MLP's first projection sees: chunks, not the 11 tokens
            gate = next(m for n, m in pol.model.named_modules() if n.endswith("layers.0.mlp.gate_proj"))
            inner = gate.forward
            gate.forward = lambda x, *a, **k: (calls.append(x.shape[1]), inner(x, *a, **k))[1]
            assert pol.n_seq_chunked > 0
        if name.startswith("seg"):
            seen = {}              # tokens per call of each layer kind's token mixer
            for layer in _decoder_layers(pol):
                mixer = layer.linear_attn if layer.block_type == "linear_attention" else layer.self_attn.q_proj
                mixer.register_forward_pre_hook(
                    lambda mod, args, kwargs, kind=layer.block_type:
                        seen.setdefault(kind, []).append((args[0] if args else kwargs["hidden_states"]).shape[1]),
                    with_kwargs=True)
        runs[name] = _grads(pol, prompt, comp)
        long_calls[name] = pol._ckpt_state["long_calls"]
        if name == "seqchunk":
            assert calls and max(calls) == 3 and sum(calls) % (len(prompt) + len(comp)) == 0, calls
        if name.startswith("seg"):
            assert set(seen) == {"linear_attention", "full_attention"}, seen
            want = {"segment": [4, 4, 3], "segtail": [5, 6], "segseq": [2, 2, 2, 2, 3]}[name]
            for kind, sizes in seen.items():       # the forward and the recomputations, per layer
                assert sizes[:len(want)] == want and len(sizes) % len(want) == 0 and set(sizes) == set(want),                     (name, kind, sizes)
            assert long_calls[name] == len(_decoder_layers(pol)), (name, long_calls[name])   # every layer
        else:
            assert long_calls[name] == 0, (name, long_calls[name])
    lp0, g0 = runs["plain"]
    assert all(g is not None for g in g0)
    scale = max(g.abs().max().item() for g in g0)
    for name in runs:
        lp, g = runs[name]
        assert torch.allclose(lp, lp0, atol=1e-5), name
        worst = max((a - b).abs().max().item() for a, b in zip(g, g0))
        # a segment boundary inside one of the delta rule's 64-token chunks changes the order of its
        # float32 sums: rounding only (test_segmented_layers_are_exact_in_float64)
        assert worst < (1e-4 * scale if name.startswith("seg") else 1e-5), (name, worst, scale)


def test_segmented_layers_are_exact_in_float64(tiny_model, tmp_path, problems_file, monkeypatch):
    """The segmented layers are the same function, not an approximation: in float64 (with a float64
    copy of the reference delta rule, which otherwise casts to float32 inside) log-probs and grads
    agree with the plain path to 1e-10 for segments of 4, 5 (5+6 of 11 tokens: the joined 1-token
    tail), 16, 64 and 2 with 3-token chunks inside."""
    import inspect
    import sys
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration
    M = sys.modules[Qwen3_5ForConditionalGeneration.__module__]
    src = inspect.getsource(M.torch_chunk_gated_delta_rule)
    src = src[src.index("def torch_chunk_gated_delta_rule"):]
    assert "torch.float32" in src
    ns = dict(M.__dict__)
    exec(src.replace("torch.float32", "torch.float64"), ns)
    monkeypatch.setattr(M, "torch_chunk_gated_delta_rule", ns["torch_chunk_gated_delta_rule"])
    init = tmp_path / "init_adapter"
    torch.manual_seed(5)
    get_peft_model(Qwen3_5ForConditionalGeneration.from_pretrained(tiny_model),
                   LoraConfig(r=4, lora_alpha=8, target_modules=REGEX, init_lora_weights=False)).save_pretrained(str(init))
    seg = {"grpo.logit_chunk": 2, "grpo.offload_min_tokens": 1, "grpo.seq_chunk": None}
    for prompt, comp, sizes in (([3, 5, 7, 9, 11, 13], [17, 19, 23, 29, 31], (4, 5)),
                                (list(range(3, 40)), list(range(50, 83)), (2, 16, 64))):
        runs = {}
        for name, over in [("plain", {"grpo.logit_chunk": 0, "grpo.offload_min_tokens": None})] +                 [("seg%d" % n, dict(seg, **{"grpo.layer_segment": n, "grpo.seq_chunk": 3 if n == 2 else None}))
                 for n in sizes]:
            cfg = make_cfg(tiny_model, tmp_path / ("%s_%d" % (name, len(prompt))), problems_file,
                           **dict(over, **{"policy.init_adapter": str(init), "model.torch_dtype": "float64"}))
            pol = policy_of(cfg)
            runs[name] = _grads(pol, prompt, comp)
            assert pol._ckpt_state["long_calls"] == (0 if name == "plain" else len(_decoder_layers(pol)))
        lp0, g0 = runs["plain"]
        scale = max(g.abs().max().item() for g in g0)
        for name, (lp, g) in runs.items():
            worst = max((a - b).abs().max().item() for a, b in zip(g, g0))
            assert (lp - lp0).abs().max().item() < 1e-10 and worst < 1e-10 * scale, (name, worst, scale)


def test_segmented_reference_pass_matches(tiny_model, tmp_path, problems_file):
    """Without grad (the KL reference) a long sequence runs through the same segmented layers and
    gives the plain path's log-probs; segment sizes cover a sequence, a 1-token tail joins."""
    assert TR._segments(11, 4) == [4, 4, 3] and TR._segments(11, 5) == [5, 6] and TR._segments(8, 4) == [4, 4]
    assert TR._segments(3, 64) == [3] and TR._segments(1, 4) == [1] and TR._segments(5, 1) == [2, 3]
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration
    init = tmp_path / "init_adapter"
    torch.manual_seed(7)
    get_peft_model(Qwen3_5ForConditionalGeneration.from_pretrained(tiny_model),
                   LoraConfig(r=4, lora_alpha=8, target_modules=REGEX, init_lora_weights=False)).save_pretrained(str(init))
    prompt, comp = list(range(3, 40)), list(range(50, 83))            # 70 tokens
    out = {}
    for name, over in (("plain", {"grpo.offload_min_tokens": None}),
                       ("seg", {"grpo.offload_min_tokens": 1, "grpo.layer_segment": 16, "grpo.seq_chunk": 8})):
        cfg = make_cfg(tiny_model, tmp_path / name, problems_file,
                       **dict(over, **{"policy.init_adapter": str(init), "grpo.kl_reference": "init"}))
        pol = policy_of(cfg)
        out[name] = pol.ref_logprobs(prompt, comp, 0.9)
        assert pol._ckpt_state["long_calls"] == (len(_decoder_layers(pol)) if name == "seg" else 0)
        assert all(p.requires_grad for p in pol.params)
    assert torch.allclose(out["plain"], out["seg"], atol=1e-5), (out["plain"] - out["seg"]).abs().max()


@pytest.mark.parametrize("state", ["recurrent", "conv"])
def test_start_up_check_catches_a_gradient_lost_at_a_segment_boundary(tiny_model, tmp_path, problems_file,
                                                                      monkeypatch, state):
    """A segment state that reaches the next segment with the right values but cut from the graph
    leaves the hidden states exact, so only the gradient part of the start-up check can see it: the
    policy must refuse to build."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration
    init = tmp_path / "init_adapter"
    torch.manual_seed(5)
    get_peft_model(Qwen3_5ForConditionalGeneration.from_pretrained(tiny_model),
                   LoraConfig(r=4, lora_alpha=8, target_modules=REGEX, init_lora_weights=False)).save_pretrained(str(init))
    if state == "recurrent":
        def cut(self, recurrent_states, layer_idx, state_idx=0, **kwargs):
            self.rec_out = recurrent_states.detach()
            return recurrent_states
        monkeypatch.setattr(TR._SegmentState, "update_recurrent_state", cut)
    else:
        inner = TR._SegmentState.update_conv_state

        def cut(self, *args, **kwargs):
            full = inner(self, *args, **kwargs)
            self.conv_out = self.conv_out.detach()
            return full
        monkeypatch.setattr(TR._SegmentState, "update_conv_state", cut)
    cfg = make_cfg(tiny_model, tmp_path / "run", problems_file,
                   **{"policy.init_adapter": str(init), "grpo.offload_min_tokens": 1, "grpo.layer_segment": 4})
    with pytest.raises(RuntimeError, match="gradients differ"):
        policy_of(cfg)


# ---------------------------------------------------------------- a full file system

class _Clock:
    """Stands in for the time module inside diskio: sleep() moves the clock, nothing waits."""

    def __init__(self):
        self.t, self.sleeps = 0.0, 0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s
        self.sleeps += 1


QUOTA = "Error while serializing: I/O error: Disk quota exceeded (os error 122)"    # safetensors, job 167222


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _same_run(a, b, step=3):
    from safetensors.torch import load_file
    for name in ("optimizer.pt", "scheduler.pt", "rng_rank0.pt"):
        x, y = (torch.load(os.path.join(str(r), "ckpt", "step_%05d" % step, name), weights_only=False) for r in (a, b))
        assert _same(x, y), name
    x, y = (load_file(os.path.join(str(r), "ckpt", "step_%05d" % step, "adapter", "adapter_model.safetensors"))
            for r in (a, b))
    assert _same(x, y)
    ma, mb = TR.read_metrics(str(a)), TR.read_metrics(str(b))
    assert sorted(ma) == sorted(mb) == list(range(step))
    assert all(ma[s]["pg"] == mb[s]["pg"] and ma[s]["kl"] == mb[s]["kl"] for s in ma)


def test_torch_save_reports_the_full_disk_itself(tmp_path, monkeypatch):
    import errno

    class FullFile:
        def __init__(self, f):
            self.f, self.n = f, 0

        def write(self, b):
            self.n += 1
            if self.n > 1:
                raise OSError(errno.ENOSPC, "No space left on device")
            return self.f.write(b)

        def __getattr__(self, k):
            return getattr(self.f, k)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.f.close()

    obj = {"w": torch.arange(4096.0), "n": 3}
    p = str(tmp_path / "optimizer.pt")
    with monkeypatch.context() as mp:
        mp.setattr(TR, "open", lambda path, mode="r": FullFile(open(path, mode)), raising=False)
        with pytest.raises(OSError) as e:
            TR.torch_save(obj, p)
    assert e.value.errno == errno.ENOSPC
    TR.torch_save(obj, p)                                 # the next attempt replaces what the failed one left
    assert _same(torch.load(p), obj)


def test_checkpoints_wait_out_a_full_file_system(tiny_model, tmp_path, problems_file, monkeypatch):
    import errno
    from llm_finetune.rl_dataxp import diskio
    TR.train(make_cfg(tiny_model, tmp_path / "full", problems_file), backend=FakeBackend(), log=lambda s: None)

    clock = _Clock()
    monkeypatch.setattr(diskio, "time", clock)
    left = {"adapter": 2, "optimizer": 1, "rng": 1, "metrics": 1}
    save_adapter, torch_save, append_metrics = TR.Policy.save_adapter, TR.torch_save, TR.append_metrics

    def full_adapter(self, path):
        if "step_00002" in path and left["adapter"]:
            left["adapter"] -= 1
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "adapter_model.safetensors"), "wb") as f:
                f.write(b"half a file")
            raise RuntimeError(QUOTA)
        return save_adapter(self, path)

    def full_torch_save(obj, path):
        for name, key in (("optimizer.pt", "optimizer"), ("rng_rank0.pt", "rng")):
            if path.endswith(os.path.join("step_00003.tmp", name)) and left[key]:
                left[key] -= 1
                with open(path, "wb") as f:
                    f.write(b"half a file")
                raise OSError(errno.ENOSPC, "No space left on device")
        return torch_save(obj, path)

    def full_metrics(run_dir, m):
        if m["step"] == 1 and left["metrics"]:
            left["metrics"] -= 1
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return append_metrics(run_dir, m)

    monkeypatch.setattr(TR.Policy, "save_adapter", full_adapter)
    monkeypatch.setattr(TR, "torch_save", full_torch_save)
    monkeypatch.setattr(TR, "append_metrics", full_metrics)
    logs = []
    TR.train(make_cfg(tiny_model, tmp_path / "part", problems_file), backend=FakeBackend(), log=logs.append)
    assert not any(left.values()) and clock.sleeps == 5
    assert any("file system full while writing checkpoint 2" in s for s in logs)
    assert any("checkpoint 3 written after waiting 1.0 min" in s for s in logs)
    assert not [d for d in os.listdir(tmp_path / "part" / "ckpt") if d.endswith(".tmp")]
    _same_run(tmp_path / "full", tmp_path / "part")
    assert os.path.exists(tmp_path / "part" / "DONE")


def test_a_disk_full_past_the_wait_leaves_no_partial_checkpoint(tiny_model, tmp_path, problems_file, monkeypatch):
    TR.train(make_cfg(tiny_model, tmp_path / "full", problems_file), backend=FakeBackend(), log=lambda s: None)
    save_adapter = TR.Policy.save_adapter

    def full_adapter(self, path):
        if "step_00002" in path:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "adapter_model.safetensors"), "wb") as f:
                f.write(b"half a file")
            raise RuntimeError(QUOTA)
        return save_adapter(self, path)

    run = tmp_path / "part"
    with monkeypatch.context() as mp:
        mp.setattr(TR.Policy, "save_adapter", full_adapter)
        with pytest.raises(RuntimeError, match="Disk quota exceeded"):
            TR.train(make_cfg(tiny_model, run, problems_file, disk_full_wait_min=0), backend=FakeBackend(),
                     log=lambda s: None)
    assert TR.latest_ckpt(str(run)) == 1 and not os.path.exists(run / "ckpt" / "step_00002")
    assert sorted(TR.read_metrics(str(run))) == [0]
    # the next job repeats step 1 and ends where an uninterrupted run does
    be = FakeBackend()
    TR.train(make_cfg(tiny_model, run, problems_file), backend=be, log=lambda s: None)
    assert [os.path.basename(os.path.dirname(a)) for a in be.adapters] == ["step_00001", "step_00002"]
    _same_run(tmp_path / "full", run)


def test_append_metrics_replaces_the_file(tmp_path):
    with open(tmp_path / "metrics.jsonl", "w") as f:
        f.write(json.dumps({"step": 0, "kl": 0.1}))                # no newline at the end
    TR.append_metrics(str(tmp_path), {"step": 1, "kl": 0.2})
    assert open(tmp_path / "metrics.jsonl").read().splitlines() == ['{"step": 0, "kl": 0.1}', '{"step": 1, "kl": 0.2}']
    assert sorted(TR.read_metrics(str(tmp_path))) == [0, 1] and sorted(os.listdir(tmp_path)) == ["metrics.jsonl"]
