"""Gradient-sketch probe (llm_finetune/rl_dataxp/grad_probe.py): the count sketch and the two-part
(rationale / action) per-reply gradients, on the tiny Qwen3_5 model of the trainer tests."""
import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")

from llm_finetune.rl_dataxp import grad_probe as GP  # noqa: E402


def test_mix64_is_splitmix64():
    def ref(z):
        z &= (1 << 64) - 1
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return z ^ (z >> 31)
    xs = [0, 1, 2, 12345, 2 ** 31, 2 ** 40 + 17, 466911231]
    got = GP._mix64(torch.tensor(xs, dtype=torch.int64)).tolist()
    assert [g & ((1 << 64) - 1) for g in got] == [ref(x) for x in xs]


def test_count_sketch_is_linear_and_keeps_dot_products():
    g = torch.Generator().manual_seed(0)
    n, dim = 400_000, 1 << 14
    x = torch.randn(n, generator=g)
    y = 0.3 * x + torch.randn(n, generator=g)
    sx, sy = GP.sketch_flat(x, dim), GP.sketch_flat(y, dim)
    # linear, and a vector sketched in pieces (with their offsets) is the vector sketched whole
    assert torch.allclose(GP.sketch_flat(x + 2 * y, dim), sx + 2 * sy, atol=1e-3)
    pieces = GP.sketch_flat(x[:1000], dim)
    GP.sketch_flat(x[1000:], dim, offset=1000, out=pieces)
    assert torch.allclose(pieces, sx, atol=1e-4)
    cos = float(x @ y / (x.norm() * y.norm()))
    cos_s = float(sx @ sy / (sx.norm() * sy.norm()))
    assert abs(cos_s - cos) < 5 / math.sqrt(dim)
    assert abs(float(sx @ sx) / float(x @ x) - 1) < 5 * math.sqrt(2 / dim)
    # buckets are used evenly and the signs are balanced
    ones = GP.sketch_flat(torch.ones(n), dim)
    assert abs(float(ones.sum())) < 5 * math.sqrt(n)
    assert float(GP.sketch_flat(torch.ones(n), dim, seed=1) @ ones) < 0.2 * float(ones @ ones)
    with pytest.raises(ValueError):
        GP.sketch_flat(x, 1000)


def test_split_point():
    assert GP.split_point([5, 9, 7, 9, 3, 4], 9) == 4          # after the LAST end-of-think token
    assert GP.split_point([5, 6, 7], 9) == 3                    # never closed: all rationale
    assert GP.split_point([5, 6, 9], 9) == 3                    # closed, nothing after
    assert GP.split_point([5, 6, 7], None) == 3


def test_probe_samples_keep_every_call_of_every_group():
    recs = [{"key": "s/b/m1", "group": "s/b", "variant": "v", "calls": [{"prompt_token_ids": [1], "token_ids": [2, 3]}]},
            {"key": "s/a/m0", "group": "s/a", "variant": "v",
             "calls": [{"prompt_token_ids": [1], "token_ids": [2]}, {"prompt_token_ids": [1], "token_ids": []},
                       {"prompt_token_ids": [4], "token_ids": [5]}]}]
    out = GP.probe_samples(recs, 7)
    assert [(s["key"], s["turn"]) for s in out] == [("s/a/m0", 0), ("s/a/m0", 2), ("s/b/m1", 0)]
    assert all(s["step"] == 7 for s in out)


# ---- on the tiny model -----------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from test_rl_dataxp_trainer import make_cfg, policy_of, problems_file, tiny_model  # noqa: E402,F401
    HAVE_MODEL = True
except BaseException:                                           # the trainer tests skip themselves here
    HAVE_MODEL = False


def _full_grad_sketch(pol, prompt, comp, T, dim, sl):
    for p in pol.params:
        p.grad = None
    lp, _ = pol.logprobs(prompt, comp, T)
    lp[sl].sum().backward()
    sk, n2 = GP.sketch_grads(pol.params, dim)
    flat = torch.cat([p.grad.flatten() for p in pol.params]).clone()
    for p in pol.params:
        p.grad = None
    return sk, n2, flat


@pytest.mark.skipif(not HAVE_MODEL, reason="needs the tiny Qwen3_5 model of the trainer tests")
@pytest.mark.parametrize("long_path", [False, True])
def test_two_part_sketches_are_the_parts_of_the_gradient(tiny_model, tmp_path, problems_file, long_path):  # noqa: F811
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration
    from test_rl_dataxp_trainer import REGEX
    init = tmp_path / "init_adapter"
    torch.manual_seed(5)
    get_peft_model(Qwen3_5ForConditionalGeneration.from_pretrained(tiny_model),
                   LoraConfig(r=4, lora_alpha=8, target_modules=REGEX, init_lora_weights=False)).save_pretrained(str(init))
    over = ({"grpo.logit_chunk": 2, "grpo.offload_min_tokens": 1, "grpo.layer_segment": 4, "grpo.seq_chunk": 3}
            if long_path else {"grpo.logit_chunk": 0, "grpo.offload_min_tokens": None})
    cfg = make_cfg(tiny_model, tmp_path, problems_file, **dict(over, **{"policy.init_adapter": str(init)}))
    cfg["grpo"]["kl_reference"] = "base"
    pol = policy_of(cfg)
    dim, T = 1 << 12, 0.8
    END = 9
    s = {"prompt_ids": [3, 5, 7, 11, 13, 17], "completion_ids": [19, 23, 29, 31, END, 37, 41, 43]}
    res = GP.sample_sketches(pol, s, T, dim, END)
    assert (res["n_r"], res["n_a"]) == (5, 3)
    assert all(p.grad is None or float(p.grad.abs().max()) == 0 for p in pol.params)      # left clean
    sk_r, g2_r, flat_r = _full_grad_sketch(pol, s["prompt_ids"], s["completion_ids"], T, dim, slice(0, 5))
    sk_a, g2_a, flat_a = _full_grad_sketch(pol, s["prompt_ids"], s["completion_ids"], T, dim, slice(5, 8))
    sk_all, g2_all, flat_all = _full_grad_sketch(pol, s["prompt_ids"], s["completion_ids"], T, dim, slice(0, 8))
    assert g2_r > 0 and g2_a > 0
    tol = 2e-3 if long_path else 1e-4
    for got, want in ((res["sk_r"], sk_r), (res["sk_a"], sk_a), (res["sk_r"] + res["sk_a"], sk_all)):
        assert float((got - want).norm() / want.norm()) < tol
    assert abs(res["g2_r"] / g2_r - 1) < tol and abs(res["g2_a"] / g2_a - 1) < tol
    assert float((flat_r + flat_a - flat_all).norm() / flat_all.norm()) < tol
    # the two parts are different directions, and the sketch sees that
    cos = float(flat_r @ flat_a / (flat_r.norm() * flat_a.norm()))
    cos_s = float(res["sk_r"] @ res["sk_a"] / (res["sk_r"].norm() * res["sk_a"].norm()))
    assert abs(cos) < 0.99 and abs(cos_s - cos) < 6 / math.sqrt(dim)
    # a reply whose rationale never closes: all rationale, an all-zero action sketch
    s2 = {"prompt_ids": [3, 5, 7], "completion_ids": [19, 23, 29]}
    r2 = GP.sample_sketches(pol, s2, T, dim, END)
    assert (r2["n_r"], r2["n_a"]) == (3, 0) and float(r2["sk_a"].abs().max()) == 0 and r2["g2_r"] > 0


@pytest.mark.skipif(not HAVE_MODEL, reason="needs the tiny Qwen3_5 model of the trainer tests")
def test_a_freed_graph_is_run_again(tiny_model, tmp_path, problems_file, monkeypatch):  # noqa: F811
    """If the first backward frees the graph, the second part is computed from a fresh forward pass."""
    cfg = make_cfg(tiny_model, tmp_path, problems_file)
    cfg["grpo"]["kl_reference"] = "base"
    pol = policy_of(cfg)
    torch.manual_seed(1)
    for p in pol.params:                                       # a fresh LoRA has B = 0: give it a gradient everywhere
        p.data.add_(0.05 * torch.randn_like(p))
    s = {"prompt_ids": [3, 5, 7, 11], "completion_ids": [19, 23, 9, 37, 41]}
    want = GP.sample_sketches(pol, s, 1.0, 1 << 10, 9)
    assert want["refwd"] == 0
    real = torch.Tensor.backward

    def no_retain(self, *a, **k):
        k["retain_graph"] = False
        return real(self, *a, **k)
    monkeypatch.setattr(torch.Tensor, "backward", no_retain)
    got = GP.sample_sketches(pol, s, 1.0, 1 << 10, 9)
    assert got["refwd"] == 1
    for name in ("sk_r", "sk_a"):
        assert float((got[name] - want[name]).norm() / want[name].norm()) < 1e-4
