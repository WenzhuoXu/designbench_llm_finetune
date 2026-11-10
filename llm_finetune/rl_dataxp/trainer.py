"""Multi-turn GRPO for dataxp: the training loop.

    torchrun --nproc_per_node N -m llm_finetune.rl_dataxp.trainer --config configs/rl/dataxp_grpo_p2_datav4.yaml \\
        run_dir=/path/run rollout.servers='["http://127.0.0.1:8001/v1"]'

One step:
  1. rank 0 publishes the current adapter (ckpt/step_k/adapter) to the vLLM servers as policy_sk;
  2. rank 0 runs the step's episodes through the harness (rollout.py, in the simulator env):
     problems_per_step problems x group_size rollouts each, horizon turns, temperature T;
  3. every rank reads the records and builds the same samples (samples.py): one sample per model
     call, weighted by its episode's group-normalised advantage; zero-variance groups are dropped;
  4. each rank takes a token-balanced share; per sample, the clipped policy-gradient term plus
     kl_coef x k3-KL to the reference (the base: the adapter disabled), summed over completion
     tokens and divided by the step's total completion tokens (one constant per step: the
     unbiased token-sum estimator up to scale);
  5. gradients are all-reduced, clipped, AdamW steps, the state is checkpointed as step k+1.

Log-probs are always recomputed by the trainer. vLLM's are recorded only for a diagnostic: on
this model bf16 engines disagree by 0.12-0.19 nats per token (DISTILLATION.md), so vLLM's
numbers are not used as the old policy. With grpo.iterations = 1 the ratio is exactly 1 and the
update is the plain policy gradient; iterations > 1 reuse the batch with PPO clipping.

Resume: every step ends with an atomic checkpoint (adapter, optimizer, scheduler, RNG, config
fingerprint, parameter checksum). A restarted job continues from the newest complete one; the
step's rollouts resume by episode key; the problem draw is a pure function of (seed, step).

Stop rules (guards): the run stops itself, writes STOPPED.json and exits 4 when the training
parse rate falls by more than guards.parse_drop, when the KL to the base grows by more than
guards.kl_increase_max, or when the policy has not moved after guards.stall_steps (learning
rate too low, the Turn-0 failure).
"""
from __future__ import annotations

import argparse
import copy
import datetime
import functools
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zlib
from typing import Callable, Optional

from .diskio import WriteErrors, fsync_tree, retry_when_full, write_text
from .samples import build_step_samples, shard_samples

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULTS = {
    "run_dir": None,
    "seed": 0,
    "model": {"path": None, "torch_dtype": "bfloat16", "attn_implementation": "sdpa",
              "trust_remote_code": True, "gradient_checkpointing": True, "cache_dir": None},
    "policy": {"init_adapter": None, "require_init_adapter": False,
               "lora": {"r": 64, "lora_alpha": 128, "lora_dropout": 0.0, "bias": "none",
                        "target_modules": None}},
    "harness": {"script": None, "argv": [], "env": {}},
    "rollout": {"backend": "subprocess", "python": None, "servers": [], "problems_file": None,
                "variants": None, "problems_per_step": 8, "group_size": 8, "horizon": 12,
                "temperature": 0.8,
                # max_tokens None: no completion cap, a reply ends at EOS or the context length
                "inject": {"return_token_ids": True, "logprobs": True, "top_p": 1.0, "top_k": 0,
                           "max_tokens": None},
                "workers": 64, "server_wait_s": 3600, "keep_full_every": 10, "max_crash_frac": 0.25},
    "reward": {"speed_bonus": 0.2},
    "grpo": {"steps": 150, "lr": 1.0e-5, "warmup_steps": 3, "betas": [0.9, 0.99], "eps": 1.0e-8,
             "weight_decay": 0.0, "max_grad_norm": 1.0, "kl_coef": 0.04, "kl_reference": "base",
             "clip_eps_low": 0.2, "clip_eps_high": 0.28, "iterations": 1, "scale_rewards": "group",
             "tis_cap": None, "max_turns_per_episode": None,
             # long sequences: lm-head + loss over the completion in chunks of logit_chunk tokens;
             # checkpointed layer inputs offloaded to CPU at or above offload_min_tokens, and at or
             # above it each layer's MLP and token-mixer LoRA projections run over the sequence in
             # (checkpointed) chunks of seq_chunk tokens; and each decoder layer is itself run in
             # pieces of layer_segment tokens (a multiple of 64, the delta rule's chunk; null = the
             # whole layer at once). Memory only: none of them changes the math.
             "logit_chunk": 2048, "offload_min_tokens": 8192, "seq_chunk": 4096, "layer_segment": 4096},
    "checkpoint": {"keep_last": 2, "keep_adapter_every": 25, "keep_adapter_steps": [50, 150]},
    "guards": {"enabled": True, "window": 5, "parse_drop": 0.10, "kl_increase_max": 0.10,
               "stall_steps": 30, "stall_kl": 0.005, "ignore_stopped": False},
    "dist": {"timeout_min": 360},
    # a write (checkpoint, metrics, rollout records) that fails on a full file system is repeated
    # every minute for this long before the run fails. 0 = fail at once. The limit is per write,
    # and the other ranks wait meanwhile for at most dist.timeout_min: inside a checkpoint two
    # writes can follow each other (2 x this must stay under dist.timeout_min); between two steps
    # more can (metrics, compaction, the rollout request, each episode record), so a pool that
    # stays full for hours and lets a write through now and then still ends the job, with a
    # timeout on the waiting ranks. Nothing is lost then: the next job resumes.
    "disk_full_wait_min": 120,
    "allow_config_change": False,
}

# what a resumed run must share with the run that wrote the checkpoint (grpo.steps may grow)
FINGERPRINT_KEYS = [("seed",), ("model", "path"), ("policy",), ("harness",), ("reward",),
                    ("rollout", "problems_per_step"), ("rollout", "group_size"), ("rollout", "horizon"),
                    ("rollout", "temperature"), ("rollout", "inject"), ("rollout", "variants"),
                    ("grpo", "lr"), ("grpo", "warmup_steps"), ("grpo", "kl_coef"), ("grpo", "kl_reference"),
                    ("grpo", "iterations"), ("grpo", "scale_rewards"), ("grpo", "clip_eps_low"),
                    ("grpo", "clip_eps_high"), ("grpo", "tis_cap"), ("grpo", "max_grad_norm"),
                    ("grpo", "max_turns_per_episode")]


# ------------------------------------------------------------------ config

def deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    import yaml
    for o in overrides:
        key, sep, val = o.partition("=")
        if not sep:
            raise SystemExit("override %r is not key=value" % o)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def load_config(path: Optional[str], overrides: list[str] = ()) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        import yaml
        deep_update(cfg, yaml.safe_load(open(path)) or {})
    return apply_overrides(cfg, list(overrides))


def fingerprint(cfg: dict, problems_sha1: str) -> str:
    sub = {}
    for path in FINGERPRINT_KEYS:
        node = cfg
        for p in path:
            node = node.get(p) if isinstance(node, dict) else None
        sub[".".join(path)] = node
    sub["problems_sha1"] = problems_sha1
    return hashlib.sha1(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ problems

def load_problems(path: str, variants: Optional[list[str]] = None) -> dict[str, list[dict]]:
    by = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by.setdefault(r["variant"], []).append(r["spec"])
    if variants:
        missing = [v for v in variants if v not in by]
        if missing:
            raise SystemExit("problems file %s has no problems for %s" % (path, missing))
        by = {v: by[v] for v in variants}
    return {v: sorted(s, key=lambda x: x["problem_id"]) for v, s in sorted(by.items())}


def _stable(*parts) -> int:
    return zlib.crc32(json.dumps(parts, default=str).encode()) & 0x7FFFFFFF


class ProblemSampler:
    """Deterministic problem draw: per_step problems per step, spread round-robin over the
    variants (the remainder rotates with the step), each variant walking its own seeded
    permutation, reshuffled every pass. A pure function of (seed, step), so resume needs no state."""

    def __init__(self, problems: dict[str, list[dict]], per_step: int, seed: int):
        self.problems = problems
        self.variants = sorted(problems)
        self.per_step = per_step
        self.seed = seed
        self._perm_cache = {}

    def counts(self, step: int) -> dict[str, int]:
        nv = len(self.variants)
        base, extra = divmod(self.per_step, nv)
        return {v: base + (1 if (i - step) % nv < extra else 0) for i, v in enumerate(self.variants)}

    def _perm(self, v: str, epoch: int) -> list[int]:
        key = (v, epoch)
        if key not in self._perm_cache:
            idx = list(range(len(self.problems[v])))
            random.Random(_stable(self.seed, v, epoch)).shuffle(idx)
            self._perm_cache[key] = idx
        return self._perm_cache[key]

    def draw(self, step: int) -> list[tuple[str, dict]]:
        before = {v: 0 for v in self.variants}
        for s in range(step):
            for v, n in self.counts(s).items():
                before[v] += n
        out = []
        for v, n in self.counts(step).items():
            P = self.problems[v]
            for j in range(n):
                epoch, k = divmod(before[v] + j, len(P))
                out.append((v, P[self._perm(v, epoch)[k]]))
        return out


def step_episodes(sampler: ProblemSampler, step: int, group_size: int, seed: int, n_servers: int) -> list[dict]:
    eps = []
    for j, (v, spec) in enumerate(sampler.draw(step)):
        g = "s%05d/%02d_%s" % (step, j, spec["problem_id"])
        for m in range(group_size):
            eps.append({"key": "%s/m%d" % (g, m), "group": g, "member": m, "variant": v, "spec": spec,
                        "seed": _stable(seed, step, j, m), "server": j % max(1, n_servers)})
    return eps


# ------------------------------------------------------------------ rollout backend

def check_crashes(recs: list[dict], max_frac: float, step: int, where: str) -> None:
    """A crashed episode is dropped from its group, so a broken setup (a wrong --code path, a
    missing simulator) would otherwise train on nothing, silently."""
    crashed = sum(1 for x in recs if x["episode"].get("crashed"))
    if crashed > max_frac * len(recs):
        raise RuntimeError("step %d: %d of %d episodes crashed (rollout.max_crash_frac %.2f); see %s"
                           % (step, crashed, len(recs), max_frac, where))


class SubprocessRollout:
    """Publishes the adapter to the vLLM servers and runs rollout.py in the simulator env."""

    def __init__(self, cfg: dict, run_dir: str):
        r = cfg["rollout"]
        if not r["servers"]:
            raise SystemExit("rollout.servers is empty")
        for k, v in (("rollout.python", r["python"]), ("harness.script", cfg["harness"]["script"])):
            if not v or not os.path.exists(v):
                raise SystemExit("%s must name an existing file, got %r" % (k, v))
        argv = list(cfg["harness"]["argv"])
        code = argv[argv.index("--code") + 1] if "--code" in argv else None
        if not code or not os.path.isdir(os.path.join(code, "dataxp")):
            raise SystemExit("harness.argv must give --code <dir holding the dataxp package>, got %r" % code)
        for flag in ("--model", "--base-url", "--temperature", "--turns", "--policy", "--loop-guard"):
            if flag in argv:
                raise SystemExit("harness.argv must not set %s (the trainer sets it per step)" % flag)
        self.cfg = cfg
        self.wait_min = float(cfg.get("disk_full_wait_min") or 0)
        self.dir = os.path.join(run_dir, "rollouts")
        retry_when_full(lambda: os.makedirs(self.dir, exist_ok=True), self.dir, self.wait_min)

    def path(self, step: int) -> str:
        return os.path.join(self.dir, "step_%05d.jsonl" % step)

    def collect(self, step: int, adapter_dir: str, episodes: list[dict]) -> list[dict]:
        from . import vllm_admin
        r = self.cfg["rollout"]
        name = "policy_s%05d" % step
        vllm_admin.wait_ready(r["servers"], r["server_wait_s"])
        vllm_admin.publish(r["servers"], name, os.path.abspath(adapter_dir))
        req = {"script": self.cfg["harness"]["script"], "harness_argv": self.cfg["harness"]["argv"],
               "servers": r["servers"], "model": name, "temperature": r["temperature"],
               "turns": r["horizon"], "inject": r["inject"], "workers": r["workers"],
               "env": self.cfg["harness"].get("env") or {}, "episodes": episodes,
               "disk_full_wait_min": self.wait_min}
        req_path = os.path.join(self.dir, "step_%05d.request.json" % step)
        retry_when_full(lambda: write_text(req_path, json.dumps(req)), "the rollout request", self.wait_min)
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        rc = subprocess.run([r["python"], "-m", "llm_finetune.rl_dataxp.rollout", "run",
                             "--request", req_path, "--out", self.path(step)], env=env, cwd=REPO_ROOT).returncode
        if rc != 0:
            raise RuntimeError("rollout for step %d exited %d" % (step, rc))
        recs = self.load(step, episodes)
        check_crashes(recs, float(r["max_crash_frac"]), step, self.path(step))
        return recs

    def load(self, step: int, episodes: list[dict]) -> list[dict]:
        recs = {}
        for line in open(self.path(step)):
            if line.strip():
                rec = json.loads(line)
                recs[rec["key"]] = rec
        missing = [e["key"] for e in episodes if e["key"] not in recs]
        if missing:
            raise RuntimeError("step %d: %d episodes missing from %s, e.g. %s"
                               % (step, len(missing), self.path(step), missing[:3]))
        return [recs[e["key"]] for e in episodes]

    def compact(self, step: int) -> None:
        """Drop the token arrays of a finished step (they are ~200 kB per episode)."""
        p = self.path(step)

        def write():
            if not os.path.exists(p):                # a repeat after the compact file replaced it
                return
            with open(p + ".compact.tmp", "w") as fo:
                for line in open(p):
                    if line.strip():
                        rec = json.loads(line)
                        rec["calls"] = [{"n_prompt": len(c.get("prompt_token_ids") or []),
                                         "n_completion": len(c.get("token_ids") or []),
                                         "finish": c.get("finish"), "seed": c.get("seed")}
                                        for c in rec.get("calls") or []]
                        fo.write(json.dumps(rec) + "\n")
                fo.flush()
                os.fsync(fo.fileno())                # the full file goes only once this one is on disk
            os.replace(p + ".compact.tmp", p.replace(".jsonl", ".compact.jsonl"))
            os.remove(p)

        retry_when_full(write, "the compacted rollouts of step %d" % step, self.wait_min)


# ------------------------------------------------------------------ long sequences

class _OffloadedCheckpoint:
    """Checkpointing that keeps a decoder layer's input on the CPU instead of the GPU.

    Gradient checkpointing keeps one hidden-state tensor per layer for the backward pass:
    64 layers x sequence x 5120 x bf16, about 26 GB at 40k tokens, which with the 27B weights and
    the LoRA/Adam state does not fit an 80 GB GPU. With no completion cap a turn can be that long,
    so above a threshold the saved input goes to pinned CPU memory and comes back for the
    recomputation. Exact: the layer is recomputed with the same inputs (dropout is off)."""

    @staticmethod
    def make():
        import torch

        class Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, run_fn, hidden):
                ctx.run_fn = run_fn
                ctx.device = hidden.device
                cpu = torch.empty(hidden.shape, dtype=hidden.dtype, device="cpu",
                                  pin_memory=hidden.is_cuda)
                cpu.copy_(hidden.detach(), non_blocking=hidden.is_cuda)
                ctx.cpu = cpu
                with torch.no_grad():
                    out = run_fn(hidden)
                if not torch.is_tensor(out):
                    raise TypeError("offloaded checkpointing expects a layer that returns one tensor")
                return out

            @staticmethod
            def backward(ctx, grad_out):
                h = ctx.cpu.to(ctx.device, non_blocking=True).detach().requires_grad_(True)
                ctx.cpu = None
                with torch.enable_grad():
                    out = ctx.run_fn(h)
                torch.autograd.backward(out, grad_out)
                return None, h.grad

        return Fn


def _segments(total: int, size: int) -> list:
    """Lengths of consecutive pieces of `size` tokens covering `total`. A 1-token tail joins the
    piece before it: with a state from earlier tokens, a 1-token call takes the model's
    single-step decode path, which is a different kernel."""
    size = max(2, int(size))
    sizes = [size] * (total // size)
    tail = total - size * len(sizes)
    if tail == 1 and sizes:
        sizes[-1] += 1
    elif tail:
        sizes.append(tail)
    return sizes


class _SegmentState:
    """Stands in for the generation cache of ONE linear-attention layer over one segment: gives the
    layer the state the segment before left (the conv's last inputs, the delta rule's recurrent
    state) and keeps the state it leaves. Same arithmetic as transformers' LinearAttentionLayer
    cache, without its in-place copies, so both states stay in the autograd graph."""

    record_past = False

    def __init__(self, layer_idx: int, rec=None, conv=None):
        self.layer_idx = layer_idx
        self.had_state = rec is not None
        self.recurrent_states = {0: rec}
        self.conv_states = {0: conv}
        self.rec_out = self.conv_out = None

    @property
    def layers(self):                       # built per access: no reference cycle through self
        return {self.layer_idx: self}

    def has_previous_state(self, layer_idx=None, state_idx=None):
        return self.had_state

    def update_conv_state(self, conv_states, layer_idx, state_idx=0, conv_kernel_size=None, **kwargs):
        import torch
        k = int(conv_kernel_size or conv_states.shape[-1])
        if self.had_state:
            full = torch.cat([self.conv_states[0], conv_states], dim=-1)
        else:
            full = conv_states
            if full.shape[-1] < k:
                full = torch.nn.functional.pad(full, (k - full.shape[-1], 0), value=0)
        # a copy, not a view: the next segment's checkpoint keeps its inputs until its backward, and a
        # view would keep this segment's whole conv input alive with it (sequence x conv width per layer)
        self.conv_out = full[..., -k:].clone()
        return full

    def update_recurrent_state(self, recurrent_states, layer_idx, state_idx=0, **kwargs):
        self.rec_out = recurrent_states
        return recurrent_states


def _long_linear_layer(layer, call, kw: dict, hidden, seg: int):
    """One linear-attention (gated delta net) decoder layer over a long sequence, a segment at a
    time. The layer is causal with a fixed-size state, so segment i needs only its own tokens and
    the state after segment i-1; with grad each segment is checkpointed, and the backward pass is
    back-propagation through time over the segments: memory O(segment), not O(sequence). The
    whole layer at once is what ran out at 54k tokens (dx38mtR, step 43): its float32 norms, the
    conv and the delta rule's backward buffers all scale with the sequence."""
    import torch
    from torch.utils.checkpoint import checkpoint
    idx = layer.linear_attn.layer_idx
    mask = kw.get("attention_mask")
    grad = torch.is_grad_enabled()

    def run(x, rec=None, conv=None, m=None):
        cache = _SegmentState(idx, rec, conv)
        out = call(x, **dict(kw, past_key_values=cache, attention_mask=m))
        if not torch.is_tensor(out) or cache.rec_out is None or cache.conv_out is None:
            raise RuntimeError("layer %d did not go through the segment state" % idx)
        return out, cache.rec_out, cache.conv_out

    sizes = _segments(hidden.shape[1], seg)
    masks = torch.split(mask, sizes, dim=1) if mask is not None else [None] * len(sizes)
    outs, rec, conv = [], None, None
    for x, m in zip(torch.split(hidden, sizes, dim=1), masks):
        fn = functools.partial(run, m=m)
        args = (x,) if rec is None else (x, rec, conv)
        out, rec, conv = checkpoint(fn, *args, use_reentrant=False) if grad else fn(*args)
        outs.append(out)
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)


def _long_attention_layer(layer, kw: dict, hidden, seg: int):
    """One full-attention decoder layer over a long sequence in three stages: what happens before
    the attention (input norm, q/k/v projections, their norms, rotary) per chunk of tokens, the
    attention itself over the whole sequence (it needs every key), and what happens after it (gate,
    output projection, residual, norm, MLP, residual) per chunk again. With grad the per-chunk
    stages are checkpointed, so only q, k, v, the gate and the attention output are kept at full
    length. The stages repeat Qwen3_5DecoderLayer.forward / Qwen3_5Attention.forward operation for
    operation; Policy.check_long_path compares them against the stock layers at start-up."""
    import torch
    from torch.utils.checkpoint import checkpoint
    att = layer.self_attn
    mod = sys.modules[type(att).__module__]
    cos, sin = kw["position_embeddings"]
    grad = torch.is_grad_enabled()
    extra = {k: v for k, v in kw.items() if k not in ("position_embeddings", "attention_mask", "past_key_values")}

    def pre(x, c, s):
        y = layer.input_layernorm(x)
        shape = (*y.shape[:-1], -1, att.head_dim)
        q, gate = torch.chunk(att.q_proj(y).view(*y.shape[:-1], -1, att.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*y.shape[:-1], -1)
        q = att.q_norm(q.view(shape)).transpose(1, 2)
        k = att.k_norm(att.k_proj(y).view(shape)).transpose(1, 2)
        v = att.v_proj(y).view(shape).transpose(1, 2)
        q, k = mod.apply_rotary_pos_emb(q, k, c, s)
        return q, k, v, gate

    def post(x, a, gate):
        a = a.reshape(*x.shape[:-1], -1).contiguous() * torch.sigmoid(gate)
        x = x + att.o_proj(a)
        return x + layer.mlp(layer.post_attention_layernorm(x))

    sizes = _segments(hidden.shape[1], seg)
    xs = torch.split(hidden, sizes, dim=1)
    parts = [checkpoint(pre, x, c, s, use_reentrant=False) if grad else pre(x, c, s)
             for x, c, s in zip(xs, torch.split(cos, sizes, dim=1), torch.split(sin, sizes, dim=1))]
    q, k, v = (torch.cat([p[i] for p in parts], dim=2) for i in range(3))
    gates = [p[3] for p in parts]
    del parts
    interface = mod.ALL_ATTENTION_FUNCTIONS.get_interface(att.config._attn_implementation,
                                                          mod.eager_attention_forward)
    a, _ = interface(att, q, k, v, kw.get("attention_mask"),
                     dropout=0.0 if not att.training else att.attention_dropout, scaling=att.scaling, **extra)
    del q, k, v
    outs = [checkpoint(post, x, ai, g, use_reentrant=False) if grad else post(x, ai, g)
            for x, ai, g in zip(xs, torch.split(a, sizes, dim=1), gates)]
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)


def _long_layer(run_fn, state: dict):
    """The segmented equivalent of one checkpointed decoder-layer call (a function of the layer's
    input), or None when the call is not one of the two layer kinds as this code knows them: the
    caller then runs the layer whole, as before."""
    import torch
    seg = state.get("layer_segment")
    call, kw = getattr(run_fn, "func", None), getattr(run_fn, "keywords", None)
    layer = getattr(call, "__self__", None)
    if not seg or layer is None or kw is None or getattr(run_fn, "args", None) or kw.get("past_key_values") is not None:
        return None
    kind, mask = getattr(layer, "block_type", None), kw.get("attention_mask")
    if kind == "linear_attention" and hasattr(getattr(layer, "linear_attn", None), "layer_idx"):
        if mask is not None and not (torch.is_tensor(mask) and mask.dim() == 2):
            return None
        return lambda h: _long_linear_layer(layer, call, kw, h, int(seg))
    if kind == "full_attention" and hasattr(layer, "self_attn") and kw.get("position_embeddings") is not None:
        att, mod = layer.self_attn, sys.modules.get(type(layer.self_attn).__module__)
        if all(hasattr(att, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "head_dim",
                                         "scaling", "attention_dropout")) \
                and all(hasattr(mod, n) for n in ("apply_rotary_pos_emb", "ALL_ATTENTION_FUNCTIONS",
                                                  "eager_attention_forward")):
            return lambda h: _long_attention_layer(layer, kw, h, int(seg))
    return None


def install_checkpointing(model, state: dict) -> int:
    """Replace every decoder layer's checkpoint function: torch's non-reentrant checkpoint below
    state['offload_min_tokens']; at or above it (state['tokens'] is set per forward) the layer's
    input is kept on the CPU and the layer runs in segments of state['layer_segment'] tokens
    (_long_layer; state['long_calls'] counts the layer calls that did). Without grad (the reference
    pass) the layer just runs, in segments, and nothing is kept. Returns the number of layers
    patched."""
    import torch
    from torch.utils.checkpoint import checkpoint
    Fn = _OffloadedCheckpoint.make()

    def ckpt(run_fn, *args, **kwargs):
        thr = state.get("offload_min_tokens")
        if thr and state.get("tokens", 0) >= thr and len(args) == 1 and torch.is_tensor(args[0]) and not kwargs:
            fn = _long_layer(run_fn, state)
            if fn is not None:
                state["long_calls"] = state.get("long_calls", 0) + 1
            fn = fn or run_fn
            return Fn.apply(fn, args[0]) if torch.is_grad_enabled() else fn(args[0])
        return checkpoint(run_fn, *args, use_reentrant=False, **kwargs)

    n = 0
    for mod in model.modules():
        if hasattr(mod, "_gradient_checkpointing_func"):
            mod._gradient_checkpointing_func = ckpt
            n += 1
    return n


# a decoder layer's per-token modules: the MLP, and the LoRA-wrapped projections of its token mixer
_SEQ_CHUNK_MODULES = re.compile(r"(^|\.)layers\.\d+\.(mlp|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)"
                                r"|self_attn\.(q_proj|k_proj|v_proj|o_proj))$")


def install_seq_chunking(model, state: dict) -> int:
    """Run each decoder layer's per-token modules over the sequence in chunks of state['seq_chunk']
    tokens when the sequence is long (state['tokens'] >= state['offload_min_tokens']).

    Recomputing one layer of a 24k-token turn with grad keeps that layer's activations: the MLP's
    17408-wide intermediates and the float32 copies of every LoRA input (PEFT casts the input to the
    adapter's float32), about 12 GiB, which is what ran out on an 80 GB GPU (dx38mtR, step 13).
    With grad, each chunk is checkpointed, so a layer keeps only its input and recomputes one chunk
    at a time in the backward; without grad (the offloaded forward, the reference pass) the chunks
    simply run one after another. The modules act on each token separately, so the result is the
    unchunked one up to float summation order. Returns the number of modules wrapped."""
    import torch
    from torch.utils.checkpoint import checkpoint

    def wrap(fwd):
        def run(x, *args, **kwargs):
            step, thr = state.get("seq_chunk"), state.get("offload_min_tokens")
            if (not step or not thr or args or kwargs or state.get("tokens", 0) < thr
                    or not torch.is_tensor(x) or x.dim() != 3 or x.shape[1] <= step):
                return fwd(x, *args, **kwargs)
            parts = range(0, x.shape[1], int(step))
            if torch.is_grad_enabled():
                return torch.cat([checkpoint(fwd, x[:, s:s + step], use_reentrant=False) for s in parts], dim=1)
            return torch.cat([fwd(x[:, s:s + step]) for s in parts], dim=1)
        return run

    n = 0
    for name, mod in model.named_modules():
        if _SEQ_CHUNK_MODULES.search(name):
            mod.forward = wrap(mod.forward)
            n += 1
    return n


# ------------------------------------------------------------------ policy

class Policy:
    """Base model + the trainable LoRA adapter 'default' (+ a frozen 'ref' copy of the initial
    adapter when grpo.kl_reference = init)."""

    def __init__(self, cfg: dict, device):
        import torch
        from peft import PeftModel
        from llm_finetune.models.loader import _apply_lora, _model_class

        self.cfg = cfg
        self.device = device
        m = cfg["model"]
        if not m["path"]:
            raise SystemExit("model.path must be set")
        cls = _model_class(m["path"], m.get("cache_dir"), {"trust_remote_code": m["trust_remote_code"]})
        dtype = getattr(torch, m["torch_dtype"])
        kw = dict(attn_implementation=m["attn_implementation"], trust_remote_code=m["trust_remote_code"])
        if m.get("cache_dir"):
            kw["cache_dir"] = m["cache_dir"]
        if device.type == "cuda":
            kw["device_map"] = {"": device.index}     # load straight onto this rank's GPU
        try:
            base = cls.from_pretrained(m["path"], dtype=dtype, **kw)
        except TypeError:
            base = cls.from_pretrained(m["path"], torch_dtype=dtype, **kw)
        base.to(device)
        self._ckpt_state = {"offload_min_tokens": cfg["grpo"].get("offload_min_tokens"),
                            "seq_chunk": cfg["grpo"].get("seq_chunk"),
                            "layer_segment": cfg["grpo"].get("layer_segment"), "tokens": 0, "long_calls": 0}
        if m["gradient_checkpointing"]:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            install_checkpointing(base, self._ckpt_state)
            if self._ckpt_state["offload_min_tokens"]:
                base.enable_input_require_grads()   # the offloaded (reentrant) path needs a grad-carrying input
        if hasattr(base, "config"):
            base.config.use_cache = False
        # the trunk (last hidden state) and the output layer, called separately so the loss can run
        # over the completion in chunks: the full logits of a long reply (tokens x 248k) never exist
        self.trunk = getattr(base, "model", None)
        self.head = base.get_output_embeddings()
        if self.trunk is None or self.head is None:
            raise RuntimeError("cannot find the decoder trunk / output layer of %s" % type(base).__name__)
        init = cfg["policy"]["init_adapter"]
        if cfg["policy"].get("require_init_adapter") and not init:
            raise SystemExit("policy.init_adapter is required by this config")
        # torch_device: PEFT's default is "cuda", which safetensors reads as cuda:0, so every rank
        # loaded the adapter (twice, with the ref copy) onto rank 0's GPU: 2.26 GiB per other rank
        # there, the memory rank 0 lacked when it ran out (dx38mtR, step 13). On the CPU instead;
        # the weights are then copied into the adapter's parameters on this rank's GPU.
        if init:
            model = PeftModel.from_pretrained(base, init, is_trainable=True, adapter_name="default",
                                              torch_device="cpu")
        else:
            lora = dict(cfg["policy"]["lora"])
            if not lora.get("target_modules"):
                raise SystemExit("policy.lora.target_modules must be set for a fresh adapter")
            torch.manual_seed(int(cfg["seed"]))
            model = _apply_lora(base, lora)
        self.ref_mode = cfg["grpo"]["kl_reference"]
        if self.ref_mode not in ("base", "init"):
            raise SystemExit("grpo.kl_reference must be base or init")
        if self.ref_mode == "init" and init:
            model.load_adapter(init, adapter_name="ref", is_trainable=False, torch_device="cpu")
            model.set_adapter("default")
        elif self.ref_mode == "init":
            self.ref_mode = "base"              # a fresh adapter starts at B = 0: init == base
        for mod in model.modules():              # no dropout: the policy's log-probs must be deterministic
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
        model.train()
        self.n_seq_chunked = install_seq_chunking(model, self._ckpt_state)
        self.model = model
        self.trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not self.trainable:
            raise RuntimeError("no trainable parameters")
        bad = [n for n, _ in self.trainable if ".default." not in n and "lora_" in n]
        if bad:
            raise RuntimeError("non-default adapter parameters are trainable: %s" % bad[:3])
        self.params = [p for _, p in self.trainable]
        self.long_path = self.check_long_path()

    def check_long_path(self, tokens: int = 300, tol: float = 0.05, grad_tol: float = 1e-3,
                        trunk_cos_min: float = 0.9) -> dict:
        """The segmented layers re-state the model's own layer code, so before training compare
        them with it on a short random sequence, plain against the long-sequence path with 64-token
        segments (called once, from __init__: it clears the parameters' .grad).
        (1) Forward: every decoder layer must take the segmented path and the hidden states must
        agree (to the model dtype's rounding: the operations are the same ones).
        (2) Gradients, one layer of each kind, whole against segmented, in float32 (_layer_grad_err):
        back-propagation across a segment boundary runs kernel code the whole layer never does (the
        delta rule's gradient to its initial state and from its final state, the conv over
        [state | segment]), and (1) says nothing about it. Limit grad_tol.
        (3) Gradients of the whole trunk to the trainable parameters in the model's own dtype,
        through the path training takes (offloaded checkpoints around the segments): the cosine must
        be at least trunk_cos_min. Loose on purpose, bf16 rounding differs between the two orders of
        operations; (2) is the sharp test."""
        import torch
        st = self._ckpt_state
        if not (self.cfg["model"]["gradient_checkpointing"] and st.get("offload_min_tokens")
                and st.get("layer_segment")):
            return {}
        layers = [m for m in self.trunk.modules()
                  if hasattr(m, "_gradient_checkpointing_func") and hasattr(m, "block_type")]
        n_layers = len(layers)
        vocab = self.model.get_input_embeddings().num_embeddings
        ids = torch.randint(0, vocab, (1, tokens), generator=torch.Generator().manual_seed(0)).to(self.device)
        seen = {}                               # layer kind -> (a layer, its input, its keyword arguments)

        def grab(mod, args, kwargs):
            if mod.block_type not in seen and args and torch.is_tensor(args[0]):
                seen[mod.block_type] = (mod, args[0].detach(), dict(kwargs))

        def hidden():
            with torch.no_grad():
                return self.trunk(input_ids=ids, use_cache=False).last_hidden_state.float()

        def grads():
            with torch.enable_grad():
                h = self.trunk(input_ids=ids, use_cache=False).last_hidden_state.float()
                w = torch.randn(h.shape, generator=torch.Generator().manual_seed(1)).to(h.device)
                (h * w).sum().backward()
            out = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in self.params]
            for p in self.params:
                p.grad = None
            return out

        hooks = [m.register_forward_pre_hook(grab, with_kwargs=True) for m in layers]
        saved = dict(st)
        try:
            st["tokens"] = 0
            plain = hidden()
            for h in hooks:
                h.remove()
            g_plain = grads()
            st.update(tokens=tokens, offload_min_tokens=1, layer_segment=64, seq_chunk=64, long_calls=0)
            seg = hidden()
            n_long = st["long_calls"]
            g_seg = grads()
        finally:
            for h in hooks:
                h.remove()
            st.clear()
            st.update(saved)
        err = float((plain - seg).abs().max() / plain.abs().max().clamp_min(1e-9))
        if n_long != n_layers or not err < tol:
            raise RuntimeError("the segmented long-sequence layers do not reproduce the model's own: %d of %d layers "
                               "segmented, relative error %.3g (limit %.3g). transformers changed? "
                               "grpo.layer_segment=null runs layers whole" % (n_long, n_layers, err, tol))
        dot = sum(float((a.double() * b.double()).sum()) for a, b in zip(g_plain, g_seg))
        na, nb, nd = (math.sqrt(sum(float(v.double().pow(2).sum()) for v in vs))
                      for vs in (g_plain, g_seg, (a - b for a, b in zip(g_plain, g_seg))))
        cos, rel = dot / max(na * nb, 1e-300), nd / max(na, 1e-300)
        del g_plain, g_seg
        grad_err = {kind: self._layer_grad_err(*seen[kind]) for kind in sorted(seen)}
        bad = {k: v for k, v in grad_err.items() if not v < grad_tol}
        if bad or set(grad_err) != {m.block_type for m in layers} or not cos >= trunk_cos_min:
            raise RuntimeError("the segmented layers' gradients differ from the whole layers': one layer of each kind, "
                               "relative error %s (limit %.3g); whole trunk, cosine %.6f (at least %.3g), relative "
                               "difference %.3g. grpo.layer_segment=null runs layers whole"
                               % (grad_err, grad_tol, cos, trunk_cos_min, rel))
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {"layers": n_long, "rel_err": err, "grad_err": grad_err, "trunk_grad_cos": cos, "trunk_grad_rel": rel}

    @staticmethod
    def _layer_grad_err(layer, x, kw: dict, seg: int = 64) -> float:
        """One decoder layer, whole against segmented, with gradients, on a float32 copy of it (in
        bf16 the rounding would hide a small error; the kernels and the code are the same ones). The
        largest of three relative differences: the output, the gradient to the layer's input, the
        gradient to its trainable parameters, for a fixed random linear function of the output. A
        boundary that drops or mangles the gradient of the carried state shows here."""
        import torch
        dt = torch.float64 if x.dtype == torch.float64 else torch.float32
        lay = copy.deepcopy(layer).to(dt)
        for m in lay.modules():                 # install_seq_chunking's wrappers call the ORIGINAL modules
            m.__dict__.pop("forward", None)

        def cast(v):
            if isinstance(v, (tuple, list)):
                return type(v)(cast(u) for u in v)
            return v.to(dt) if torch.is_tensor(v) and v.is_floating_point() else v

        kw = {k: cast(v) for k, v in kw.items()}
        call = torch.nn.Module.__call__.__get__(lay)        # the layer itself, past its checkpointing wrapper
        long_fn = _long_layer(functools.partial(call, **kw), {"layer_segment": seg})
        if long_fn is None:
            raise RuntimeError("no segmented path for a %s layer" % getattr(layer, "block_type", "?"))
        ps = [p for p in lay.parameters() if p.requires_grad]
        w = torch.randn(x.shape, generator=torch.Generator().manual_seed(1)).to(x.device, dt)
        res = []
        for fn in (lambda h: call(h, **kw), long_fn):
            h = x.detach().clone().to(dt).requires_grad_(True)
            with torch.enable_grad():
                out = fn(h)
                (out * w).sum().backward()
            res.append([out.detach(), h.grad] + ([torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p))
                                                             .flatten() for p in ps])] if ps else []))
            for p in ps:
                p.grad = None
        return max(float((a - b).double().norm() / a.double().norm().clamp_min(1e-30)) for a, b in zip(*res))

    def _restore_grad_flags(self):
        names = {n for n, _ in self.trainable}
        for n, p in self.model.named_parameters():
            p.requires_grad_(n in names)

    def logprobs(self, prompt_ids, completion_ids, temperature: float, raw: bool = False):
        """Per-token log-probs of the completion at temperature T (with grad when enabled), and
        optionally at T=1 (detached; vLLM reports unscaled log-probs by default). The trunk runs
        once; the output layer and the loss run over the completion in checkpointed chunks."""
        import torch
        import torch.nn.functional as F
        from torch.utils.checkpoint import checkpoint
        P, C = len(prompt_ids), len(completion_ids)
        ids = torch.tensor([list(prompt_ids) + list(completion_ids)], device=self.device)
        self._ckpt_state["tokens"] = P + C
        hid = self.trunk(input_ids=ids, use_cache=False).last_hidden_state[0, P - 1:P - 1 + C]
        tgt = torch.tensor(list(completion_ids), device=self.device)
        step = int(self.cfg["grpo"].get("logit_chunk") or C) or C
        head = self.head

        def chunk_lp(h, t):
            return -F.cross_entropy(head(h).float() / temperature, t, reduction="none")

        parts, raws = [], []
        for s in range(0, C, step):
            h, t = hid[s:s + step], tgt[s:s + step]
            parts.append(checkpoint(chunk_lp, h, t, use_reentrant=False) if torch.is_grad_enabled() else chunk_lp(h, t))
            if raw:
                with torch.no_grad():
                    raws.append(-F.cross_entropy(head(h.detach()).float(), t, reduction="none"))
        lp = torch.cat(parts)
        return lp, (torch.cat(raws) if raw else None)

    def ref_logprobs(self, prompt_ids, completion_ids, temperature: float):
        import torch
        with torch.no_grad():
            if self.ref_mode == "base":
                with self.model.disable_adapter():
                    lp, _ = self.logprobs(prompt_ids, completion_ids, temperature)
            else:
                self.model.set_adapter("ref")
                try:
                    lp, _ = self.logprobs(prompt_ids, completion_ids, temperature)
                finally:
                    self.model.set_adapter("default")
                    self._restore_grad_flags()
        return lp.detach()

    def checksum(self) -> float:
        return float(sum(p.detach().double().sum().item() for p in self.params))

    def save_adapter(self, path: str) -> None:
        self.model.save_pretrained(path, selected_adapters=["default"])

    def load_adapter_weights(self, path: str) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(os.path.join(path, "adapter_model.safetensors"))
        res = set_peft_model_state_dict(self.model, sd, adapter_name="default")
        unexpected = getattr(res, "unexpected_keys", None) or []
        if unexpected:
            raise RuntimeError("adapter keys not in the model: %s" % unexpected[:3])


# ------------------------------------------------------------------ distributed helpers

class Dist:
    def __init__(self, timeout_min: int):
        import torch
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local = int(os.environ.get("LOCAL_RANK", "0"))
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda", self.local) if self.cuda else torch.device("cpu")
        if self.cuda:
            torch.cuda.set_device(self.device)
        self.cpu_group = None
        if self.world > 1:
            import torch.distributed as dist
            t = datetime.timedelta(minutes=timeout_min)
            dist.init_process_group("nccl" if self.cuda else "gloo", timeout=t)
            # waits on rank 0's rollouts happen on a CPU group: no GPU spin, long timeout
            self.cpu_group = dist.new_group(backend="gloo", timeout=t) if self.cuda else None

    def barrier(self):
        if self.world > 1:
            import torch.distributed as dist
            dist.barrier(group=self.cpu_group)

    def broadcast_object(self, obj):
        if self.world == 1:
            return obj
        import torch.distributed as dist
        box = [obj]
        dist.broadcast_object_list(box, src=0, group=self.cpu_group)
        return box[0]

    def all_reduce_(self, t, op: str = "sum"):
        if self.world > 1:
            import torch.distributed as dist
            dist.all_reduce(t, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
        return t

    def all_reduce_grads(self, params, bucket_bytes: int = 256 << 20):
        if self.world == 1:
            return
        import torch
        import torch.distributed as dist
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        bucket, size = [], 0
        for p in params + [None]:
            if p is not None:
                bucket.append(p.grad)
                size += p.grad.numel() * p.grad.element_size()
            if bucket and (p is None or size >= bucket_bytes):
                flat = _flatten_dense_tensors(bucket)
                dist.all_reduce(flat)
                for g, s in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
                    g.copy_(s)
                bucket, size = [], 0

    def broadcast_params(self, params):
        if self.world > 1:
            import torch.distributed as dist
            for p in params:
                dist.broadcast(p.data, src=0)

    def close(self):
        if self.world > 1:
            import torch.distributed as dist
            dist.destroy_process_group()


# ------------------------------------------------------------------ checkpoints

def ckpt_dir(run_dir: str, step: int) -> str:
    return os.path.join(run_dir, "ckpt", "step_%05d" % step)


def latest_ckpt(run_dir: str) -> Optional[int]:
    root = os.path.join(run_dir, "ckpt")
    best = None
    if os.path.isdir(root):
        for d in os.listdir(root):
            if d.startswith("step_") and not d.endswith(".tmp"):
                p = os.path.join(root, d)
                if os.path.exists(os.path.join(p, "COMPLETE")) and os.path.exists(os.path.join(p, "optimizer.pt")):
                    best = max(best or 0, int(d[5:]))
    return best


def torch_save(obj, path: str) -> None:
    """torch.save, flushed to disk; a failed write raises the OSError itself (torch reports a full
    disk as "PytorchStreamWriter failed writing file", which does not say why)."""
    import torch
    with open(path, "wb") as f:
        w = WriteErrors(f)
        try:
            torch.save(obj, w)
            f.flush()
            os.fsync(f.fileno())
        except BaseException as e:
            if w.error is not None and e is not w.error:
                raise w.error from e
            raise


def save_ckpt(run_dir, step, policy, opt, sched, dist, state: dict, wait_min: float = 0.0,
              log: Callable[[str], None] = print) -> None:
    """Write ckpt/step_N.tmp, flush it to disk, mark it COMPLETE and rename it: a checkpoint
    either is whole or does not exist. Each write is repeated for up to wait_min minutes while
    the file system is full (the other ranks wait at the barriers)."""
    import torch
    d = ckpt_dir(run_dir, step)
    tmp = d + ".tmp"
    name = "checkpoint %d" % step

    def start():
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)

    def body():
        policy.save_adapter(os.path.join(tmp, "adapter"))
        fsync_tree(os.path.join(tmp, "adapter"))
        torch_save(opt.state_dict(), os.path.join(tmp, "optimizer.pt"))
        torch_save(sched.state_dict(), os.path.join(tmp, "scheduler.pt"))
        write_text(os.path.join(tmp, "state.json"), json.dumps(dict(state, step=step, checksum=policy.checksum())))

    def commit():
        write_text(os.path.join(tmp, "COMPLETE"), time.strftime("%F %T"))
        if os.path.exists(d):
            shutil.rmtree(d)
        os.replace(tmp, d)

    if dist.rank == 0:
        retry_when_full(start, name, wait_min, log)
    dist.barrier()
    if dist.rank == 0:
        retry_when_full(body, name, wait_min, log)
    rng = {"python": random.getstate(), "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    retry_when_full(lambda: torch_save(rng, os.path.join(tmp, "rng_rank%d.pt" % dist.rank)),
                    name + " (rank %d rng)" % dist.rank, wait_min,
                    log if dist.rank == 0 else print)          # the log of the other ranks is silent
    dist.barrier()
    if dist.rank == 0:
        retry_when_full(commit, name, wait_min, log)
    dist.barrier()


def append_metrics(run_dir: str, m: dict) -> None:
    """Add a line to metrics.jsonl by replacing the file, so a failed write leaves no half line."""
    p = os.path.join(run_dir, "metrics.jsonl")
    old = open(p).read() if os.path.exists(p) else ""
    if old and not old.endswith("\n"):
        old += "\n"
    write_text(p, old + json.dumps(m) + "\n")


def load_ckpt(run_dir, step, policy, opt, sched, dist) -> dict:
    import torch
    d = ckpt_dir(run_dir, step)
    policy.load_adapter_weights(os.path.join(d, "adapter"))
    opt.load_state_dict(torch.load(os.path.join(d, "optimizer.pt"), map_location=policy.device, weights_only=False))
    sched.load_state_dict(torch.load(os.path.join(d, "scheduler.pt"), weights_only=False))
    state = json.load(open(os.path.join(d, "state.json")))
    got, want = policy.checksum(), state["checksum"]
    if abs(got - want) > 1e-6 * max(1.0, abs(want)):
        raise RuntimeError("checkpoint %s: parameter checksum %.9g, saved %.9g" % (d, got, want))
    rp = os.path.join(d, "rng_rank%d.pt" % dist.rank)
    if os.path.exists(rp):
        rng = torch.load(rp, weights_only=False)
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    return state


def prune_ckpts(run_dir: str, cfg: dict, newest: int) -> None:
    c = cfg["checkpoint"]
    root = os.path.join(run_dir, "ckpt")
    steps = sorted(int(d[5:]) for d in os.listdir(root) if d.startswith("step_") and not d.endswith(".tmp"))
    full = set(steps[-int(c["keep_last"]):])
    for s in steps:
        if s in full or s > newest:
            continue
        d = ckpt_dir(run_dir, s)
        keep_adapter = s == 0 or s in set(c["keep_adapter_steps"] or []) or (
            c["keep_adapter_every"] and s % int(c["keep_adapter_every"]) == 0)
        if keep_adapter:
            for f in os.listdir(d):
                if f not in ("adapter", "state.json", "COMPLETE"):
                    p = os.path.join(d, f)
                    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        else:
            shutil.rmtree(d)


# ------------------------------------------------------------------ metrics and guards

def read_metrics(run_dir: str) -> dict[int, dict]:
    p = os.path.join(run_dir, "metrics.jsonl")
    out = {}
    if os.path.exists(p):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                out[r["step"]] = r                # a rerun step overwrites its earlier line
    return out


def check_guards(history: dict[int, dict], g: dict) -> Optional[str]:
    """A reason to stop, or None. history: step -> metrics (with parse_rate and kl)."""
    if not g["enabled"]:
        return None
    steps = sorted(history)
    w = int(g["window"])
    if len(steps) < 2 * w:
        return None

    def mean(key, ss):
        xs = [history[s][key] for s in ss if history[s].get(key) is not None]
        return sum(xs) / len(xs) if xs else None

    first, last = steps[:w], steps[-w:]
    p0, p1 = mean("parse_rate", first), mean("parse_rate", last)
    if p0 is not None and p1 is not None and p0 - p1 > g["parse_drop"]:
        return "parse rate fell from %.3f to %.3f (limit %.2f)" % (p0, p1, g["parse_drop"])
    k0, k1 = mean("kl", first), mean("kl", last)
    if k0 is not None and k1 is not None and k1 - k0 > g["kl_increase_max"]:
        return "KL to the reference rose from %.4f to %.4f (limit +%.2f)" % (k0, k1, g["kl_increase_max"])
    if k0 is not None and len(steps) >= int(g["stall_steps"]):
        moved = max(abs(history[s]["kl"] - k0) for s in steps if history[s].get("kl") is not None)
        if moved < g["stall_kl"]:
            return ("the policy has not moved after %d steps (max |KL - KL0| = %.5f < %.3f): raise the learning rate"
                    % (len(steps), moved, g["stall_kl"]))
    return None


# ------------------------------------------------------------------ the step

def train_on_samples(policy: Policy, samples: list[dict], dist: Dist, cfg: dict, opt, sched) -> dict:
    """One GRPO update (grpo.iterations passes over this step's samples). Returns summed stats."""
    import torch
    g = cfg["grpo"]
    T = float(cfg["rollout"]["temperature"])
    mine = shard_samples(samples, dist.world, dist.rank)
    # the per-step constant: completion tokens of the full (not subsampled) set, in expectation
    n_tok = sum(s.get("weight", 1.0) * len(s["completion_ids"]) for s in samples)
    stats = torch.zeros(9, dtype=torch.float64, device=policy.device)
    # 0 tokens 1 pg 2 kl 3 logp 4 clipped 5 gap_abs 6 gap_n 7 adv*tokens 8 samples
    if n_tok == 0:
        return {"skipped_update": True}
    cuda = policy.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(policy.device)
    old, ref = {}, {}
    gn = 0.0
    for it in range(int(g["iterations"])):
        for i, s in enumerate(mine):
            if it == 0 and g["kl_coef"]:
                ref[i] = policy.ref_logprobs(s["prompt_ids"], s["completion_ids"], T)
            lp, raw = policy.logprobs(s["prompt_ids"], s["completion_ids"], T, raw=(it == 0))
            if it == 0:
                old[i] = lp.detach()
            ratio = torch.exp(lp - old[i])
            A = float(s["advantage"])
            pg1 = -A * ratio
            pg2 = -A * torch.clamp(ratio, 1 - g["clip_eps_low"], 1 + g["clip_eps_high"])
            pg = torch.maximum(pg1, pg2)
            vl = s.get("vllm_logprobs")
            if g.get("tis_cap") and vl and len(vl) == len(s["completion_ids"]):
                w = torch.exp(old[i] - torch.tensor(vl, device=lp.device, dtype=lp.dtype)).clamp(max=float(g["tis_cap"]))
                pg = pg * w
            loss_tok = pg
            if g["kl_coef"]:
                d = ref[i] - lp
                kl = torch.exp(d) - d - 1
                loss_tok = loss_tok + g["kl_coef"] * kl
            (float(s.get("weight", 1.0)) * loss_tok.sum() / n_tok).backward()
            if it == 0:
                with torch.no_grad():
                    stats[0] += lp.numel()
                    stats[1] += pg.sum().double()
                    if g["kl_coef"]:
                        stats[2] += kl.sum().double()
                    stats[3] += lp.sum().double()
                    stats[4] += ((ratio < 1 - g["clip_eps_low"]) | (ratio > 1 + g["clip_eps_high"])).sum().double()
                    if vl and len(vl) == len(s["completion_ids"]) and raw is not None:
                        v = torch.tensor(vl, device=raw.device, dtype=torch.float64)
                        stats[5] += (raw.double() - v).abs().sum()
                        stats[6] += v.numel()
                    stats[7] += A * lp.numel()
                    stats[8] += 1
        dist.all_reduce_grads(policy.params)
        gn = float(torch.nn.utils.clip_grad_norm_(policy.params, float(g["max_grad_norm"])))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
    dist.all_reduce_(stats)
    # the memory headroom: the largest peak over the ranks, and the longest sample it had to fit
    peak = torch.tensor([torch.cuda.max_memory_allocated(policy.device) / 2 ** 30 if cuda else 0.0],
                        dtype=torch.float64, device=policy.device)
    dist.all_reduce_(peak, op="max")
    t = max(1.0, float(stats[0]))
    return {"tokens": int(stats[0]), "pg": float(stats[1]) / t, "kl": float(stats[2]) / t if g["kl_coef"] else None,
            "logp_mean": float(stats[3]) / t, "clip_frac": float(stats[4]) / t,
            "vllm_gap_abs": float(stats[5]) / float(stats[6]) if stats[6] > 0 else None,
            "adv_token_mean": float(stats[7]) / t, "trained_samples": int(stats[8]), "grad_norm": gn,
            "lr": sched.get_last_lr()[0], "skipped_update": False,
            "peak_mem_gib": round(float(peak[0]), 2) if cuda else None,
            "max_sample_tokens": max(len(s["prompt_ids"]) + len(s["completion_ids"]) for s in samples)}


# ------------------------------------------------------------------ main loop

def train(cfg: dict, backend=None, log: Callable[[str], None] = print) -> dict:
    """Run (or resume) the GRPO loop. Returns the metrics history. ``backend`` (tests) replaces
    the vLLM + harness rollout: an object with collect(step, adapter_dir, episodes) and
    load(step, episodes)."""
    import torch

    run_dir = cfg["run_dir"]
    for key, val in (("run_dir", run_dir), ("model.path", cfg["model"]["path"]),
                     ("rollout.problems_file", cfg["rollout"]["problems_file"])):
        if not val:
            raise SystemExit("%s must be set" % key)
    dist = Dist(int(cfg["dist"]["timeout_min"]))
    wait_min = float(cfg.get("disk_full_wait_min") or 0)
    if dist.rank != 0:
        log = lambda s: None                                   # noqa: E731
    try:
        if dist.rank == 0:
            retry_when_full(lambda: os.makedirs(run_dir, exist_ok=True), run_dir, wait_min, log)
        dist.barrier()
        done = os.path.join(run_dir, "DONE")
        if os.path.exists(done) and json.load(open(done)).get("steps", 0) >= int(cfg["grpo"]["steps"]):
            log("run %s is already complete" % run_dir)
            return read_metrics(run_dir)
        sp = os.path.join(run_dir, "STOPPED.json")
        if os.path.exists(sp) and not cfg["guards"]["ignore_stopped"]:
            raise SystemExit("run stopped earlier: %s (set guards.ignore_stopped=true to continue)" % open(sp).read())
        r = cfg["rollout"]
        problems = load_problems(r["problems_file"], r["variants"])
        problems_sha1 = hashlib.sha1(open(r["problems_file"], "rb").read()).hexdigest()
        fp = fingerprint(cfg, problems_sha1)
        sampler = ProblemSampler(problems, int(r["problems_per_step"]), int(cfg["seed"]))
        random.seed(int(cfg["seed"]) + dist.rank)
        torch.manual_seed(int(cfg["seed"]))
        policy = Policy(cfg, dist.device)
        if policy.long_path:
            lp = policy.long_path
            log("long-sequence path checked: %d layers segmented, hidden states relative error %.3g; gradients, "
                "one layer whole vs segmented in float32: %s; whole trunk: cosine %.6f, relative difference %.3g"
                % (lp["layers"], lp["rel_err"], ", ".join("%s %.3g" % kv for kv in sorted(lp["grad_err"].items())),
                   lp["trunk_grad_cos"], lp["trunk_grad_rel"]))
        dist.broadcast_params(policy.params)
        g = cfg["grpo"]
        opt = torch.optim.AdamW(policy.params, lr=float(g["lr"]), betas=tuple(g["betas"]), eps=float(g["eps"]),
                                weight_decay=float(g["weight_decay"]))
        warm = max(0, int(g["warmup_steps"]))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: min(1.0, (n + 1) / (warm + 1)) if warm else 1.0)
        start = latest_ckpt(run_dir)
        if start is None:
            start = 0
            save_ckpt(run_dir, 0, policy, opt, sched, dist, {"fingerprint": fp}, wait_min, log)
            log("initial checkpoint written (adapter %s, checksum %.6g)"
                % (cfg["policy"]["init_adapter"] or "fresh LoRA", policy.checksum()))
        else:
            st = load_ckpt(run_dir, start, policy, opt, sched, dist)
            if st.get("fingerprint") != fp and not cfg["allow_config_change"]:
                raise SystemExit("config fingerprint %s differs from the checkpoint's %s (allow_config_change=true "
                                 "to override)" % (fp, st.get("fingerprint")))
            log("resumed from step %d" % start)
        if dist.rank == 0:
            resolved = json.dumps(dict(cfg, fingerprint=fp, started_at_step=start), indent=1)
            retry_when_full(lambda: write_text(os.path.join(run_dir, "config.resolved.json"), resolved),
                            "config.resolved.json", wait_min, log)
        if backend is None:
            backend = SubprocessRollout(cfg, run_dir) if dist.rank == 0 else _Reader(run_dir)
        history = read_metrics(run_dir)
        history = {s: m for s, m in history.items() if s < start}
        n_servers = len(r["servers"]) or 1
        for step in range(start, int(g["steps"])):
            t0 = time.time()
            eps = step_episodes(sampler, step, int(r["group_size"]), int(cfg["seed"]), n_servers)
            err = None
            if dist.rank == 0:
                try:
                    records = backend.collect(step, os.path.join(ckpt_dir(run_dir, step), "adapter"), eps)
                except Exception as e:                       # noqa: BLE001
                    err = "%s: %s" % (type(e).__name__, e)
            err = dist.broadcast_object(err)
            if err:
                raise RuntimeError("rollouts failed at step %d: %s" % (step, err))
            if dist.rank != 0:
                records = backend.load(step, eps)
            t_roll = time.time() - t0
            samples, sstats = build_step_samples(records, int(r["horizon"]), float(cfg["reward"]["speed_bonus"]),
                                                 g["scale_rewards"], g.get("max_turns_per_episode"),
                                                 int(cfg["seed"]), step)
            t1 = time.time()
            tstats = train_on_samples(policy, samples, dist, cfg, opt, sched)
            t_train = time.time() - t1
            m = {"step": step, **sstats, **tstats, "t_rollout": round(t_roll, 1), "t_train": round(t_train, 1),
                 "time": time.strftime("%F %T")}
            history[step] = m
            stop = check_guards(history, cfg["guards"])
            t2 = time.time()
            save_ckpt(run_dir, step + 1, policy, opt, sched, dist, {"fingerprint": fp}, wait_min, log)
            m["t_ckpt"] = round(time.time() - t2, 1)
            if dist.rank == 0:
                retry_when_full(lambda: append_metrics(run_dir, m), "metrics.jsonl", wait_min, log)
                retry_when_full(lambda: prune_ckpts(run_dir, cfg, step + 1), "ckpt/ (removing old checkpoints)",
                                wait_min, log)
                kf = int(r.get("keep_full_every") or 0)
                if hasattr(backend, "compact") and not (kf and step % kf == 0):
                    backend.compact(step)
                log("step %d | reward %s feas %s | groups %d/%d used (%d zero-var) | samples %d tok %d | pg %.4f kl %s "
                    "gn %.3f lr %.2e | parse %.3f | roll %.0fs train %.0fs ckpt %.0fs | longest %s tok peak %s GiB" % (
                        step, _f(m["reward_mean"]), _f(m["feasible_rate"]), m["groups_used"], m["groups"],
                        m["groups_zero_var"], m["samples"], m["completion_tokens"], m.get("pg") or 0.0,
                        _f(m.get("kl")), m.get("grad_norm") or 0.0, m.get("lr") or 0.0, m["parse_rate"],
                        t_roll, t_train, m["t_ckpt"], m.get("max_sample_tokens", "-"), m.get("peak_mem_gib", "-")))
            if stop:
                if dist.rank == 0:
                    stopped = json.dumps({"step": step, "reason": stop})
                    retry_when_full(lambda: write_text(sp, stopped), "STOPPED.json", wait_min, log)
                    log("STOPPED at step %d: %s" % (step, stop))
                dist.barrier()
                raise GuardStop(stop)
        if dist.rank == 0:
            done_text = json.dumps({"steps": int(g["steps"]), "time": time.strftime("%F %T")})
            retry_when_full(lambda: write_text(done, done_text), "DONE", wait_min, log)
        return history
    except GuardStop:
        raise                                   # every rank stops at the same step, after a barrier
    except BaseException:
        if dist.world > 1:
            # One rank failed; the others are waiting in a collective it will never join, and
            # destroy_process_group would wait for them until the NCCL timeout (6 h lost after the
            # dx38mtR OOM). Report and leave now: torchrun sees the exit and stops the other ranks.
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        raise
    finally:
        dist.close()


class GuardStop(RuntimeError):
    pass


class _Reader:
    """Non-zero ranks: read the step's records that rank 0 wrote."""

    def __init__(self, run_dir):
        self.dir = os.path.join(run_dir, "rollouts")

    def load(self, step, episodes):
        recs = {}
        for line in open(os.path.join(self.dir, "step_%05d.jsonl" % step)):
            if line.strip():
                rec = json.loads(line)
                recs[rec["key"]] = rec
        return [recs[e["key"]] for e in episodes]


def _f(x) -> str:
    return "-" if x is None else "%.3f" % x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="multi-turn GRPO on dataxp (see module docstring)")
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key.sub=value (YAML values)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    try:
        train(cfg)
    except GuardStop:
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
