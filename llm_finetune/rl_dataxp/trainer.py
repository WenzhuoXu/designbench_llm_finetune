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
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import zlib
from typing import Callable, Optional

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
                "inject": {"return_token_ids": True, "logprobs": True, "top_p": 1.0, "top_k": 0},
                "workers": 64, "server_wait_s": 3600, "keep_full_every": 10, "max_crash_frac": 0.25},
    "reward": {"speed_bonus": 0.2},
    "grpo": {"steps": 150, "lr": 1.0e-5, "warmup_steps": 3, "betas": [0.9, 0.99], "eps": 1.0e-8,
             "weight_decay": 0.0, "max_grad_norm": 1.0, "kl_coef": 0.04, "kl_reference": "base",
             "clip_eps_low": 0.2, "clip_eps_high": 0.28, "iterations": 1, "scale_rewards": "group",
             "tis_cap": None},
    "checkpoint": {"keep_last": 2, "keep_adapter_every": 25, "keep_adapter_steps": [50, 150]},
    "guards": {"enabled": True, "window": 5, "parse_drop": 0.10, "kl_increase_max": 0.10,
               "stall_steps": 30, "stall_kl": 0.005, "ignore_stopped": False},
    "dist": {"timeout_min": 360},
    "allow_config_change": False,
}

# what a resumed run must share with the run that wrote the checkpoint (grpo.steps may grow)
FINGERPRINT_KEYS = [("seed",), ("model", "path"), ("policy",), ("harness",), ("reward",),
                    ("rollout", "problems_per_step"), ("rollout", "group_size"), ("rollout", "horizon"),
                    ("rollout", "temperature"), ("rollout", "inject"), ("rollout", "variants"),
                    ("grpo", "lr"), ("grpo", "warmup_steps"), ("grpo", "kl_coef"), ("grpo", "kl_reference"),
                    ("grpo", "iterations"), ("grpo", "scale_rewards"), ("grpo", "clip_eps_low"),
                    ("grpo", "clip_eps_high"), ("grpo", "tis_cap"), ("grpo", "max_grad_norm")]


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
        self.dir = os.path.join(run_dir, "rollouts")
        os.makedirs(self.dir, exist_ok=True)

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
               "env": self.cfg["harness"].get("env") or {}, "episodes": episodes}
        req_path = os.path.join(self.dir, "step_%05d.request.json" % step)
        with open(req_path + ".tmp", "w") as f:
            json.dump(req, f)
        os.replace(req_path + ".tmp", req_path)
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
        if not os.path.exists(p):
            return
        with open(p + ".compact.tmp", "w") as fo:
            for line in open(p):
                if line.strip():
                    rec = json.loads(line)
                    rec["calls"] = [{"n_prompt": len(c.get("prompt_token_ids") or []),
                                     "n_completion": len(c.get("token_ids") or []),
                                     "finish": c.get("finish"), "seed": c.get("seed")} for c in rec.get("calls") or []]
                    fo.write(json.dumps(rec) + "\n")
        os.replace(p + ".compact.tmp", p.replace(".jsonl", ".compact.jsonl"))
        os.remove(p)


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
        if m["gradient_checkpointing"]:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(base, "config"):
            base.config.use_cache = False
        init = cfg["policy"]["init_adapter"]
        if cfg["policy"].get("require_init_adapter") and not init:
            raise SystemExit("policy.init_adapter is required by this config")
        if init:
            model = PeftModel.from_pretrained(base, init, is_trainable=True, adapter_name="default")
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
            model.load_adapter(init, adapter_name="ref", is_trainable=False)
            model.set_adapter("default")
        elif self.ref_mode == "init":
            self.ref_mode = "base"              # a fresh adapter starts at B = 0: init == base
        for mod in model.modules():              # no dropout: the policy's log-probs must be deterministic
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
        model.train()
        self.model = model
        self.trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not self.trainable:
            raise RuntimeError("no trainable parameters")
        bad = [n for n, _ in self.trainable if ".default." not in n and "lora_" in n]
        if bad:
            raise RuntimeError("non-default adapter parameters are trainable: %s" % bad[:3])
        self.params = [p for _, p in self.trainable]

    def _restore_grad_flags(self):
        names = {n for n, _ in self.trainable}
        for n, p in self.model.named_parameters():
            p.requires_grad_(n in names)

    def logprobs(self, prompt_ids, completion_ids, temperature: float, raw: bool = False):
        """Per-token log-probs of the completion at temperature T (with grad when enabled), and
        optionally at T=1 (detached; vLLM reports unscaled log-probs by default)."""
        import torch
        import torch.nn.functional as F
        P, C = len(prompt_ids), len(completion_ids)
        ids = torch.tensor([list(prompt_ids) + list(completion_ids)], device=self.device)
        try:
            out = self.model(input_ids=ids, use_cache=False, logits_to_keep=C + 1)
        except TypeError:
            out = self.model(input_ids=ids, use_cache=False)
        lg = out.logits[0]
        if lg.shape[0] == C + 1:
            lg = lg[:C]
        elif lg.shape[0] == P + C:
            lg = lg[P - 1:P - 1 + C]
        else:
            raise RuntimeError("unexpected logits length %d for prompt %d + completion %d" % (lg.shape[0], P, C))
        tgt = torch.tensor(list(completion_ids), device=self.device)
        lp = -F.cross_entropy(lg.float() / temperature, tgt, reduction="none")
        rawlp = None
        if raw:
            with torch.no_grad():
                rawlp = -F.cross_entropy(lg.detach().float(), tgt, reduction="none")
        return lp, rawlp

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

    def all_reduce_(self, t):
        if self.world > 1:
            import torch.distributed as dist
            dist.all_reduce(t)
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


def save_ckpt(run_dir, step, policy, opt, sched, dist, state: dict) -> None:
    import torch
    d = ckpt_dir(run_dir, step)
    tmp = d + ".tmp"
    if dist.rank == 0:
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
    dist.barrier()
    if dist.rank == 0:
        policy.save_adapter(os.path.join(tmp, "adapter"))
        torch.save(opt.state_dict(), os.path.join(tmp, "optimizer.pt"))
        torch.save(sched.state_dict(), os.path.join(tmp, "scheduler.pt"))
        json.dump(dict(state, step=step, checksum=policy.checksum()), open(os.path.join(tmp, "state.json"), "w"))
    rng = {"python": random.getstate(), "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    torch.save(rng, os.path.join(tmp, "rng_rank%d.pt" % dist.rank))
    dist.barrier()
    if dist.rank == 0:
        open(os.path.join(tmp, "COMPLETE"), "w").write(time.strftime("%F %T"))
        if os.path.exists(d):
            shutil.rmtree(d)
        os.replace(tmp, d)
    dist.barrier()


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
    n_tok = sum(len(s["completion_ids"]) for s in samples)
    stats = torch.zeros(9, dtype=torch.float64, device=policy.device)
    # 0 tokens 1 pg 2 kl 3 logp 4 clipped 5 gap_abs 6 gap_n 7 adv*tokens 8 samples
    if n_tok == 0:
        return {"skipped_update": True}
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
            (loss_tok.sum() / n_tok).backward()
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
    t = max(1.0, float(stats[0]))
    return {"tokens": int(stats[0]), "pg": float(stats[1]) / t, "kl": float(stats[2]) / t if g["kl_coef"] else None,
            "logp_mean": float(stats[3]) / t, "clip_frac": float(stats[4]) / t,
            "vllm_gap_abs": float(stats[5]) / float(stats[6]) if stats[6] > 0 else None,
            "adv_token_mean": float(stats[7]) / t, "trained_samples": int(stats[8]), "grad_norm": gn,
            "lr": sched.get_last_lr()[0], "skipped_update": False}


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
    if dist.rank != 0:
        log = lambda s: None                                   # noqa: E731
    try:
        if dist.rank == 0:
            os.makedirs(run_dir, exist_ok=True)
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
        dist.broadcast_params(policy.params)
        g = cfg["grpo"]
        opt = torch.optim.AdamW(policy.params, lr=float(g["lr"]), betas=tuple(g["betas"]), eps=float(g["eps"]),
                                weight_decay=float(g["weight_decay"]))
        warm = max(0, int(g["warmup_steps"]))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: min(1.0, (n + 1) / (warm + 1)) if warm else 1.0)
        start = latest_ckpt(run_dir)
        if start is None:
            start = 0
            save_ckpt(run_dir, 0, policy, opt, sched, dist, {"fingerprint": fp})
            log("initial checkpoint written (adapter %s, checksum %.6g)"
                % (cfg["policy"]["init_adapter"] or "fresh LoRA", policy.checksum()))
        else:
            st = load_ckpt(run_dir, start, policy, opt, sched, dist)
            if st.get("fingerprint") != fp and not cfg["allow_config_change"]:
                raise SystemExit("config fingerprint %s differs from the checkpoint's %s (allow_config_change=true "
                                 "to override)" % (fp, st.get("fingerprint")))
            log("resumed from step %d" % start)
        if dist.rank == 0:
            json.dump(dict(cfg, fingerprint=fp, started_at_step=start),
                      open(os.path.join(run_dir, "config.resolved.json"), "w"), indent=1)
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
                                                 g["scale_rewards"])
            t1 = time.time()
            tstats = train_on_samples(policy, samples, dist, cfg, opt, sched)
            t_train = time.time() - t1
            m = {"step": step, **sstats, **tstats, "t_rollout": round(t_roll, 1), "t_train": round(t_train, 1),
                 "time": time.strftime("%F %T")}
            history[step] = m
            stop = check_guards(history, cfg["guards"])
            t2 = time.time()
            save_ckpt(run_dir, step + 1, policy, opt, sched, dist, {"fingerprint": fp})
            m["t_ckpt"] = round(time.time() - t2, 1)
            if dist.rank == 0:
                with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
                    f.write(json.dumps(m) + "\n")
                prune_ckpts(run_dir, cfg, step + 1)
                kf = int(r.get("keep_full_every") or 0)
                if hasattr(backend, "compact") and not (kf and step % kf == 0):
                    backend.compact(step)
                log("step %d | reward %s feas %s | groups %d/%d used (%d zero-var) | samples %d tok %d | pg %.4f kl %s "
                    "gn %.3f lr %.2e | parse %.3f | roll %.0fs train %.0fs ckpt %.0fs" % (
                        step, _f(m["reward_mean"]), _f(m["feasible_rate"]), m["groups_used"], m["groups"],
                        m["groups_zero_var"], m["samples"], m["completion_tokens"], m.get("pg") or 0.0,
                        _f(m.get("kl")), m.get("grad_norm") or 0.0, m.get("lr") or 0.0, m["parse_rate"],
                        t_roll, t_train, m["t_ckpt"]))
            if stop:
                if dist.rank == 0:
                    json.dump({"step": step, "reason": stop}, open(sp, "w"))
                    log("STOPPED at step %d: %s" % (step, stop))
                dist.barrier()
                raise GuardStop(stop)
        if dist.rank == 0:
            json.dump({"steps": int(g["steps"]), "time": time.strftime("%F %T")}, open(os.path.join(run_dir, "DONE"), "w"))
        return history
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
