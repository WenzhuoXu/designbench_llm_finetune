"""Which training signal has a direction? Per-turn gradient sketches of a run's own rollouts.

The policy gradient is linear in the per-sample weights: g = sum_i w_i * grad(sum of log-probs of
reply i). So one pass that stores a sketch of every reply's score gradient lets any labelling
(episode advantage, per-turn labels, ...) be evaluated afterwards without a GPU: the gradient of a
set of problems under a labelling is a weighted sum of sketches, and whether two disjoint sets of
problems agree on a direction is a dot product of two such sums.

The sketch is a count sketch (every coordinate goes to one of ``dim`` buckets with a random sign):
linear, and dot products are preserved without bias, with standard error ~ |x||y| / sqrt(dim).
Each reply is split at its last ``</think>`` token into the rationale part and the action part,
and the two parts are sketched separately (their sum is the whole reply's gradient).

    python -m llm_finetune.rl_dataxp.grad_probe --config CFG --run-dir RUN --steps 100,80,60 \\
        --out DIR --shard 0 --shards 4 [--dim 65536] [--limit N] [key=value ...]

Reads rollouts/step_<s>.jsonl (the uncompacted files) and ckpt/step_<k>/adapter (the kept adapter
nearest to s). Writes DIR/s<step>_k<shard>.{r,a}.npy (float32, samples x dim) and .meta.jsonl; a
finished (step, shard) is skipped on a rerun. Nothing is written to the run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

from .samples import shard_samples

_M1, _M2 = -0x40A7B892E31B1A47, -0x6B2FB644ECCEEE15      # splitmix64's multipliers as signed 64-bit


def _mix64(x):
    """splitmix64's finaliser on an int64 tensor (wrapping multiplication, logical shifts)."""
    x = (x ^ ((x >> 30) & 0x3FFFFFFFF)) * _M1
    x = (x ^ ((x >> 27) & 0x1FFFFFFFFF)) * _M2
    return x ^ ((x >> 31) & 0x1FFFFFFFF)


def sketch_flat(vec, dim: int, offset: int = 0, out=None, seed: int = 0):
    """Count sketch of a flat tensor whose first coordinate has global index ``offset``.
    ``dim`` must be a power of two. Adds into ``out`` (float32, on vec's device) and returns it."""
    import torch
    if dim & (dim - 1):
        raise ValueError("dim must be a power of two")
    if out is None:
        out = torch.zeros(dim, dtype=torch.float32, device=vec.device)
    h = _mix64(torch.arange(offset, offset + vec.numel(), device=vec.device, dtype=torch.int64) + seed * 0x9E3779B1)
    idx = h & (dim - 1)
    sign = ((h >> 40) & 1).to(torch.float32) * 2 - 1
    out.index_add_(0, idx, vec.detach().reshape(-1).to(torch.float32) * sign)
    return out


def sketch_grads(params, dim: int, seed: int = 0):
    """(sketch, squared norm) of the gradient now in ``params`` (a missing .grad counts as zero)."""
    import torch
    out = torch.zeros(dim, dtype=torch.float32, device=params[0].device)
    n2 = torch.zeros((), dtype=torch.float64, device=params[0].device)
    off = 0
    for p in params:
        if p.grad is not None:
            sketch_flat(p.grad, dim, off, out, seed)
            n2 += p.grad.detach().double().pow(2).sum()
        off += p.numel()
    return out, float(n2)


def split_point(completion_ids, think_end_id: Optional[int]) -> int:
    """Index of the first action token: one past the last ``</think>``. No such token: the whole
    reply is rationale (the index is the length)."""
    if think_end_id is None:
        return len(completion_ids)
    for i in range(len(completion_ids) - 1, -1, -1):
        if completion_ids[i] == think_end_id:
            return i + 1
    return len(completion_ids)


def probe_samples(records: list[dict], step: int) -> list[dict]:
    """One sample per model call of every episode (also of groups whose episodes all ended alike:
    per-turn labellings use them). Deterministic order: records by key, calls in turn order."""
    out = []
    for r in sorted(records, key=lambda r: r["key"]):
        for k, c in enumerate(r.get("calls") or []):
            if c.get("token_ids"):
                out.append({"step": step, "key": r["key"], "group": r["group"], "variant": r["variant"], "turn": k,
                            "prompt_ids": c["prompt_token_ids"], "completion_ids": c["token_ids"]})
    return out


def _zero(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


def sample_sketches(policy, s: dict, temperature: float, dim: int, think_end_id: Optional[int], seed: int = 0) -> dict:
    """Sketches of grad(sum log-prob) over the rationale tokens and over the action tokens of one
    reply. The two backward passes share one forward pass when the graph allows it; if not (the
    offloaded long-sequence path frees its buffers), the forward pass is repeated."""
    import torch
    C = len(s["completion_ids"])
    a0 = split_point(s["completion_ids"], think_end_id)
    res = {"n_r": a0, "n_a": C - a0, "refwd": 0}
    lp, _ = policy.logprobs(s["prompt_ids"], s["completion_ids"], temperature)
    res["lp_r"], res["lp_a"] = float(lp.detach()[:a0].sum()), float(lp.detach()[a0:].sum())
    parts = [("r", slice(0, a0)), ("a", slice(a0, C))]
    parts = [(n, sl) for n, sl in parts if sl.stop > sl.start]
    # the offloaded long-sequence path (trainer.install_checkpointing) gives its CPU buffers back in
    # the first backward pass, so there the graph cannot be kept
    st = getattr(policy, "_ckpt_state", None) or {}
    keep = not (st.get("offload_min_tokens") and len(s["prompt_ids"]) + C >= st["offload_min_tokens"])
    for j, (name, sl) in enumerate(parts):
        _zero(policy.params)
        last = j == len(parts) - 1
        if j and not keep:
            lp, _ = policy.logprobs(s["prompt_ids"], s["completion_ids"], temperature)
            res["refwd"] = 1
        try:
            lp[sl].sum().backward(retain_graph=keep and not last)
        except Exception as e:
            if isinstance(e, torch.OutOfMemoryError) or j == 0 or res["refwd"]:
                raise
            _zero(policy.params)                 # the first backward freed the graph after all: run it again
            lp, _ = policy.logprobs(s["prompt_ids"], s["completion_ids"], temperature)
            lp[sl].sum().backward()
            res["refwd"] = 1
        sk, n2 = sketch_grads(policy.params, dim, seed)
        res["sk_" + name], res["g2_" + name] = sk.cpu(), n2
    del lp
    _zero(policy.params)
    for name in ("r", "a"):
        if "sk_" + name not in res:
            res["sk_" + name], res["g2_" + name] = torch.zeros(dim), 0.0
    return res


def nearest_adapter(run_dir: str, step: int) -> tuple[int, str]:
    """The kept adapter closest to ``step`` (ties: the earlier one)."""
    d = os.path.join(run_dir, "ckpt")
    have = sorted(int(n[5:]) for n in os.listdir(d)
                  if n.startswith("step_") and n[5:].isdigit()
                  and os.path.exists(os.path.join(d, n, "adapter", "adapter_model.safetensors")))
    if not have:
        raise SystemExit("no adapter under %s" % d)
    k = min(have, key=lambda x: (abs(x - step), x))
    return k, os.path.join(d, "step_%05d" % k, "adapter")


def main(argv=None) -> int:
    import numpy as np
    import torch
    from . import trainer as TR
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", required=True, help="comma-separated rollout steps, in the order to do them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--dim", type=int, default=65536)
    ap.add_argument("--limit", type=int, default=0, help="at most this many samples per step (a smoke run)")
    ap.add_argument("--think-end", default="</think>", help="token that ends the rationale ('' = no split)")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = TR.load_config(a.config, a.overrides)
    cfg["run_dir"] = a.run_dir
    cfg["grpo"]["kl_reference"] = "base"                     # no reference pass here: no second adapter
    T = float(cfg["rollout"]["temperature"])
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(a.out, exist_ok=True)
    think_end_id = None
    if a.think_end:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"]["path"], trust_remote_code=cfg["model"]["trust_remote_code"])
        ids = tok.encode(a.think_end, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit("%r is %d tokens in this tokenizer, expected 1" % (a.think_end, len(ids)))
        think_end_id = ids[0]
    tag = "[shard %d/%d]" % (a.shard, a.shards)
    print(tag, "think_end_id", think_end_id, "dim", a.dim, "T", T, flush=True)
    torch.manual_seed(int(cfg["seed"]))
    policy = TR.Policy(cfg, dev)
    loaded = None
    for step in [int(x) for x in a.steps.split(",") if x.strip()]:
        base = os.path.join(a.out, "s%05d_k%d" % (step, a.shard))
        if os.path.exists(base + ".meta.jsonl"):
            print(tag, "step", step, "already done", flush=True)
            continue
        path = os.path.join(a.run_dir, "rollouts", "step_%05d.jsonl" % step)
        recs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        mine = shard_samples(probe_samples(recs, step), a.shards, a.shard)
        if a.limit:
            mine = mine[-a.limit:]                            # the shortest ones
        k, adir = nearest_adapter(a.run_dir, step)
        if loaded != k:
            policy.load_adapter_weights(adir)
            loaded = k
        print(tag, "step %d: %d episodes, %d samples here, adapter of step %d" % (step, len(recs), len(mine), k), flush=True)
        R = np.zeros((len(mine), a.dim), dtype=np.float32)
        A = np.zeros((len(mine), a.dim), dtype=np.float32)
        meta, t0 = [], time.time()
        for i, s in enumerate(mine):
            t1 = time.time()
            m = {"step": step, "key": s["key"], "group": s["group"], "variant": s["variant"], "turn": s["turn"],
                 "n_prompt": len(s["prompt_ids"]), "adapter": k}
            try:
                res = sample_sketches(policy, s, T, a.dim, think_end_id)
            except torch.OutOfMemoryError as e:
                _zero(policy.params)
                torch.cuda.empty_cache()
                m["error"] = "oom: %s" % str(e)[:160]
                print(tag, "OUT OF MEMORY on", s["key"], s["turn"], len(s["prompt_ids"]) + len(s["completion_ids"]), flush=True)
            else:
                R[i], A[i] = res.pop("sk_r").numpy(), res.pop("sk_a").numpy()
                m.update(res)
            m["t"] = round(time.time() - t1, 2)
            meta.append(m)
            if i % 20 == 0 or i == len(mine) - 1:
                print(tag, "step %d sample %d/%d  %.0f s  last %s" % (step, i + 1, len(mine), time.time() - t0,
                                                                    json.dumps({x: m.get(x) for x in ("n_r", "n_a", "t", "refwd")})),
                      flush=True)
        for name, arr in (("r", R), ("a", A)):
            tmp = "%s.%s.tmp.npy" % (base, name)
            np.save(tmp, arr)
            os.replace(tmp, "%s.%s.npy" % (base, name))
        tmp = base + ".meta.jsonl.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for m in meta:
                f.write(json.dumps(m) + "\n")
        os.replace(tmp, base + ".meta.jsonl")              # written last: its presence marks the step done
        print(tag, "step %d done: %d samples in %.0f s" % (step, len(mine), time.time() - t0), flush=True)
    print(tag, "GRAD_PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
