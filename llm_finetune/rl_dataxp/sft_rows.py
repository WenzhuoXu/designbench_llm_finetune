"""A supervised round on training rows, continuing an adapter: the cross-entropy that trained the
dx38 adapter (the assistant_after_think target), on the model, long-sequence path, checkpoints and
multi-GPU reduction of the RL trainer.

A row is a tier row of dataxp.ownrows.build_tiers / scripts/rl_screen/build_rows.py: messages =
[system, user, assistant{reasoning_content, content}]. Its prompt is the served prompt of the
messages before the assistant turn (it ends in '<think>\\n'); its completion is
    reasoning.strip() + '\\n</think>\\n\\n' + content.strip() + '<|im_end|>'
and the loss of a row is the mean negative log-probability of its completion tokens at temperature
1. An optimizer step averages sft.rows_per_step rows (64: the batch-1, accumulation-64 recipe of
dx38, where every row weighs the same whatever its length), AdamW, gradient clipping, cosine
schedule with a linear warm-up. One difference from that recipe: LoRA dropout is off (dx38 trained
with 0.05). The RL trainer's long-sequence path recomputes layer segments in the backward pass and
was validated without dropout only; sft.lora_dropout > 0 is accepted but not validated on it.

    sft:
      rows_file: train.jsonl        # one row per line; the order is a seeded shuffle per epoch
      dev_file: null                # optional: rows scored (no gradient) every dev_every steps
      epochs: 1.0
      rows_per_step: 64
      lr: 1.0e-4
      warmup_ratio: 0.03
      weight_decay: 0.01
      max_grad_norm: 1.0
      lora_dropout: 0.0
      temperature: 1.0              # the temperature of the log-probabilities in the loss (see below)
      seed: 42
      save_every: 25                # a full checkpoint every this many steps (and at the end)
      dev_every: 0
      lazy_rows: false              # true: index the rows file and read a row when its step draws it (large files)
      check_rows: true              # false: skip tokenising every row before training (the builder checked them)
      resume_dir: null              # an exported checkpoint (adapter/ + state.json, no optimizer): used when it is
                                    # newer than run_dir's newest full checkpoint; the optimizer restarts
      rewarm_steps: 0               # after such a resume, the lr ramps up linearly over this many steps

Resuming from resume_dir (con1 full-set chains, docs/con1_full_rows.md): the adapter weights, the step (hence the
data position and the scheduled lr) come from the copy; AdamW's moments restart at zero, so the first steps would
move every weight by about the full lr (Adam's first update is lr * sign(g)); rewarm_steps scales the scheduled lr by
(k + 1) / (rewarm_steps + 1) for the k-th step after the resume. The copy's fingerprint must match the run's.

sft.temperature: 1.0 is the dx38 recipe. Rows that are the policy's own replies drawn at a
temperature T < 1 share, under a loss at 1.0, a direction that has nothing to do with which reply
was kept: making any own reply more likely sharpens the policy (measured on 713 first-move states
of dx38mtR, replies drawn at 0.8: 27% of a 64-row step, the same for screened and unscreened
replies; the part only the screening has is under 1% of the step). With the loss at the sampling
temperature an unscreened own reply has a zero expected gradient (the score-function identity), so
what the rows share is the screening alone.

    python -m llm_finetune.rl_dataxp.sft_rows --config configs/rl/<cfg>.yaml [key=value ...]
(under torchrun for several GPUs). Resumes from the newest complete checkpoint of run_dir; the
final adapter is run_dir/final.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Callable

from . import trainer as TR
from .diskio import retry_when_full, write_text
from .samples import shard_samples

END = "<|im_end|>"
SFT_DEFAULTS = {"rows_file": None, "dev_file": None, "epochs": 1.0, "rows_per_step": 64, "lr": 1.0e-4,
                "warmup_ratio": 0.03, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1.0e-8,
                "max_grad_norm": 1.0, "lora_dropout": 0.0, "seed": 42, "save_every": 25, "dev_every": 0,
                "keep_adapter_every": 50, "temperature": 1.0, "keep_last": 2, "lazy_rows": False,
                "check_rows": True, "resume_dir": None, "rewarm_steps": 0}


def _flag(x) -> bool:
    return x if isinstance(x, bool) else str(x).strip().lower() in ("1", "true", "yes", "on")


def file_sha1(path: str) -> str:
    """sha1 of a file's bytes, read in blocks (the same digest as hashing the whole file at once)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class RowFile:
    """A rows file indexed by line offsets: rows[i] reads and parses line i when asked (a full-set rows file is several
    GB; four ranks holding it parsed would take tens of GB each)."""

    def __init__(self, path: str):
        import re
        self.path = path
        self.offsets = []
        self.domains: dict = {}
        pat = re.compile(rb'"domain":\s*"([^"\\]*)"')
        with open(path, "rb") as f:
            pos = 0
            for line in f:
                if line.strip():
                    self.offsets.append(pos)
                    m = None
                    for m in pat.finditer(line):
                        pass                               # the row's own key comes after its messages: the last match
                    k = m.group(1).decode() if m else "None"
                    self.domains[k] = self.domains.get(k, 0) + 1
                pos += len(line)
        self._f = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, i: int) -> dict:
        if self._f is None:
            self._f = open(self.path, "rb")
        self._f.seek(self.offsets[i])
        return json.loads(self._f.readline().decode("utf-8"))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def completion_text(message: dict) -> str:
    """The supervised text of an assistant turn under the assistant_after_think target."""
    return (message.get("reasoning_content") or "").strip() + "\n</think>\n\n" + (message.get("content") or "").strip() + END


def row_sample(row: dict, tok) -> dict:
    """prompt_ids / completion_ids of a row, tokenised as it is trained. A row with supervise = 'every_assistant'
    (con1 consequence rows, conseq_format.py) is a conversation whose every assistant turn is supervised: its sample
    carries mask (1 on answer tokens) and the loss is the mean over the masked tokens."""
    if row.get("supervise") == "every_assistant":
        from .conseq_format import encode
        e = encode(tok, row["messages"])
        if not any(e["mask"]):
            raise ValueError("a consequence row without an answer")
        return dict(e, domain=row.get("domain"), tier=row.get("tier"))
    msgs = row["messages"]
    if not msgs or msgs[-1].get("role") != "assistant":
        raise ValueError("the row's last message is not the assistant turn")
    ids = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True)
    ids = list(ids if isinstance(ids, list) else ids["input_ids"])
    if not tok.decode(ids[-4:]).endswith("<think>\n"):
        raise ValueError("the served prompt does not end with '<think>\\n': %r" % tok.decode(ids[-8:]))
    comp = tok.encode(completion_text(msgs[-1]), add_special_tokens=False)
    return {"prompt_ids": ids, "completion_ids": list(comp), "domain": row.get("domain"), "tier": row.get("tier")}


def row_order(n: int, epochs: float, seed: int) -> list[int]:
    """The rows in training order: a seeded shuffle of all rows per epoch, cut at n * epochs rows."""
    total = int(round(n * float(epochs)))
    out, e = [], 0
    while len(out) < total:
        perm = list(range(n))
        random.Random("%d/%d" % (int(seed), e)).shuffle(perm)
        out.extend(perm)
        e += 1
    return out[:total]


def lr_factor(step: int, steps: int, warmup_ratio: float) -> float:
    """Linear warm-up over ceil(steps * warmup_ratio) steps, then a cosine to zero at the last step."""
    warm = int(math.ceil(steps * float(warmup_ratio)))
    if step < warm:
        return (step + 1) / (warm + 1)
    span = max(1, steps - warm)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warm) / span)))


def read_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def set_lora_dropout(model, p: float) -> int:
    import torch
    n = 0
    for name, mod in model.named_modules():
        if "lora_dropout" in name and isinstance(mod, torch.nn.Dropout):
            mod.p = float(p)
            n += 1
    return n


def _row_nll(lp, sample: dict):
    """(row loss, weight sum, weighted nll sum, informative nll sum, informative token count) of one row. A sample
    without a mask weighs every completion token 1; a consequence row's mask holds per-token weights (0 on the user
    turns between answers, conseq_format: 1 on informative answer tokens, a small weight on fixed wording); the row
    loss is the weighted mean. 'Informative' = tokens of weight >= 1."""
    import torch
    mask = sample.get("mask")
    if mask is None:
        n = lp.numel()
        return -lp.mean(), float(n), -lp.sum(), -lp.sum(), float(n)
    if len(mask) != lp.numel():
        raise RuntimeError("mask of %d tokens for a completion of %d" % (len(mask), lp.numel()))
    w = torch.tensor(mask, dtype=lp.dtype, device=lp.device)
    tot = w.sum()
    info = w >= 1.0
    return -(lp * w).sum() / tot, float(tot), -(lp * w).sum(), -lp[info].sum(), float(info.sum())


def _split_losses(stats, prefix: str) -> dict:
    """Row-mean loss of the masked (consequence) rows and of the other rows, when a batch holds any of them, and the
    per-token nll of the informative tokens of the consequence rows."""
    out = {}
    for name, k in (("conseq", 4), ("policy", 6)):
        if float(stats[k]) > 0:
            out["%sloss_%s" % (prefix, name)] = float(stats[k + 1]) / float(stats[k])
            out["%srows_%s" % (prefix, name)] = int(stats[k])
    if float(stats[9]) > 0:
        out["%snll_info_token" % prefix] = float(stats[8]) / float(stats[9])
    return out


def _accumulate(stats, s, nll, wsum, wnll, inll, icnt):
    stats[0] += 1
    stats[1] += nll.double()
    stats[2] += wsum
    stats[3] += wnll.double()
    k = 4 if s.get("mask") is not None else 6
    stats[k] += 1
    stats[k + 1] += nll.double()
    if s.get("mask") is not None:
        stats[8] += inll.double()
        stats[9] += icnt


def train_step(policy, samples: list[dict], dist, opt, max_grad_norm: float, temperature: float = 1.0) -> dict:
    """One optimizer step on the mean per-row cross-entropy of `samples` (this rank's share), the
    log-probabilities taken at `temperature`."""
    import torch
    mine = shard_samples(samples, dist.world, dist.rank)
    # rows, row-mean nll, token weight, weighted token nll; rows / row-mean nll of consequence rows and of the others;
    # informative-token nll sum and count of the consequence rows
    stats = torch.zeros(10, dtype=torch.float64, device=policy.device)
    cuda = policy.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(policy.device)
    for s in mine:
        lp, _ = policy.logprobs(s["prompt_ids"], s["completion_ids"], float(temperature))
        nll, wsum, wnll, inll, icnt = _row_nll(lp, s)
        if not bool(torch.isfinite(nll)):
            raise RuntimeError("non-finite loss on a row of %d + %d tokens (%s)" % (
                len(s["prompt_ids"]), len(s["completion_ids"]), s.get("domain")))
        (nll / len(samples)).backward()
        with torch.no_grad():
            _accumulate(stats, s, nll.detach(), wsum, wnll.detach(), inll.detach(), icnt)
    dist.all_reduce_grads(policy.params)
    gn = float(torch.nn.utils.clip_grad_norm_(policy.params, float(max_grad_norm)))
    opt.step()
    opt.zero_grad(set_to_none=True)
    dist.all_reduce_(stats)
    peak = torch.tensor([torch.cuda.max_memory_allocated(policy.device) / 2 ** 30 if cuda else 0.0],
                        dtype=torch.float64, device=policy.device)
    dist.all_reduce_(peak, op="max")
    return {"rows": int(stats[0]), "loss": float(stats[1]) / max(1.0, float(stats[0])), "tokens": int(stats[2]),
            "nll_per_token": float(stats[3]) / max(1e-9, float(stats[2])), "grad_norm": gn,
            **_split_losses(stats, ""),
            "peak_mem_gib": round(float(peak[0]), 2) if cuda else None,
            "max_row_tokens": max(len(s["prompt_ids"]) + len(s["completion_ids"]) for s in samples)}


def score_rows(policy, samples: list[dict], dist, temperature: float = 1.0) -> dict:
    """Mean per-row and per-token negative log-probability, no gradient, dropout off."""
    import torch
    mine = shard_samples(samples, dist.world, dist.rank)
    stats = torch.zeros(10, dtype=torch.float64, device=policy.device)
    policy.model.eval()
    try:
        with torch.no_grad():
            for s in mine:
                lp, _ = policy.logprobs(s["prompt_ids"], s["completion_ids"], float(temperature))
                nll, wsum, wnll, inll, icnt = _row_nll(lp, s)
                _accumulate(stats, s, nll, wsum, wnll, inll, icnt)
    finally:
        policy.model.train()
    dist.all_reduce_(stats)
    return {"dev_rows": int(stats[0]), "dev_loss": float(stats[1]) / max(1.0, float(stats[0])),
            "dev_nll_per_token": float(stats[3]) / max(1e-9, float(stats[2])), **_split_losses(stats, "dev_")}


def exported_state(path) -> dict | None:
    """state.json of an exported checkpoint (adapter/ + state.json, written by scripts/con1/orchard_full.sh), or None."""
    if not path:
        return None
    sp, ap = os.path.join(path, "state.json"), os.path.join(path, "adapter", "adapter_model.safetensors")
    if not (os.path.exists(sp) and os.path.exists(ap)):
        return None
    st = json.load(open(sp))
    return st if "step" in st and "checksum" in st else None


def sft_fingerprint(cfg: dict, rows_sha1: str) -> str:
    s = cfg["sft"]
    sub = {"rows_sha1": rows_sha1, "model": cfg["model"]["path"], "policy": cfg["policy"],
           **{k: s[k] for k in ("epochs", "rows_per_step", "lr", "warmup_ratio", "weight_decay", "betas", "eps",
                                "max_grad_norm", "lora_dropout", "seed", "temperature")}}
    return hashlib.sha1(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:16]


def train(cfg: dict, log: Callable[[str], None] = print) -> dict:
    import torch
    from transformers import AutoTokenizer

    s = dict(SFT_DEFAULTS, **(cfg.get("sft") or {}))
    # a command-line override such as sft.lr=5e-5 arrives as a string (YAML reads 5e-5 as one)
    for k in ("epochs", "lr", "warmup_ratio", "weight_decay", "eps", "max_grad_norm", "lora_dropout", "temperature"):
        s[k] = float(s[k])
    for k in ("rows_per_step", "seed", "save_every", "dev_every", "keep_adapter_every", "keep_last", "rewarm_steps"):
        s[k] = int(s[k])
    for k in ("lazy_rows", "check_rows"):
        s[k] = _flag(s[k])
    if s["resume_dir"] in ("", "null", "None"):
        s["resume_dir"] = None
    s["betas"] = [float(b) for b in s["betas"]]
    cfg["sft"] = s
    run_dir = cfg["run_dir"]
    for key, val in (("run_dir", run_dir), ("model.path", cfg["model"]["path"]), ("sft.rows_file", s["rows_file"])):
        if not val:
            raise SystemExit("%s must be set" % key)
    cfg["grpo"]["kl_reference"] = "base"                 # no reference pass here: no second adapter is loaded
    # checkpoints: the RL trainer's rule, on this run's own cadence
    cfg["checkpoint"] = {"keep_last": int(s["keep_last"]), "keep_adapter_every": int(s["keep_adapter_every"]),
                         "keep_adapter_steps": []}
    dist = TR.Dist(int(cfg["dist"]["timeout_min"]))
    wait_min = float(cfg.get("disk_full_wait_min") or 0)
    if dist.rank != 0:
        log = lambda msg: None                             # noqa: E731
    try:
        if dist.rank == 0:
            retry_when_full(lambda: os.makedirs(run_dir, exist_ok=True), run_dir, wait_min, log)
        dist.barrier()
        rows = RowFile(s["rows_file"]) if s["lazy_rows"] else read_rows(s["rows_file"])
        dev = read_rows(s["dev_file"]) if s["dev_file"] else []
        rows_sha1 = file_sha1(s["rows_file"])
        fp = sft_fingerprint(cfg, rows_sha1)
        order = row_order(len(rows), s["epochs"], s["seed"])
        per = int(s["rows_per_step"])
        steps = int(math.ceil(len(order) / per))
        tok = AutoTokenizer.from_pretrained(cfg["model"]["path"], trust_remote_code=cfg["model"]["trust_remote_code"])
        if dist.rank == 0 and not s["check_rows"]:
            for k, r in enumerate(dev):                    # the dev rows are still checked (they are few)
                row_sample(r, tok)
            log("rows check skipped (sft.check_rows false): %d training rows, sha1 %s" % (len(rows), rows_sha1))
        if dist.rank == 0 and s["check_rows"]:
            # every row is tokenised once before the model is loaded: a row that cannot be trained on
            # stops the run here, not at the step that draws it
            longest, n_tok = 0, 0
            import itertools
            for k, r in enumerate(itertools.chain(rows, dev)):
                try:
                    smp = row_sample(r, tok)
                    if not smp["completion_ids"]:
                        raise ValueError("empty completion")
                except Exception as e:                       # noqa: BLE001
                    raise SystemExit("row %d of %s cannot be trained on: %s" % (
                        k if k < len(rows) else k - len(rows), s["rows_file"] if k < len(rows) else s["dev_file"], e))
                longest = max(longest, len(smp["prompt_ids"]) + len(smp["completion_ids"]))
                n_tok += len(smp["completion_ids"]) if k < len(rows) else 0
            log("rows checked: longest %d tokens, %d supervised tokens in the training file" % (longest, n_tok))
        random.seed(int(s["seed"]) + dist.rank)
        torch.manual_seed(int(s["seed"]) + dist.rank)      # dropout masks differ between the ranks
        policy = TR.Policy(cfg, dist.device)
        n_drop = set_lora_dropout(policy.model, s["lora_dropout"])
        dist.broadcast_params(policy.params)
        opt = torch.optim.AdamW(policy.params, lr=float(s["lr"]), betas=tuple(s["betas"]), eps=float(s["eps"]),
                                weight_decay=float(s["weight_decay"]))
        rewarm = {"from": None}                            # the step of an optimizer-less resume (resume_dir)

        def factor(n):
            f = lr_factor(n, steps, s["warmup_ratio"])
            if rewarm["from"] is not None and s["rewarm_steps"] > 0 and n >= rewarm["from"]:
                f *= min(1.0, (n - rewarm["from"] + 1) / (s["rewarm_steps"] + 1))
            return f

        start = TR.latest_ckpt(run_dir)
        ex = exported_state(s["resume_dir"])
        if ex is not None and (start is None or int(ex["step"]) > start):
            if ex.get("fingerprint") != fp and not cfg["allow_config_change"]:
                raise SystemExit("config / rows fingerprint %s differs from the exported checkpoint's %s (%s)" % (
                    fp, ex.get("fingerprint"), s["resume_dir"]))
            start = int(ex["step"])
            policy.load_adapter_weights(os.path.join(s["resume_dir"], "adapter"))
            got = policy.checksum()
            if abs(got - ex["checksum"]) > 1e-6 * max(1.0, abs(ex["checksum"])):
                raise RuntimeError("exported adapter %s: parameter checksum %.9g, saved %.9g" % (
                    s["resume_dir"], got, ex["checksum"]))
            rewarm["from"] = start
            for g in opt.param_groups:
                g["initial_lr"] = float(s["lr"])
            # last_epoch = start - 1: the constructor's first step sets the lr of step `start`
            sched = torch.optim.lr_scheduler.LambdaLR(opt, factor, last_epoch=start - 1)
            log("resumed from the exported adapter of step %d (%s): optimizer restarted, lr re-warmed over %d steps, "
                "lr now %.3e" % (start, s["resume_dir"], s["rewarm_steps"], sched.get_last_lr()[0]))
        else:
            sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
            if start is None:
                start = 0
                TR.save_ckpt(run_dir, 0, policy, opt, sched, dist, {"fingerprint": fp}, wait_min, log)
            else:
                st = TR.load_ckpt(run_dir, start, policy, opt, sched, dist)
                if st.get("fingerprint") != fp and not cfg["allow_config_change"]:
                    raise SystemExit("config / rows fingerprint %s differs from the checkpoint's %s" % (fp, st.get("fingerprint")))
                rewarm["from"] = st.get("rewarm_from")
                log("resumed from step %d" % start)
        log("rows %d (%s), dev rows %d, %g epoch(s) = %d rows in %d steps of %d; lr %g, warm-up %d steps, LoRA dropout %g "
            "on %d modules; adapter %s; fingerprint %s" % (
                len(rows), " ".join("%s %d" % kv for kv in sorted(
                    (rows.domains if isinstance(rows, RowFile) else _count(rows, "domain")).items())), len(dev), s["epochs"],
                len(order), steps, per, s["lr"], int(math.ceil(steps * float(s["warmup_ratio"]))), s["lora_dropout"], n_drop,
                cfg["policy"]["init_adapter"] or "fresh LoRA", fp))
        if dist.rank == 0:
            resolved = json.dumps(dict(cfg, fingerprint=fp, started_at_step=start, steps=steps), indent=1, default=str)
            retry_when_full(lambda: write_text(os.path.join(run_dir, "config.resolved.json"), resolved),
                            "config.resolved.json", wait_min, log)
        dev_samples = [row_sample(r, tok) for r in dev]
        history = {k: m for k, m in TR.read_metrics(run_dir).items() if k < start}
        if dev_samples and start == 0:
            d0 = score_rows(policy, dev_samples, dist, s["temperature"])
            log("before training: dev loss %.4f (per token %.4f) on %d rows" % (d0["dev_loss"], d0["dev_nll_per_token"], d0["dev_rows"]))
            if dist.rank == 0:
                retry_when_full(lambda: write_text(os.path.join(run_dir, "dev_start.json"), json.dumps(d0)), "dev_start.json", wait_min, log)
        for step in range(start, steps):
            t0 = time.time()
            batch = [rows[i] for i in order[step * per:(step + 1) * per]]
            samples = [row_sample(r, tok) for r in batch]
            m = {"step": step, **train_step(policy, samples, dist, opt, s["max_grad_norm"], s["temperature"]),
                 "lr": sched.get_last_lr()[0], "by_tier": _count(batch, "tier"), "by_domain": _count(batch, "domain")}
            sched.step()
            last = step + 1 == steps
            if dev_samples and (last or (int(s["dev_every"]) and (step + 1) % int(s["dev_every"]) == 0)):
                m.update(score_rows(policy, dev_samples, dist, s["temperature"]))
            m["t_train"] = round(time.time() - t0, 1)
            m["time"] = time.strftime("%F %T")
            history[step] = m
            if last or (step + 1) % int(s["save_every"]) == 0:
                TR.save_ckpt(run_dir, step + 1, policy, opt, sched, dist, {"fingerprint": fp, "rewarm_from": rewarm["from"]},
                             wait_min, log)
                if dist.rank == 0:
                    retry_when_full(lambda: TR.prune_ckpts(run_dir, cfg, step + 1), "ckpt/ (removing old checkpoints)", wait_min, log)
            if dist.rank == 0:
                retry_when_full(lambda: TR.append_metrics(run_dir, m), "metrics.jsonl", wait_min, log)
                log("step %d/%d | loss %.4f (per token %.4f)%s | rows %d tok %d | gn %.3f lr %.2e | %s%.0fs | longest %d tok peak %s GiB" % (
                    step + 1, steps, m["loss"], m["nll_per_token"],
                    "".join(" %s %.4f" % (k, m[k]) for k in ("loss_conseq", "loss_policy", "nll_info_token",
                                                              "dev_loss_conseq", "dev_loss_policy",
                                                              "dev_nll_info_token") if k in m),
                    m["rows"], m["tokens"], m["grad_norm"], m["lr"],
                    ("dev %.4f | " % m["dev_loss"]) if "dev_loss" in m else "", m["t_train"], m["max_row_tokens"], m["peak_mem_gib"]))
        if dist.rank == 0:
            final = os.path.join(run_dir, "final")
            retry_when_full(lambda: policy.save_adapter(final), "final adapter", wait_min, log)
            done = json.dumps({"steps": steps, "rows": len(rows), "time": time.strftime("%F %T")})
            retry_when_full(lambda: write_text(os.path.join(run_dir, "DONE"), done), "DONE", wait_min, log)
            log("SFT_ROWS_DONE %d steps; final adapter %s" % (steps, final))
        dist.barrier()
        return history
    except BaseException:
        if dist.world > 1:                                 # as the RL trainer: leave at once, torchrun stops the rest
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        raise
    finally:
        dist.close()


def _count(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key.sub=value (YAML values)")
    a = ap.parse_args(argv)
    train(TR.load_config(a.config, a.overrides))
    return 0


if __name__ == "__main__":
    sys.exit(main())
