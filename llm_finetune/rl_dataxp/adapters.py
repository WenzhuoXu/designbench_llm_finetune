"""Exact sums of LoRA adapters, as one adapter.

    python -m llm_finetune.rl_dataxp.adapters --adapter R=/path/with_reasoning --adapter A=/path/moves_only \\
        --weight 1.0 --weight 0.5 --out /path/merged_beta05

A weight-space merge "base + dR + beta dA" of two LoRA fine-tunes of the same base (the
collaborator's merged model) is itself a LoRA of rank r1 + r2: stack the A matrices and
put each adapter's scale (weight * lora_alpha / r) into its B block,

    sum_i w_i (alpha_i / r_i) B_i A_i  =  [w_1 s_1 B_1 | w_2 s_2 B_2] @ [A_1 ; A_2],

with lora_alpha = r so the combined scale is 1. The result is exact (no SVD truncation), keeps
both halves of every packed layer vLLM needs (in_proj_qkv + in_proj_z, in_proj_b + in_proj_a),
serves through --enable-lora --max-lora-rank r1+r2, and can be trained further. Disabling it
gives back the base model, which is what the GRPO trainer's KL reference needs.

Trainer env (torch + safetensors).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys

A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"


def _load(path: str):
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    st = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(st):
        raise FileNotFoundError("%s has no adapter_model.safetensors" % path)
    return cfg, load_file(st)


def _check_cfg(cfgs: list[dict], paths: list[str]) -> None:
    c0 = cfgs[0]
    for c, p in zip(cfgs, paths):
        if c.get("peft_type", "LORA") != "LORA":
            raise ValueError("%s: not a LoRA adapter (%s)" % (p, c.get("peft_type")))
        if c.get("use_dora") or c.get("use_rslora"):
            raise ValueError("%s: DoRA / rsLoRA adapters are not supported" % p)
        if c.get("rank_pattern") or c.get("alpha_pattern"):
            raise ValueError("%s: per-module rank/alpha patterns are not supported" % p)
        if c.get("base_model_name_or_path") != c0.get("base_model_name_or_path"):
            raise ValueError("different base models: %s vs %s" % (c0.get("base_model_name_or_path"),
                                                                  c.get("base_model_name_or_path")))
        if c.get("modules_to_save"):
            raise ValueError("%s: modules_to_save (full copies of modules) cannot be summed as LoRA" % p)


def concat_adapters(paths: list[str], weights: list[float], out: str) -> dict:
    import torch
    from safetensors.torch import save_file
    if len(paths) != len(weights) or not paths:
        raise ValueError("one weight per adapter")
    loaded = [_load(p) for p in paths]
    cfgs = [c for c, _ in loaded]
    sds = [s for _, s in loaded]
    _check_cfg(cfgs, paths)
    prefixes = [sorted(k[: -len(A_SUFFIX)] for k in sd if k.endswith(A_SUFFIX)) for sd in sds]
    for p, pre, sd in zip(paths, prefixes, sds):
        if pre != prefixes[0]:
            raise ValueError("%s adapts a different set of modules than %s" % (p, paths[0]))
        extra = [k for k in sd if not (k.endswith(A_SUFFIX) or k.endswith(B_SUFFIX))]
        if extra:
            raise ValueError("%s has non-LoRA tensors, e.g. %s" % (p, extra[:3]))
    ranks = [int(c["r"]) for c in cfgs]
    scales = [w * float(c["lora_alpha"]) / int(c["r"]) for w, c in zip(weights, cfgs)]
    merged = {}
    for pre in prefixes[0]:
        As = [sd[pre + A_SUFFIX].float() for sd in sds]
        Bs = [sd[pre + B_SUFFIX].float() * s for sd, s in zip(sds, scales)]
        for A, B, r, p in zip(As, Bs, ranks, paths):
            if A.shape[0] != r or B.shape[1] != r:
                raise ValueError("%s: %s has rank %d/%d, config says %d" % (p, pre, A.shape[0], B.shape[1], r))
        merged[pre + A_SUFFIX] = torch.cat(As, 0).contiguous()
        merged[pre + B_SUFFIX] = torch.cat(Bs, 1).contiguous()
    rank = sum(ranks)
    cfg = copy.deepcopy(cfgs[0])
    cfg.update(r=rank, lora_alpha=rank, lora_dropout=0.0)
    os.makedirs(out, exist_ok=True)
    save_file(merged, os.path.join(out, "adapter_model.safetensors"))
    json.dump(cfg, open(os.path.join(out, "adapter_config.json"), "w"), indent=2)
    prov = {"sum": [{"path": os.path.abspath(p), "weight": w, "r": r, "lora_alpha": c["lora_alpha"],
                     "sha1": hashlib.sha1(open(os.path.join(p, "adapter_model.safetensors"), "rb").read()).hexdigest()}
                    for p, w, r, c in zip(paths, weights, ranks, cfgs)],
            "rank": rank, "modules": len(prefixes[0])}
    json.dump(prov, open(os.path.join(out, "MERGE_PROVENANCE.json"), "w"), indent=2)
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="exact sum of LoRA adapters as one LoRA adapter")
    ap.add_argument("--adapter", action="append", required=True, help="[label=]path, repeated")
    ap.add_argument("--weight", action="append", type=float, required=True, help="one per --adapter")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    paths = [x.partition("=")[2] if "=" in x else x for x in a.adapter]
    prov = concat_adapters(paths, a.weight, a.out)
    print("rank %d over %d modules -> %s" % (prov["rank"], prov["modules"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
