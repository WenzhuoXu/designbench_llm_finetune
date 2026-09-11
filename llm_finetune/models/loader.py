"""
Model and tokenizer loading with advanced infrastructure:
  - Flash Attention 2 (mandatory for H100, 3-5x memory/speed improvement)
  - torch.compile with inductor backend (~20% throughput on H100)
  - Gradient checkpointing for memory-efficient training
  - Optional LoRA via PEFT
  - Thinking model special token handling (Qwen3, DeepSeek-R1, Phi-4-reasoning)
  - BF16 precision throughout

Usage:
    from llm_finetune.models.loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg.model)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

log = logging.getLogger(__name__)

# HF cache on the HPC storage system
DEFAULT_HF_CACHE = "/ocean/projects/mch250030p/wxu7/hf_models"

# Models that support/require thinking tokens
THINKING_MODELS = {
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-0528",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "microsoft/Phi-4-reasoning",
}


def load_model_and_tokenizer(
    cfg: DictConfig,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load model and tokenizer with full H100-optimized infrastructure.

    Args:
        cfg: Model config from configs/model/*.yaml. Must contain:
            - model_name_or_path: HF model ID or local path
            - torch_dtype: "bfloat16" (recommended for H100)
            - attn_implementation: "flash_attention_2" (recommended)
            - trust_remote_code: bool
            - cache_dir: HF cache directory
            - use_compile: bool — enable torch.compile
            - use_gradient_checkpointing: bool
            - use_lora: bool — inject LoRA adapters
            - lora: DictConfig with LoRA hyperparameters (if use_lora=True)

    Returns:
        (model, tokenizer) tuple ready for training.
    """
    model_id = cfg.model_name_or_path
    cache_dir = cfg.get("cache_dir", DEFAULT_HF_CACHE)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", cache_dir)

    log.info(f"Loading tokenizer: {model_id}")
    tokenizer = _load_tokenizer(model_id, cfg, cache_dir)

    log.info(f"Loading model: {model_id} (dtype={cfg.torch_dtype})")
    model = _load_model(model_id, cfg, cache_dir)

    if cfg.get("use_gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        log.info("Gradient checkpointing enabled")

    if cfg.get("use_lora", False):
        model = _apply_lora(model, cfg.lora)

    if cfg.get("use_compile", False):
        log.info("Applying torch.compile (inductor, max-autotune) ...")
        model = torch.compile(model, backend="inductor", mode="max-autotune")
        log.info("torch.compile applied")

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    log.info(f"Model params: {n_params:.2f}B total, {n_trainable:.2f}B trainable")

    return model, tokenizer


def _load_tokenizer(model_id: str, cfg: DictConfig, cache_dir: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=cfg.get("trust_remote_code", True),
        padding_side="right",  # for SFT; DataCollatorForRL will switch to left-pad
    )

    # Ensure pad token exists (many models don't have one by default)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        log.info(f"Set pad_token = {tokenizer.pad_token!r}")

    # Thinking model: ensure <think>/<|think|> tokens are in vocabulary
    if model_id in THINKING_MODELS or cfg.get("thinking_mode", False):
        _ensure_thinking_tokens(tokenizer, model_id)

    return tokenizer


def _load_model(model_id: str, cfg: DictConfig, cache_dir: str) -> AutoModelForCausalLM:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)

    attn_impl = cfg.get("attn_implementation", "flash_attention_2")

    # Verify flash-attn is available
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            log.warning(
                "flash-attn not installed. Falling back to sdpa. "
                "Install with: pip install flash-attn --no-build-isolation"
            )
            attn_impl = "sdpa"

    model_kwargs = dict(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        trust_remote_code=cfg.get("trust_remote_code", True),
        cache_dir=cache_dir,
    )

    # Only pass device_map when explicitly set. Must be None/absent for DeepSpeed.
    device_map = cfg.get("device_map", None)
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    log.info(f"attn_implementation={attn_impl}, dtype={torch_dtype}, device_map={device_map}")
    model = _model_class(model_id, cache_dir, cfg).from_pretrained(**model_kwargs)

    return model


def _model_class(model_id: str, cache_dir: str, cfg: DictConfig):
    """Resolve the class the checkpoint actually declares.

    AutoModelForCausalLM is right for a plain decoder, but newer Qwen releases
    (qwen3_5, which Qwen3.6-27B and Qwen3.8-27B both use) ship as multimodal
    wrappers -- Qwen3_5ForConditionalGeneration, with the text tower under
    config.text_config. Auto hands the outer config to the text model and dies
    on a missing vocab_size. Honour the declared architecture when transformers
    exposes it, and fall back to Auto otherwise.
    """
    import transformers
    from transformers import AutoConfig

    try:
        conf = AutoConfig.from_pretrained(
            model_id, cache_dir=cache_dir,
            trust_remote_code=cfg.get("trust_remote_code", True))
        arch = (getattr(conf, "architectures", None) or [None])[0]
    except Exception:
        arch = None
    if arch and arch != "AutoModelForCausalLM":
        cls = getattr(transformers, arch, None)
        if cls is not None:
            log.info(f"loading declared architecture {arch}")
            return cls
        log.warning(f"config declares {arch}, not found in transformers; using AutoModelForCausalLM")
    return AutoModelForCausalLM


def _apply_lora(model: torch.nn.Module, lora_cfg: DictConfig) -> torch.nn.Module:
    """Inject LoRA adapters via PEFT."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        raise ImportError("peft not installed. Run: pip install peft")

    # A string is a REGEX matched against full module paths, not a list of names.
    # That distinction matters on multimodal checkpoints: Qwen3.8-27B carries a
    # vision tower whose blocks use the leaf names qkv/proj/linear_fc1/linear_fc2,
    # and a bare name list would attach adapters to it even though this corpus is
    # text only. list() on a string would also silently shatter it into single
    # characters, which PEFT would then match against nothing.
    tm = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
    target_modules = tm if isinstance(tm, str) else list(tm)

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        target_modules=target_modules,
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Report what actually got adapted. A target list that silently matches
    # nothing still "succeeds" and trains a model that cannot learn.
    hit = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    log.info(f"LoRA attached to {len(hit)} modules; first: {hit[:3]}")
    if not hit:
        raise RuntimeError(
            f"LoRA target_modules matched nothing: {target_modules!r}. "
            "Check the module names for this architecture.")
    log.info(f"LoRA applied: r={lora_cfg.get('r', 64)}, alpha={lora_cfg.get('lora_alpha', 128)}")
    return model


def _ensure_thinking_tokens(tokenizer: AutoTokenizer, model_id: str) -> None:
    """Ensure model-specific thinking tokens are properly registered."""
    if "Qwen3" in model_id or "qwen3" in model_id.lower():
        # Qwen3 uses <think> and </think> — should already be in vocab
        for tok in ["<think>", "</think>"]:
            if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id:
                log.warning(f"Token {tok!r} not in Qwen3 vocabulary — adding as special token")
                tokenizer.add_special_tokens({"additional_special_tokens": [tok]})

    elif "DeepSeek-R1" in model_id:
        # DeepSeek-R1 uses <think> and </think>
        for tok in ["<think>", "</think>"]:
            if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id:
                tokenizer.add_special_tokens({"additional_special_tokens": [tok]})

    elif "Phi-4-reasoning" in model_id:
        # Phi-4-reasoning uses <think> and </think>
        for tok in ["<think>", "</think>"]:
            if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id:
                tokenizer.add_special_tokens({"additional_special_tokens": [tok]})


def get_model_memory_footprint(model: torch.nn.Module) -> dict[str, float]:
    """Return per-GPU memory usage in GB."""
    try:
        import pynvml
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        stats = {}
        for i in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats[f"gpu_{i}_used_gb"] = info.used / 1e9
            stats[f"gpu_{i}_total_gb"] = info.total / 1e9
        return stats
    except Exception:
        return {}
