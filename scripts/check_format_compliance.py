"""
Post-training format compliance check for warmstart SFT checkpoints.

Generates completions from a trained checkpoint on dev-set prompts and
measures what fraction contain parseable grammar actions. Used as a
gate-keeping check before submitting GRPO RL training.

Why this is a separate script (not a trainer callback):
  - Multi-GPU training with DeepSpeed ZeRO can't safely run model.generate()
    inside the training loop. Running it post-training on 1 GPU avoids all
    distributed-generation hazards.

Usage:
    # Check the default warmstart checkpoint
    python scripts/check_format_compliance.py \
      --checkpoint checkpoints/sft/warmstart_qwen3_14b_38930650/final \
      --dev /ocean/projects/mch250030p/wxu7/DesignBench/data/sft/dev.jsonl \
      --n 50 \
      --save-results logs/compliance/warmstart_qwen3_14b_38930650.json

    # Output:
    #   Checked 50 prompts from .../dev.jsonl
    #   format_compliance: 92.0% (46/50 parseable)
    #   ready_for_grpo: True  (threshold=90%)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

# Make llm_finetune importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Grammar pattern — accepts <action>ACTION(...)</action> or bare ACTION(...)
_ACTION_TAG = re.compile(
    r"<action>\s*"
    r"(SCALE_PARAM|SCALE_MULTI_PARAM|ADD_MEMBER|MODIFY_PARAM|REMOVE_MEMBER|MOVE_JOINT|OPTIMAL_STATE)"
    r"(?:\([^)]*\))?"
    r"\s*</action>"
)
_BARE_ACTION = re.compile(
    r"(SCALE_PARAM|SCALE_MULTI_PARAM|ADD_MEMBER|MODIFY_PARAM|REMOVE_MEMBER|MOVE_JOINT|OPTIMAL_STATE)"
    r"\([^)]*\)"
)


def main():
    parser = argparse.ArgumentParser(description="Check warmstart format compliance")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to warmstart SFT checkpoint directory")
    parser.add_argument("--dev", required=True,
                        help="Path to dev.jsonl (for prompt extraction)")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of dev prompts to check")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Required compliance rate to be RL-ready")
    parser.add_argument("--device", default="cuda",
                        help="Device to run generation on")
    parser.add_argument("--save-results", default=None,
                        help="Path to write per-sample JSON results (optional)")
    args = parser.parse_args()

    # Lazy imports: these pull in torch/transformers which we don't want on login node
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm_finetune.data.processors.chat_formatter import ChatFormatter
    from llm_finetune.data.processors.warmstart_transform import WarmstartTransform

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    log.info(f"Loading tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Loading model from {ckpt_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",  # keep it simple for eval
    ).to(args.device)
    model.eval()

    # Use the checkpoint's model_name_or_path for chat template selection.
    # Fall back to generic formatter if unknown.
    try:
        config_path = ckpt_path / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        model_id = config.get("_name_or_path", "")
    except Exception:
        model_id = ""
    formatter = ChatFormatter.from_model_id(model_id, tokenizer)

    transform = WarmstartTransform()

    log.info(f"Loading dev prompts from {args.dev}")
    prompts = []
    prompt_meta = []   # parallel list of metadata dicts
    with open(args.dev) as f:
        for line in f:
            if len(prompts) >= args.n:
                break
            ex = json.loads(line)
            messages = ex.get("messages", [])
            if not messages:
                continue
            # Apply the same transform used during training
            transformed = transform.transform_messages(messages)
            # Use only the system message (the problem prompt) — ask the model
            # to produce its first assistant turn
            system_msg = next((m for m in transformed if m["role"] == "system"), None)
            if system_msg is None:
                continue
            prompt_messages = [system_msg]
            prompts.append(prompt_messages)
            prompt_meta.append({
                "problem_id": ex.get("problem_id", ""),
                "trace_id": ex.get("trace_id", ""),
                "trace_quality": ex.get("trace_quality", None),
                "strategy_type": ex.get("strategy_type", ""),
            })

    log.info(f"Generating {len(prompts)} completions ...")
    n_parseable = 0
    n_tag_wrapped = 0
    examples_shown = 0
    per_sample_results = []

    # Patterns for richer per-sample analysis
    _THINK_TAG = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    _ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    _ACTION_TYPE = re.compile(
        r"(SCALE_PARAM|SCALE_MULTI_PARAM|ADD_MEMBER|MODIFY_PARAM|REMOVE_MEMBER|MOVE_JOINT|OPTIMAL_STATE)"
    )

    for i, prompt_messages in enumerate(prompts):
        input_ids = formatter.apply_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(args.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(
            output[0][input_ids.shape[1]:],
            skip_special_tokens=False,
        )

        tag_match = _ACTION_TAG.search(gen_text)
        bare_match = _BARE_ACTION.search(gen_text)

        if tag_match:
            n_parseable += 1
            n_tag_wrapped += 1
        elif bare_match:
            n_parseable += 1

        # Per-sample analysis
        think_blocks = _THINK_TAG.findall(gen_text)
        answer_blocks = _ANSWER_TAG.findall(gen_text)
        action_type_match = _ACTION_TYPE.search(gen_text)
        output_tokens = output.shape[1] - input_ids.shape[1]

        status = "TAG" if tag_match else ("BARE" if bare_match else "FAIL")
        sample = {
            "idx": i,
            **prompt_meta[i],
            "status": status,
            "action_type": action_type_match.group(1) if action_type_match else None,
            "has_think": len(think_blocks) > 0,
            "n_think_blocks": len(think_blocks),
            "has_answer": len(answer_blocks) > 0,
            "n_answer_blocks": len(answer_blocks),
            "output_tokens": output_tokens,
            "gen_preview": gen_text[:300].replace("\n", " "),
        }
        per_sample_results.append(sample)

        if examples_shown < 3:
            preview = gen_text[:200].replace("\n", " ")
            log.info(f"  [{i}] ✓ {status}: {preview}")
            examples_shown += 1

    n_total = len(prompts)
    compliance = n_parseable / n_total if n_total > 0 else 0.0
    tag_rate = n_tag_wrapped / n_total if n_total > 0 else 0.0
    ready = compliance >= args.threshold

    think_rate = sum(1 for s in per_sample_results if s["has_think"]) / n_total
    answer_rate = sum(1 for s in per_sample_results if s["has_answer"]) / n_total
    mean_output_tokens = sum(s["output_tokens"] for s in per_sample_results) / n_total

    print()
    print(f"Checked {n_total} prompts from {args.dev}")
    print(f"format_compliance: {compliance:.1%} ({n_parseable}/{n_total} parseable)")
    print(f"action_tag_rate:   {tag_rate:.1%} ({n_tag_wrapped}/{n_total} wrapped in <action>)")
    print(f"think_tag_rate:    {think_rate:.1%} ({sum(1 for s in per_sample_results if s['has_think'])}/{n_total} with <think>)")
    print(f"answer_tag_rate:   {answer_rate:.1%} ({sum(1 for s in per_sample_results if s['has_answer'])}/{n_total} with <answer>)")
    print(f"mean_output_tokens:{mean_output_tokens:.1f}")
    print(f"ready_for_grpo:    {ready}  (threshold={args.threshold:.0%})")
    print()

    if args.save_results:
        out_path = Path(args.save_results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "checkpoint": str(args.checkpoint),
            "dev_file": str(args.dev),
            "n_total": n_total,
            "format_compliance": compliance,
            "action_tag_rate": tag_rate,
            "think_tag_rate": think_rate,
            "answer_rate": answer_rate,
            "mean_output_tokens": mean_output_tokens,
            "ready_for_grpo": ready,
            "threshold": args.threshold,
            "samples": per_sample_results,
        }
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Per-sample results saved to {out_path}")

    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
