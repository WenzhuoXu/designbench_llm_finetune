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


from llm_finetune.data.grammar import (  # noqa: E402
    ACTION_TAG_PATTERN,
    count_answer_tags,
    count_think_tags,
    find_actions,
    validate_action,
)


def evaluate_generation_text(gen_text: str, *, problem_or_state: dict | None = None) -> dict:
    """Evaluate one generated response under the strict gold warmstart gates."""
    actions = find_actions(gen_text)
    action_tag_count = len(ACTION_TAG_PATTERN.findall(gen_text or ""))
    answer_count = count_answer_tags(gen_text)
    think_count = count_think_tags(gen_text)
    single_action = len(actions) == 1 and (action_tag_count in (0, 1))
    validation = validate_action(actions[0], problem_or_state) if actions else None
    semantic_ok = bool(single_action and validation and validation.is_valid)
    if semantic_ok:
        invalid_reason = ""
    elif not actions:
        invalid_reason = "no_action"
    elif not single_action:
        invalid_reason = "multiple_actions"
    else:
        invalid_reason = validation.reason if validation is not None else "invalid_action"
    return {
        "status": "STRICT" if semantic_ok else "FAIL",
        "semantic_ok": semantic_ok,
        "executable_ok": semantic_ok,
        "single_action": single_action,
        "action": validation.action if validation else "",
        "canonical_action": validation.canonical_action if validation else "",
        "action_type": validation.action_type if validation else None,
        "invalid_reason": invalid_reason,
        "n_actions": len(actions),
        "n_action_tags": action_tag_count,
        "has_think": think_count > 0,
        "n_think_blocks": think_count,
        "has_answer": answer_count > 0,
        "n_answer_blocks": answer_count,
        "answer_spam": answer_count > 0,
    }


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
    parser.add_argument("--semantic-compliance-min", type=float, default=0.95)
    parser.add_argument("--executable-action-min", type=float, default=0.95)
    parser.add_argument("--single-action-min", type=float, default=0.98)
    parser.add_argument("--answer-spam-max", type=float, default=0.02)
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

    # Use the checkpoint's base model for chat template selection.
    # LoRA checkpoints usually have adapter_config.json rather than config.json.
    try:
        config_path = ckpt_path / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        model_id = config.get("_name_or_path", "")
    except Exception:
        model_id = ""
    if not model_id:
        try:
            adapter_config_path = ckpt_path / "adapter_config.json"
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            model_id = adapter_config.get("base_model_name_or_path", "")
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
            # Apply the same transform used during training.
            transformed = transform.transform_messages(messages)
            first_assistant_idx = next(
                (idx for idx, m in enumerate(transformed) if m["role"] == "assistant"),
                None,
            )
            if first_assistant_idx is None or first_assistant_idx == 0:
                continue
            prompt_messages = transformed[:first_assistant_idx]
            prompts.append(prompt_messages)
            prompt_meta.append({
                "problem_id": ex.get("problem_id", ""),
                "trace_id": ex.get("trace_id", ""),
                "trace_quality": ex.get("trace_quality", None),
                "strategy_type": ex.get("strategy_type", ""),
            })

    log.info(f"Generating {len(prompts)} completions ...")
    examples_shown = 0
    per_sample_results = []

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

        eval_result = evaluate_generation_text(gen_text)
        output_tokens = output.shape[1] - input_ids.shape[1]

        sample = {
            "idx": i,
            **prompt_meta[i],
            **eval_result,
            "messages": prompt_messages,
            "source": "greedy",
            "rank": 0,
            "parsed_action": eval_result["action"],
            "rejection_reason": eval_result["invalid_reason"],
            "raw_output": gen_text,
            "output_tokens": output_tokens,
            "gen_preview": gen_text[:300].replace("\n", " "),
        }
        per_sample_results.append(sample)

        if examples_shown < 3:
            preview = gen_text[:200].replace("\n", " ")
            log.info(f"  [{i}] {sample['status']}: {preview}")
            examples_shown += 1

    n_total = len(prompts)
    if n_total == 0:
        log.error("No prompts were checked; zero-candidate compliance runs are invalid.")
        sys.exit(1)

    semantic_rate = sum(1 for s in per_sample_results if s["semantic_ok"]) / n_total
    executable_rate = sum(1 for s in per_sample_results if s["executable_ok"]) / n_total
    single_action_rate = sum(1 for s in per_sample_results if s["single_action"]) / n_total
    tag_rate = sum(1 for s in per_sample_results if s["n_action_tags"] == 1) / n_total
    think_rate = sum(1 for s in per_sample_results if s["has_think"]) / n_total
    answer_spam_rate = sum(1 for s in per_sample_results if s["answer_spam"]) / n_total
    invalid_placeholder_or_range = sum(
        1
        for s in per_sample_results
        if s["invalid_reason"] in {"placeholder_token", "range_member_ids"}
    )
    mean_output_tokens = sum(s["output_tokens"] for s in per_sample_results) / n_total
    semantic_min = max(args.threshold, args.semantic_compliance_min)
    ready = (
        semantic_rate >= semantic_min
        and executable_rate >= args.executable_action_min
        and single_action_rate >= args.single_action_min
        and answer_spam_rate <= args.answer_spam_max
        and invalid_placeholder_or_range == 0
    )

    print()
    print(f"Checked {n_total} prompts from {args.dev}")
    print(f"semantic_compliance: {semantic_rate:.1%}")
    print(f"executable_action:   {executable_rate:.1%}")
    print(f"single_action_rate:  {single_action_rate:.1%}")
    print(f"action_tag_rate:     {tag_rate:.1%}")
    print(f"think_tag_rate:      {think_rate:.1%}")
    print(f"answer_spam_rate:    {answer_spam_rate:.1%}")
    print(f"invalid_placeholder_or_range: {invalid_placeholder_or_range}")
    print(f"mean_output_tokens:{mean_output_tokens:.1f}")
    print(f"ready_for_grpo:    {ready}")
    print()

    if args.save_results:
        out_path = Path(args.save_results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "checkpoint": str(args.checkpoint),
            "dev_file": str(args.dev),
            "n_total": n_total,
            "format_compliance": semantic_rate,
            "semantic_compliance": semantic_rate,
            "executable_action": executable_rate,
            "single_action_rate": single_action_rate,
            "action_tag_rate": tag_rate,
            "think_tag_rate": think_rate,
            "answer_spam_rate": answer_spam_rate,
            "invalid_placeholder_or_range": invalid_placeholder_or_range,
            "mean_output_tokens": mean_output_tokens,
            "ready_for_grpo": ready,
            "threshold": semantic_min,
            "gold_gates": {
                "semantic_compliance_min": semantic_min,
                "executable_action_min": args.executable_action_min,
                "single_action_min": args.single_action_min,
                "answer_spam_max": args.answer_spam_max,
            },
            "samples": per_sample_results,
        }
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Per-sample results saved to {out_path}")

    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
