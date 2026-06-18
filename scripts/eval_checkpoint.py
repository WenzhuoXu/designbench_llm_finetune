"""
Checkpoint evaluation: load a fine-tuned model and run DesignBench benchmark.

Uses DesignBench's ValidationRunner to evaluate the model on all 10 benchmark
problems, computing feasibility rate, FOS improvement, mass reduction, and
grammar compliance.

Usage:
    python scripts/eval_checkpoint.py \
        --checkpoint checkpoints/sft/qwen3_sft/final \
        --model-config configs/model/qwen3_14b.yaml \
        --problems-dir /ocean/projects/mch250030p/wxu7/DesignBench/data/problems \
        --output-dir results/eval/qwen3_sft_001

    # Or evaluate base model (no checkpoint):
    python scripts/eval_checkpoint.py --model-config configs/model/qwen3_14b.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/DesignBench")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

_MT_DBG = {"n": 0}  # logs a few sample CoT turns across the eval for inspection


def _load_model_config(path):
    """Load a model config YAML with its Hydra ``defaults:`` list resolved.

    ``OmegaConf.load()`` alone does NOT process Hydra's ``defaults:`` list, so a
    config that inherits from ``base.yaml`` (e.g. ``qwen3_14b.yaml``) ends up
    missing every base key (``torch_dtype``, ``attn_implementation``,
    ``device_map``, ...). That is exactly what crashed the May-27 eval batch
    with ``Missing key torch_dtype``. This merges each default — resolved
    relative to the config's own directory — *under* the named config, which
    mimics Hydra composition for the common single-level ``defaults: [base]``
    case (and recurses for nested defaults). ``_self_`` and missing files are
    skipped.
    """
    from omegaconf import OmegaConf

    path = Path(path)
    cfg = OmegaConf.load(path)
    defaults = cfg.pop("defaults", []) or []
    merged = OmegaConf.create({})
    for d in defaults:
        if not isinstance(d, str) or d == "_self_":
            continue
        dep = path.parent / f"{d}.yaml"
        if dep.exists():
            merged = OmegaConf.merge(merged, _load_model_config(dep))
    merged = OmegaConf.merge(merged, cfg)
    return merged


def _extract_action_followup(model, tokenizer, formatter, messages, cot_completion, device):
    """Parse the action 'another way' for reasoning models that produce excellent
    CoT but won't emit the <action> tag inline. We keep the full CoT and ask the
    SAME model, in one focused follow-up, to commit its decision to a single
    grammar action. Returns (parsed_action_or_None, followup_text)."""
    import torch
    followup = list(messages) + [
        {"role": "assistant", "content": cot_completion},
        {"role": "user", "content":
         "Based on your reasoning above, output EXACTLY ONE grammar action and nothing else, "
         "in the form:\n<action>GRAMMAR_ACTION(...)</action>"},
    ]
    ids = formatter.apply_template(followup, add_generation_prompt=True, tokenize=True)
    t = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model.generate(t, max_new_tokens=1024, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    from llm_finetune.envs.truss_env import parse_grammar_action
    resp = tokenizer.decode(out[0][t.shape[1]:], skip_special_tokens=True)
    return parse_grammar_action(resp), resp


def _eval_problem_multiturn(spec, model, tokenizer, formatter, problem_text,
                            max_steps, temperature, max_new_tokens, few_shot=False,
                            extract_action=False):
    """Multi-turn rollout — the protocol the model was trained for and that the
    master doc (§3, horizon H≈9) assumes: generate ONE action, apply it via FEA,
    feed the result back as the next user turn, repeat until feasible or
    max_steps.

    This mirrors TrussRolloutEnv.run_completion's internal truss chaining
    (_load_truss_and_goals / _apply_action / _analyze_truss) but interleaves a
    fresh model generation between every action. The legacy single-turn path
    (run_completion on one big completion) forces the model to emit a whole
    trajectory in one shot — out-of-distribution for a model SFT'd one-action-
    per-turn — then guesses step boundaries by splitting on "\\n\\n", which
    yielded ~4% grammar success and ~0% feasibility in the S0 baseline.
    """
    import torch
    from llm_finetune.data.processors import designbench_prompt
    from llm_finetune.envs.truss_env import (
        _load_truss_and_goals, _apply_action, _analyze_truss, parse_grammar_action,
    )

    truss, goals = _load_truss_and_goals(spec)
    initial_state = _analyze_truss(truss, goals)
    state = initial_state
    device = next(model.parameters()).device

    action_history = []
    parse_success = []

    for _t in range(max_steps):
        if state.get("is_feasible", False):
            break
        messages = designbench_prompt.build_messages(spec, initial_state, action_history, formatter, few_shot=few_shot)
        prompt_ids = formatter.apply_template(messages, add_generation_prompt=True, tokenize=True)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(output[0][input_tensor.shape[1]:], skip_special_tokens=True)
        thinking, _ = formatter.extract_thinking(completion)
        parsed = parse_grammar_action(completion)
        followup = ""
        if parsed is None and extract_action:  # reasoning model: extract action from its CoT
            parsed, followup = _extract_action_followup(
                model, tokenizer, formatter, messages, completion, device)
        # keep the FULL CoT as the reasoning for this turn (this is what we distill)
        thinking = completion.strip()
        if _MT_DBG["n"] < 4:  # log a few sample CoT turns for inspection
            _MT_DBG["n"] += 1
            log.info(f"[CoT SAMPLE {spec.get('problem_id')} turn={_t} ntok={len(completion.split())} "
                     f"parsed={parsed!r}]\n>>> {completion[:500]}\n... [followup-extract]: {followup[:200]}")
        if parsed is None:
            parse_success.append(False)
            action_history.append({"action": completion.strip()[:120], "fea_result": state, "thinking": thinking})
            continue
        try:
            truss = _apply_action(truss, parsed)
            state = _analyze_truss(truss, goals)
            parse_success.append(True)
        except Exception as e:  # noqa: BLE001
            log.debug(f"action {parsed!r} failed: {e}")
            parse_success.append(False)
        action_history.append({"action": parsed, "fea_result": state, "thinking": thinking})

    return {
        "problem_id": spec.get("problem_id", ""),
        "reaches_solution": bool(state.get("is_feasible", False)),
        "n_steps": len(action_history),
        "final_fos_buckling": state.get("fos_buckling", 0),
        "final_fos_yielding": state.get("fos_yielding", 0),
        "final_mass": state.get("mass", 0),
        "initial_fos_buckling": initial_state.get("fos_buckling", 0),
        "initial_mass": initial_state.get("mass", 0),
        "grammar_success_rate": sum(parse_success) / max(len(parse_success), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on DesignBench")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint directory (None = use base model)")
    parser.add_argument("--model-config", required=True,
                        help="Path to model config YAML (e.g., configs/model/qwen3_14b.yaml)")
    parser.add_argument("--problems-dir",
                        default="/ocean/projects/mch250030p/wxu7/DesignBench/data/problems",
                        help="Directory with problem spec JSON files")
    parser.add_argument("--output-dir", default="results/eval",
                        help="Directory to save evaluation results")
    parser.add_argument("--wandb-project", default="designbench-training",
                        help="W&B project for logging results")
    parser.add_argument("--wandb-run-name", default=None,
                        help="W&B run name for this eval")
    parser.add_argument("--n-gpus", type=int, default=1,
                        help="Number of GPUs for model inference")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Max grammar actions per episode")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Samples per problem")
    parser.add_argument("--rollout", choices=["multi", "single"], default="multi",
                        help="multi: generate→FEA→generate loop, one action/turn "
                             "(matches training/inference distribution). single: one big "
                             "completion parsed for all actions (legacy; OOD, mismatched).")
    parser.add_argument("--few-shot", action="store_true",
                        help="Append a worked example to the system prompt (format-enable a base teacher model).")
    parser.add_argument("--extract-action", action="store_true",
                        help="For reasoning models that produce CoT but no inline <action>: when "
                             "no grammar action is parseable, ask the model once more to commit its "
                             "reasoning to a single grammar action (keeps the full CoT).")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max new tokens per model turn (multi-turn rollout).")
    parser.add_argument("--max-problems", type=int, default=0,
                        help="If >0, evaluate only the first N problems (quick baseline).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model config (resolves Hydra `defaults: [base]` — see _load_model_config)
    model_cfg = _load_model_config(args.model_config)
    base_model_id = model_cfg.model_name_or_path

    # Eval-time overrides (single-GPU inference, no DeepSpeed):
    #   device_map=auto  → place the model on the GPU. base.yaml uses null for
    #                      DeepSpeed; left as-is the model would sit on CPU and
    #                      mismatch the .cuda() input tensors below.
    #   grad-ckpt off    → gradient_checkpointing_enable() forces use_cache=False,
    #                      which makes generate() crawl. Off for eval.
    #   use_lora=False   → we attach the *trained* adapter below, not a fresh one.
    model_cfg.device_map = "auto"
    model_cfg.use_gradient_checkpointing = False
    model_cfg.use_compile = False
    model_cfg.use_lora = False

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    is_adapter = bool(ckpt and (ckpt / "adapter_config.json").exists())
    if ckpt and not is_adapter:
        # Full-weights checkpoint: load directly from the checkpoint directory.
        model_cfg.model_name_or_path = str(ckpt)

    model_id = str(ckpt) if ckpt else base_model_id
    log.info(f"Evaluating model: {model_id} (base={base_model_id}, adapter={is_adapter})")

    from llm_finetune.models.loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    # Attach the trained checkpoint. The v2 checkpoints are LoRA adapters on top
    # of the base model; the previous code loaded ONLY the base model and never
    # the adapter, so it would have scored raw Qwen3-14B (a silent, meaningless
    # baseline). Apply + merge the adapter, and adopt the checkpoint's own
    # tokenizer so the chat template matches what was used during training.
    if is_adapter:
        from peft import PeftModel
        log.info(f"Loading + merging LoRA adapter from {ckpt}")
        model = PeftModel.from_pretrained(model, str(ckpt))
        model = model.merge_and_unload()
        if (ckpt / "tokenizer_config.json").exists():
            from transformers import AutoTokenizer
            ck_tok = AutoTokenizer.from_pretrained(str(ckpt))
            if ck_tok.pad_token is None and ck_tok.eos_token is not None:
                ck_tok.pad_token = ck_tok.eos_token
                ck_tok.pad_token_id = ck_tok.eos_token_id
            tokenizer = ck_tok

    model.eval()
    log.info(f"Model loaded: {model_id}")

    # NOTE: We intentionally do NOT construct DesignBench's LocalModelAdapter here.
    # Its __init__ re-loads the model from `model_name` via AutoTokenizer/AutoModel
    # (model_name="final" → HF hub → 401), and the object was never used downstream
    # anyway — the eval loop below drives `model.generate` directly. Removing it
    # avoids a wasteful second load and the crash it caused.

    # Run benchmark evaluation problem by problem
    problems_dir = Path(args.problems_dir)
    problem_files = sorted(problems_dir.glob("auto_problem_*.json"))
    if not problem_files:
        problem_files = sorted(problems_dir.glob("*.json"))
    log.info(f"Found {len(problem_files)} problems in {problems_dir}")

    from llm_finetune.envs.truss_env import TrussRolloutEnv
    from llm_finetune.data.processors.chat_formatter import ChatFormatter
    from llm_finetune.data.datasets.rl_dataset import _spec_to_problem_text
    import torch

    # Use the base HF id (not the checkpoint path) so thinking mode resolves via
    # MODEL_THINKING_MODE — a checkpoint path would default to NONE and silently
    # drop Qwen3 enable_thinking.
    formatter = ChatFormatter.from_model_id(base_model_id, tokenizer)
    env = TrussRolloutEnv(max_steps=args.max_steps)
    device = next(model.parameters()).device

    if args.max_problems and args.max_problems > 0:
        problem_files = problem_files[:args.max_problems]
        log.info(f"Limiting to first {len(problem_files)} problems (--max-problems)")
    log.info(f"Rollout mode: {args.rollout}")

    all_results = []
    for pf in problem_files:
        with open(pf) as f:
            spec = json.load(f)
        pid = spec.get("problem_id", pf.stem)
        log.info(f"Evaluating problem: {pid}")
        problem_text = _spec_to_problem_text(spec)

        if args.rollout == "multi":
            result = _eval_problem_multiturn(
                spec, model, tokenizer, formatter, problem_text,
                max_steps=args.max_steps, temperature=args.temperature,
                max_new_tokens=args.max_new_tokens, few_shot=args.few_shot,
                extract_action=args.extract_action,
            )
            result["problem_id"] = pid
        else:
            # Legacy single-turn: one big completion, parse all actions out of it.
            messages = formatter.build_messages(problem_text=problem_text)
            prompt_ids = formatter.apply_template(messages, add_generation_prompt=True, tokenize=True)
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
            with torch.no_grad():
                output = model.generate(
                    input_tensor,
                    max_new_tokens=8192,
                    temperature=args.temperature if args.temperature > 0 else None,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion = tokenizer.decode(output[0][input_tensor.shape[1]:], skip_special_tokens=True)
            rollout = env.run_completion(spec, completion)
            result = {
                "problem_id": pid,
                "reaches_solution": rollout.reaches_solution,
                "n_steps": rollout.n_steps,
                "final_fos_buckling": rollout.final_state.get("fos_buckling", 0),
                "final_fos_yielding": rollout.final_state.get("fos_yielding", 0),
                "final_mass": rollout.final_state.get("mass", 0),
                "initial_fos_buckling": rollout.initial_state.get("fos_buckling", 0),
                "initial_mass": rollout.initial_state.get("mass", 0),
                "grammar_success_rate": sum(rollout.parse_success) / max(len(rollout.parse_success), 1),
            }

        all_results.append(result)
        log.info(f"  {pid}: feasible={result['reaches_solution']}, "
                 f"fos_b={result['final_fos_buckling']:.3f}, n_steps={result['n_steps']}, "
                 f"gram={result['grammar_success_rate']:.2f}")

    # Aggregate metrics
    n = len(all_results)
    agg = {
        "n_problems": n,
        "rollout": args.rollout,
        "feasibility_rate": sum(r["reaches_solution"] for r in all_results) / n,
        "mean_grammar_success_rate": sum(r["grammar_success_rate"] for r in all_results) / n,
        "mean_final_fos_buckling": sum(r["final_fos_buckling"] for r in all_results) / n,
        "mean_steps": sum(r["n_steps"] for r in all_results) / n,
    }
    log.info(f"\nAggregated results:\n" + json.dumps(agg, indent=2))

    # Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump({"aggregate": agg, "per_problem": all_results}, f, indent=2)
    log.info(f"Results saved to {results_path}")

    # Log to W&B
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"eval_{model_id.split('/')[-1]}",
            config={"checkpoint": args.checkpoint, "model": model_id},
        )
        wandb.log({"eval/" + k: v for k, v in agg.items()})
        # Log as W&B table
        table = wandb.Table(columns=list(all_results[0].keys()))
        for r in all_results:
            table.add_data(*r.values())
        wandb.log({"eval/per_problem": table})
        run.finish()
        log.info("W&B eval run logged")
    except Exception as e:
        log.warning(f"W&B logging failed: {e}")


if __name__ == "__main__":
    main()
