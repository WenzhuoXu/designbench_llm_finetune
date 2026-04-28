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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model config
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.load(args.model_config)
    model_id = args.checkpoint or model_cfg.model_name_or_path
    log.info(f"Evaluating model: {model_id}")

    # Load model and tokenizer
    from llm_finetune.models.loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_cfg)
    log.info(f"Model loaded: {model_id}")

    # Load DesignBench validation infrastructure
    try:
        # Use DesignBench's local model adapter pattern
        from validation.model_adapters import LocalModelAdapter
        adapter = LocalModelAdapter(
            model=model,
            tokenizer=tokenizer,
            model_name=model_id.split("/")[-1],
        )
        log.info("DesignBench LocalModelAdapter created")
    except ImportError as e:
        log.error(f"Failed to import DesignBench validation: {e}")
        log.error("Ensure DesignBench is in sys.path and dependencies are installed")
        sys.exit(1)

    # Run benchmark evaluation problem by problem
    problems_dir = Path(args.problems_dir)
    problem_files = sorted(problems_dir.glob("auto_problem_*.json"))
    if not problem_files:
        problem_files = sorted(problems_dir.glob("*.json"))
    log.info(f"Found {len(problem_files)} problems in {problems_dir}")

    all_results = []
    for pf in problem_files:
        with open(pf) as f:
            spec = json.load(f)
        pid = spec.get("problem_id", pf.stem)
        log.info(f"Evaluating problem: {pid}")

        # Simple validation loop using TrussRolloutEnv
        from llm_finetune.envs.truss_env import TrussRolloutEnv
        from llm_finetune.data.processors.chat_formatter import ChatFormatter
        from llm_finetune.data.datasets.rl_dataset import _spec_to_problem_text

        formatter = ChatFormatter.from_model_id(model_id, tokenizer)
        env = TrussRolloutEnv(max_steps=args.max_steps)

        problem_text = _spec_to_problem_text(spec)
        messages = formatter.build_messages(problem_text=problem_text)
        prompt_ids = formatter.apply_template(messages, add_generation_prompt=True, tokenize=True)

        import torch
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long).cuda()

        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=8192,
                temperature=args.temperature if args.temperature > 0 else None,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        completion_ids = output[0][input_tensor.shape[1]:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

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
                 f"fos_b={result['final_fos_buckling']:.3f}, n_steps={result['n_steps']}")

    # Aggregate metrics
    n = len(all_results)
    agg = {
        "n_problems": n,
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
