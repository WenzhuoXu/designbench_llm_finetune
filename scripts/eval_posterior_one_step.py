"""
One-step online posterior evaluator.

Runs live LLM candidate generation on real problem states, evaluates tree-expanded
lookahead online with FEA transitions, and writes per-state metrics as JSONL.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/DesignBench")

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-step online posterior evaluation")
    parser.add_argument(
        "--model-config",
        default="configs/model/qwen25_0p5b.yaml",
        help="Path to model YAML config",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint path overriding model_name_or_path",
    )
    parser.add_argument(
        "--problems-dir",
        default="/ocean/projects/mch250030p/wxu7/DesignBench/data/problems",
        help="DesignBench problems directory",
    )
    parser.add_argument("--n-states", type=int, default=8)
    parser.add_argument("--sampling-mode", default="rollin_policy")
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--reference-depth", type=int, default=2)
    parser.add_argument("--level1-branching", type=int, default=5)
    parser.add_argument("--deeper-branching", type=int, default=3)
    parser.add_argument("--output", default="logs/posterior/one_step_eval.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()

    from llm_finetune.data.datasets.rl_dataset import RLPromptDataset
    from llm_finetune.data.processors.chat_formatter import ChatFormatter
    from llm_finetune.models.loader import load_model_and_tokenizer
    from llm_finetune.training.rl.posterior.evaluator import (
        OneStepOnlinePosteriorEvaluator,
        PosteriorEvalConfig,
        save_records_jsonl,
    )

    model_cfg = OmegaConf.load(args.model_config)
    formatter_model_id = model_cfg.model_name_or_path
    if args.checkpoint:
        model_cfg.model_name_or_path = args.checkpoint

    model, tokenizer = load_model_and_tokenizer(model_cfg)
    formatter = ChatFormatter.from_model_id(formatter_model_id, tokenizer)
    dataset = RLPromptDataset.from_problems_dir(
        problems_dir=args.problems_dir,
        tokenizer=tokenizer,
        formatter=formatter,
    )

    eval_cfg = PosteriorEvalConfig(
        sampling_mode=args.sampling_mode,
        n_states=args.n_states,
        rollout_steps=args.rollout_steps,
        candidate_count=args.candidate_count,
        depth=args.depth,
        reference_depth=args.reference_depth,
        level1_branching=args.level1_branching,
        deeper_branching=args.deeper_branching,
        seed=args.seed,
    )

    evaluator = OneStepOnlinePosteriorEvaluator(
        model=model,
        tokenizer=tokenizer,
        formatter=formatter,
        dataset=dataset,
        config=eval_cfg,
    )
    records = evaluator.evaluate()
    save_records_jsonl(records, args.output)
    log.info("Saved %d posterior records to %s", len(records), args.output)


if __name__ == "__main__":
    main()
