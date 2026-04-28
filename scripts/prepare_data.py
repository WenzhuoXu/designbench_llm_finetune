"""
Data preparation script: tokenize and cache DesignBench data as HF datasets.

Precomputes tokenized training data for all configured models, saving to disk
as HuggingFace Dataset objects. This avoids re-tokenization at every training
run and enables fast DataLoader initialization.

Usage:
    # Tokenize for Qwen3-14B, full benchmark
    python scripts/prepare_data.py --model qwen3_14b --data sft_traces

    # Dry run: verify format without saving
    python scripts/prepare_data.py --model qwen3_14b --dry-run --num-examples 10

    # All models
    python scripts/prepare_data.py --all-models --data sft_traces
"""

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

DESIGNBENCH_PATH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
SFT_TRAIN_JSONL = DESIGNBENCH_PATH / "data/sft/train.jsonl"
SFT_DEV_JSONL = DESIGNBENCH_PATH / "data/sft/dev.jsonl"

ALL_MODELS = [
    "qwen3_14b", "deepseek_r1_14b", "phi4_reasoning",
    "llama4_scout", "phi4", "gemma3_12b", "qwen25_14b",
    "qwen25_0p5b",  # smoke-test / pipeline verification
]


def main():
    parser = argparse.ArgumentParser(description="Prepare DesignBench training data")
    parser.add_argument("--model", default="qwen3_14b", choices=ALL_MODELS + ["all"])
    parser.add_argument("--data", default="sft_traces",
                        choices=["sft_traces", "mcts_trees", "rl_prompts"])
    parser.add_argument("--output-dir", default="data/processed",
                        help="Output directory for tokenized datasets")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify format without saving to disk")
    parser.add_argument("--num-examples", type=int, default=None,
                        help="Limit number of examples (for --dry-run)")
    parser.add_argument("--all-models", action="store_true",
                        help="Prepare data for all models")
    args = parser.parse_args()

    models = ALL_MODELS if args.all_models or args.model == "all" else [args.model]

    for model_name in models:
        log.info(f"\n{'='*60}")
        log.info(f"Preparing {args.data} data for model: {model_name}")
        log.info(f"{'='*60}")
        prepare_for_model(model_name, args)


def prepare_for_model(model_name: str, args):
    from omegaconf import OmegaConf

    # Load model config
    config_path = Path(f"configs/model/{model_name}.yaml")
    if not config_path.exists():
        log.error(f"Model config not found: {config_path}")
        return

    try:
        model_cfg = OmegaConf.load(config_path)
    except Exception as e:
        log.error(f"Failed to load config {config_path}: {e}")
        return

    # Load tokenizer
    try:
        from llm_finetune.data.processors.chat_formatter import ChatFormatter
        from transformers import AutoTokenizer

        model_id = model_cfg.model_name_or_path
        cache_dir = model_cfg.get("cache_dir", "/ocean/projects/mch250030p/wxu7/hf_models")

        log.info(f"Loading tokenizer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=model_cfg.get("trust_remote_code", True),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        formatter = ChatFormatter.from_model_id(model_id, tokenizer)
        log.info(f"Chat formatter: thinking_mode={formatter.thinking_mode}")
    except Exception as e:
        log.error(f"Failed to load tokenizer for {model_name}: {e}")
        return

    if args.data == "sft_traces":
        _prepare_sft_traces(tokenizer, formatter, model_name, args)
    elif args.data == "mcts_trees":
        _prepare_mcts_trees(tokenizer, formatter, model_name, args)
    elif args.data == "rl_prompts":
        _prepare_rl_prompts(tokenizer, formatter, model_name, args)


def _prepare_sft_traces(tokenizer, formatter, model_name: str, args):
    from llm_finetune.data.processors.sft_jsonl_processor import SFTJsonlProcessor
    from llm_finetune.training.sft.targets import build_target_from_config

    if not SFT_TRAIN_JSONL.exists():
        log.error(f"SFT JSONL not found: {SFT_TRAIN_JSONL}")
        return

    target = build_target_from_config("warmstart_reasoning")
    processor = SFTJsonlProcessor(
        tokenizer=tokenizer,
        formatter=formatter,
        target=target,
        max_seq_len=8192,
    )

    log.info(f"Processing SFT JSONL: {SFT_TRAIN_JSONL}")
    examples = processor.process_jsonl(SFT_TRAIN_JSONL)

    if args.num_examples:
        examples = examples[: args.num_examples]

    # Print statistics
    lengths = [len(e["input_ids"]) for e in examples]
    log.info(f"Examples: {len(examples)}")
    log.info(f"Sequence lengths: min={min(lengths)}, mean={sum(lengths)//len(lengths)}, max={max(lengths)}")
    log.info(f"Mean quality: {sum(e['trace_quality'] for e in examples)/len(examples):.3f}")

    if examples:
        ex = examples[0]
        log.info(f"\nFirst example ({ex['problem_id']}):")
        log.info(f"  input_ids length: {len(ex['input_ids'])}")
        log.info(f"  Decoded first 200 chars: {tokenizer.decode(ex['input_ids'][:50])!r}")

    if args.dry_run:
        log.info("Dry run complete — data not saved.")
        return

    # Save as HF dataset
    try:
        from datasets import Dataset
        output_dir = Path(args.output_dir) / model_name / "sft_traces"
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = Dataset.from_list(examples)
        dataset.save_to_disk(str(output_dir))
        log.info(f"Saved {len(examples)} examples to {output_dir}")
    except ImportError:
        log.warning("datasets not installed — saving as JSON instead")
        output_path = Path(args.output_dir) / model_name / "sft_traces.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(examples, f)
        log.info(f"Saved {len(examples)} examples to {output_path}")


def _prepare_mcts_trees(tokenizer, formatter, model_name: str, args):
    from llm_finetune.data.datasets.mcts_dataset import MCTSDataset

    trees_dir = DESIGNBENCH_PATH / "data/modification_trees"
    if not trees_dir.exists():
        log.warning(f"Trees directory not found: {trees_dir}")
        return

    dataset = MCTSDataset.from_tree_dir(
        tree_dir=trees_dir,
        tokenizer=tokenizer,
        formatter=formatter,
        sampling_strategy="flat",
    )
    stats = dataset.get_stats()
    log.info(f"MCTS dataset stats: {stats}")

    if args.dry_run:
        log.info("Dry run complete — MCTS data not saved.")
        return

    log.info(f"MCTS dataset ready: {len(dataset)} samples")


def _prepare_rl_prompts(tokenizer, formatter, model_name: str, args):
    from llm_finetune.data.datasets.rl_dataset import RLPromptDataset

    dataset = RLPromptDataset.from_problems_dir(
        problems_dir=DESIGNBENCH_PATH / "data/problems",
        tokenizer=tokenizer,
        formatter=formatter,
    )
    log.info(f"RL prompt dataset: {len(dataset)} prompts")

    if args.dry_run:
        sample = dataset[0]
        log.info(f"  Sample prompt length: {len(sample['input_ids'])} tokens")
        log.info("Dry run complete.")


if __name__ == "__main__":
    main()
