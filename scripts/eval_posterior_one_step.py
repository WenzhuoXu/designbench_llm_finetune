"""
One-step online posterior evaluator.

Runs live LLM candidate generation on real problem states, evaluates tree-expanded
lookahead online with FEA transitions, and writes per-state metrics as JSONL.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/DesignBench")

log = logging.getLogger(__name__)


def load_model_config(config_path: str | Path):
    """Load a model config YAML and merge any simple local defaults."""
    config_path = Path(config_path)
    cfg = OmegaConf.load(config_path)

    merged = OmegaConf.create()
    for default in cfg.get("defaults", []):
        if isinstance(default, str):
            default_name = default
        elif isinstance(default, dict) and len(default) == 1:
            _, default_name = next(iter(default.items()))
        else:
            continue

        if default_name in {"_self_", None}:
            continue

        default_path = config_path.parent / f"{default_name}.yaml"
        if default_path.exists():
            merged = OmegaConf.merge(merged, OmegaConf.load(default_path))

    cfg_no_defaults = OmegaConf.create(
        {k: v for k, v in OmegaConf.to_container(cfg, resolve=False).items() if k != "defaults"}
    )
    return OmegaConf.merge(merged, cfg_no_defaults)


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
    parser.add_argument("--heartbeat-seconds", type=int, default=15)
    return parser.parse_args()


def _describe_model_placement(model) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        placements = sorted(set(str(device) for device in device_map.values()))
        return f"hf_device_map={device_map}, unique_devices={placements}"
    try:
        first_device = next(model.parameters()).device
    except StopIteration:
        first_device = torch.device("cpu")
    return f"parameter_device={first_device}"


def _assert_model_on_cuda(model) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        unique_devices = {str(device) for device in device_map.values()}
        if any(device.startswith("cuda") for device in unique_devices):
            return
        raise RuntimeError(f"Model did not load onto CUDA. hf_device_map={device_map}")

    try:
        first_device = next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters to inspect for CUDA placement") from exc

    if first_device.type != "cuda":
        raise RuntimeError(f"Model is not on CUDA. First parameter device: {first_device}")


def _heartbeat_loop(evaluator, interval_seconds: int, stop_event: threading.Event) -> None:
    while not stop_event.wait(interval_seconds):
        snapshot = evaluator.snapshot_progress()
        log.info(
            "Heartbeat: context=%d/%d problem_id=%s stage=%s stage_elapsed=%.1fs records_written=%d",
            snapshot.get("context_index", 0),
            snapshot.get("context_total", 0),
            snapshot.get("problem_id", "") or "<unknown>",
            snapshot.get("stage", "unknown"),
            snapshot.get("stage_elapsed_s", 0.0),
            snapshot.get("records_written", 0),
        )


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

    model_cfg = load_model_config(args.model_config)
    formatter_model_id = model_cfg.model_name_or_path
    if args.checkpoint:
        model_cfg.model_name_or_path = args.checkpoint
    if torch.cuda.is_available() and model_cfg.get("device_map") is None:
        model_cfg.device_map = "auto"
    log.info(
        "Eval config: model_config=%s checkpoint=%s output=%s n_states=%d sampling_mode=%s "
        "rollout_steps=%d candidate_count=%d depth=%d reference_depth=%d",
        args.model_config,
        args.checkpoint or "<config default>",
        args.output,
        args.n_states,
        args.sampling_mode,
        args.rollout_steps,
        args.candidate_count,
        args.depth,
        args.reference_depth,
    )
    log.info(
        "Runtime environment: cuda_available=%s cuda_visible_devices=%s configured_device_map=%s",
        torch.cuda.is_available(),
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        model_cfg.get("device_map"),
    )

    model, tokenizer = load_model_and_tokenizer(model_cfg)
    log.info("Model placement: %s", _describe_model_placement(model))
    _assert_model_on_cuda(model)
    if torch.cuda.is_available():
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        allocated_gb = torch.cuda.memory_allocated(gpu_index) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(gpu_index) / (1024 ** 3)
        log.info(
            "CUDA ready: device=%d name=%s allocated=%.2fGB reserved=%.2fGB",
            gpu_index,
            gpu_name,
            allocated_gb,
            reserved_gb,
        )
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
    evaluator.update_progress(stage="initializing evaluator", records_written=0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(evaluator, args.heartbeat_seconds, stop_event),
        daemon=True,
    )
    heartbeat_thread.start()

    try:
        evaluator.update_progress(stage="sampling contexts", records_written=0)
        contexts = evaluator.sample_contexts()
        log.info("Posterior evaluator sampled %d contexts", len(contexts))
        started_at = time.perf_counter()
        records = []
        total = len(contexts)
        evaluator.update_progress(stage="ready to evaluate contexts", context_total=total)
        for index, context in enumerate(contexts, start=1):
            context_start = time.perf_counter()
            evaluator.update_progress(
                stage="starting context evaluation",
                context_index=index,
                context_total=total,
                problem_id=context.problem_id,
                records_written=len(records),
            )
            log.info(
                "Starting context %d/%d: problem_id=%s step=%s history_len=%d feasible=%s",
                index,
                total,
                context.problem_id,
                context.current_state.get("step", 0),
                len(context.action_history),
                context.current_state.get("is_feasible", False),
            )
            record = evaluator.evaluate_context(context)
            context_elapsed = time.perf_counter() - context_start
            if record is None:
                log.warning(
                    "Skipping context %d/%d after %.1fs: no posterior record for problem_id=%s",
                    index,
                    total,
                    context_elapsed,
                    context.problem_id,
                )
                continue

            records.append(record)
            save_records_jsonl(records, output_path)
            evaluator.update_progress(
                stage="context completed",
                context_index=index,
                context_total=total,
                problem_id=context.problem_id,
                records_written=len(records),
            )
            total_elapsed = time.perf_counter() - started_at
            remaining = total - index
            eta_seconds = (total_elapsed / index) * remaining if index else 0.0
            log.info(
                "Finished context %d/%d in %.1fs: tree_best=%s reference_best=%s "
                "tree_fea=%d reference_fea=%d tree_elapsed=%.1fs reference_elapsed=%.1fs "
                "written_records=%d eta=%.1f min",
                index,
                total,
                context_elapsed,
                record.tree_best_class,
                record.reference_best_class,
                record.tree_total_fea_calls,
                record.reference_total_fea_calls,
                record.tree_elapsed_s,
                record.reference_elapsed_s,
                len(records),
                eta_seconds / 60.0,
            )

        save_records_jsonl(records, args.output)
        evaluator.update_progress(
            stage="completed",
            context_index=total,
            context_total=total,
            records_written=len(records),
        )
        log.info("Saved %d posterior records to %s", len(records), args.output)
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=1)


if __name__ == "__main__":
    main()
