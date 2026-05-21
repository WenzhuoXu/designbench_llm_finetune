"""
Empirical verification of §3.2-3.6 of plan_posterior_reward_walkthrough.md.

Uses the live OneStepOnlinePosteriorEvaluator (real model + FEA) to collect
per-state records, then computes and reports the theory-predicted quantities:

  §3.2  Greedy myopia gap Δ̄_G — expected ≈ 0.20
  §3.3  Myopia gap std σ_G    — expected ≈ 0.05
  §3.4  Branching miss rate   — expected ≈ 1 - b/5
  §3.5  FEA cost per state    — analytic + empirical averages
  §3.6  LLM-vs-tree ρ         — fraction where top-model action == tree-best action

Usage (see also slurm/eval_posterior_one_step_h100.sbatch for GPU setup):

    python scripts/verify_tree_expansion.py \
        --model-config configs/model/qwen3_14b.yaml \
        --checkpoint checkpoints/sft/warmstart_qwen3_14b_39083858/final \
        --n-states 20 \
        --depth 1 \
        --reference-depth 3 \
        --output logs/posterior/verify_tree_expansion.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/DesignBench")

log = logging.getLogger(__name__)


# ── config loader (same as eval_posterior_one_step.py) ────────────────────────

def load_model_config(config_path: str | Path):
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


# ── argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empirical verification of §3.2-3.6 (requires GPU + DesignBench)"
    )
    parser.add_argument("--model-config", default="configs/model/qwen25_0p5b.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--problems-dir",
        default="/ocean/projects/mch250030p/wxu7/DesignBench/data/problems",
    )
    parser.add_argument("--n-states", type=int, default=20,
                        help="Number of states to evaluate (§3.2-3.6 need ≥20 for reliable estimates)")
    parser.add_argument("--sampling-mode", default="rollin_policy",
                        choices=["initial", "rollin_policy", "rollin_mixed"])
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--candidate-count", type=int, default=4,
                        help="K candidates for §3.6 ρ measurement")
    parser.add_argument("--depth", type=int, default=1,
                        help="Tree depth D for main evaluation (§3.5 cost: D=1 b=5 → 24 FEA/state)")
    parser.add_argument("--reference-depth", type=int, default=3,
                        help="Reference depth for V^* estimate (§3.2: higher=less biased V^*)")
    parser.add_argument("--level1-branching", type=int, default=5,
                        help="b at level 1 — §3.4 recommendation: 5 for full action-class coverage")
    parser.add_argument("--deeper-branching", type=int, default=3)
    parser.add_argument("--output", default="logs/posterior/verify_tree_expansion.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── §3.2-3.6 metric computation ───────────────────────────────────────────────

def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def compute_verification_metrics(records: list[dict]) -> dict:
    """Aggregate §3.2-3.6 metrics from a list of PosteriorEvalRecord dicts."""
    if not records:
        return {}

    # §3.2 — Greedy myopia gap Δ_G(s) = V^*(s) - V^tree(s)
    # Proxy: V^*(s) ≈ max(reference_action_values), V^G(s) ≈ max(tree_action_values)
    delta_g_values: list[float] = []
    for rec in records:
        ref_vals = list(rec.get("reference_action_values", {}).values())
        tree_vals = list(rec.get("tree_action_values", {}).values())
        if ref_vals and tree_vals:
            delta_g = max(ref_vals) - max(tree_vals)
            delta_g_values.append(max(0.0, delta_g))

    # §3.3 — σ_G = std of per-candidate Δ_G(T(s, a_k)) across the K candidates
    # For each state: compare each candidate's reference value vs tree value
    sigma_g_values: list[float] = []
    for rec in records:
        ref_vals = rec.get("reference_action_values", {})
        tree_vals = rec.get("tree_action_values", {})
        common_actions = set(ref_vals) & set(tree_vals)
        if len(common_actions) >= 2:
            per_action_delta = [
                max(0.0, ref_vals[a] - tree_vals[a]) for a in common_actions
            ]
            sigma_g_values.append(_std(per_action_delta))

    # §3.4 — Optimal-class miss rate
    # optimal_class_missed = 1 when reference_best_class not in tree-evaluated classes
    n_missed = sum(1 for r in records if r.get("optimal_class_missed", 0))
    miss_rate = n_missed / len(records) if records else 0.0

    # §3.5 — FEA calls per state (empirical)
    tree_fea_per_state = [r.get("tree_total_fea_calls", 0) for r in records]
    ref_fea_per_state = [r.get("reference_total_fea_calls", 0) for r in records]

    # §3.6 — LLM-vs-tree agreement rate ρ
    rho_values = [r.get("rho_tree_agreement", 0) for r in records]
    rho = sum(rho_values) / len(rho_values) if rho_values else 0.0

    # §3.6 — ranking match vs deeper reference
    ranking_matches = [r.get("ranking_match_vs_reference", 0) for r in records]
    ranking_accuracy = sum(ranking_matches) / len(ranking_matches) if ranking_matches else 0.0

    # Difficulty distribution
    difficulties = [r.get("difficulty", 0.0) for r in records]

    return {
        # §3.2
        "n_states": len(records),
        "delta_g_mean": sum(delta_g_values) / len(delta_g_values) if delta_g_values else 0.0,
        "delta_g_std": _std(delta_g_values),
        "delta_g_max": max(delta_g_values) if delta_g_values else 0.0,
        "delta_g_fraction_zero": sum(1 for v in delta_g_values if v < 1e-6) / len(delta_g_values) if delta_g_values else 0.0,
        # §3.3
        "sigma_g_mean": sum(sigma_g_values) / len(sigma_g_values) if sigma_g_values else 0.0,
        "sigma_g_std": _std(sigma_g_values),
        # §3.4
        "optimal_class_miss_rate": miss_rate,
        "theory_miss_rate_b5": 0.0,  # 1 - 5/5 = 0 with stratified sampling
        # §3.5
        "tree_fea_calls_mean": sum(tree_fea_per_state) / len(tree_fea_per_state) if tree_fea_per_state else 0.0,
        "ref_fea_calls_mean": sum(ref_fea_per_state) / len(ref_fea_per_state) if ref_fea_per_state else 0.0,
        "tree_elapsed_mean_s": sum(r.get("tree_elapsed_s", 0) for r in records) / len(records),
        "ref_elapsed_mean_s": sum(r.get("reference_elapsed_s", 0) for r in records) / len(records),
        # §3.6
        "rho_llm_tree_agreement": rho,
        "ranking_accuracy_vs_reference": ranking_accuracy,
        # Context
        "difficulty_mean": sum(difficulties) / len(difficulties) if difficulties else 0.0,
        "class_entropy_mean": sum(r.get("class_entropy", 0) for r in records) / len(records),
        "class_coverage_mean": sum(r.get("class_coverage", 0) for r in records) / len(records),
    }


def _analytic_fea_cost_table(depth: int, b1: int, bd: int) -> dict:
    """Analytic FEA call count for tree config (D, b1, bd)."""
    if depth == 0:
        return {"depth": 0, "fea_per_state": 0, "formula": "D=0: no expansion"}
    total = b1
    for d in range(1, depth):
        total += b1 * (bd ** d)
    return {
        "depth": depth,
        "level1_branching": b1,
        "deeper_branching": bd,
        "fea_per_state": total,
        "formula": f"b1 + b1*bd^1 + ... = {total}",
    }


def print_verification_report(metrics: dict, args: argparse.Namespace) -> None:
    sep = "=" * 64
    print(sep)
    print("TREE-EXPANSION VERIFICATION REPORT  (§3.2-3.6)")
    print(sep)
    print(f"  States evaluated : {metrics['n_states']}")
    print(f"  Tree depth D     : {args.depth}  (reference depth: {args.reference_depth})")
    print(f"  Branching b      : {args.level1_branching} (level 1), {args.deeper_branching} (deeper)")
    print(f"  Candidate count K: {args.candidate_count}")
    print()

    print("§3.2  Greedy Myopia Gap  Δ̄_G  (theory: ≈ 0.20)")
    print(f"  mean(Δ_G)        : {metrics['delta_g_mean']:.4f}   [theory: 0.20]")
    print(f"  std(Δ_G)         : {metrics['delta_g_std']:.4f}")
    print(f"  max(Δ_G)         : {metrics['delta_g_max']:.4f}")
    print(f"  fraction Δ_G=0   : {metrics['delta_g_fraction_zero']:.2%}  (greedy = optimal)")
    print()

    print("§3.3  Myopia Gap Std  σ_G  (theory: ≈ 0.05)")
    print(f"  mean(σ_G)        : {metrics['sigma_g_mean']:.4f}   [theory: 0.05]")
    print(f"  std(σ_G)         : {metrics['sigma_g_std']:.4f}")
    print()

    print("§3.4  Branching Coverage  miss_rate(b=5)  (theory: 0.00 with stratified)")
    print(f"  empirical miss   : {metrics['optimal_class_miss_rate']:.2%}  [theory: {metrics['theory_miss_rate_b5']:.2%}]")
    print(f"  class coverage   : {metrics['class_coverage_mean']:.1f} / 5 classes (mean)")
    print(f"  class entropy    : {metrics['class_entropy_mean']:.3f} nats")
    print()

    cost_main = _analytic_fea_cost_table(args.depth, args.level1_branching, args.deeper_branching)
    cost_ref = _analytic_fea_cost_table(args.reference_depth, args.level1_branching, args.deeper_branching)
    print("§3.5  FEA Cost")
    print(f"  analytic tree  D={args.depth}, b={args.level1_branching}: {cost_main['fea_per_state']} calls/state")
    print(f"  analytic ref   D={args.reference_depth}, b={args.level1_branching}: {cost_ref['fea_per_state']} calls/state")
    print(f"  empirical tree : {metrics['tree_fea_calls_mean']:.1f} calls/state  ({metrics['tree_elapsed_mean_s']:.1f}s avg)")
    print(f"  empirical ref  : {metrics['ref_fea_calls_mean']:.1f} calls/state  ({metrics['ref_elapsed_mean_s']:.1f}s avg)")

    analytic_tables = []
    for d, b1, bd in [(0, 1, 1), (1, 5, 3), (2, 5, 3), (3, 5, 3)]:
        c = _analytic_fea_cost_table(d, b1, bd)
        analytic_tables.append(f"    D={d}: {c['fea_per_state']:4d} calls/state")
    print("  Reference table (b=5/3):")
    for row in analytic_tables:
        print(row)
    print()

    print("§3.6  LLM-vs-Tree Agreement  ρ  (target: ≥ 0.85 at convergence)")
    print(f"  ρ (model agrees w/ tree-best) : {metrics['rho_llm_tree_agreement']:.3f}")
    print(f"  tree ranks match reference     : {metrics['ranking_accuracy_vs_reference']:.3f}")
    print()

    print("Context")
    print(f"  mean difficulty  : {metrics['difficulty_mean']:.3f}")
    print(sep)


# ── main ──────────────────────────────────────────────────────────────────────

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
        raise RuntimeError("Model has no parameters") from exc
    if first_device.type != "cuda":
        raise RuntimeError(f"Model is not on CUDA. First parameter device: {first_device}")


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
        append_record_jsonl,
        save_records_jsonl,
    )
    from dataclasses import asdict

    model_cfg = load_model_config(args.model_config)
    formatter_model_id = model_cfg.model_name_or_path
    if args.checkpoint:
        model_cfg.model_name_or_path = args.checkpoint
    if torch.cuda.is_available() and model_cfg.get("device_map") is None:
        model_cfg.device_map = "auto"

    log.info(
        "Verify tree expansion: model_config=%s checkpoint=%s n_states=%d "
        "depth=%d reference_depth=%d level1_branching=%d",
        args.model_config,
        args.checkpoint or "<config default>",
        args.n_states,
        args.depth,
        args.reference_depth,
        args.level1_branching,
    )

    model, tokenizer = load_model_and_tokenizer(model_cfg)
    _assert_model_on_cuda(model)
    if torch.cuda.is_available():
        gpu_index = torch.cuda.current_device()
        allocated_gb = torch.cuda.memory_allocated(gpu_index) / (1024 ** 3)
        log.info("CUDA: device=%d name=%s allocated=%.2fGB",
                 gpu_index, torch.cuda.get_device_name(gpu_index), allocated_gb)

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
        # Use wider reference tree for better V^* estimate (§3.2)
        reference_level1_branching=args.level1_branching,
        reference_deeper_branching=args.deeper_branching,
    )

    evaluator = OneStepOnlinePosteriorEvaluator(
        model=model,
        tokenizer=tokenizer,
        formatter=formatter,
        dataset=dataset,
        config=eval_cfg,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    contexts = evaluator.sample_contexts()
    log.info("Sampled %d contexts for verification", len(contexts))
    started_at = time.perf_counter()
    records: list[dict] = []
    total = len(contexts)

    for index, context in enumerate(contexts, start=1):
        context_start = time.perf_counter()
        log.info(
            "Context %d/%d: problem_id=%s step=%d feasible=%s",
            index, total,
            context.problem_id,
            context.current_state.get("step", 0),
            context.current_state.get("is_feasible", False),
        )
        record = evaluator.evaluate_context(context)
        context_elapsed = time.perf_counter() - context_start
        if record is None:
            log.warning("Context %d/%d produced no record (%.1fs)", index, total, context_elapsed)
            continue

        record_dict = asdict(record)
        records.append(record_dict)
        append_record_jsonl(record, output_path)

        elapsed = time.perf_counter() - started_at
        eta = (elapsed / index) * (total - index) if index else 0.0
        log.info(
            "Context %d/%d done in %.1fs: tree=%s ref=%s ρ=%d "
            "tree_fea=%d ref_fea=%d eta=%.1fmin",
            index, total, context_elapsed,
            record.tree_best_class, record.reference_best_class,
            record.rho_tree_agreement,
            record.tree_total_fea_calls, record.reference_total_fea_calls,
            eta / 60.0,
        )

    log.info("Evaluation complete. %d records written to %s", len(records), output_path)

    if records:
        metrics = compute_verification_metrics(records)
        print_verification_report(metrics, args)
        summary_path = output_path.with_suffix(".summary.json")
        with summary_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        log.info("Summary written to %s", summary_path)
    else:
        log.warning("No records collected — cannot compute verification metrics")


if __name__ == "__main__":
    main()
