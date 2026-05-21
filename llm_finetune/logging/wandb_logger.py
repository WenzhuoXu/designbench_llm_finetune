"""
W&B Logger: comprehensive Weights & Biases integration for DesignBench training.

Features:
  - Full config logging (Hydra OmegaConf → wandb.config)
  - Per-step training metrics: loss, grad_norm, lr, throughput (tok/s)
  - Per-GPU stats via pynvml: utilization, memory, temperature, power
  - Rollout tables: W&B Table with per-rollout breakdown
  - Eval metrics: feasibility_rate, FOS improvement, mass reduction
  - Model artifacts: checkpoint → W&B Artifact (versioned, linked to run)
  - W&B Alerts: notify on anomalies (memory spike, loss explosion)

Projects:
  - Training runs: designbench-training
  - Evaluation runs: design-bench (same as DesignBench validation project)

Reuses patterns from DesignBench/validation/wandb_logger.py.

Usage:
    logger = WandbLogger(
        project="designbench-training",
        run_name="qwen3_sft_warmstart_001",
        config=OmegaConf.to_container(cfg),
    )
    logger.init()
    logger.log_training_step(step=10, metrics={"loss": 0.5, "lr": 2e-5})
    logger.log_rollout_table(rollouts, rewards)
    logger.save_artifact("checkpoints/step_100", "sft-step-100", "model")
    logger.finish()
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return numeric if math.isfinite(numeric) else default


def _finite_or_none(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _sanitize_metric_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            clean[key] = _finite_float(value, 0.0)
        else:
            clean[key] = value
    return clean


class WandbLogger:
    """Weights & Biases integration for DesignBench LLM training."""

    def __init__(
        self,
        project: str = "designbench-training",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        log_every_n_steps: int = 10,
        gpu_stats_every_n_steps: int = 50,
    ):
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.log_every_n_steps = log_every_n_steps
        self.gpu_stats_every_n_steps = gpu_stats_every_n_steps
        self._run = None
        self._rollout_table = None
        self._rollout_table_step = 0
        self._last_gpu_poll = 0.0

    @property
    def is_active(self) -> bool:
        return self._run is not None

    def init(self) -> None:
        """Initialize W&B run."""
        try:
            import wandb
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.run_name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
                reinit=True,
            )
            log.info(f"W&B run initialized: {self._run.url}")
            self._init_rollout_table()
        except ImportError:
            log.warning("wandb not installed. W&B logging disabled.")
        except Exception as e:
            log.warning(f"W&B init failed: {e}. Continuing without W&B.")

    def finish(self) -> None:
        """Finish W&B run."""
        if self._run is not None:
            self._flush_rollout_table()
            self._run.finish()
            self._run = None
            log.info("W&B run finished")

    def log_training_step(self, step: int, metrics: dict[str, Any]) -> None:
        """Log per-step training metrics.

        Logs: loss, perplexity, grad_norm, lr, throughput (tok/s), seq_len.
        Also logs GPU stats every gpu_stats_every_n_steps steps.
        """
        if not self.is_active:
            return
        if step % self.log_every_n_steps != 0:
            return

        import wandb

        metrics = _sanitize_metric_dict(metrics)
        log_dict = {"train/" + k: v for k, v in metrics.items()}

        # Add perplexity if loss is present
        if "loss" in metrics:
            import math
            try:
                log_dict["train/perplexity"] = math.exp(metrics["loss"])
            except Exception:
                pass

        # GPU stats (periodic)
        if step % self.gpu_stats_every_n_steps == 0:
            gpu_stats = self._get_gpu_stats()
            log_dict.update({"system/" + k: v for k, v in gpu_stats.items()})

        wandb.log(log_dict, step=step)

    def log_rollout_table(
        self,
        rollouts: list,
        rewards: list[float],
        cost_fn=None,
        step: Optional[int] = None,
    ) -> None:
        """Log rollout results as a W&B Table.

        Columns: problem_id, n_steps, actions_summary, delta_FOS, delta_mass,
                 feasible, reward, cost (if cost_fn), grammar_success_rate.
        """
        if not self.is_active or not rollouts:
            return

        if self._rollout_table is None:
            self._init_rollout_table()

        self._rollout_table_step += 1
        for rollout, reward in zip(rollouts, rewards):
            if rollout is None:
                continue

            initial = getattr(rollout, "initial_state", {}) or {}
            final = getattr(rollout, "final_state", {}) or {}
            actions = getattr(rollout, "action_sequence", [])
            parse_ok = getattr(rollout, "parse_success", [])

            delta_fos_b = _finite_float(final.get("fos_buckling")) - _finite_float(
                initial.get("fos_buckling")
            )
            delta_fos_y = _finite_float(final.get("fos_yielding")) - _finite_float(
                initial.get("fos_yielding")
            )
            delta_mass = _finite_float(final.get("mass")) - _finite_float(
                initial.get("mass")
            )
            grammar_rate = (
                sum(parse_ok) / len(parse_ok) if parse_ok else 0.0
            )

            cost_val = 0.0
            if cost_fn is not None:
                try:
                    cost_val = _finite_float(cost_fn.compute(rollout, {}))
                except Exception:
                    pass

            tree_adv = _finite_or_none(
                (getattr(rollout, "tree_metrics", None) or {}).get("tree_advantage")
            )
            phi_H = _finite_or_none(
                (getattr(rollout, "tree_metrics", None) or {}).get("phi_H")
            )
            row = [
                getattr(rollout, "problem_id", ""),
                len(actions),
                "; ".join(actions[:3]) + ("..." if len(actions) > 3 else ""),
                round(delta_fos_b, 4),
                round(delta_fos_y, 4),
                round(delta_mass, 4),
                bool(getattr(rollout, "reaches_solution", False)),
                round(_finite_float(reward), 4),
                round(cost_val, 4),
                round(grammar_rate, 4),
                round(phi_H, 4) if phi_H is not None else None,
                round(tree_adv, 4) if tree_adv is not None else None,
            ]
            self._rollout_table.add_data(*row)

        # Log table periodically (W&B has row limits per log call)
        if self._rollout_table_step % 10 == 0:
            self._flush_rollout_table()

    def log_reward_breakdown(
        self,
        step: int,
        means: dict[str, float],
        stds: Optional[dict[str, float]] = None,
    ) -> None:
        """Log per-component reward mean and std to W&B.

        Keys logged: rewards/<component>_mean, rewards/<component>_std.
        Std is logged only when provided and non-zero — it is the primary
        signal for whether a component is actually differentiating rollouts.
        """
        if not self.is_active or not means:
            return
        import wandb
        log_dict: dict = {}
        for k, v in means.items():
            log_dict[f"rewards/{k}_mean"] = _finite_float(v)
        if stds:
            for k, v in stds.items():
                log_dict[f"rewards/{k}_std"] = _finite_float(v)
        wandb.log(log_dict, step=step)

    def log_cost_breakdown(
        self,
        step: int,
        means: dict[str, float],
        stds: Optional[dict[str, float]] = None,
    ) -> None:
        """Log per-component cost mean and std to W&B as costs/<component>_mean/std."""
        if not self.is_active or not means:
            return
        import wandb
        log_dict: dict = {}
        for k, v in means.items():
            log_dict[f"costs/{k}_mean"] = _finite_float(v)
        if stds:
            for k, v in stds.items():
                log_dict[f"costs/{k}_std"] = _finite_float(v)
        wandb.log(log_dict, step=step)

    def log_rho(self, step: int, rho: float) -> None:
        """Log rho(t) — the training-time proxy for LLM-vs-tree-best agreement.

        rho = fraction of rollout groups where argmax(Φ_H) == argmax(total_reward).
        Rises toward 1 as the composite reward aligns with the Φ criterion.
        See docs/reward_rho_connection.md §ρ(t).
        """
        if not self.is_active:
            return
        import wandb
        wandb.log({"eval/rho_tree_agreement": _finite_float(rho)}, step=step)

    def log_eval_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        """Log DesignBench evaluation metrics.

        Expected keys: feasibility_rate, mean_fos_improvement, mean_mass_reduction,
                       grammar_success_rate, mean_steps_to_solution.
        """
        if not self.is_active:
            return
        import wandb
        log_dict = {"eval/" + k: v for k, v in _sanitize_metric_dict(metrics).items()}
        wandb.log(log_dict, step=step)
        log.info(f"W&B eval metrics logged at step {step}: {metrics}")

    def log_gpu_stats(self, step: int) -> None:
        """Explicitly log GPU statistics (called from training loop if needed)."""
        if not self.is_active:
            return
        import wandb
        gpu_stats = self._get_gpu_stats()
        if gpu_stats:
            wandb.log({"system/" + k: v for k, v in gpu_stats.items()}, step=step)

    def save_artifact(
        self,
        path: str | Path,
        name: str,
        artifact_type: str = "model",
        metadata: Optional[dict] = None,
    ) -> None:
        """Save a checkpoint or dataset as a W&B Artifact.

        Args:
            path: Local path to checkpoint directory or file.
            name: Artifact name (e.g., "sft-qwen3-step-100").
            artifact_type: "model", "dataset", or "checkpoint".
            metadata: Optional dict attached to artifact.
        """
        if not self.is_active:
            return
        try:
            import wandb
            artifact = wandb.Artifact(
                name=name,
                type=artifact_type,
                metadata=metadata or {},
            )
            path = Path(path)
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            self._run.log_artifact(artifact)
            log.info(f"W&B Artifact saved: {name} ({artifact_type})")
        except Exception as e:
            log.warning(f"Failed to save W&B artifact {name}: {e}")

    def alert(self, title: str, text: str, level: str = "WARN") -> None:
        """Send a W&B alert (email notification)."""
        if not self.is_active:
            return
        try:
            import wandb
            wandb.alert(title=title, text=text, level=getattr(wandb.AlertLevel, level, "WARN"))
        except Exception as e:
            log.warning(f"W&B alert failed: {e}")

    def watch_model(self, model, log_freq: int = 100) -> None:
        """Watch model parameters/gradients in W&B."""
        if not self.is_active:
            return
        try:
            import wandb
            wandb.watch(model, log="gradients", log_freq=log_freq)
            log.info(f"W&B model watch active (log_freq={log_freq})")
        except Exception as e:
            log.warning(f"W&B watch failed: {e}")

    def _init_rollout_table(self) -> None:
        """Initialize W&B Table for rollout logging."""
        try:
            import wandb
            self._rollout_table = wandb.Table(columns=[
                "problem_id", "n_steps",
                "actions_summary",
                "delta_FOS_buckling", "delta_FOS_yielding",
                "delta_mass",
                "feasible",
                "reward", "cost",
                "grammar_success_rate",
                "phi_H",
                "tree_advantage",
            ])
        except Exception:
            pass

    def _flush_rollout_table(self) -> None:
        """Log current rollout table to W&B and reset."""
        if not self.is_active or self._rollout_table is None:
            return
        try:
            import wandb
            wandb.log({"rollouts/table": self._rollout_table})
            self._init_rollout_table()
        except Exception as e:
            log.debug(f"Failed to flush rollout table: {e}")

    def _get_gpu_stats(self) -> dict[str, float]:
        """Poll pynvml for per-GPU statistics."""
        now = time.time()
        # Rate limit: max once per 5 seconds
        if now - self._last_gpu_poll < 5.0:
            return {}
        self._last_gpu_poll = now
        try:
            import pynvml
            pynvml.nvmlInit()
            n_gpus = pynvml.nvmlDeviceGetCount()
            stats = {}
            total_util = 0
            total_mem_pct = 0
            for i in range(n_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
                except Exception:
                    power_w = 0.0
                stats[f"gpu{i}_util_pct"] = util.gpu
                stats[f"gpu{i}_mem_gb"] = round(mem.used / 1e9, 2)
                stats[f"gpu{i}_mem_pct"] = round(mem.used / mem.total * 100, 1)
                stats[f"gpu{i}_temp_c"] = temp
                stats[f"gpu{i}_power_w"] = round(power_w, 1)
                total_util += util.gpu
                total_mem_pct += mem.used / mem.total * 100

            # Aggregate across all GPUs
            stats["gpu_mean_util_pct"] = total_util / n_gpus
            stats["gpu_mean_mem_pct"] = total_mem_pct / n_gpus

            # Alert on high memory
            if stats["gpu_mean_mem_pct"] > 90:
                self.alert(
                    title="GPU Memory Warning",
                    text=f"Mean GPU memory usage: {stats['gpu_mean_mem_pct']:.1f}%",
                    level="WARN",
                )
            return stats
        except Exception:
            return {}


def build_wandb_logger(cfg, run_name: Optional[str] = None) -> WandbLogger:
    """Build WandbLogger from Hydra config.

    Args:
        cfg: Full Hydra config.
        run_name: Override run name.

    Returns:
        Initialized WandbLogger (not yet started — call .init()).
    """
    log_cfg = cfg.get("logging", {})
    model_name = cfg.get("model", {}).get("model_name_or_path", "unknown").split("/")[-1]

    if run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_{timestamp}"

    from omegaconf import OmegaConf
    config_dict = OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, "keys") else {}

    return WandbLogger(
        project=log_cfg.get("wandb_project", "designbench-training"),
        entity=log_cfg.get("wandb_entity", None),
        run_name=run_name,
        config=config_dict,
        tags=log_cfg.get("wandb_tags", []),
        notes=log_cfg.get("wandb_notes", None),
        log_every_n_steps=log_cfg.get("log_every_n_steps", 10),
        gpu_stats_every_n_steps=log_cfg.get("gpu_stats_every_n_steps", 50),
    )
