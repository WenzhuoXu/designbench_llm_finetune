"""
Local Logger: Rich console + rotating file + JSONL metrics + torch.profiler.

Provides beautiful terminal output and persistent local logs for every training run.

Features:
  - Rich console: color-coded live tables, progress bars, step metrics
  - Rotating file handler: logs/{run_name}/train.log (up to 100MB, 3 backups)
  - JSONL metrics: logs/{run_name}/metrics.jsonl (one JSON per step, queryable)
  - torch.profiler: optional profiling for first N steps (detect bottlenecks)
  - GPU stats: periodic pynvml poll for utilization, memory, temp, power

Usage:
    logger = LocalLogger(run_name="qwen3_sft_warmstart", log_dir="logs/")
    logger.start()
    logger.log_step(step=10, metrics={"loss": 0.5, "lr": 2e-5})
    logger.log_eval(step=100, metrics={"feasibility_rate": 0.3})
    logger.stop()
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


class LocalLogger:
    """Multi-sink logger for training runs."""

    def __init__(
        self,
        run_name: str,
        log_dir: str | Path = "logs",
        log_every_n_steps: int = 10,
        profile_steps: int = 0,       # Set > 0 to enable torch.profiler for first N steps
        gpu_stats_every_n_steps: int = 50,
    ):
        self.run_name = run_name
        self.log_dir = Path(log_dir) / run_name
        self.log_every_n_steps = log_every_n_steps
        self.profile_steps = profile_steps
        self.gpu_stats_every_n_steps = gpu_stats_every_n_steps

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = None
        self._profiler = None
        self._console = None
        self._file_handler = None
        self._metrics_file = None
        self._step_table = None

    def start(self) -> None:
        """Initialize all logging sinks."""
        self._start_time = time.time()
        self._setup_rich()
        self._setup_file_logger()
        self._setup_metrics_jsonl()
        if self.profile_steps > 0:
            self._setup_profiler()
        log.info(f"LocalLogger started: {self.log_dir}")

    def stop(self) -> None:
        """Flush and close all logging sinks."""
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)
        if self._metrics_file is not None:
            self._metrics_file.close()
        if self._file_handler is not None:
            self._file_handler.close()
        log.info(f"LocalLogger stopped. Logs saved to {self.log_dir}")

    def log_step(self, step: int, metrics: dict[str, Any]) -> None:
        """Log training step metrics."""
        if step % self.log_every_n_steps != 0:
            return

        elapsed = time.time() - (self._start_time or time.time())
        metrics["step"] = step
        metrics["elapsed_s"] = round(elapsed, 1)
        metrics["timestamp"] = datetime.now().isoformat()

        # GPU stats
        if step % self.gpu_stats_every_n_steps == 0:
            gpu_stats = self._get_gpu_stats()
            metrics.update(gpu_stats)

        # Rich console
        if self._console is not None:
            self._print_step_rich(step, metrics)

        # File logger
        log.info(f"Step {step}: " + " | ".join(f"{k}={v}" for k, v in metrics.items()
                                                 if not k.startswith("timestamp")))

        # JSONL
        self._write_jsonl({"type": "train", **metrics})

        # torch.profiler step
        if self._profiler is not None and step < self.profile_steps:
            self._profiler.step()

    def log_eval(self, step: int, metrics: dict[str, Any]) -> None:
        """Log evaluation metrics."""
        metrics["step"] = step
        metrics["timestamp"] = datetime.now().isoformat()

        if self._console is not None:
            self._print_eval_rich(step, metrics)

        log.info(
            f"EVAL Step {step}: "
            + " | ".join(f"{k}={v}" for k, v in metrics.items()
                         if not k.startswith("timestamp"))
        )
        self._write_jsonl({"type": "eval", **metrics})

    def log_rollout(self, step: int, rollouts: list, rewards: list[float]) -> None:
        """Log RL rollout summary."""
        if not rollouts:
            return
        n_feasible = sum(1 for r in rollouts if getattr(r, "reaches_solution", False))
        summary = {
            "step": step,
            "n_rollouts": len(rollouts),
            "feasibility_rate": n_feasible / len(rollouts),
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
        }
        log.info(f"Rollout Step {step}: " + str(summary))
        self._write_jsonl({"type": "rollout", **summary})

    def _setup_rich(self) -> None:
        """Set up Rich console for colored output."""
        try:
            from rich.console import Console
            from rich.logging import RichHandler

            self._console = Console()
            rich_handler = RichHandler(
                console=self._console,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
            )
            rich_handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(rich_handler)
        except ImportError:
            log.warning("rich not installed — using plain logging")

    def _setup_file_logger(self) -> None:
        """Set up rotating file handler."""
        log_path = self.log_dir / "train.log"
        self._file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=100 * 1024 * 1024,  # 100 MB
            backupCount=3,
            encoding="utf-8",
        )
        self._file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(self._file_handler)

    def _setup_metrics_jsonl(self) -> None:
        """Open JSONL file for metric logging."""
        jsonl_path = self.log_dir / "metrics.jsonl"
        self._metrics_file = open(jsonl_path, "a", encoding="utf-8")

    def _setup_profiler(self) -> None:
        """Set up torch.profiler for performance analysis."""
        try:
            import torch
            profile_dir = str(self.log_dir / "profiler")
            os.makedirs(profile_dir, exist_ok=True)
            self._profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            self._profiler.__enter__()
            log.info(f"torch.profiler active for first {self.profile_steps} steps → {profile_dir}")
        except Exception as e:
            log.warning(f"Failed to set up torch.profiler: {e}")

    def _get_gpu_stats(self) -> dict[str, float]:
        """Poll pynvml for per-GPU statistics."""
        try:
            import pynvml
            pynvml.nvmlInit()
            n_gpus = pynvml.nvmlDeviceGetCount()
            stats = {}
            for i in range(n_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # W
                except Exception:
                    power = 0.0
                stats[f"gpu{i}_util_pct"] = util.gpu
                stats[f"gpu{i}_mem_gb"] = mem.used / 1e9
                stats[f"gpu{i}_mem_pct"] = mem.used / mem.total * 100
                stats[f"gpu{i}_temp_c"] = temp
                stats[f"gpu{i}_power_w"] = power
            return stats
        except Exception:
            return {}

    def _print_step_rich(self, step: int, metrics: dict) -> None:
        """Print a colored step summary to Rich console."""
        if self._console is None:
            return
        key_metrics = {k: v for k, v in metrics.items()
                       if k in ("loss", "reward", "lr", "grad_norm",
                                "feasibility_rate", "tok_per_s")}
        parts = [f"[bold cyan]Step {step}[/]"]
        for k, v in key_metrics.items():
            if isinstance(v, float):
                parts.append(f"[green]{k}[/]=[yellow]{v:.4f}[/]")
            else:
                parts.append(f"[green]{k}[/]=[yellow]{v}[/]")
        self._console.print(" | ".join(parts))

    def _print_eval_rich(self, step: int, metrics: dict) -> None:
        if self._console is None:
            return
        self._console.print(f"[bold magenta]EVAL Step {step}[/]: {metrics}")

    def _write_jsonl(self, record: dict) -> None:
        if self._metrics_file is not None:
            self._metrics_file.write(json.dumps(record) + "\n")
            self._metrics_file.flush()
