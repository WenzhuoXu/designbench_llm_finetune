"""
Monitor a running GRPO SLURM job and kill it if training looks unhealthy.

Kill conditions (any one triggers):
  1. NaN/Inf reward at any step
  2. grad_norm > 50 for 3 consecutive W&B polls
  3. kl_divergence > 20 (policy collapsed away from reference)
  4. Reward trend: mean(last_20) < mean(first_10) - 0.5 and step > 50
  5. SLURM job no longer in queue (completed/failed)

Usage:
    python scripts/monitor_grpo_job.py <JOB_ID> \\
        --run-name qwen3_14b_grpo_posterior_<JOB_ID> \\
        --wandb-project designbench-training \\
        --poll-interval 60

    # Typically launched by submit_and_monitor.sh
    nohup python scripts/monitor_grpo_job.py 12345678 \\
        --run-name qwen3_14b_grpo_posterior_12345678 \\
        > logs/monitor_12345678.log 2>&1 &
"""

from __future__ import annotations

import argparse
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [monitor] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Kill thresholds ────────────────────────────────────────────────────────────
GRAD_NORM_LIMIT = 50.0          # kill if grad_norm exceeds this
GRAD_NORM_CONSECUTIVE = 3       # ... for this many consecutive polls
KL_LIMIT = 20.0                 # kill if kl > this (policy diverged)
REWARD_WARMUP_STEPS = 30        # ignore reward trend before this step
REWARD_DECLINE_MARGIN = 0.5     # kill if last_20_mean < first_10_mean - margin
REWARD_HISTORY_MIN = 20         # need this many data points before trend check


def _squeue_state(job_id: str) -> Optional[str]:
    """Return SLURM job state string, or None if not in queue."""
    try:
        out = subprocess.check_output(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out if out else None
    except subprocess.CalledProcessError:
        return None


def _scancel(job_id: str, reason: str) -> None:
    log.warning("KILLING job %s — %s", job_id, reason)
    try:
        subprocess.run(["scancel", job_id], check=True)
        log.info("scancel %s succeeded", job_id)
    except subprocess.CalledProcessError as e:
        log.error("scancel failed: %s", e)


def _wandb_alert(run, title: str, text: str) -> None:
    try:
        import wandb
        wandb.alert(title=title, text=text, level=wandb.AlertLevel.WARN)
        log.info("W&B alert sent: %s", title)
    except Exception as e:
        log.debug("W&B alert failed (non-fatal): %s", e)


def _fetch_wandb_history(run_name: str, project: str, entity: Optional[str]) -> Optional[list[dict]]:
    """Fetch recent training history rows from W&B API."""
    try:
        import wandb
        api = wandb.Api(timeout=30)
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(path, filters={"display_name": run_name}, order="-created_at")
        if not runs:
            log.debug("W&B run '%s' not found yet in project '%s'", run_name, project)
            return None
        run = runs[0]
        # Fetch last 100 steps of key metrics
        history = run.scan_history(
            keys=["train/reward", "train/grad_norm", "train/kl", "_step"],
            min_step=0,
        )
        rows = list(history)
        return rows, run
    except Exception as e:
        log.debug("W&B API fetch failed: %s", e)
        return None


def _check_conditions(rows: list[dict]) -> Optional[str]:
    """Return kill reason string if any kill condition is met, else None."""
    if not rows:
        return None

    rewards = [r.get("train/reward") for r in rows if r.get("train/reward") is not None]
    grad_norms = [r.get("train/grad_norm") for r in rows if r.get("train/grad_norm") is not None]
    kls = [r.get("train/kl") for r in rows if r.get("train/kl") is not None]
    steps = [r.get("_step", 0) for r in rows]
    max_step = max(steps) if steps else 0

    # 1. NaN/Inf reward
    for r in rewards:
        if math.isnan(r) or math.isinf(r):
            return f"NaN/Inf reward detected ({r}) at step ~{max_step}"

    # 2. Gradient explosion (consecutive)
    if len(grad_norms) >= GRAD_NORM_CONSECUTIVE:
        recent = grad_norms[-GRAD_NORM_CONSECUTIVE:]
        if all(g > GRAD_NORM_LIMIT for g in recent):
            return (
                f"grad_norm > {GRAD_NORM_LIMIT} for {GRAD_NORM_CONSECUTIVE} consecutive "
                f"polls (last values: {[round(g,1) for g in recent]})"
            )

    # 3. KL explosion
    if kls and max(kls[-5:]) > KL_LIMIT:
        return f"KL divergence {max(kls[-5:]):.1f} > {KL_LIMIT} at step ~{max_step}"

    # 4. Reward trend (only after warmup and sufficient history)
    if max_step >= REWARD_WARMUP_STEPS and len(rewards) >= REWARD_HISTORY_MIN:
        first_10_mean = sum(rewards[:10]) / 10
        last_20_mean = sum(rewards[-20:]) / 20
        if last_20_mean < first_10_mean - REWARD_DECLINE_MARGIN:
            return (
                f"Reward declining: first-10 mean={first_10_mean:.3f}, "
                f"last-20 mean={last_20_mean:.3f} (step {max_step})"
            )

    return None


def monitor(
    job_id: str,
    run_name: str,
    project: str,
    entity: Optional[str],
    poll_interval: int,
) -> None:
    log.info("Monitor started — job %s | run %s | project %s", job_id, run_name, project)
    log.info("Kill conditions: grad_norm>%g×%d, kl>%g, NaN reward, reward decline",
             GRAD_NORM_LIMIT, GRAD_NORM_CONSECUTIVE, KL_LIMIT)

    consecutive_high_grad = 0
    active_wandb_run = None

    while True:
        time.sleep(poll_interval)

        # ── Check SLURM state ──────────────────────────────────────────────
        state = _squeue_state(job_id)
        if state is None:
            log.info("Job %s no longer in SLURM queue — exiting monitor.", job_id)
            break
        log.info("SLURM job %s state: %s", job_id, state)
        if state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"):
            log.warning("Job %s ended with state %s — exiting monitor.", job_id, state)
            break
        if state != "RUNNING":
            log.info("Job %s is %s (not yet running) — waiting...", job_id, state)
            continue

        # ── Fetch W&B metrics ──────────────────────────────────────────────
        result = _fetch_wandb_history(run_name, project, entity)
        if result is None:
            log.info("W&B history not available yet — will retry next poll.")
            continue
        rows, active_wandb_run = result
        if not rows:
            log.info("W&B history empty — training may not have logged yet.")
            continue

        latest_step = max(r.get("_step", 0) for r in rows)
        latest_reward = next(
            (r["train/reward"] for r in reversed(rows) if "train/reward" in r), None
        )
        latest_grad = next(
            (r["train/grad_norm"] for r in reversed(rows) if "train/grad_norm" in r), None
        )
        latest_kl = next(
            (r["train/kl"] for r in reversed(rows) if "train/kl" in r), None
        )
        log.info(
            "step=%d  reward=%.4f  grad_norm=%s  kl=%s",
            latest_step,
            latest_reward if latest_reward is not None else float("nan"),
            f"{latest_grad:.2f}" if latest_grad is not None else "n/a",
            f"{latest_kl:.3f}" if latest_kl is not None else "n/a",
        )

        # ── Evaluate kill conditions ───────────────────────────────────────
        kill_reason = _check_conditions(rows)
        if kill_reason:
            log.error("Kill condition met: %s", kill_reason)
            if active_wandb_run is not None:
                _wandb_alert(
                    active_wandb_run,
                    title=f"Job {job_id} killed by monitor",
                    text=kill_reason,
                )
            _scancel(job_id, kill_reason)
            log.info("Monitor exiting after kill.")
            break

    log.info("Monitor finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor GRPO SLURM job and kill if unhealthy")
    parser.add_argument("job_id", help="SLURM job ID to monitor")
    parser.add_argument("--run-name", required=True, help="W&B run display name")
    parser.add_argument("--wandb-project", default="designbench-training")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between metric polls (default 60)")
    args = parser.parse_args()

    monitor(
        job_id=args.job_id,
        run_name=args.run_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
