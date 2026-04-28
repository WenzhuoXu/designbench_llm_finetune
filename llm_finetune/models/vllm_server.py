"""
vLLM server management for fast GRPO rollout generation.

vLLM provides 5-10x faster generation vs HF generate, critical for GRPO where
you need K rollouts per prompt (group_size=8-16) at each training step.

TRL >= 0.12 supports VLLMClient natively — this module manages the server
lifecycle and provides weight synchronization (actor → vLLM) after each
gradient update (online GRPO).

Usage:
    server = VLLMServer(model_path, tensor_parallel_size=8, port=8000)
    server.start()
    completions = server.generate(prompts, max_new_tokens=512, temperature=0.8, n=8)
    server.sync_weights(trainer.model.state_dict())
    server.stop()
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)

DEFAULT_PORT = 8000
HEALTH_CHECK_RETRIES = 60
HEALTH_CHECK_INTERVAL = 5.0  # seconds


@dataclass
class SamplingParams:
    """Parameters for vLLM generation (mirrors vllm.SamplingParams)."""
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = -1
    max_tokens: int = 1024
    n: int = 1             # number of completions per prompt (= GRPO group_size)
    stop: list[str] = field(default_factory=list)
    repetition_penalty: float = 1.0
    seed: Optional[int] = None


class VLLMServer:
    """Manages a vLLM OpenAI-compatible server subprocess for GRPO rollouts.

    The vLLM server runs as a subprocess on the same node, using tensor
    parallelism across all 8 H100s. After each training step, weights are
    synced from the actor model to vLLM's engine (online RL).

    Architecture on 8-GPU node:
        - vLLM server: launched on all 8 GPUs via tensor_parallel_size=8
        - Training process: uses all 8 GPUs via DeepSpeed
        - GRPO trainer alternates: (1) sync weights to vLLM, (2) generate
          rollouts via HTTP, (3) compute rewards, (4) gradient update

    Note: For multi-node GRPO, one node runs vLLM and another runs training.
    See slurm/grpo_h100.sbatch for multi-node configuration.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 8,
        port: int = DEFAULT_PORT,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        dtype: str = "bfloat16",
        enable_prefix_caching: bool = True,
        trust_remote_code: bool = True,
        served_model_name: Optional[str] = None,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.enable_prefix_caching = enable_prefix_caching
        self.trust_remote_code = trust_remote_code
        self.served_model_name = served_model_name or "actor"
        self._process: Optional[subprocess.Popen] = None
        self._client = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"

    def start(self, wait_ready: bool = True) -> None:
        """Launch vLLM server subprocess."""
        if self._process is not None:
            log.warning("vLLM server already running")
            return

        cmd = self._build_command()
        log.info(f"Starting vLLM server: {' '.join(cmd)}")

        env = os.environ.copy()
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        log.info(f"vLLM server PID: {self._process.pid}")

        if wait_ready:
            self._wait_until_ready()

    def stop(self) -> None:
        """Terminate vLLM server subprocess."""
        if self._process is not None:
            log.info("Stopping vLLM server ...")
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            self._client = None
            log.info("vLLM server stopped")

    def is_alive(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    def generate(
        self,
        prompts: list[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> list[list[str]]:
        """Generate completions for a batch of prompts.

        Args:
            prompts: List of prompt strings.
            sampling_params: Generation parameters. Defaults to SamplingParams().

        Returns:
            List of lists: completions[i] = list of n completions for prompts[i].
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        client = self._get_client()
        completions = []
        for prompt in prompts:
            response = client.completions.create(
                model=self.served_model_name,
                prompt=prompt,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                n=sampling_params.n,
                stop=sampling_params.stop or None,
                seed=sampling_params.seed,
            )
            completions.append([choice.text for choice in response.choices])
        return completions

    def generate_chat(
        self,
        messages_batch: list[list[dict]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> list[list[str]]:
        """Generate chat completions for a batch of message sequences.

        Args:
            messages_batch: List of message lists (OpenAI chat format).
            sampling_params: Generation parameters.

        Returns:
            List of lists: completions[i][j] = j-th completion for messages_batch[i].
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        client = self._get_client()
        completions = []
        for messages in messages_batch:
            response = client.chat.completions.create(
                model=self.served_model_name,
                messages=messages,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                n=sampling_params.n,
                stop=sampling_params.stop or None,
                seed=sampling_params.seed,
            )
            completions.append([choice.message.content for choice in response.choices])
        return completions

    def sync_weights(self, state_dict: dict[str, Any]) -> None:
        """Sync actor model weights to vLLM engine (for online GRPO).

        Uses TRL's VLLMClient weight sync if available, otherwise falls back
        to server restart (offline GRPO mode).

        Args:
            state_dict: Actor model state dict (from trainer.model.state_dict()).
        """
        try:
            from trl.trainer.grpo_trainer import VLLMClient  # type: ignore
            if self._client is None:
                self._client = VLLMClient(host="localhost", server_port=self.port)
            self._client.update_named_parameters(
                [(name, param) for name, param in state_dict.items()]
            )
            log.debug("vLLM weights synced via TRL VLLMClient")
        except (ImportError, AttributeError):
            log.warning(
                "TRL VLLMClient weight sync not available. "
                "Using offline GRPO (weights not synced mid-training). "
                "Upgrade to trl>=0.12 for online GRPO."
            )

    def _build_command(self) -> list[str]:
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--dtype", self.dtype,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--served-model-name", self.served_model_name,
        ]
        if self.max_model_len is not None:
            cmd += ["--max-model-len", str(self.max_model_len)]
        if self.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        return cmd

    def _wait_until_ready(self) -> None:
        """Poll health endpoint until server is ready."""
        import urllib.request
        health_url = f"{self.base_url}/health"
        log.info(f"Waiting for vLLM server at {health_url} ...")
        for attempt in range(HEALTH_CHECK_RETRIES):
            try:
                with urllib.request.urlopen(health_url, timeout=5) as resp:
                    if resp.status == 200:
                        log.info(f"vLLM server ready (attempt {attempt + 1})")
                        return
            except Exception:
                pass
            if not self.is_alive():
                raise RuntimeError("vLLM server process died during startup")
            time.sleep(HEALTH_CHECK_INTERVAL)
        raise TimeoutError(
            f"vLLM server not ready after {HEALTH_CHECK_RETRIES * HEALTH_CHECK_INTERVAL:.0f}s"
        )

    def _get_client(self):
        """Lazy-initialize OpenAI client pointing at local vLLM server."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required for vLLM client. pip install openai")
            self._client = OpenAI(base_url=f"{self.base_url}/v1", api_key="EMPTY")
        return self._client

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        if self._process is not None and self.is_alive():
            self.stop()
