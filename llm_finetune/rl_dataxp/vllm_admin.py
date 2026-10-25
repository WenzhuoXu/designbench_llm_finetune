"""Publishing the current policy adapter to the vLLM servers.

The servers run with --enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING=True. Each step's
adapter gets its own name (policy_sNNNNN) instead of an in-place reload: a new name is a new
LoRA id, so no prefix-cache entry computed under an older adapter can be reused. The previous
step's adapter is unloaded afterwards (vLLM keeps the weights in memory once loaded, so the
checkpoint directory may be pruned).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def _root(base_url: str) -> str:
    u = base_url.rstrip("/")
    return u[:-3] if u.endswith("/v1") else u


def _request(url: str, body: dict | None = None, timeout: float = 600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode(errors="replace")


def served_models(base_url: str) -> list[str]:
    return [m["id"] for m in _request(_root(base_url) + "/v1/models", timeout=30)["data"]]


def wait_ready(base_urls: list[str], timeout_s: float = 3600, poll_s: float = 10) -> None:
    t0 = time.time()
    pending = list(base_urls)
    while pending:
        still = []
        for u in pending:
            try:
                served_models(u)
            except Exception:                     # noqa: BLE001
                still.append(u)
        pending = still
        if pending:
            if time.time() - t0 > timeout_s:
                raise TimeoutError("vLLM servers not ready after %.0f s: %s" % (timeout_s, pending))
            time.sleep(poll_s)


def load_adapter(base_url: str, name: str, path: str) -> None:
    if name in served_models(base_url):
        return
    try:
        _request(_root(base_url) + "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path})
    except urllib.error.HTTPError as e:
        raise RuntimeError("loading adapter %s from %s on %s failed: %s %s"
                           % (name, path, base_url, e.code, e.read()[:500])) from e
    if name not in served_models(base_url):
        raise RuntimeError("adapter %s not listed by %s after loading" % (name, base_url))


def unload_adapter(base_url: str, name: str) -> None:
    try:
        _request(_root(base_url) + "/v1/unload_lora_adapter", {"lora_name": name})
    except urllib.error.HTTPError:
        pass                                      # already gone


def publish(base_urls: list[str], name: str, path: str, prefix: str = "policy_s") -> None:
    """Load ``path`` as ``name`` on every server, then unload every other adapter whose name
    starts with ``prefix``."""
    for u in base_urls:
        load_adapter(u, name, path)
    for u in base_urls:
        for m in served_models(u):
            if m.startswith(prefix) and m != name:
                unload_adapter(u, m)
