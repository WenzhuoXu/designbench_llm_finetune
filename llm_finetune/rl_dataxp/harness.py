"""Glue to the dataxp evaluation harness, so that training rollouts ARE evaluation episodes.

Nothing of the harness is re-implemented here. ``run_one`` calls the harness's own
``run_episode`` with the harness's own ``LLM`` client and the argparse Namespace the harness's
own ``main()`` builds from the same command-line flags the evaluation driver passes. The only
change is at the HTTP layer (``Recorder``): each chat-completion request gains
``return_token_ids`` (and optionally logprobs, a per-call seed and sampling overrides), and each
response's token ids are recorded. The harness parses the response exactly as in evaluation;
vLLM only adds fields it ignores.

Runs in the simulator env (stdlib + the harness's dataxp code); no torch.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import urllib.request
from typing import Optional

MODULE_NAME = "dx_harness"


def load_harness(script: str):
    """Import dataxp_policy_eval_v3.py from its path (it puts its own directory on sys.path for
    assemble_dataxp_sft, as it does when run as a script)."""
    script = os.path.abspath(script)
    mod = sys.modules.get(MODULE_NAME)
    if mod is not None and os.path.abspath(getattr(mod, "__file__", "")) == script:
        return mod
    spec = importlib.util.spec_from_file_location(MODULE_NAME, script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def harness_args(H, argv: list[str]):
    """The Namespace the harness's main() builds for ``episodes <argv>``, captured by standing in
    for cmd_episodes for the duration of the call, so every default is the harness's own."""
    captured = {}
    orig_cmd, orig_argv = H.cmd_episodes, sys.argv
    H.cmd_episodes = lambda a: captured.__setitem__("a", a)
    sys.argv = [H.__file__, "episodes"] + [str(x) for x in argv]
    try:
        H.main()
    finally:
        H.cmd_episodes, sys.argv = orig_cmd, orig_argv
    return captured["a"]


def instruction_for_variant(H, a, D, domain) -> str:
    """The instruction string cmd_episodes hands run_episode for this variant (its instr_of)."""
    instruction = open(a.instruction_file).read() if a.instruction_file else ""
    return "" if H.speaks_verbs(D, domain) else H.instruction_for(instruction, domain, a.param_line)


def call_record(j: dict, body: dict) -> dict:
    """What one chat-completion response contributes to training."""
    ch = (j.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    content = lp.get("content") if isinstance(lp, dict) else None
    return {"prompt_token_ids": j.get("prompt_token_ids"), "token_ids": ch.get("token_ids"),
            "logprobs": [x.get("logprob") for x in content] if content else None,
            "finish": ch.get("finish_reason"), "model": j.get("model"), "seed": body.get("seed")}


class Recorder:
    """Context manager that stands in for urllib.request.urlopen inside one episode process.

    inject      fields merged into every chat-completion body (return_token_ids, logprobs,
                top_p / top_k overrides, ...)
    seed_base   if not None, call k gets seed (seed_base + 1000003 k) mod 2^31; a retried call
                (the harness retries on an exception) keeps its seed
    """

    def __init__(self, inject: dict, seed_base: Optional[int] = None):
        self.inject = dict(inject)
        self.seed_base = seed_base
        self.calls: list[dict] = []
        self._orig = None

    def __enter__(self):
        self._orig = urllib.request.urlopen
        urllib.request.urlopen = self._urlopen
        return self

    def __exit__(self, *exc):
        urllib.request.urlopen = self._orig
        return False

    def _urlopen(self, req, *args, **kw):
        body = None
        data = getattr(req, "data", None)
        if data:
            try:
                body = json.loads(data)
            except ValueError:
                body = None
        if not (isinstance(body, dict) and "messages" in body):
            return self._orig(req, *args, **kw)
        body.update(self.inject)
        if self.seed_base is not None:
            body["seed"] = (int(self.seed_base) + 1000003 * len(self.calls)) % (2 ** 31)
        req = urllib.request.Request(req.full_url, json.dumps(body).encode(), dict(req.header_items()))
        with self._orig(req, *args, **kw) as r:
            raw = r.read()
        self.calls.append(call_record(json.loads(raw), body))
        return io.BytesIO(raw)


def run_one(payload: dict) -> dict:
    """One episode, in this process. payload: script, argv (harness flags, including --model,
    --base-url, --temperature, --turns), variant, spec, inject, seed."""
    H = load_harness(payload["script"])
    a = harness_args(H, payload["argv"])
    if a.policy != "llm":
        raise ValueError("RL rollouts need --policy llm, got %r" % a.policy)
    if getattr(a, "loop_guard", False):
        raise ValueError("--loop-guard re-asks the model inside a turn; it is not supported in RL rollouts")
    mods = H.import_dataxp(a.code)
    D, _, _, get_domain = mods
    domain = get_domain(payload["variant"])
    instruction = instruction_for_variant(H, a, D, domain)
    with Recorder(payload.get("inject") or {}, payload.get("seed")) as rec:
        ep = H.run_episode(mods, payload["variant"], payload["spec"], a, H.LLM(a), instruction)
    rl_error = None
    if any(not c.get("prompt_token_ids") or c.get("token_ids") is None for c in rec.calls):
        rl_error = "no_token_ids"          # the server ignored return_token_ids: cannot train
    elif len(rec.calls) != len(ep.get("turns", [])):
        rl_error = "calls_turns_mismatch"
    return {"episode": ep, "calls": rec.calls, "rl_error": rl_error}
