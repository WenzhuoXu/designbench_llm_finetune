"""Multi-turn GRPO on dataxp, with the evaluation harness as the environment.

The harness (``dataxp_policy_eval_v3.py``) asks the model once per turn with a fresh
system + user prompt. Each of those calls becomes one training sample: the exact token ids
vLLM saw and generated, weighted by the advantage of the episode it belongs to. The reward
is the evaluation outcome itself (``reward.py``); the potential Phi is never part of it.

Modules, and the environment each one runs in:

  reward.py      episode reward and validity                      stdlib, both envs
  samples.py     group advantages, episode -> per-turn samples   stdlib, both envs
  harness.py     loads the harness, records vLLM token ids        stdlib + harness (simulator env)
  rollout.py     runs a step's episodes, resumable                stdlib + harness (simulator env)
  problems.py    builds the training-problem file                 stdlib + harness (simulator env)
  vllm_admin.py  publishes adapters to the vLLM servers           stdlib
  adapters.py    exact concatenation of LoRA adapters             torch + safetensors (trainer env)
  trainer.py     the GRPO loop, checkpoints, resume               torch + transformers + peft (trainer env)

Nothing here imports torch at package import time, so the simulator env (no torch) can run
the rollout side. See docs/rl_dataxp_infra.md.
"""
