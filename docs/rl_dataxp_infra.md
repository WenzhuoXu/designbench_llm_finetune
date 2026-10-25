# RL infrastructure for dataxp: multi-turn GRPO with the evaluation harness as the environment

Built 2026-10-25 for option B of `docs/rl_pipeline_options.md`. Code: `llm_finetune/rl_dataxp/`.
Launcher: `scripts/rl/orchard_dataxp_grpo.sh` (+ `slurm/orchard_dataxp_grpo.sbatch`). Configs:
`configs/rl/dataxp_grpo_p2_datav4.yaml` (datav4 protocol, ready) and
`configs/rl/dataxp_grpo_p1_v2.yaml` (the v2 protocol of the October 22 artifact; needs three inputs,
listed at the end of this file).

## Design in one paragraph

The harness asks the model once per turn with a fresh system + user prompt. Rollouts therefore
call the harness's own `run_episode`, with its own `LLM` client and the argparse namespace its own
`main()` builds from the evaluation flags. Only the HTTP layer changes: requests gain
`return_token_ids` (plus logprobs, a per-call seed and `top_p=1, top_k=0`), and the token ids vLLM
saw and produced are recorded. Each model call is one training sample, weighted by its episode's
group-normalised advantage. The trainer recomputes every log-prob itself and never uses vLLM's as
the old policy, because on this model bf16 engines disagree by 0.12–0.19 nats/token
(DISTILLATION.md). The adapter is published to the vLLM servers under a new name each step, so no
prefix-cache entry from an older adapter is reused.

```
 rank 0 (d27o env)                         simulator env (corpus_o)            vLLM (vllm029)
 publish ckpt/step_k/adapter  ─────────────────────────────────────────────▶ load_lora_adapter policy_sk
 rollout.py run  ──▶ one child per episode: harness run_episode ──HTTP──▶   chat/completions (+token ids)
 read rollouts/step_k.jsonl ◀── appended per finished episode
 all ranks: samples → token-balanced shards → PG + β·KL(policy‖base) → all-reduce → AdamW
 checkpoint step_{k+1} (atomic) → metrics.jsonl → guards
```

## Pieces

| file | does | env |
|---|---|---|
| `reward.py` | R = 1 + 0.2·(12 − t)/12 if first feasible at turn t ≤ 12, else 0. Crashed, server-error and feasible-at-start episodes are dropped, not scored 0 | any |
| `samples.py` | group advantages (std-normalised; zero-variance groups dropped); one sample per call; stats (feasible rate, parse rate, cap share, fails per variant); token-balanced sharding | any |
| `harness.py` | loads the harness by path; captures its argparse namespace; `Recorder` stands in for `urlopen`; `run_one` = one episode | simulator |
| `rollout.py` | runs a step's episodes in child processes (crash isolation, as the harness does); appends per episode; resumes by key; cuts a half-written last line; exit 4 if the server returned no token ids | simulator |
| `problems.py` | training-problem file: refuses held-out families; drops evaluation ids and canonical specs, duplicates, feasible starts; writes a summary with input sha1s | simulator |
| `vllm_admin.py` | wait for servers, load/unload adapters | any |
| `adapters.py` | exact sum of LoRA adapters as one adapter (rank r1 + r2): the collaborator's "merged" model becomes a single trainable, servable adapter | trainer |
| `trainer.py` | the loop; config (YAML + `key=value` overrides); deterministic problem sampler; checkpoints and resume; guards | trainer |

## Training step and loss

- **Batch.** `problems_per_step` problems (round-robin over variants, a seeded permutation per
  variant, a pure function of `(seed, step)`), each played `group_size` times at temperature T for
  `horizon` turns.
- **Loss.** For each sample, the per-token clipped policy-gradient term plus `kl_coef` × the k3
  estimator of KL to the reference. The sum over all completion tokens is divided by the step's
  total completion tokens, which is the unbiased token-sum estimator up to one constant per step.
  - With `iterations: 1` the ratio is exactly 1 and this is the plain policy gradient. `iterations`
    > 1 reuses the batch with PPO clipping (`clip_eps_low/high`).
  - `tis_cap` optionally caps a truncated importance weight trainer/vLLM. It is off by default,
    because of the engine noise above.
- **Reference.** `kl_reference: base` disables the adapter, giving the un-tuned model (RL's Razor:
  forgetting tracks KL to the base). `init` uses a frozen copy of the starting adapter.
- **Policy at temperature T.** Log-probs are computed at temperature T (logits / T), matching
  sampling with `top_p=1, top_k=0`. Evaluation is greedy and unaffected.
- **Optimizer.** AdamW, constant learning rate after a linear warm-up, gradient clipping.
  Gradients are all-reduced in 256 MB buckets.

## Restart safety

- **Checkpoint every step.** Each step ends with `ckpt/step_{k+1}`, written to `.tmp` and renamed
  once complete. It holds the adapter, optimizer, scheduler, per-rank RNG, the config fingerprint
  and a parameter checksum that is verified on load.
- **What a restart does.** It resumes from the newest complete checkpoint and reruns only that
  step's missing episodes. It refuses a changed config (`allow_config_change=true` overrides; only
  `grpo.steps` may change freely, so a finished run can be extended).
- **What is kept.** The last 2 checkpoints in full; adapters only for step 0, every 25th step and
  steps 50 / 150. Token arrays of finished steps are compacted, except every 10th step.

## Guards (the run stops itself, writes STOPPED.json, exits 4)

- **Parse rate falls.** The training parse rate (last 5 steps vs first 5) falls by more than 0.10:
  the format drift seen in truss Turn 2.
- **KL to the reference grows.** It grows by more than 0.10 nats/token.
- **No movement.** KL has moved less than 0.005 from its start after 30 steps: the learning rate
  is too low (truss Turn 0).

A stopped run will not restart unless `guards.ignore_stopped=true`.

## What is verified (local, 2026-10-25)

`pytest tests/test_rl_dataxp_core.py` (15 tests, no dependencies). Run the harness tests with
`DX_HARNESS_DIR=<copy of Orchard's corpus/harness>` (6 tests), and the trainer tests with
transformers 5.17 + peft 0.21 + omegaconf (9 tests, CPU, a tiny `Qwen3_5ForConditionalGeneration`).

- **Parity with evaluation.** The same 3 problems (circuit2, thermal2, frame3d2, verb-spelled,
  thinking off) through `dataxp_policy_eval_v3.py episodes` and through RL rollouts, at T=0
  against the same deterministic server, give identical episodes turn for turn, with moves
  applied.
- **Requests and recording.** Requests carry `return_token_ids`, a distinct seed per call, the
  sampling overrides and the step's adapter name. Exactly one recorded call per turn.
- **Rollout resume.** It reruns only missing episodes and cuts a half-written line. A server that
  ignores `return_token_ids` fails the step loudly.
- **Problem builder.** It drops evaluation ids and specs, duplicates and feasible starts, and
  refuses a held-out family.
- **Real architecture.** On the tiny Qwen3_5 model, the production LoRA regex reaches the
  linear-attention projections (and not the vision tower). The `logits_to_keep` path equals a full
  forward. A fresh adapter's KL to the base is 0. `init` vs `base` references behave as specified,
  and the reference copy never trains.
- **Learning direction.** Positive-advantage completions gain log-prob and negative ones lose it.
- **Resume is bit-identical.** An interrupted-then-resumed run equals an uninterrupted one bit for
  bit. A leftover `.tmp` checkpoint is ignored, a changed config is refused, and a stalled run is
  stopped by the guard.
- **Two ranks equal one.** Two processes (gloo) give the single-process update to within 1e-5.
- **Adapter sum is exact.** It matches explicit weight merging to 1e-5, and disabling it gives the
  base.

## Not verified yet (needs the GPU canary)

- vLLM 0.29 returning `prompt_token_ids` / `token_ids` for Qwen3.8-27B chat completions, and
  runtime LoRA loading of a PEFT adapter saved by this trainer. The flags exist in the installed
  source; behaviour is unchecked.
- NCCL gradient all-reduce with a gloo group for the rollout wait (only gloo-only was tested), and
  memory and throughput of the 27B trainer on H100: 54 GB of weights plus LoRA/Adam state is about
  60 GB before activations.
- Rollout throughput per step.

**Canary** (`SMOKE=1`: 3 steps, 2 problems × 4 rollouts, 3 turns; ~1 h on 4 GPUs) passes when:

- every episode has token ids;
- the per-token gap between vLLM and trainer log-probs (`vllm_gap_abs`) is of the documented size
  (≲ 0.2 nats);
- `grad_norm` is finite and above 0;
- the step-k adapter is the one served at step k (`policy_s0000k` in the server log);
- a kill after step 1 resumes at step 1 or 2 without rerunning finished episodes.

## Running (datav4 protocol, P2)

Storage rule: code lives in git (this repo, pushed to GitHub). Persistent files on Orchard live in
`$HOME` (`/home/wxu2/rl`), never in `/project/community`, which is wiped every two weeks; only run
outputs and caches go there.

Staged 2026-10-25:
- **Orchard:** the code is at `/home/wxu2/rl/repo`. The corpus problem files of the 8 training
  families (bridges2 `datav4_corpus/*/accepted*`, 47 files) are at `/home/wxu2/rl/corpus_problems`.
  The training-problem file `/home/wxu2/rl/problems/train_p2_datav4.jsonl` (sha1 `cf9712047712`)
  was built by job 161611 (`scripts/rl/build_problems_p2.sbatch`, 7 min CPU).
- **Durable copies:** the problem files also sit on bridges2 under
  `/ocean/projects/mch250030p/wxu7/rl/problems/`.

It holds 5,752 problems:

| family | read | kept | dropped |
|---|---|---|---|
| circuit2 | 910 | 795 | 115 feasible starts |
| frame3d2 | 910 | 910 | � |
| grid2 | 910 | 910 | � |
| seir2 | 910 | 910 | � |
| traffic2 | 910 | 907 | 3 feasible starts |
| grid2op | 491 | 431 | 60 evaluation ids |
| engibench | 520 | 434 | 60 evaluation ids, 26 feasible starts |
| photonics | 520 | 455 | 60 evaluation ids, 5 feasible starts |

No v2 evaluation problem occurs in the corpora. The batches being appended now (b012, b013) are
not included.

```bash
# 1. this repo on Orchard (code only)
#    upload to /home/wxu2/rl/repo (Orchard $HOME; /project/community is wiped)
# 2. training problems (simulator env, CPU; corpus problem files from bridges2 datav4_corpus)
/project/community/wxu2/envs/corpus_o/bin/python -m llm_finetune.rl_dataxp.problems \
  --script /project/community/wxu2/corpus/harness/scripts/dataxp_policy_eval_v3.py \
  --code /project/community/wxu2/corpus/harness/dataxp_code_v3 \
  --corpus circuit2=.../circuit2/accepted_b005/problems.jsonl ... \
  --eval-problems /project/community/wxu2/corpus/harness/problems_v2.jsonl /project/community/wxu2/corpus/harness/problems_v3_native.jsonl \
  --out /home/wxu2/rl/problems/train_p2_datav4.jsonl
# 3. canary, then the run
sbatch --export=ALL,RUN_DIR=/project/community/wxu2/rl/runs/p2_smoke,CONFIG=configs/rl/dataxp_grpo_p2_datav4.yaml,SMOKE=1 slurm/orchard_dataxp_grpo.sbatch
sbatch --export=ALL,RUN_DIR=/project/community/wxu2/rl/runs/p2_base_s0,CONFIG=configs/rl/dataxp_grpo_p2_datav4.yaml slurm/orchard_dataxp_grpo.sbatch
```

**Evaluation of a checkpoint.** Use the unchanged harness driver:
`ADAPTERS="p2_s50=<run>/ckpt/step_00050/adapter" ARMS="p2_s50:p2_s50:nothink" dx_eval_orchard.sh`.
That driver hard-codes `--max-lora-rank 64`, so an adapter of rank > 64 (a merged start) needs that
one flag raised there first.

## Open items

- **P1 (v2 protocol) inputs, not on Orchard:**
  - the pre-verb dataxp copy (`dataxp_code_v2_w1fix`, on drc);
  - S's two adapters from the collaborator, summed with `adapters.py` (rank 128, so
    `MAX_LORA_RANK=128`);
  - a problems file built against that copy.

  The config refuses to start without them.
- **grid2op game-over.** The harness keeps asking after a game-over (an open harness item). In
  training that only wastes turns (such an episode is never feasible, R = 0), but it costs
  rollout time.
- **Orchard cleanup.** `/project/community` is wiped every two weeks: copy kept adapters and
  `metrics.jsonl` off it (to bridges2 `/ocean` or local) as soon as a run ends. The harness,
  environments and model snapshot that other tracks keep under `/project/community/wxu2` are
  exposed to the same wipe.
- **Bridges-2.** `configs/rl/dataxp_grpo_p2_bridges2.yaml` + `scripts/rl/bridges2_dataxp_grpo.sbatch`
  run the same launcher on a whole 8�H100 node. It covers the v2 families only, because
  Bridges-2 has no native simulators.
- **Truss code untouched.** The truss GRPO code (`llm_finetune/training/rl/`, `scripts/train_grpo.py`)
  is left as it is: the ledger's results and `tests/test_rewards.py` /
  `tests/test_shaping_invisibility.py` depend on it. Nothing in `rl_dataxp` imports it.
