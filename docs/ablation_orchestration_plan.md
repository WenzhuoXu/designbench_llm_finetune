# Ablation Orchestration Plan — Autonomous Agent Runbook

> **Audience.** An autonomous Claude Code agent acting as **Orchestrator/Administrator**
> for the DesignBench SFT+GRPO ablation study on Bridges-2.
> **Goal.** Drive `eval/feasibility_rate` (held-out truss problems) to **> 90%**, or to a
> reasoned stop, by iteratively running reward/CoT/model ablations on the HPC.
> **Theory anchor.** `docs/plan_posterior_reward_walkthrough.md` (the "master doc").
> Every experiment must map to a section there; every result must be re-aligned against it.

---

## 0. Role of the Orchestrator

The main agent is an **administrator, not a worker**. It does **not** run training or model
code itself. It:

1. **Plans** the next batch of experiments from theory + last turn's results.
2. **Generates configs and SLURM scripts**, then **submits batches** of jobs.
3. **Sets up alerts** for job start/completion and **monitors** them.
4. **Evaluates** finished checkpoints, **analyzes** results, **diagnoses failure modes**.
5. **Re-aligns** findings with the master doc and **proposes the next stage**.
6. **Reports** to the human when the target is hit or progress stalls.

Sub-work (log parsing, config generation, metric extraction) may be delegated to
sub-agents, but **decisions, submissions, and the experiment ledger stay with the
orchestrator**.

### 0.1 Hard rules (from project memory — do not violate)

- **Never run model/GPU/transformers code on the login node.** All training and eval go
  through `sbatch`. Login-node Python is for log parsing, config edits, `squeue`, JSON only.
- **Every Python invocation that imports torch/transformers** must first
  `module load anaconda3/2024.10-1 && conda activate my_env` — but this only happens
  *inside sbatch scripts*, never interactively on login.
- **SLURM discipline:** read the existing `.sbatch` before reusing it; audit every API
  param against the installed TRL/Accelerate/vLLM versions; **never leave duplicate jobs
  queued** for the same cell. `squeue -u $USER` before every submit; cancel stale dupes.
- **SFT is warmstart-only** — it teaches output format, not reasoning/policy. Do not
  "improve" SFT reasoning to chase eval accuracy; the policy is learned by GRPO.
- The framework must **generalize beyond truss** — prefer config/hook changes over
  truss-specific hacks in shared code.

---

## 1. Current State (as of 2026-06-12 — verify before acting)

### 1.1 SFT — DONE, warmstart only
- Gold warmstart checkpoint (May 21):
  `checkpoints/sft/gold_warmstart_qwen3_14b_fixed_20260521_001258/final`
- Compliance (`logs/compliance/..._40954766.json`): format = semantic = executable =
  single-action = **1.0**, `ready_for_grpo: true`, `mean_output_tokens ≈ 50`.
- **Read:** format is solved. The 50-token output confirms this is *format-only* warmstart;
  reasoning depth is GRPO's job, not SFT's.

### 1.2 GRPO ablation v2 — RAN, but policy barely moved
- 6 configs, two full passes (`logs/qwen3_14b_v2_*_20260523_*`, `..._20260525_*`),
  200 steps each, ~5.5 h/run on 2-GPU LoRA (`slurm/grpo_qwen3_lora_2gpu.sbatch`):
  | Config | Theory step (§6.3) | Knob |
  |---|---|---|
  | `grpo_abl_v2_01a/b/c` | Step 1 baseline | α ∈ {2, 5, 10}, r_env only |
  | `grpo_abl_v2_02_tree` | Step 2 (PRIMARY) | tree expansion D=1, b=5 |
  | `grpo_abl_v2_04_tree_d2` | Step 2 | tree expansion D=2 |
  | `grpo_abl_v2_03_base` | SFT-prior control | no warmstart |
- **Failure signals in the metrics** (`metrics.jsonl`):
  - `kl ≈ 0.001`, `clip_ratio/region_mean = 0` across the run → **the policy is not
    moving**. LoRA + LR may be too weak, or advantages are near-zero-variance.
  - `rewards/compute_rewards/mean` is noisy (10→22→9→15), **no upward trend**.
  - **`eval/rho_tree_agreement` (ρ(t)) is NOT in `metrics.jsonl`** even though the
    config comments and `submit_ablations_v2.sh` claim to track it — the falsifiable
    headline metric of §3.6 is currently **uninstrumented**. This is a blocking gap.
  - `gpu0_util = 0%` while `gpu1 = 100%` → 2-GPU split is imbalanced (likely vLLM idle
    on one card or train/gen serialization). Throughput risk, not correctness — note it.

### 1.3 Eval — NOT DONE (the critical gap)
- `results/eval/` is **empty**. **No checkpoint has ever been scored on the real task.**
- `scripts/eval_checkpoint.py` emits `eval/feasibility_rate = mean(reaches_solution)` —
  **this is the ">90%" target metric.** Eval harness: `slurm/eval_ablation_v2.sbatch`
  (1×H100, `--max-steps 20 --temperature 0.0`).
- **Turn-0 priority: we are flying blind. Before any new training, evaluate the existing
  v2 checkpoints to establish the baseline number.**

### 1.4 Available models (configs/model/*.yaml)
- Primary: `qwen3_14b`. Also present: `qwen3_30b_a3b_thinking_2507` (the "qwen3.5/large"
  slot), `gemma3_12b`, `gemma4_31b`, `gemma4_26b_a4b`, `deepseek_r1_14b/32b`,
  `phi4`, `phi4_reasoning`, `mistral_small_24b`, `llama4_scout`.
- Model ablation axis (user request): **qwen3_14b → qwen3_30b_a3b_thinking → gemma4**.
  Only widen the model axis **after** the reward recipe is validated on qwen3_14b
  (cheapest), to avoid burning H100-hours on a moving target.

---

## 2. Experiment Roadmap

Follow the master doc's **§6.3 recommended ablation sequence** as the spine. Each stage is
one "turn" of the loop (§4). Do not add a reward term until the previous one is verified
(§7.1 of the master doc gives the per-component check for each).

### 2.1 Reward / advantage axis (primary research variable)
Ordered by §6.3; each builds on the previous winner:

| Stage | Adds | Config knob | §7.1 verification | Stop-and-diagnose if… |
|---|---|---|---|---|
| **S0 Eval baseline** | nothing (eval existing ckpts) | — | get a number at all | (always proceed) |
| **S1 r_env + α sweep** | Lagrangian potential only | `reward_weights.lagrangian_potential`, `posterior.alpha ∈ {2,5,10}` | feasible-rate monotone in α; ≥90% at α≥5 | feasibility flat in α |
| **S2 Tree expansion** | D=1,b=5 advantage (PRIMARY) | `use_tree_expansion`, `tree_expansion.{depth,branching}` | **ρ(t) rises → ≥0.85**; Φ(s_H) ↑ vs S1 | ρ(t) stalls < 0.85 → §3.6 (a/b/c) diagnosis |
| **S3 SFT priors** | action-class KL + sibling DPO | warmstart variant (§2.1/§2.2) | SFT action-class KL drops; DPO pair-sat ≥80% | — |
| **S4 Pattern signals** | r_macro + r_dead | `reward_weights.{macro,dead_end}` (β₂=0.05, ξ=0.20) | macro-completion ↑, dead-end overlap ↓ | — |
| **S5 LLM-capability** | r_escape + difficulty λ | `reward_weights.escape`, `lambda` | **CRITICAL:** hard-init catches up; advantage gap appears | **no gap → PAUSE, report.** LLM isn't using priors |
| **S6 Reasoning grounding** | r_pred (CoT→prediction) | `reward_weights.pred` (μ=0.10) | counterfactual CoT perturbation shifts action | — |
| **S7 Strategy adaptation** | ρ_adapt | `nu=0.05`, multi-problem batch | per-problem action-class KL ↑ | — |

**CoT complexity / depth axis** (user request) maps to:
- **CoT depth** = tree-expansion depth `D` (S2: D=1 vs D=2; deeper continuation policy
  `continuation: greedy → llm_sampled` per §3.3 — the doc says continuation depth beats
  tree depth, so test `continuation` before pushing D≥2).
- **CoT complexity** = `r_pred` forward-prediction grounding (S6) forces the CoT to commit
  to a predicted post-action state; and the prompt change requiring predicted Δm/feasibility.
- **Reward composition** = the `reward_weights` map; sweep the §6.2 default coefficients
  (α, β₂, ξ, μ, κ, λ, ν) one axis at a time once the structure is fixed.

### 2.2 Model axis (secondary — only after reward recipe validated)
After the best reward recipe is fixed on `qwen3_14b`, re-run that single recipe across:
`qwen3_14b` (anchor) → `qwen3_30b_a3b_thinking_2507` → `gemma4_31b` (and/or `gemma4_26b_a4b`).
This isolates **"does the recipe transfer across model families/sizes?"** — a clean
generalization claim, and the headline for a model-scaling figure.

### 2.3 Each stage = a *batch* of jobs (parallelism is mandatory)
The user runs on an 8×H100 cluster — **submit the whole sweep at once**, not serially.
E.g. S1 = 3 jobs (α∈{2,5,10}); S2 = {D1, D2, D1+llm-continuation}; model axis = 3 jobs.
Aim to keep the queue saturated with independent cells, each writing to a distinct
`RUN_NAME` so W&B and `results/eval/<RUN_NAME>` stay separable.

---

## 3. Instrumentation Prerequisites (fix BEFORE the next training batch)

These gate the whole study — a turn that can't measure its headline metric is wasted:

1. **ρ(t) logging.** `eval/rho_tree_agreement` must be written to `metrics.jsonl` during
   GRPO (§3.6). Locate the tree-expansion advantage code (`rl/posterior/` per the master
   doc's feature path) and emit ρ(t) per logging step. Without it, S2 is unfalsifiable.
2. **Policy-movement sanity.** Confirm `kl` and `clip_ratio/region_mean` become nonzero
   when the policy actually updates. If they stay 0 at LR=5e-6 LoRA, the recipe can't
   learn — escalate LR, widen LoRA rank, or check advantage variance (`reward_std`) before
   spending more GPU-hours.
3. **Eval wiring.** Confirm `eval_ablation_v2.sbatch` runs end-to-end and writes
   `results/eval/<RUN_NAME>/...json` with `aggregate.feasibility_rate`. This is the
   acceptance metric — it must be reproducible on demand.

Treat §3 as **Turn 0**, bundled with the S0 eval baseline.

---

## 4. The Per-Turn Loop (the core algorithm)

Repeat until the termination criteria in §6 are met.

```
TURN n:
  1. PLAN
     - Read the ledger (§5) + last turn's eval results + the master doc section
       for this stage.
     - Decide the batch: list of (config, model, run_name) cells for this stage.
     - Write/patch the needed configs/<rl|model>.yaml. Keep diffs minimal & reviewable.

  2. SUBMIT (batch, in parallel)
     - squeue -u $USER  → cancel any stale duplicate of a cell you're about to submit.
     - Submit all cells of the stage with sbatch --export=... (distinct RUN_NAME each).
     - Record every job id + cell mapping in the ledger immediately.

  3. ALERT + MONITOR  (see §5)
     - Arm start/finish alerts for the submitted job ids.
     - Poll squeue on an interval; on completion, pull metrics.jsonl + slurm .err.
     - Cheap correctness gate while RUNNING: nonzero kl by step ~20, reward_std > 0,
       no OOM/NCCL in .err. Kill + fix a cell that is clearly broken; don't wait 5 h.

  4. EVALUATE
     - For each finished checkpoint, submit eval_ablation_v2.sbatch.
     - Collect eval/feasibility_rate (+ ρ(t), Φ(s_H), per-problem table).

  5. ANALYZE + DIAGNOSE FAILURE MODES
     - Compare cells within the stage and vs. previous best.
     - Name the failure mode (reward-hacking §8.4, ρ(t) stall §3.6, KL-collapse,
       hard-init no-gap §6.3-step5, GPU imbalance, format regression).
     - Cross-check the relevant §7.1 verification row for this stage.

  6. RE-ALIGN with the master doc
     - Does the result confirm or contradict the theory's prediction for this section?
     - If contradict: is it an instrumentation bug, a coefficient miscalibration
       (§6.2), or a real theory gap (→ note in ledger as an open problem, §8)?

  7. PROPOSE next stage
     - Update the ledger with the decision + rationale.
     - If target met or stalled → §6 (report). Else → TURN n+1 with the proposal.
```

**Discipline:** one structural change per turn (one new reward term *or* one model *or*
one coefficient axis). Mixing changes makes attribution impossible — the whole point of an
ablation.

---

## 5. Alerting, Monitoring, and the Ledger

### 5.1 Job start/finish alerts
Two complementary mechanisms — use both:
- **SLURM mail** (passive): add `#SBATCH --mail-type=BEGIN,END,FAIL` and
  `#SBATCH --mail-user=wxu2@andrew.cmu.edu` to submitted scripts. Survives agent downtime.
- **Active poll** (agent-driven): after submitting, schedule a wake-up to re-check
  `squeue -u $USER`. Pick the interval by what you're waiting for, not round numbers:
  - GRPO cell ~5.5 h → poll every ~20–30 min (long fallback; cache-cold is fine).
  - Eval cell ~tens of min → poll ~5 min while it's the only thing pending.
  - When a job's state flips `R → (gone)`, immediately pull its
    `logs/<run>/metrics.jsonl` and `logs/slurm/*_<jobid>.err`.

  On completion the harness re-invokes the agent; use that to run step 4–7. Do **not**
  busy-wait in the foreground.

### 5.2 The experiment ledger (single source of truth)
Maintain `docs/ablation_ledger.md` (create it on Turn 0). One row per cell:

```
| turn | date | stage | config | model | run_name | job_id | status |
  feasibility_rate | rho_t | kl_final | reward_trend | failure_mode | decision |
```

Append-only narrative below the table: per turn, 3–6 lines —
*hypothesis → result → theory re-alignment → next proposal*. This is what makes the run
auditable and lets a fresh agent resume mid-study.

### 5.3 What to watch per job (from CLAUDE.md + master doc §7.1)
- Training: `kl` (>0.01 by step 100), `clip_ratio/region_mean` (>0 = policy moving),
  `rewards/compute_rewards/mean` (↑ within-condition), `reward_std` (>0 = learnable
  advantages), `entropy` (not collapsing to 0).
- Theory: `eval/rho_tree_agreement` (→≥0.85 for tree cells).
- System: `gpu*_util_pct` (both cards busy — the 0%/100% split is a known throughput bug).
- Eval: `eval/feasibility_rate` (the target), per-problem table (which problems fail).

---

## 6. Termination Criteria & Reporting

**Stop and report to the human when ANY of:**

1. **Success:** a checkpoint reaches `eval/feasibility_rate > 0.90` on the held-out
   problem set (confirm with a second eval seed / temperature to rule out variance).
2. **Diminishing returns:** two consecutive turns yield < 1 absolute-point feasibility
   improvement *and* the next planned term's §7.1 check already passed (nothing left that
   theory predicts will help).
3. **Theory wall (§6.3 step 5):** the LLM-capability stage shows **no advantage gap** on
   hard-init/OOD slices → the policy isn't using priors; further reward shaping is futile.
   Pause and report per the master doc's explicit instruction.
4. **Hard blocker:** an instrumentation/infra problem the agent cannot resolve
   autonomously (e.g., persistent KL-collapse at all LR/LoRA settings).

**Report contents:** best checkpoint + its eval number, the winning reward recipe and
coefficients, the ablation table (contribution of each term), ρ(t) trajectory for tree
cells, named failure modes, and which master-doc predictions held vs. broke (candidate
§8 open-problem updates). Attach the ledger.

---

## 7. Turn-0 Concrete Checklist (start here)

1. Create `docs/ablation_ledger.md` (§5.2 schema).
2. **Fix instrumentation (§3):** ρ(t) logging + verify eval harness writes
   `feasibility_rate`. Add `--mail-type`/`--mail-user` to the sbatch scripts.
3. **Establish the baseline:** submit `eval_ablation_v2.sbatch` for the existing v2
   checkpoints (`checkpoints/grpo/qwen3_14b_v2_*_20260525_1228/*`). One batch, parallel.
   This is the first real feasibility number — record it.
4. From those numbers, decide whether to **re-run S1/S2 with the policy-movement fix**
   (likely, given KL≈0) or proceed to S3. Propose in the ledger; begin Turn 1.

> Until §3 instrumentation is fixed, **do not launch a fresh GRPO sweep** — a run whose
> headline metric (ρ(t)) and acceptance metric (feasibility) aren't captured is a wasted
> 5.5 h × N-cells of H100 time.
