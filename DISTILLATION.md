# Distilling the design search into an LLM

The goal is a language model that does iterative engineering design: read a
simulated state, emit one tool call, repeat. A model-free search is the teacher;
the distilled model is the deliverable. The claim under test is that a search
costing 137–726 simulations per problem can be distilled into a model that
reaches comparable feasibility in roughly eight model calls.

Everything below is runnable as written on Bridges-2 from this directory with
`conda activate my_env`.

---

## Layout

| Path | What it is |
|---|---|
| `design_agent/` | The portable domain layer and the search. `da_domain.py` defines the `Domain` interface; `da_truss/synth/pipe/catalogue/cases` implement five of them; `da_search*.py` is the Φ-argmax search; `da_serial.py` is the one serialisation every domain shares. |
| `scripts/gen_corpus.py` | One search pass per problem → `sft.jsonl` + `value.jsonl` + `pref.jsonl`, balanced on turns, every trajectory replay-verified. |
| `scripts/make_tiers.py` | Splits the corpus into nested training tiers and the two held-out sets. |
| `scripts/eval_distilled.py` | The evaluator. Every arm runs through one episode loop at one action space. |
| `scripts/report_data.py` | Extracts every number the progress figures plot. |
| `configs/model/qwen38_27b.yaml` | The target model and its LoRA configuration. |
| `slurm/` | Batch scripts. CPU work goes to `RM-small -q low`; training and serving to `GPU`. |

## Getting the data

Code is in git; the corpus and the problem sets are not. They live in the
project's shared folder, readable by anyone in `mch250030p`:

```
/ocean/projects/mch250030p/shared/designbench/
  problems/problems_hard/          580 problems -- THE EVALUATION SET
  problems/problems_gen_{c..g}/    49,889 minted training problems
  corpus_v2/<domain>/              sft + value + pref, five domains
  tiers/                           what training actually reads
```

Symlink it into your checkout rather than copying -- the 405 GB storage quota is
pooled across every user in the project, so a second copy costs the group 4 GB
for nothing. See the README in that folder for the exact commands.

`problems_hard` is the one irreplaceable item: it is in neither repo, it is the
set every measured baseline below is anchored to, and none of its 580 instances
starts feasible. The minted problems regenerate from `slurm/mint_truss.sbatch`
with seeds 20260911 and 20260921-24; the corpus and tiers regenerate from the
commands below.

## The pipeline

```bash
# 1. Mint truss problems (the only domain with finite supply).
sbatch slurm/mint_truss.sbatch          # 10k problems, ~22 min on 46 cores

# 2. Build the corpus. One domain per job; the four procedural domains are
#    cheap and over-generate, truss is FEA-bound and sets the ceiling.
sbatch slurm/corpus_v2_cheap.sbatch
DOMAIN=truss sbatch slurm/corpus_v2.sbatch

# 3. Split into nested tiers. Balance is on turns and is exact.
python scripts/make_tiers.py --tiers 1000,10000,100000,0

# 4. Train. One tier per invocation; everything else is held fixed.
bash scripts/launch_curve.sh 1k          # then 10k, 100k, all

# 5. Evaluate a checkpoint against the in-script controls.
MODEL_PATH=<checkpoint> ARMS=model,base,fsd,native,search \
  sbatch slurm/eval_vllm.sbatch
```

## Where things stand

Measured in `scripts/eval_distilled.py`, n = 60 per domain, identical problems
and identical action space for every arm, paired exact sign tests:

| domain | search | generic sizing pass | domain heuristic | W–L | p |
|---|---|---|---|---|---|
| truss | 0.9000 | 0.4667 | — | 26–0 | 3.0e-08 |
| synth | 0.9000 | 0.2500 | — | 39–0 | 3.6e-12 |
| pipe | 1.0000 | 0.0000 | — | 60–0 | 1.7e-18 |
| catalogue | 1.0000 | 0.0000 | 0.7833 | 13–0 | 2.4e-04 |
| cases | 1.0000 | 0.0500 | 0.1000 | 54–0 | 1.1e-16 |

242–0 against the generic control across 300 problems; the search loses no
problem to any control in any domain. Corpus: 232,373 supervised turns over
129,179 trajectories, balanced to exactly 0.250 per domain, 571.7M tokens at two
epochs. The training runs themselves have not started — they are queued.

## Things that will bite you

These are all failures we hit, not hypotheticals.

- **`problems_hard` is the evaluation set.** `gen_corpus.py` deliberately draws
  truss training problems only from `problems_gen_*`. Putting `problems_hard`
  back into training supply invalidates every number in the table above.
- **The system prompt's worked example is load-bearing.** Without a concrete
  `<tool>NAME(args)</tool>` example, an un-finetuned model emits
  `<SCALE ids=[E4,E6] factor=1.3></SCALE>` and parses at 0 of 12 calls. The
  stored corpus rows carry the same prompt; keep them in sync or you are
  measuring format compliance rather than design ability.
- **LoRA targets must stay a regex.** `qwen3_5` interleaves 48 linear-attention
  layers with 16 full-attention layers. The conventional leaf-name list
  (`q/k/v/o_proj` + MLP) reaches only 72% of linear parameters and freezes all 48
  linear-attention layers. The regex in the model config is anchored on
  `model.language_model`, which also keeps the vision tower frozen.
  `_apply_lora` raises if the target matches nothing — do not soften that.
- **`value.jsonl` and `pref.jsonl` store a turn index, not the observation.**
  Writing the rendered state into all 22 value records and 14 preference records
  per turn made those files 20× the size of the trajectories they annotate and
  exhausted the 405 GB project quota at 21 GB.
- **Quota exhaustion does not surface cleanly.** It arrived as `tqdm` raising
  `OSError` while writing a progress bar, leaving a 53 GB download alive but
  transferring nothing for fifty minutes. Disable progress bars on long
  transfers.
- **Scheduler**: `RM`/`RM-small` require `-q low`, and that QOS allows only five
  submitted jobs at a time. The `GPU` partition refuses partial-node
  allocations — request all eight GPUs or the submission is denied.
