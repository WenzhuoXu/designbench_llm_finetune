# Generating DesignBench truss problems that contain a design decision

Status: measured, 2026-08-27. Everything below is a number produced by a script in
this repo on the Bridges-2 CPU queue; nothing is estimated.

Artefacts:

| what | where |
|---|---|
| generator (legacy + `--mode hard`) | `scripts/generate_truss_problems.py` |
| validity checker (raw-simulator degeneracy) | `scripts/validate_generated_problems.py` |
| measured-difficulty labeller | `scripts/label_difficulty.py` |
| one-global-action reachability probe | `scripts/one_global_action_probe.py` |
| SLURM (generate + validate + ladder + label) | `slurm/gen_hard_cpu.sbatch` |
| SLURM (reference ladders) | `slurm/refs_ladder_cpu.sbatch` |
| SLURM (one-global probe) | `slurm/one_global_probe_cpu.sbatch` |
| the batch | `DesignBench/data/problems_hard/` |

All jobs run on `--partition=RM-shared --qos=low`. No GPU is used anywhere in this
pipeline.

---

## 1. What the originals actually are

Measured over `DesignBench/data/problems/auto_problem_*.json` (n = 100) plus the
30 hand/test specs that share the directory (n = 130 loadable problem specs).

* 2-row Warren **ground structure**, `num_bays` in {2,3,4} → 6/8/10 joints and
  11/16/21 members (bottom chord + top chord + verticals + both diagonals per bay
  = 5·bays + 1).
* span ~ U(6, 14) m, height ~ U(1.5, 4.0) m, planar (z = 0).
* supports: joint 0 pinned, joint `nx-1` roller_y.
* one material per problem, uniform over {6061_T6_Aluminum, A36_Steel} (49/51).
* Pipe sections everywhere; declared bounds `r ∈ [0.005, 0.15]`,
  `t ∈ [0.001, 0.02]`, initial `r = 0.03, t = 0.005`.
* 1–3 downward point loads on distinct **top-row** joints, U(30 000, 80 000) N.
* goals: `minimum_fos_buckling = minimum_fos_yielding = 1.5` and `maximum_mass`,
  with **`maximum_mass / _metadata.optimal_mass = 1.100000` for all 100**.
* member proportions (1455 members): `r` p10/median/p90 = 0.0126 / 0.0265 / 0.0435 m;
  `t/r` p10/median/p90 = 0.083 / 0.129 / 0.167.

Two defects matter for benchmarking.

**(a) 31 of the 130 are degenerate mechanisms.** Read through the RAW simulator
(`analyze_truss(load_truss_from_problem(spec), goals)`, *not* `_analyze_truss`,
which clamps an infinite deflection to exactly 1.0), 11 report `deflection = inf`
and 17 report 1e13–1e14 m while clean problems top out at 0.106 m. They are
kinematically unstable, and every policy scores 0 on them because the search's
own validity guard rejects every successor state:

```
                 greedy_critical  lookahead_v2_d1  lookahead_v2_d2
originals clean   (n=98/99)  0.949           1.000            1.000
originals degen   (n=31)     0.000           0.000            0.000
```

So the original set's headline difficulty (0.723 / 0.762 / 0.762) is **not
difficulty**. It is 31 broken problems dragging down a set that the classical
sizing heuristic otherwise solves 95% of and depth-1 lookahead solves 100% of.

**(b) The mass budget does not bind, so one scalar solves the problem.**
`_metadata.optimal_mass` is computed at the fixed pipe aspect `t/r = 1/6`. That is
not the lightest design the problem's own bounds allow: for a thin-walled pipe
`A ≈ 2πrt` and `I ≈ πr³t`, so `I = A·r²/2` and buckling capacity per unit mass
grows as `r²`. Measured on 8 problems, a certified free-aspect optimum is
0.36–0.83× the fixed-aspect `optimal_mass` (aspect floor 1/60 … 1/12). With
`maximum_mass = 1.1 × optimal_mass` the search has ~27% of unusable slack — the
previous generated batch reached feasibility at `mass/optimal_mass = 0.800`,
i.e. 0.73× of its own budget.

The direct consequence, measured by `scripts/one_global_action_probe.py` (a
21-point grid per parameter over factors [0.5, 4.0], applied to every member at
once, i.e. `SCALE_MULTI_PARAM([all], [r:a])` / `[t:b]` / both):

| set | one global rescale reaches feasibility |
|---|---|
| originals, all 130 | 79 / 130 = **0.608** |
| originals, 99 non-degenerate | 79 / 99 = **0.798** |
| originals, 31 degenerate | 0 / 31 |
| previous generated (250) | 14 / 250 = 0.056 |

(Bounds mode `declared`: the rescale must keep every member inside the problem's
own `shape_params`, which is exactly the filter `search_ladder.candidate_actions`
applies. Dropping that filter — mode `physical`, an upper bound on what any agent
emitting raw grammar could do — takes the originals to 91/130 = 0.700.)

Four fifths of the *working* original problems are one scalar wide.

---

## 2. Why the previous generated batch was 100% solvable by depth-1 lookahead

`problems_gen` (250) fixed the degeneracy — 250/250 pass the raw-simulator check —
but made the sizing problem trivial in a different way. Its degradation samples
uniformly from {`shrink_all` (every member's `t ×= U(0.75,0.90)`), `shrink_random`
(one member's `t`), `remove_member`} and stops as soon as min FOS < 1.5. The posed
design is therefore the optimum with its **relative proportions essentially
intact**: correctly proportioned, uniformly under-sized. Its `optimal_mass` is
still the fixed-aspect one, so nothing punishes over-building.

Measured: `lookahead_v2_d1` solved **250/250 in a median of 2 steps**, and
`lookahead_v2_d2` added nothing (also 250/250). Under Φ_v2 the global all-member
action is the search argmax at 48.4% of visited states; under Φ_v1 at 97.3%.

Note the brief's "one global action solves them" is not literally true of that
batch (5.6% by the probe above) — what is true is that it falls to **two** greedy
lookahead moves with no allocation decision anywhere.

---

## 3. The fix

Two changes, both in `scripts/generate_truss_problems.py` under `--mode hard`.
The legacy path is untouched, so `problems_gen` remains reproducible.

### 3.1 A mass reference that binds (`certified_optimum_free`, `min_area_section`)

`optimal_mass` is no longer the fixed-aspect FSD result. The generator runs a
**two-variable fully-stressed design**: from each member's own utilisation it takes
the required area factor (yielding) and inertia factor (buckling) and asks
`min_area_section` for the lightest `Pipe(r, t)` inside the declared bounds that
meets both, subject to an aspect floor `t/r ≥ aspect_min` (default 1/25 — an
un-floored optimiser walks to a 1 mm wall on a 150 mm pipe, which the
linear-elastic simulator accepts and no real pipe survives). Every iterate is
certified by the real simulator; the lightest certified state is `certified_mass`.

The posed budget is then

```
optimal_mass  = certified_mass × mass_slack        (mass_slack ≥ 1 is the difficulty dial)
maximum_mass  = 1.1 × optimal_mass                 (the invariant every original satisfies)
```

This keeps `maximum_mass / optimal_mass = 1.1` exactly — the validator, the
potential's `objective_ref`, and any downstream consumer see the same interface —
while moving the actual tightness into `mass_slack`, which is recorded in
`_metadata` along with `certified_mass` and the fixed-aspect mass for audit.
V4 (a feasible design provably exists, inside the declared bounds, at or under the
budget) holds by construction because the certificate is held in hand.

### 3.2 Degradation by MISALLOCATION, not uniform thinning (`degrade_misallocate`)

Starting from the fixed-aspect optimum (so the posed design has realistic
proportions), each problem gets:

* a random subset (`frac_down` of members) **shrunk**, each by its *own* factor
  drawn independently from `down_range` — so the repair factor differs per member
  and any single global factor has to be the max over them, over-building
  everything else;
* a disjoint random subset (`frac_up`) **inflated**, each by its own factor from
  `up_range` — these members are already strong, so a global scale-up spends the
  budget on material that was never needed, and the other one-action escape (a
  global scale-*down*) kills the members in the first subset;
* the parameter touched drawn per member from {r}, {t}, {r,t}, so the aspect ratio
  varies member to member and no single-parameter sweep fixes the set;
* optionally one stability-checked **member removal** and one **interior joint
  jitter**, which change the force paths so the posed proportions are not a
  rescaling of *any* optimum of the posed structure.

The repair therefore has to move material *from* the inflated members *to* the
shrunk ones under a budget that binds. That is an allocation decision, not a
scalar. Mechanism-creating removals and jitters are rejected (exact rank test on
the geometric equilibrium matrix), which is why the batch stays clean.

---

## 4. Calibration: the pilot sweep

Each stratum is a knob setting, **not** a difficulty claim. Difficulty is measured
afterwards. Pilots of 40 problems (50 for s1), identical policies, 20 max steps,
`--tau 0.05 --alpha 5.0`:

| stratum | `mass_slack` | frac down / up | removals | jitter | n | greedy_critical | lookahead_v2_d1 | lookahead_v2_d2 | Φ_v2 global-argmax |
|---|---|---|---|---|---|---|---|---|---|
| s0  | 2.80 | 0.30 / 0.20 | 0 | 0.0 | 40 | 0.550 | 1.000 | 1.000 | 51.0% |
| s1  | 2.10 | 0.35 / 0.25 | 0 | 0.0 | 50 | 0.360 | 1.000 | 1.000 | 37.9% |
| s2  | 1.70 | 0.45 / 0.30 | 0 | 0.3 | 40 | 0.225 | 0.925 | 0.925 | 27.1% |
| s2b | 1.55 | 0.45 / 0.32 | 1 | 0.3 | 40 | 0.075 | 0.725 | 0.925 | 26.9% |
| s3  | 1.40 | 0.50 / 0.35 | 1 | 0.3 | 40 | 0.025 | 0.500 | 0.750 | 12.2% |
| s4  | 1.20 | 0.55 / 0.40 | 1 | 0.5 | 40 | 0.000 | 0.150 | 0.150 |  3.1% |
| s5  | 1.05 | 0.60 / 0.45 | 1 | 0.5 | 40 | 0.000 | 0.075 | 0.100 |  6.9% |

Readings:

* `mass_slack` is a clean monotone difficulty dial once the mass reference is the
  free-aspect one. It is the dominant knob; the misallocation shape parameters
  move things far less.
* s0 is already back in the pathology zone (Φ_v2 picks the global action half the
  time) — it is included only in small proportion, to keep genuinely easy content
  in the set.
* s4 and s5 are past the useful end: 83% and 90% of their problems are solved by
  no policy in the ladder. s5 is excluded from the batch; s4 is included at low
  weight as the unsolved tail.
* **s2b and s3 are where depth-2 lookahead earns its keep** (0.725→0.925 and
  0.500→0.750). On the originals and on `problems_gen`, d2 = d1 exactly; this is
  the first truss family in this repo on which the extra depth buys anything.

One-shot global reachability per stratum (declared bounds, pilot dirs):
s0 0.675, s1 0.140, s2 0.025, s2b 0.000, s3 0.000, s4 0.000. The dial that fixes
the solve rate is the same dial that kills the global action.

Chosen mix for the batch: `s0:1, s1:2, s2:2, s2b:4, s3:4, s4:1` (14 units).

---

## 5. Measured solve rates — the headline table

Identical policies, identical settings on all three sets: 20 max steps,
`alpha = 5.0`, `tau = 0.05`, `feasibility_offset = 0`, `search_ladder.py`,
32 CPU workers, no LLM anywhere.

| set | n | `greedy_critical` | `lookahead_v2_d1` | `lookahead_v2_d2` |
|---|---|---|---|---|
| originals (`data/problems`) | 130 | 0.723 | 0.762 | 0.762 |
| originals, 99 non-degenerate only | 99 | **0.949** | **1.000** | **1.000** |
| previous generated (`data/problems_gen`) | 250 | 0.236 | **1.000** | **1.000** |
| **new batch (`data/problems_hard`)** | **580** | **0.116** | **0.734** | **0.800** |

Target from the brief: `lookahead_v2_d1` in 0.4–0.8 (**0.734 ✓**, and not 1.000),
`greedy_critical` meaningfully below it (**0.116 vs 0.734, a 0.618 gap ✓**), and a
tail that d1 fails and d2 solves (**44 problems, 7.6% of the batch ✓** — the first
set in this repo where d2 ≠ d1 at all).

Per stratum, on the batch itself:

| stratum | n | greedy_critical | lookahead_v2_d1 | lookahead_v2_d2 | easy | medium | hard | open |
|---|---|---|---|---|---|---|---|---|
| s0  |  41 | 0.439 | 1.000 | 1.000 | 18 |  23 |  0 |   0 |
| s1  |  83 | 0.289 | 1.000 | 1.000 | 24 |  59 |  0 |   0 |
| s2  |  83 | 0.108 | 0.892 | 0.928 |  9 |  65 |  5 |   4 |
| s2b | 166 | 0.060 | 0.801 | 0.867 | 10 | 123 | 13 |  20 |
| s3  | 166 | 0.036 | 0.512 | 0.627 |  6 |  79 | 21 |  60 |
| s4  |  41 | 0.000 | 0.244 | 0.366 |  0 |  10 |  5 |  26 |
| **ALL** | **580** | **0.116** | **0.734** | **0.800** | **67** | **359** | **44** | **110** |

**Difficulty counts (measured, written into `_metadata.difficulty`):**
easy 67 (11.6%), medium 359 (61.9%), hard 44 (7.6%), open 110 (19.0%).

Validity: **580/580 accepted, 580/580 PASS**, 0 rejected for any reason —
0 degenerate mechanisms under the raw-simulator check (`analyze_truss` on
`load_truss_from_problem`, not `_analyze_truss`). For comparison the same checker
rejects 28/100 of the original `auto_problem_*` as degenerate mechanisms and
0/250 of `problems_gen`. Generation acceptance was 580/580 candidates — no
rejects at all, because the mechanism tests run *inside* the degradation loop
rather than as a post-filter. The validator was re-run after
`label_difficulty.py` rewrote the specs: still 580/580 PASS.

---

## 6. What the batch looks like

`DesignBench/data/problems_hard/`, 580 problems, ids `hard_problem_0000` …
`hard_problem_0579`, plus `manifest.jsonl` (one audit record per candidate).

Generated with

```bash
sbatch --export=ALL,N=580,ATTEMPTS=580,\
OUTDIR=/ocean/projects/mch250030p/wxu7/DesignBench/data/problems_hard,\
STRATA="s0:1+s1:2+s2:2+s2b:4+s3:4+s4:1",SEED=20260828,TAG=hard_v1,\
PREFIX=hard_problem,WRITE_LABELS=1,ONE_GLOBAL=1 slurm/gen_hard_cpu.sbatch
```

(`STRATA` uses `+` as its separator because `sbatch --export` splits on commas —
this bit us once and cost a pilot round.)

Realised strata: s0 41, s1 83, s2 83, s2b 166, s3 166, s4 41.
Member counts: 10 (110), 11 (71), 15 (126), 16 (72), 20 (137), 21 (64).

Distributional facts about the posed designs (from `manifest.jsonl`):

| quantity | p10 | p50 | p90 |
|---|---|---|---|
| initial mass / `maximum_mass` | 0.62 | 0.95 | 1.50 |
| `certified_mass` (free aspect) / fixed-aspect optimum | 0.52 | 0.56 | 0.67 |

**43.1% of the batch starts already over the mass budget.** No amount of scaling
material *up* can fix those; the agent has to take material off the inflated
members first. That is the mechanical reason a single global scale-up dies, and
it is a property the previous batch had on 1/250 of its problems and the
originals on 0/89 of their finite mass-family problems (measured
initial-mass/budget: previous gen p10/p50/p90 = 0.42/0.62/0.82, max 1.08;
originals 0.44/0.65/0.76, max 0.88).

Each spec carries, in `_metadata`:

```json
"optimal_mass": 94.76, "certified_mass": 67.69, "certified_aspect_min": 0.04,
"fixed_aspect_mass": 131.59, "mass_slack": 1.4, "stratum": "s3",
"degradation": {"removed": 1, "jittered": 1, "n_down": 5, "n_up": 4,
                "down_ids": [...], "up_ids": [...], "min_fos": 0.022},
"difficulty": "hard",
"policy_solved": {"greedy_critical": false, "lookahead_v2_d1": false,
                  "lookahead_v2_d2": true}
```

`difficulty` is written by `scripts/label_difficulty.py` from the measured ladder
results, never from the generation knobs:

* **easy** — `greedy_critical` solves it (classical sizing is enough)
* **medium** — greedy fails, `lookahead_v2_d1` solves it (depth-1 search is the value)
* **hard** — d1 fails, `lookahead_v2_d2` solves it (needs multi-step lookahead)
* **open** — no ladder policy solves it, though V4 holds a certificate that a
  feasible design exists inside the declared bounds and under the budget

---

## 7. Is the global action still the answer? — the diagnostics

Two independent measurements, both on all three sets.

### 7.1 Search argmax (`lookahead_diagnostics.jsonl`, written by `search_ladder.py`)

At every state the greedy policy visits, the diagnostic scores every candidate
sizing action and records whether the argmax is the global all-member
`SCALE_MULTI_PARAM`.

| set | Φ_v1 global argmax | Φ_v2 global argmax | states |
|---|---|---|---|
| originals (130) | 98.4% | **81.9%** | 641 |
| previous generated (250) | 97.3% | **48.4%** | 4 086 |
| **new batch (580)** | 96.7% | **20.9%** | 10 679 |

Φ_v2 (the goal-aligned Lagrangian the ladder actually searches with) drops
81.9% → 20.9%. **Φ_v1 barely moves (98.4% → 96.7%), and that is a property of
Φ_v1, not of the problems**: v1 prices no mass cap at all and a phantom
0.01 m deflection limit, so "make everything bigger" is always its argmax
whatever the problem is. Read the v2 column; the v1 column is a control that
shows the diagnostic is sensitive to the potential's specification.

### 7.2 One-shot global reachability (`one_global_action_probe.py`)

The blunt version: does **any** single uniform rescale reach feasibility?

| set | declared bounds | physical bounds |
|---|---|---|
| originals, all 130 | 0.608 | 0.700 |
| originals, 99 non-degenerate | **0.798** | **0.919** |
| previous generated (250) | 0.056 | 0.380 |
| **new batch (580)** | **0.067** | **0.181** |
| new batch, 456 problems outside the deliberately-easy s0/s1 | **0.000** | — |

All 39 declared-bounds hits in the new batch sit in s0 (26/41) and s1 (13/83),
the two strata included precisely to keep easy content in the set. **Zero of the
456 problems in s2 and harder fall to one global action.**

Honest wrinkle: on the *declared-bounds* metric the previous batch (0.056) is
already as low as the new one (0.067), so this probe is not what separates them.
What separates them is (i) the physical-bounds metric, 0.380 → 0.181; (ii) the
Φ_v2 argmax, 48.4% → 20.9%; and (iii) decisively, the solve rates in §5 —
`lookahead_v2_d1` falls from 1.000 to 0.734 and depth-2 starts to matter.

---

## 8. Caveats and things that did not work

1. **`optimal_mass` is not the true optimum, and never was.** It is
   `certified_mass × mass_slack`, an *attainable* mass (a certified design exists
   at or below it) but not a proven minimum — exactly the semantics the originals
   have, where `optimal_mass` sits ~1.7× above what a free-aspect optimiser
   reaches. Anything that reads `mass/optimal_mass` as "times the true optimum"
   is over-reading it on every DesignBench truss set, old or new.
2. **The aspect floor `t/r ≥ 1/25` is a modelling choice, not physics from the
   simulator.** trussme does not model local wall buckling, so an unfloored
   optimiser drives every member to a 1 mm wall on a 150 mm pipe and the
   "optimum" becomes an exploit. 1/25 keeps sections in the range the originals
   occupy (their `t/r` p10–p90 is 0.083–0.167). Raising the floor loosens every
   budget; lowering it tightens them. It is a CLI flag (`--aspect-min`) and it is
   recorded per problem in `_metadata.certified_aspect_min`.
3. **A search policy can still legally beat `certified_mass`** by going below the
   aspect floor, since the floor constrains the generator, not the agent. The
   budget is therefore not a hard wall against thin-wall exploitation; it is
   tight enough that uniform scaling fails, which is what the fix needed.
4. **3.6% of the batch (21/580) starts with a deflection above 0.15 m**
   (p50 0.019, p90 0.078, max 0.717). These are floppy-but-valid structures: the
   posed problems state no deflection goal, and every one of them is under the
   repo's own mechanism bound of `0.1 × span`. The originals' clean problems top
   out at 0.106 m, so this is a genuine distribution shift at the tail. If that
   matters for a downstream use, filter on
   `_metadata` / re-run the validator with a tighter `DEFLECTION_SANE_M`.
5. **s5 (`mass_slack` 1.05) was generated and discarded.** 90% of its problems
   are solved by no policy in the ladder. Tightening the budget past s4 stops
   producing harder *design* problems and starts producing problems whose only
   solution is the thin-wall exploit of caveat 3.
6. **Depth-2 lookahead buys nothing on the reference sets.** d2 = d1 exactly on
   the originals (0.762/0.762) and on `problems_gen` (1.000/1.000). The `hard`
   stratum in this batch is the first place in this repo where the extra depth
   changes an outcome, and it does so on a minority of problems — do not read the
   `hard` count as a general statement about search depth.
7. **The difficulty labels are policy-relative and budget-relative.** They are
   defined by three fixed non-LLM policies at 20 steps with `alpha=5.0, tau=0.05`.
   Change the step budget, the potential, or the action set and the labels move.
   `_metadata.policy_solved` records the raw outcomes so a relabel needs no re-run.
8. **The one-global-action probe is an upper bound on one-move reachability, not
   a policy result.** It sweeps 21 factors per parameter over [0.5, 4] and the
   full 21×21 outer product for the joint sweep — far denser than the search's own
   five factors — so a 0 there is strong evidence, while a hit does not mean any
   policy would find it.
9. **Not checked: whether an LLM finds these problems *readable*.** The specs are
   in the same format and the same numeric ranges as the originals (member radii
   5–50 mm, wall 1–15 mm), and one was read by hand, but no model was asked to
   solve one. The Bedrock API was available and was not used: nothing in the fix
   is a language question.
